# H3-RefReuse：MiniMax-H3 参考复用推理

面向 Ascend A5 的八步 Ref2VA 实验方案，原实验名为 `LoRA-Q8-SP-mix`。
`RefReuse` 直接表达“跨去噪步复用参考信息”，避免把旧名中的 Q8 误读为 INT8 量化。
本分支冻结 D045 的实现基底 `5917730a4b02aedb44d374173cdc1b10484ccd52`，提供可直接拉取的完整代码。
**本次不包含 RBS（参考图像块选择）、Ref 保留率调节或后续稀疏率调度。**

## 优化点

- **参考 KV 跨步复用**：第 0 步在 50 个主 DiT 层捕获视觉参考 K/V，第 1–7 步复用。
  缓存放在 NPU，复用时移除参考行的投影、MLP 等计算，保留非参考 query 对完整参考 KV 的访问。
- **短 Q / 长 KV 的矩形 attention**：直接调用 A5 RainFusion/Eagle 算子，避免补虚拟参考 Q；
  沿用原 KV 排列、128-token 块及保护块语义。稀疏率为 0.8，mix 使用 INT8 Q/K、FP8 V，
  Transformer 另用 MXFP8；参考 KV 池本身仍为 BF16。
- **序列并行对齐**：compact 序列只在通信阶段补齐，attention 前剥离补齐行，输出后恢复。
  限纯 Ulysses，ring degree=1；不把补齐行作为真实 attention key。
- **降低缓存工程开销**：参考池 view、刷新直接写池、复用拼接缓冲和请求内静态 mask。
- **保留八步 LoRA**：v4-600 EMA，strength=1，259 对 BF16 因子，保留连续 Q/K/V 顺序。
  LoRA 采用运行时 hook；它自身有额外开销，不能将少步数收益或 LoRA 收益归给参考缓存。

参考表示随目标分支变化，因此跨步复用属于近似推理。NPU 保存全层参考 KV 会增加 HBM 占用，
这不是省显存方案。缓存对象是视觉参考行，不默认包含参考音频。

## D045 实验结果

8×Ascend 950PR，短边 768，24 fps，8 步；每例每方案 1 次预热 + 3 次正式，正式中位数如下。
对照保留相同 LoRA、MXFP8 和 mix，只关闭参考复用。

| 用例 | 参考输入 | 请求时长 s | LoRA 对照 E2E s | H3-RefReuse E2E s | 耗时减少 | 对照 / 本方案 DiT s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 图 | 4 | 31.512510 | 25.990828 | 17.52% | 22.376327 / 16.841669 |
| 2 | 2 图 + 音频 | 8 | 18.627391 | 18.187716 | 2.36% | 13.192239 / 12.752174 |
| 3 | 5 图 + 音频 | 11 | 30.394250 | 29.497199 | 2.95% | 22.217832 / 21.387640 |
| 4 | 6 图 + 音频 | 13 | 40.010846 | 38.577943 | 3.58% | 30.109574 / 28.648701 |
| 5 | 3 图 + 音频 | 15 | 58.034483 | 53.213185 | 8.31% | 44.628986 / 39.901605 |
| 6 | 2 图 + 视频 | 9 | 55.416239 | 45.161609 | 18.50% | 38.378190 / 28.232789 |

六例四方案共 96/96 请求成功；本方案占 24 次。**这是单 seed、分块顺序、每组 n=3 的历史数据，
未采集 msProf，D045 画质仍待人工评价。** 相对不带 LoRA 的 OFF-better，本方案仅例 1/5/6 更快，
不能宣称全面加速。完整四方案、逐次结果和官方 18 字段对应关系见 [结果说明](RESULTS.md)、
[benchmark.csv](benchmark.csv) 和 [results.json](results.json)。

## 拉取及启动

```bash
git clone --branch h3-ref-reuse --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
python recipes/h3_ref_reuse/verify.py
```

准备兼容的 Ascend 容器、完整模型、LoRA 和输入文件，版本和安装步骤见 [复现环境](ENVIRONMENT.md)。
下面脚本要求 `IMAGE` 已安装匹配的 vLLM/Omni/MindIE-SD 依赖，源码挂载到镜像原 editable 路径。

```bash
export IMAGE=your-compatible-ascend-image
export MODEL_DIR=/absolute/path/MiniMax-H3
export LORA_PATH=/absolute/path/minimax_h3_turbo_v4_step600_ema.safetensors
export INPUT_DIR=/absolute/path/reference-media
export DEVICES=0,1,2,3,4,5,6,7
bash recipes/h3_ref_reuse/run_container.sh 2>&1 | tee server.log
```

另开终端，保留同一 `INPUT_DIR`，等待服务健康后按同一输入各运行 1 预热 + 3 正式：

```bash
curl --fail http://127.0.0.1:19136/health
bash recipes/h3_ref_reuse/request.sh recipes/h3_ref_reuse/requests/case01.json warmup.mp4
for n in 1 2 3; do
    bash recipes/h3_ref_reuse/request.sh recipes/h3_ref_reuse/requests/case01.json "measure-$n.mp4"
done
```

六例 prompt、参数和参考顺序均在 [requests](requests)。按 [inputs.json](inputs.json) 的相对路径
准备媒体，并核对 SHA256；输入文件、模型和输出视频不在本仓库分发。没有原媒体时，可以改用自己的
参考和 prompt 验证推理流程，但不能称为逐输入重现 D045 数字。

比较时分别停止服务，再用 `USE_REF_CACHE=0` 启动同 LoRA 对照；`USE_LORA=0` 可关闭 LoRA。
每种配置采用独立进程/容器生命周期，不进行常驻 worker 热切换。4 卡可设 `DEVICES=0,1,2,3`，
但不是 D045 同环境复现；低容量卡及大参考输入可能 OOM，不自动降规格或卸载。

## 验证和来源

[验证记录](VALIDATION.md) 区分本次 CPU/静态验证与历史 NPU 实测。
源码是在 D045 实际挂载 15 文件基础上整理 SPDX、导入、格式及类型注解，并删除 return 后不可达的旧分支；
推理计算路径不变，保留研究 helper 名称和路由计数，
便于核对历史日志。[原文件 SHA](source_original.sha256) 与 [发布文件 SHA](source_published.sha256) 分开保存。

基础 Ref KV 机制源自 [vLLM-Omni PR #8072](https://github.com/vllm-project/vllm-omni/pull/8072)；
LoRA 来自 [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)；
矩形路径使用 MindIE-SD 现有算子，未新写 CANN 内核。本次代码和文档整理使用 Codex 辅助。
