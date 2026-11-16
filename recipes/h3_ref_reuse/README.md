# H3-RefReuse-RBS 快速启动

MiniMax-H3 Ref2VA 的八步 LoRA、NPU 参考 KV 复用、mix attention 和参考图片块筛选配置。
使用已有的 Ascend A5 推理环境，依赖版本和配套 MindIE-SD 安装见 [环境说明](ENVIRONMENT.md)。

## 优化点

- **八步 LoRA**：加载 MiniMax-H3 Turbo v4-600 EMA adapter，以八步完成去噪。
- **参考 KV 复用**：首步计算各层参考 K/V，以 BF16 保留在 NPU，后七步复用。
- **跳过参考行计算**：复用步仅对非参考行执行投影和 MLP，attention 继续读取完整 K/V。
- **mix attention**：Q/K 使用 INT8，V 使用 FP8。图片稀疏开启时补 dummy 参考 Q；关闭时使用矩形路径。
- **序列并行对齐**：仅为通信补齐序列，进入 attention 前移除补齐行，计算后恢复。
- **参考块筛选**：对每张参考图、每个注意力头和合格视频 query 块分别选取高分参考 key 块，
  保留文本、音频、混合边界块及受保护的 query 连接；首步和后续复用步均启用。

旧RBS的逐例数据、对照条件和测量范围见 [历史对比结果](RESULTS.md)，新版接口尚未重新执行NPU验证。

## 安装

安装 [新版 MindIE-SD 图片稀疏接口](ENVIRONMENT.md#mindie-sd-图片稀疏接口)，
再按 vLLM-Omni 的常规源码安装方式使用本分支：

```bash
git clone --branch lq-slim --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

## 启动服务

以下配置可直接写入现有的服务启动脚本。替换模型和 LoRA 路径：

```bash
export MODEL=/path/to/MiniMax-H3
export RESEARCH_LARRY_LORA_PATH=/path/to/minimax_h3_turbo_v4_step600_ema.safetensors
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "$MODEL" --omni --task-type ref2va \
    --host 127.0.0.1 --port 9098 --trust-remote-code \
    --num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8 \
    --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8 \
    --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}' \
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":0,"end_step":0,"image_sparsity":0.5}}}' \
    --kv-offload-config '{"mode":"on","kv_refresh_interval":8,"kv_host_quantization":"none","kv_storage":"npu"}'
```

- `--kv-offload-config` 启用参考 KV 复用；`kv_storage=npu` 将 BF16 参考缓存保留在 NPU，
  `kv_refresh_interval=8` 对应八步请求的首步计算、后七步复用。
- `RESEARCH_LARRY_LORA_PATH` 指向
  [v4-600 EMA LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)。
  该 adapter 由本分支加载，strength 为 1。
- 删除 `--kv-offload-config` 即关闭参考复用；取消 LoRA 环境变量即可关闭 LoRA。

## 调整图片稀疏率

修改 `block_sparse.image_sparsity`，无需额外环境变量：

| 参数值 | 行为 |
| --- | --- |
| `0.2`、`0.5`、`0.8` | 分别删除每张参考图候选块的20%、50%、80% |
| `0.0`（默认） | 关闭图片筛选 |

范围为 `[0,1]`，保留块数向上取整且至少一块，包括删除率1的情况。
图片按独立区域分块，所有视频区域的合格查询行参与选择，首帧及边界保护继续生效。
比例不裁剪参考KV缓存，也不等于整个attention mask的保留比例。
开启图片筛选的复用步补齐dummy参考Q并丢弃参考输出；关闭图片筛选保留矩形路径。
历史RBS性能数据对应旧分块及旧接口，不能视为当前实现的性能验证。

原始无 RBS 版本保留在
[`h3-ref-reuse`](https://github.com/march7th-26/vllm-omni/tree/h3-ref-reuse) 分支。
对比时保持硬件、权重、请求和其他参数一致；四卡运行时将 `--num-gpus`、`--usp`、
`--text-encoder-tp-size`、`--vae-patch-parallel-size` 同时改为 `4`，并确认输入适合可用显存。

## 发送请求

服务就绪后，使用标准同步视频接口。替换参考图路径和 prompt：

```bash
curl --fail-with-body http://127.0.0.1:9098/v1/videos/sync \
    --form-string 'prompt=Use the object in <Picture 1> as the subject. A steady close-up in soft studio lighting, with a slow camera push-in and quiet room ambience.' \
    --form-string 'short_edge=768' \
    --form-string 'aspect_ratio=9:16' \
    --form-string 'fps=24' \
    --form-string 'num_inference_steps=8' \
    --form-string 'flow_shift=12.0' \
    --form-string 'audio_flow_shift=3.0' \
    --form-string 'seed=42' \
    --form-string 'extra_params={"task":"ref2va","duration":4}' \
    --form 'input_references=@/path/to/reference.jpg;type=image/jpeg' \
    --output output.mp4
```

生成视频保存为 `output.mp4`。多个参考文件按 prompt 中的编号顺序重复传入 `input_references`。
参考复用属于近似推理，并增加显存占用；当前配置使用纯 Ulysses（ring=1）、DiT TP=1。
