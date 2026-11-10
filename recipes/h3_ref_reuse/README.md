# H3-RefReuse-RBS 快速启动

MiniMax-H3 Ref2VA 的八步 LoRA、NPU 参考 KV 复用、矩形 mix attention 和
RBS（Reference Block Selection，参考块筛选）配置。
使用已有的 Ascend A5 推理环境，依赖版本和配套 MindIE-SD 安装见 [环境说明](ENVIRONMENT.md)。

## 优化点

- **八步 LoRA**：加载 MiniMax-H3 Turbo v4-600 EMA adapter，以八步完成去噪。
- **参考 KV 复用**：首步计算各层参考 K/V，以 BF16 保留在 NPU，后七步复用。
- **跳过参考行计算**：复用步仅对非参考行执行投影和 MLP，attention 继续读取完整 K/V。
- **矩形 mix attention**：短 Q、完整 K/V 直接计算，不补虚拟参考 Q；Q/K 使用 INT8，V 使用 FP8。
- **序列并行对齐**：仅为通信补齐序列，进入 attention 前移除补齐行，计算后恢复。
- **参考块筛选**：对每张参考图、每个注意力头和目标 query 块分别选取高分参考 key 块，
  保留文本、音频、混合边界块及受保护的 query 连接；首步和后续复用步均启用。

RBS 的耗时收益依输入而异，逐例数据、对照条件和测量范围见 [对比结果](RESULTS.md)。

## 安装

安装 [配套 MindIE-SD RBS 版本](ENVIRONMENT.md#mindie-sd-rbs)，
再按 vLLM-Omni 的常规源码安装方式使用本分支：

```bash
git clone --branch h3-ref-reuse-rbs --single-branch https://github.com/march7th-26/vllm-omni.git
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
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":0,"end_step":0,"ref_keep_ratio":0.5}}}' \
    --kv-offload-config '{"mode":"on","kv_refresh_interval":8,"kv_host_quantization":"none","kv_storage":"npu"}'
```

- `--kv-offload-config` 启用参考 KV 复用；`kv_storage=npu` 将 BF16 参考缓存保留在 NPU，
  `kv_refresh_interval=8` 对应八步请求的首步计算、后七步复用。
- `RESEARCH_LARRY_LORA_PATH` 指向
  [v4-600 EMA LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)。
  该 adapter 由本分支加载，strength 为 1。
- 删除 `--kv-offload-config` 即关闭参考复用；取消 LoRA 环境变量即可关闭 LoRA。

## 调整 RBS 比例

直接修改上述 `block_sparse.ref_keep_ratio`，无需额外环境变量：

| 参数值 | 行为 |
| --- | --- |
| `0.2`、`0.5`、`0.8` | 分别保留每张参考图候选块的 20%、50%、80%，向上取整且至少保留一块 |
| `1.0`（默认） | 关闭 RBS，使用完整参考块连接，便于比较 H3-RefReuse |

范围为 `(0, 1]`；RBS 要求 `sparsity > 0`。
比例只作用于完全属于单张参考图的 128-token 候选块，不裁剪参考 KV 缓存，
不等于整个 attention mask 的保留比例；参考视频不按此比例裁剪。
减小比例会改变 attention 连接，需结合自己的输入检查画质和耗时。

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
