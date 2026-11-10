# H3-RefReuse 快速启动

MiniMax-H3 Ref2VA 的八步 LoRA、NPU 参考 KV 复用和矩形 mix attention 配置。
使用已有的 Ascend 950PR 推理环境，依赖版本见 [环境说明](ENVIRONMENT.md)。

## 优化点

- **八步 LoRA**：加载 MiniMax-H3 Turbo v4-600 EMA adapter，以八步完成去噪。
- **参考 KV 复用**：首步计算各层参考 K/V，以 BF16 保留在 NPU，后七步复用。
- **跳过参考行计算**：复用步仅对非参考行执行投影和 MLP，attention 继续读取完整 K/V。
- **矩形 mix attention**：短 Q、完整 K/V 直接计算，不补虚拟参考 Q；Q/K 使用 INT8，V 使用 FP8。
- **序列并行对齐**：仅为通信补齐序列，进入 attention 前移除补齐行，计算后恢复。

六例 LoRA 对照的 E2E 耗时减少 **2.36%–18.50%**，DiT 耗时减少 **3.34%–26.44%**。
配置、逐例数据和测量范围见 [对比结果](RESULTS.md)。

## 安装

按 vLLM-Omni 的常规源码安装方式使用本分支：

```bash
git clone --branch h3-ref-reuse --single-branch https://github.com/march7th-26/vllm-omni.git
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
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":0,"end_step":0}}}' \
    --kv-offload-config '{"mode":"on","kv_refresh_interval":8,"kv_host_quantization":"none","kv_storage":"npu"}'
```

- `--kv-offload-config` 启用参考 KV 复用；`kv_storage=npu` 将 BF16 参考缓存保留在 NPU，
  `kv_refresh_interval=8` 对应八步请求的首步计算、后七步复用。
- `RESEARCH_LARRY_LORA_PATH` 指向
  [v4-600 EMA LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)。
  该 adapter 由本分支加载，strength 为 1。
- 删除 `--kv-offload-config` 即关闭参考复用；取消 LoRA 环境变量即可关闭 LoRA。

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
