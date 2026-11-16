# LQ-Slim 组合版

在[新版 LQ-Slim](../lq_slim/README.md)上增加 head chunk 通算重叠、AudioVAE
Conv1d 改写和本地 sparse attention 的 NPU Graph。当前为待 NPU 验证的实现，
不提供性能或画质结论。接口与执行边界见[融合设计](DESIGN.md)。

## 安装和启动

沿用原有 Ascend A5 环境、模型和 LoRA，MindIE-SD 必须支持新版
`image_spans/image_sparsity`，安装要求见[环境说明](../h3_ref_reuse/ENVIRONMENT.md)。
本分支不要求使用 MindIE 的等长量化通信入口。

```bash
git clone --branch lq-slim-stack --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation

export MODEL=/path/to/MiniMax-H3
export RESEARCH_LARRY_LORA_PATH=/path/to/minimax_h3_turbo_v4_step600_ema.safetensors
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_MINIMAX_H3_AUDIO_TCONV_CONV1D=1

vllm serve "$MODEL" --omni --task-type ref2va \
    --host 127.0.0.1 --port 9098 --trust-remote-code \
    --num-gpus 4 --usp 4 --ring 1 --text-encoder-tp-size 4 \
    --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 4 \
    --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}' \
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"step_sparsities":[0.8,0.8,0.85,0.85,0.85,0.85,0.85,0.85],"image_sparsity":0.9,"precision":"mix","start_step":0,"end_step":0,"head_chunk_size":1,"sparse_graph":true}}}' \
    --kv-offload-config '{"mode":"on","kv_refresh_interval":8,"kv_host_quantization":"none","kv_storage":"npu"}'
```

按常规设备配置选择四张 NPU。显存不足时，沿用原服务的文本编码器卸载配置：

```bash
--diffusion-offload-config '{"mode":"module","components":["text_encoder"],"pin_memory":true}'
```

请求使用原[同步视频接口示例](../lq_slim/README.md#发送请求)。
本配置不启用 msProf。

## 开关和边界

| 配置 | 默认值 | 组合版设置及作用 |
| --- | --- | --- |
| `block_sparse.head_chunk_size` | `0` | `1`：每次处理一个 rank-local head；严格 Ulysses、resident Ref KV、ring1 |
| `block_sparse.sparse_graph` | `false` | `true`：捕获本地稀疏计算；刷新/复用、通信和缓存注入留在图外 |
| `VLLM_OMNI_MINIMAX_H3_AUDIO_TCONV_CONV1D` | `0` | `1`：将 AudioVAE decoder 的普通/分组转置卷积及 alias-free 上采样改为 polyphase Conv1d |

三个开关可独立关闭；关闭后恢复原路径，LoRA、MXFP8、Ref interval8、图片删除率0.9
及视频分步删除率均不随开关变化。通算入口遇到不支持的布局直接报错，不静默关闭 Ref。
没有视觉参考的请求不适用该通算入口。

通信采用 BF16，attention 仍为本地 mix。它与 MindIE 原生 INT8/FP8 量化通信方案不同，
也不包含整个 DiT block 的 Graph。新版图片路径继续补 dummy Q，尚未实现真实矩形图片稀疏。
chunk 内量化的归约范围可能不同，Conv1d 的浮点累加顺序也会变化；精度、画质和音频质量待评。

Graph 在同一 worker 的同一设备、线程和计算流上共享，最多保留三个几何/精度/稀疏率条目。
每个新条目第一次调用走 eager，第二次捕获并 replay。几何切换会淘汰旧条目；
首次请求可能包含捕获开销。Graph 捕获失败直接向上传递，不能视作已验证支持。

## 本地验证

CPU 数学、配置、Ref 缓存与 Graph 控制流测试：

```bash
python -m pytest --noconftest -o addopts='' \
    tests/diffusion/models/minimax_h3/test_lq_slim_stack.py \
    tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py \
    tests/diffusion/models/minimax_h3/test_reference_block_selection.py -q
```

这些检查不执行 NPU collective、原生 sparse kernel 或真实 Graph。
组合版尚未运行服务器实验，与 OFF-better-LoRA 的收益对比仍待同条件业务测试。
