# LQ-Slim

MiniMax-H3 Ref2VA 的八步 LoRA、参考 KV 复用、矩形 mix attention 和分步稀疏配置。
在 H3-RefReuse-RBS 上，将参考图候选块保留率设为 10%，前两步稀疏率为 0.8，后六步为 0.85。
完整 Q 和复用步的矩形 Q 使用相同的逐步调度，保护块规则保持。

## 安装

准备已有的 Ascend A5 推理环境，安装
[配套 MindIE-SD RBS 版本](../h3_ref_reuse/ENVIRONMENT.md#mindie-sd-rbs)，
再按常规源码安装方式安装本分支：

```bash
git clone --branch codex/lq-slim --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

MindIE-SD 继续使用原 RBS 的 3.2.0 Python 修改和已有原生算子，LQ-Slim 不新增 MindIE-SD 修改。
完整模型使用 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)，LoRA 使用
[v4-step600 EMA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)。
LoRA 文件和依赖版本见[环境说明](../h3_ref_reuse/ENVIRONMENT.md)。

## 启动服务

替换模型和 LoRA 路径。四卡需在启动前通过 Ascend 的常规设备配置选择：

```bash
export MODEL=/path/to/MiniMax-H3
export RESEARCH_LARRY_LORA_PATH=/path/to/minimax_h3_turbo_v4_step600_ema.safetensors
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "$MODEL" --omni --task-type ref2va \
    --host 127.0.0.1 --port 9098 --trust-remote-code \
    --num-gpus 4 --usp 4 --ring 1 --text-encoder-tp-size 4 \
    --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 4 \
    --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}' \
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"step_sparsities":[0.8,0.8,0.85,0.85,0.85,0.85,0.85,0.85],"ref_keep_ratio":0.1,"precision":"mix","start_step":0,"end_step":0}}}' \
    --kv-offload-config '{"mode":"on","kv_refresh_interval":8,"kv_host_quantization":"none","kv_storage":"npu"}'
```

显存不足时，可在同一命令后追加现有模块卸载配置：

```bash
--diffusion-offload-config '{"mode":"module","components":["text_encoder"],"pin_memory":true}'
```

它将文本编码器权重卸载到 CPU；DiT 和 BF16 参考 KV 仍保留在 NPU。
RoPE 提前准备会激活已有卸载钩子，避免位置张量与频率 buffer 的设备不一致。
参考块筛选改变 attention 连接，不缩小参考 KV 池或序列并行通信。

## 发送请求

服务就绪后使用标准同步视频接口，替换 prompt 和参考图路径：

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

多个参考文件按 prompt 的编号顺序重复传入 `input_references`。

## 配置语义

| 字段 | LQ-Slim 配置 | 含义 |
| --- | --- | --- |
| `step_sparsities` | `[0.8,0.8,0.85,0.85,0.85,0.85,0.85,0.85]` | 零基去噪步的稀疏率；长度必须匹配请求步数 |
| `sparsity` | `0.8` | 不设置调度时使用的标量；`0` 仍关闭稀疏 |
| `ref_keep_ratio` | `0.1` | 每图、每头、每个合格目标 Q 块独立选取的参考图候选块比例 |
| `precision` | `mix` | Q/K 使用 INT8，V 使用 FP8；输入和参考 KV 为 BF16 |
| `kv_refresh_interval` | `8` | 首步刷新参考 KV，后七步复用；不表示 INT8 缓存 |

删除 `step_sparsities` 即恢复固定稀疏率，设 `ref_keep_ratio=0.2` 即使用原 RBS 对照工作点。
调度每项必须为有限数且严格位于 `(0,1)`；空列表等同于不设置调度。
稀疏调用缺少步号、步号越界或总步数不匹配会报错；无稀疏几何的 dense 调用不读取调度。
首尾 dense 步、跳过层、首帧和边界保护规则继续生效。

0.8/0.85 是全部 pooled K 列上的标称丢弃比例，保护连接和 RBS 筛选会改变实际掩码。
参考图 10% 仅作用于完全属于该图的合格整块，向上取整且至少保留一块；
混合块、参考视频、参考音频和受保护 query 行不按此比例裁剪。

## 验证范围

基础配置曾在 4×Ascend 950DT、84 GiB/卡、ARM64、MindIE-SD 3.2.0 上，
以原六个 Ref2VA 输入分别执行每方案 1 预热＋2 正式；RBS 与 LQ-Slim 共 36 请求成功。
两方案均为八步、MXFP8、SP4、ring1、文本 TP4、VAE tile4；第六例统一文本编码器 CPU 卸载。
未采集 msProf，精度、画质及音画同步尚未验收。

本分支将相同调度和 RoPE 放置逻辑整理到已发布 RBS 源码上；整理后尚未重新运行 NPU 端到端。
CPU 检查覆盖配置往返、非法调度、完整/矩形 Q 八步分发、dense 路由、RoPE 钩子顺序，
以及既有 RBS 几何、参考 KV 复用和 SP 补齐，共 51 项通过。
CPU 检查的原生 attention 入口使用调用桩，不证明设备算子或输出画质正确。

依赖 Python 3.12、pytest 和 CPU PyTorch，最小本地检查：

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py -q
```

安装完整仓库测试依赖后，按现有 CI 标记执行：

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py -m 'core_model and cpu' --run-level core_model -q
```
