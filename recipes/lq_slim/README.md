# LQ-Slim

MiniMax-H3 Ref2VA 的八步 LoRA、参考 KV 复用、mix attention 和分步稀疏配置。
参考图使用独立 image spans，每图候选块标称保留 10%，视频前两步删除率为 0.8，后六步为 0.85。
开启图片稀疏时，复用步在 Ulysses 后补 dummy 参考 Q，走新版等长 Q/K/V 路径并丢弃参考输出。
参考投影与参考 KV 仍复用；关闭图片稀疏时保留原矩形路径。

通算重叠、AudioVAE 和局部 NPU Graph 的可选组合见[组合版说明](../lq_slim_stack/README.md)。

## 安装

准备已有的 Ascend A5 推理环境，安装
[支持新版图片接口的 MindIE-SD](../h3_ref_reuse/ENVIRONMENT.md#mindie-sd-图片稀疏接口)，
再按常规源码安装方式安装本分支：

```bash
git clone --branch lq-slim --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

MindIE-SD 的公共 `sparse_attention` 需显式支持 `image_spans/image_sparsity`，
且 rf_v2/rf_v3 使用匹配的图片重排与选块实现。按该源码版本的常规安装方式安装，
原 RBS 两文件补丁不满足此接口。
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
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"step_sparsities":[0.8,0.8,0.85,0.85,0.85,0.85,0.85,0.85],"image_sparsity":0.9,"precision":"mix","start_step":0,"end_step":0}}}' \
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
| `image_sparsity` | `0.9` | 每图、每头、每个合格视频 Q 块删除约90%的参考图候选块，保留约10% |
| `precision` | `mix` | Q/K 使用 INT8，V 使用 FP8；输入和参考 KV 为 BF16 |
| `kv_refresh_interval` | `8` | 首步刷新参考 KV，后七步复用；不表示 INT8 缓存 |

删除 `step_sparsities` 即恢复固定视频稀疏率；`image_sparsity=0` 关闭图片筛选。
`image_sparsity` 范围为 `[0,1]`，表示删除率；`0.8` 对应标称保留20%，但新分块不等价于旧RBS。
调度每项必须为有限数且严格位于 `(0,1)`；空列表等同于不设置调度。
稀疏调用缺少步号、步号越界或总步数不匹配会报错；无稀疏几何的 dense 调用不读取调度。
首尾 dense 步、跳过层、首帧和边界保护规则继续生效。

0.8/0.85 是全部 pooled K 列上的标称丢弃比例，保护连接和图片筛选会改变实际掩码。
参考图 10% 作用于独立图片区域的完整候选块，向上取整且至少保留一块；
即使 `image_sparsity=1` 也至少保留一块。尾块、混合块及受保护 query 行不裁。
图片筛选作用于所有 `video_spans` 的合格查询行，包括参考视频；音频和图片查询行保持保护。

## 验证范围

基础配置曾在 4×Ascend 950DT、84 GiB/卡、ARM64、MindIE-SD 3.2.0 上，
以原六个 Ref2VA 输入分别执行每方案 1 预热＋2 正式；RBS 与 LQ-Slim 共 36 请求成功。
两方案均为八步、MXFP8、SP4、ring1、文本 TP4、VAE tile4；第六例统一文本编码器 CPU 卸载。
未采集 msProf，精度、画质及音画同步尚未验收。

上述36请求使用旧RBS实现，不能作为新版图片稀疏的性能或画质结论。
新版CPU回归覆盖图片布局及缓存压缩/恢复、配置与非法参数、完整/复用步八步调度、
dummy Q 裁剪、关闭图片稀疏时的矩形路由、Ref KV复用、SP补齐、实际MindIE图片分块/动态mask与公共参数转发。
原生attention入口使用调用桩；新版尚未运行NPU端到端、性能或质量验收。

依赖 Python 3.12、pytest 和 CPU PyTorch，最小本地检查：

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py -q
```

安装完整仓库测试依赖后，按现有 CI 标记执行：

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py -m 'core_model and cpu' --run-level core_model -q
```

匹配MindIE源码的跨库CPU检查（执行真实重排、pooling和mask，省略NPU导入）：

```bash
MINDIE_SD_SOURCE=/path/to/MindIE-SD python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_image_sparse_mask.py -q
```
