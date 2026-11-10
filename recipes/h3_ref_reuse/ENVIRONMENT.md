# 环境与依赖

使用已有的 Ascend A5 推理环境，按仓库的
[NPU 安装指南](../../docs/getting_started/installation/npu.md) 安装依赖。

## H3-RefReuse 验证环境

| 项目 | 版本 / 配置 |
| --- | --- |
| 硬件 | 8×Ascend 950PR，每卡 128 GiB HBM |
| 系统 / Python | Linux x86_64 / Python 3.12.13 |
| 驱动 / CANN | 25.9.rc1.8 / 9.1.0 |
| vLLM | `0.32.0+empty` |
| vLLM-Ascend | `0.19.1rc2.dev2959+g450e55698.d20261020` |
| PyTorch / torch_npu | `2.10.0+cpu` / `2.10.0.post4` |
| MindIE-SD | 3.2.0，具备多 span 和 Eagle 接口 |
| Transformers / Diffusers | 5.19.0 / 0.40.0 |
| 并行 | USP8、ring1、文本编码 TP8、VAE tile/patch8；DiT TP1 |

以上版本来自性能测试环境。其他依赖组合需自行验证兼容性。

RBS 开关对照使用 Linux ARM64、4×Ascend 950DT（84 GiB/卡）、驱动 25.6.rc2.b001、
Python 3.12.13、CANN 9.1.0、PyTorch `2.10.0+cpu`、torch_npu `2.10.0.post4`、
MindIE-SD 3.2.0；并行为 USP4、ring1、文本编码 TP4、VAE tile/patch4、DiT TP1。
结果见 [RBS 对照](RESULTS.md#rbs-开关对照)。

## 模型与算子

- 使用 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 的完整模型及依赖权重。
- LoRA 使用 [MiniMax-H3-Turbo-LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) 的
  `minimax_h3_turbo_v4_step600_ema.safetensors`，SHA256 为
  `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`。
- MindIE-SD 的 `sparse_attention` 需接受 `precision` 和 `video_spans`，
  并包含 rf_v2 多 span helper、rf_v3 量化接口与原生
  `eagle_quant_block_sparse_attention` 算子；RBS 还需以下配套修改。

## MindIE-SD RBS

使用 [MindIE-SD RBS 分支](https://gitcode.com/qq_48159150/MindIE-SD/tree/rbs-experiment-source)，
固定版本为 `bbf372dabee190c72826b0d52077dc2e00b79be2`，基于官方 MindIE-SD 3.2.0。
在已具备上述 MindIE-SD 3.2.0 和原生算子的环境中，安装两份 RBS Python 文件：

```bash
git clone --branch rbs-experiment-source https://gitcode.com/qq_48159150/MindIE-SD.git
cd MindIE-SD
git checkout bbf372dabee190c72826b0d52077dc2e00b79be2
MINDIESD_DIR=$(python -c 'from importlib.metadata import distribution; print(distribution("mindiesd").locate_file("mindiesd"))')
cp mindiesd/layers/flash_attn/ref_sparse_mask.py "$MINDIESD_DIR/layers/flash_attn/"
cp mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py "$MINDIESD_DIR/layers/flash_attn/"
```

需要构建 wheel 时，使用该版本的
[安装说明](https://gitcode.com/qq_48159150/MindIE-SD/blob/bbf372dabee190c72826b0d52077dc2e00b79be2/reproduction/rbs/README.md#环境和安装)。
上述两文件安装方式保留已有原生库；新构建的原生库需另做 NPU 回归。
缺少 RBS helper 时，`ref_keep_ratio < 1` 会报错，不会静默忽略参数。

## 配置范围

当前配置面向 Ref2VA、batch=1、BF16 激活、head_dim=128、纯 Ulysses、DiT TP=1。
参考 KV 在 NPU 上以 BF16 保存，不启用 CPU 卸载。
缓存约占 `2 × 层数 × 参考行数 × 每 rank heads × head_dim × BF16 字节数`，
输入规模需与可用显存匹配。启动和请求参数见 [快速启动](README.md)。
