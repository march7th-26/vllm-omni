# 复现环境与依赖

## D045 实测环境

| 项目 | 版本 / 配置 |
| --- | --- |
| 硬件 | 8×Ascend 950PR，每卡 131072 MiB HBM；物理 0–7 |
| 宿主 | Linux x86_64，Debian 12，kernel 5.15.152.ve.13-amd64 |
| 驱动 | 25.9.rc1.8；ascendhal 7.35.23 |
| 容器 | Linux amd64；Ubuntu 22.04 标签；Python 3.12.13 |
| 原镜像身份 | `sha256:c02dc466c75fef4e79aee0a602288a219441ba9a0390589520355d03cb653ffa` |
| Omni 实验基底 | `5917730a4b02aedb44d374173cdc1b10484ccd52` + 本分支的 D045 源码 |
| vLLM | `0.32.0+empty` |
| vLLM-Ascend | `0.19.1rc2.dev2959+g450e55698.d20261020`，镜像内开发构建 |
| PyTorch / torch_npu | `2.10.0+cpu` / `2.10.0.post4` |
| CANN / MindIE-SD | 9.1.0 / 3.2.0（需下述多 span 和 Eagle 接口） |
| Transformers / Diffusers | 5.19.0 / 0.40.0 |
| 并行 | USP8、ring1、文本编码 TP8、VAE tile/patch8；DiT TP1 |
| 模型配置 | 完整 MiniMax-H3、Transformer MXFP8；无 DLO/CPU 卸载、Cache-DiT 或 TaylorSeer |
| 请求 | 8 步，seed 580331635，flow shift 12、audio shift 3；参数见 requests |
| attention | mix，sparsity 0.8，start_step=0/end_step=0，八步都启用稀疏路径 |
| 参考池 / LoRA | NPU BF16、interval8 / v4-600 EMA、strength1 |

vLLM、vLLM-Ascend、Transformers 和 Diffusers 的版本取自归档容器的包元数据；
硬件、驱动和宿主取自当次/紧邻迁移归档。原镜像身份用于核对历史环境，不保证今天可从公开 registry
拉取；本次没有测试镜像下载。不同镜像、原生库或驱动下需重新验证，不能仅凭包版本相同认定同环境。

## 安装与启动前核对

1. 使用已匹配的 Ascend A5 驱动、CANN、torch/torch_npu 和 vLLM-Ascend 环境，保留原生扩展。
   项目通用步骤见 [NPU 安装指南](../../docs/getting_started/installation/npu.md)。
   该指南中的旧版通用镜像不能替代上表的 D045 环境。
2. 本分支是完整源码。若环境尚未安装此 checkout，在容器内的可写源码目录执行：

   ```bash
   VLLM_OMNI_TARGET_DEVICE=npu pip install --no-deps --no-build-isolation -e .
   ```

   `run_container.sh` 采用原镜像的 `/vllm-workspace/vllm-omni` editable 路径；其他安装布局需调整挂载，
   然后核对 `vllm_omni.__file__`，避免导入镜像里的另一份代码。
3. MindIE-SD 需要多 span 排列、显式 `precision`、rf_v3 和 Eagle 算子。可获取的非 RBS 源码锚点为
   [8701dabb](https://github.com/march7th-26/MindIE-SD/tree/8701dabb0aba4ba52d3cf81d8798c73abaa187df)，
   按该版本的 [安装指导](https://github.com/march7th-26/MindIE-SD/blob/8701dabb0aba4ba52d3cf81d8798c73abaa187df/docs/zh/installation.md)
   为当前 CANN/PyTorch 编译安装。此 revision 位于 RBS 提交之前；无需应用 RBS 补丁。
   它提供所需 Python 接口，**本次未将该 revision 新构建的 wheel 重新跑 NPU**；
   D045 的原生二进制来自历史镜像，不能把接口兼容等同于二进制或性能复现。
4. 准备 [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) 的完整 diffusers 格式模型和依赖权重，
   不是只有一个 DiT 文件。另从 [LoRA 发布页](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
   下载 `minimax_h3_turbo_v4_step600_ema.safetensors`（779849816 字节），SHA256 必须为
   `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`。
5. 在实际推理容器中、此 checkout 根目录执行：

   ```bash
   python recipes/h3_ref_reuse/verify.py --runtime
   npu-smi info
   ```

   检查包含源码 SHA、实际 Omni 导入位置、A5、MindIE 私有 helper 和原生 Eagle 注册状态。
   它不加载图片/视频，不生成模型请求，也不替代多 rank HCCL 和算子数值验证。
   启动脚本清除继承的 msProf 变量，仅启用 pipeline 标量计时。

运行后应从日志核对每 rank 的 259 LoRA 目标、八步 1 刷新/7 复用、
`full_query_mix_success=50`、`rectangular_mix_success=350`、虚拟参考 Q 为 0。
实际日志字段见 `research_larry_lora.py`、`denoise_loop.py` 和 RainFusion 后端。
SP padding 的数量由输入与并行度决定，不要求每例非零。

## 容量和适用范围

仅声明本配置下的 Ref2VA、batch1、BF16 激活、head_dim128、A5 BF16/mix 矩形路径。
不扩展到 Ring、DiT TP>1、任意 adapter、动态切换或其他 Ascend 型号。
参考 KV 存储量约为 `2 × 50层 × 参考行数 × 每rank heads × head_dim × BF16字节数`。
D045 本方案各例离散设备采样最大 HBM 为 67769/67028/67311/67474/67833/74427 MiB，
包含设备/进程开销，不是 allocator 峰值，也不是常驻权重显存。
