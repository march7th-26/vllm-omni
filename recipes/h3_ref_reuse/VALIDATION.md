# 发布验证记录

2026-11-10；个人 fork 的研究复现分支，基底 `5917730a4b02aedb44d374173cdc1b10484ccd52`。
目标是让其他使用者取得独立于 RBS 的完整实现、明确的配置和可核查的历史性能记录。
本次未向上游提交 PR，未启动新的服务器实验。

## 本次检查

| 检查 | 结果及覆盖范围 |
| --- | --- |
| 原源码身份 | D045 实际挂载的 15 文件 SHA256 全部核对；原/发布 SHA 分开保存 |
| 有效计算路径 | 15 文件经语法树核对一致，核对时仅归一导入、恒等 typing.cast、两处变量注解及 return/raise 后不可达语句 |
| CPU 行为 | 19 项通过：八步缓存内容与清理、未完成捕获的请求清理、带污染补齐行的通信布局、配置拒绝规则 |
| 启动/请求参数 | 12 项通过：四种 LoRA/Ref 开关、两组设备参数、六例 multipart 字段及参考顺序，与 D045 实际 fields.json 对照 |
| 历史数据 | 96 次请求标量记录与原档逐项一致；24 行 benchmark，18 个官方表头一致；六例 prompt/参数保持，仅归一媒体路径 |
| 语法和格式 | 17 个 Python 文件编译；Ruff check/format；3 个 shell 的 bash -n 和 ShellCheck 0.9.0 通过 |
| 文档 | Markdownlint 通过，检查本地链接存在、输出中不含私有服务器地址/本机路径或密钥模式 |
| 本地项目门禁 | SPDX、禁用 import、torch.cuda、测试标记及 TTS ratchet 通过；未修改 allowlist 或降低门槛 |
| 类型检查 | 仓库 mypy-3.10 wrapper 通过：10 个生产文件、1 个测试文件；模型目录遵循已有排除规则 |

CPU 环境为 Windows x86_64、Python 3.12.10、PyTorch 2.6.0+cpu。
命令：

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py -q
python recipes/h3_ref_reuse/verify.py
```

测试直接加载发布文件和真实 CPU 张量；缓存模块仅替换导入时的日志和未调用的平台对象，
没有模拟 NPU 运算。隔离运行避开需要完整 vLLM 的全仓 fixture 和本机未安装的 xdist，
缺 pytest-asyncio 带来一项配置告警，19 项用例均通过。
启动脚本检查使用无副作用的 CLI 替身；`--runtime` 在该检查中替换为源码身份检查，
没有真实启动 vLLM、Docker、NPU 算子或发送 HTTP 推理请求。

整体 `pre-commit run` 在无关 actionlint 环境安装阶段长时间无进展，已停止本次自有安装进程；
上表是另行直接运行的适用检查，不声称整套 pre-commit 已通过。没有绕过已安装的 commit hook。

## 整理差异与适用边界

未改变 D045 的 LoRA/量化/缓存/稀疏配置或采样算法。除了 SPDX、格式与导入顺序，
仅补静态类型信息并删除矩形路径 return 后永远不可达的旧 dummy-Q 分支。
保留研究模块名、计数和请求日志，以便对应原实测；这不是一次面向最新上游的重新移植。
未新增 model-specific Python example，默认缓存仍关闭，默认 Host 存储不变。

自检保留的维护边界：实验 LoRA 使用固定 hook；计数器保留原方法包装；
旧 Host/非 A5 helper 与宽泛的复制流清理异常处理保留在历史实现中。
不将这些兼容代码解释为本分支新增的性能收益或新的支持范围。
缓存是请求级状态；失败请求会在 denoise 的 finally 中关闭，不支持在半捕获对象上续步。

NPU 通信、原生算子和完整模型的证据来自历史 D045，**本次整理后未重新跑 NPU**。
Python 有效计算语法树一致不能证明新构建的依赖二进制、跨驱动数值或端到端性能一致。
运行前按 [环境说明](ENVIRONMENT.md) 核对实际容器；性能/画质界限见 [结果说明](RESULTS.md)。
本次未采 msProf，没有打开参考图、截图或视频帧，也没有计算像素或感知指标。

整理使用 Codex 辅助；发布记录提供可复核的差分与检查结果，供使用者和作者审阅。
