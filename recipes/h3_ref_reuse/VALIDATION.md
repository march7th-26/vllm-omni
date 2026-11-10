# 验证

## CPU 检查

34 项测试通过，覆盖参考 KV 八步复用与清理、序列并行补齐、RBS 参数边界、
每张参考图的缓存位置映射、完整 query 的参数传递和矩形 query 块编号映射。
RBS 测试从生产源文件加载配置和几何定义；未加载 vLLM 或原生 NPU attention。

依赖 Python、pytest 和 CPU PyTorch。本地检查使用 Python 3.12.10、PyTorch 2.6.0+cpu：

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py -q
```

在已安装完整仓库测试依赖的环境中，按现有 L1 标记执行：

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py -m 'core_model and cpu' --run-level core_model -q
```

## NPU 证据范围

[结果](RESULTS.md) 中的 NPU 数字来自既有完整模型测试，分别列出四卡 RBS 开关对照与
八卡 H3-RefReuse 对照。当前分支整理后未重新执行 NPU 端到端测试；CPU 检查不能代替
原生算子、完整模型或画质验收。新构建的 MindIE-SD wheel 也未完成本分支的 NPU 验证。
