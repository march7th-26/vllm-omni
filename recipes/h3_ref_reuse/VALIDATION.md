# 验证

## 本地检查

- CPU 测试：19 项通过，覆盖八步参考 KV 复用、请求清理、序列并行补齐和配置校验。
- 使用真实 CPU 张量；仅替换导入时的日志和未调用的平台对象，不模拟 NPU 运算。
- 启动参数和 multipart 请求字段通过本地命令替身核对，未实际启动服务或发送推理请求。
- Markdownlint、命令语法和本地链接检查通过。

CPU 测试使用 Python 3.12.10、PyTorch 2.6.0+cpu：

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_ref_reuse_snapshot.py -q
```

运行该隔离测试时，未安装 pytest-asyncio 会产生一项配置告警。
完整仓库 pre-commit 套件未完成；以上为单独执行的检查。

## NPU 证据范围

NPU 通信、原生算子和完整模型结果来自既有性能测试，见 [实验结果](RESULTS.md)。
代码整理后未重新运行 NPU，也未验证新构建的 MindIE-SD 二进制。
依赖接口一致不能证明数值和性能一致，画质仍待人工评价。
未采集 msProf。
