# LQ-Slim 融合设计

## 保留的计算语义

基于新版 LQ-Slim：八步 LoRA、MXFP8、BF16 resident Ref KV interval8，
视频删除率0.8/0.8/0.85×6、图片删除率0.9。缓存刷新与参考投影跳过规则保持不变。
本实现不依赖“上游合入”等同于“Ref 与图片 ABI 已适配”的假设。

## 执行顺序和数据所有权

1. 模型继续在主 block 前压缩参考行。刷新步传输完整 Q/K/V；复用步仅传输紧凑序列。
2. 严格 Ulysses 将每个目的 rank 的同一个局部 head 范围打包，使用现有
   `all_to_all_4D` 完成 BF16 Q/K/V 通信。所有 rank 按同一顺序提交三个 forward collective。
3. 通信流为每个 chunk 记录 ready event。计算流等待该事件，去掉 SP transport padding，
   刷新时按全局参考位置填入对应 head 的缓存切片；复用时仅拼接该切片与当前紧凑 K/V。
4. 调用现有 RainFusion consumer。图片路径在本地补 dummy 参考 Q，沿用公共图片重排、
   mask 和 mix kernel，再裁掉参考输出；没有图片选择时保留原矩形路由。
5. 本地结果恢复 transport padding。通信流等待计算完成后逆向 all-to-all，
   按原 head 下标写回输出。返回前计算流等待所有逆向通信。

缓存的物理布局仍为 `[2, layers, reference_rows, rank_local_heads, head_dim]`。
每层所有 head 恰好处理一次后才更新原有 layer 统计和完成标记；缺失、重复或乱序 chunk 报错。
这避免把一个 chunk 当作完整层，从而导致下一层读取未初始化缓存。

发送、接收、KV 拼接和输出 tensor 在函数内保留所有者，并对跨流使用调用 `record_stream`。
任一提交后异常不切换到原路径、不重发 collective；异常清理先等待两个流，随后向上传递。
该实现只支持单请求 BSND、BF16、严格整除 head 的纯 Ulysses，与 ring、joint QKV、
gate compression、paged KV、Host Ref 缓存的组合会报错。

## Graph 边界

Graph 只包住参数已经解析完成的公共 `sparse_attention`，不捕获 Ref 状态、层计数、
通信、SP padding、dummy Q 构造和 KV 拼接。输入先复制到固定地址的 Q/K/V buffer；
相同 geometry、precision、sparsity、shape、dtype 和 device 可以跨 head、层及请求共享。
这使 replay 每次读取当前请求数据，不保留某个 Ref pool 的地址。

worker 级 LRU 最多三个条目，覆盖常用的刷新、前期复用和后期复用。
每个条目先进行一次实际 eager 调用，再捕获并 replay；输出立即在计算流复制为调用者所有，
避免下一 head replay 覆盖正在逆向通信的静态输出。条目切换和淘汰先同步计算流。
不同计算流、线程或设备复用同一 cache 会报错。

接口依据 Ascend 的 [NPUGraph 和 graph 上下文实现](https://gitee.com/ascend/pytorch/blob/master/torch_npu/npu/graphs.py)。
公共 MindIE 源码中存在的动态 mask 会在捕获时记录相应 tensor 操作，不能把上一 head 的
最终 mask 作为常量传入；本实现不传入外部 `cached_mask`。实际 kernel 能否在目标版本捕获，
需要原生 NPU 验证。

## AudioVAE

仅在显式环境开关开启时替换 decoder 的 `nn.ConvTranspose1d.forward` 和匹配
checkpoint 接口的 alias-free `UpSample1d.forward`，不替换参数或
state dict，不改参考编码。对 stride 的每个相位提取权重，转为 `stride × out_channels`
个 Conv1d 输出通道，恢复时间交错排列后按原 padding/output_padding 裁剪并加 bias。
支持普通及分组卷积；dilation 不支持的层保留原实现，显式 output_size 也使用原 forward。
alias-free upsampler 保留 replicate padding、比例系数及左右裁剪，不修改全局函数。
每次调用重新读取权重，兼容 weight_norm pre-hook 和 CPU/NPU 权重搬运。
该改写数学等价，浮点结果不保证逐位一致。

## 与 MindIE 原生 pipeline 的差别

参考接口为 MindIE-SD dev `e63893d3d8572f7987aa7c26f0c0323130c90b3b` 的
`mindiesd/parallel`。该版本分布式入口仍要求等长 Q/K/V，使用共享重排几何；
其量化 wire 不能直接提供现有 Ref pool 所需的 BF16 KV，也未接入公共 image spans consumer。

本实现复用 Omni 的通信原语和 RainFusion consumer，自行接入 chunk 事件依赖，
没有复制 MindIE pipeline 或声称使用它的 mixed packet/global-scale 协议。
BF16 wire 保住紧凑序列通信收益，但通信字节数大于原生量化 wire；更多 collective、
Python 调度、Graph 输入/输出复制和 chunk 内量化归约均可能抵消收益。

后续量化通信需要独立的刷新填池路径，并让参考统计参加当前 Q/K/V 的 scale 协议；
不能直接复用上一步的 FP8 V 载荷。真实短 Q 图片稀疏还需要独立 Q/KV 重排与块边界协议。
这两项不包含在本次兼容组合实现中。

## 验证边界

CPU 测试覆盖原/分 chunk 缓存八步值与层统计一致、余数 chunk、缺失与重复 head、
四/八 rank 的 head 打包与恢复、AudioVAE 数学等价和动态权重、标准配置透传、
Graph 静态输入更新、几何切换、LRU 和跨流拒绝。Graph 测试使用调用桩。
尚无 NPU 通算、真实捕获、端到端、性能或媒体质量结论。
