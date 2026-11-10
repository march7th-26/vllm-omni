# H3-RefReuse 快速启动

在 Ascend 950PR 上运行 MiniMax-H3 Ref2VA，默认启用八步 LoRA、NPU 参考 KV 复用和矩形 mix attention。

## 环境准备

- Linux、Docker 和 Ascend 容器运行时，8 张 Ascend 950PR。
- 已安装匹配依赖的容器镜像：CANN 9.1.0、PyTorch 2.10.0、torch_npu 2.10.0.post4、
  vLLM 0.32.0、vLLM-Ascend 和支持多 span / Eagle 接口的 MindIE-SD。
  具体版本、安装和接口检查见 [环境说明](ENVIRONMENT.md)。
- 完整的 [MiniMax-H3 模型](https://huggingface.co/MiniMaxAI/MiniMax-H3)。
- [MiniMax-H3-Turbo-LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) 中的
  `minimax_h3_turbo_v4_step600_ema.safetensors`。
- 一张自己的参考图，用于下面的请求示例。

## 拉取代码

```bash
git clone --branch h3-ref-reuse --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
python recipes/h3_ref_reuse/verify.py
```

## 启动服务

替换以下镜像名和绝对路径，将参考图保存为 `$INPUT_DIR/reference.jpg`。
镜像需将 vLLM-Omni 以 editable 方式安装在 `/vllm-workspace/vllm-omni`；其他布局见环境说明。

```bash
export IMAGE=your-compatible-ascend-image
export MODEL_DIR=/absolute/path/MiniMax-H3
export LORA_PATH=/absolute/path/minimax_h3_turbo_v4_step600_ema.safetensors
export INPUT_DIR=/absolute/path/reference-media
export DEVICES=0,1,2,3,4,5,6,7
bash recipes/h3_ref_reuse/run_container.sh
```

脚本会检查源码和运行依赖，并启动 `http://127.0.0.1:19136`。

## 发送请求

另开终端进入仓库根目录，设置相同的参考图目录，等待健康检查成功后发送请求：

```bash
export INPUT_DIR=/absolute/path/reference-media
curl --fail http://127.0.0.1:19136/health

cat > request.json <<'JSON'
{
  "prompt": "Use the object in <Picture 1> as the subject. A steady close-up in soft studio lighting, with a slow camera push-in and quiet room ambience.",
  "seconds": 4,
  "target": {"short_edge": 768, "aspect_ratio": "9:16"},
  "num_inference_steps": 8,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 42,
  "conditions": [{"type": "image", "uri": "reference.jpg", "role": "reference"}]
}
JSON

bash recipes/h3_ref_reuse/request.sh request.json output.mp4
```

生成视频保存为 `output.mp4`，请求状态码和耗时保存为 `output.mp4.timing.txt`。
修改 `prompt` 和 `conditions` 可替换描述与参考文件；`uri` 相对于 `INPUT_DIR`。
默认输出为短边 768、24 fps，示例请求时长为 4 秒。

如需关闭参考 KV 复用或 LoRA，停止服务后分别设置 `USE_REF_CACHE=0` 或 `USE_LORA=0`，
再运行启动脚本。参考 KV 缓存会增加显存占用，输入规模需与设备容量匹配。
