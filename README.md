# video-stt

把视频/音频文件转成 `.srt` 字幕文件。项目默认支持两个本地 Whisper large-v3 模型：

| 模型代号 | 引擎 | 默认本机路径 | 其他机器覆盖方式 | CLI 覆盖 |
|---|---|---|---|---|
| `faster-large-v3` | faster-whisper / CTranslate2 | `/Users/a123/models/faster-whisper-large-v3` | `VIDEO_STT_FASTER_LARGE_V3_DIR` | `--model-path` |
| `openai-large-v3` | Transformers + OpenAI Whisper 权重 | `/Users/a123/models/whisper-large-v3` | `VIDEO_STT_OPENAI_LARGE_V3_DIR` | `--model-path` |

> 给 AI Agent：仓库根目录有 `AGENTS.md`，里面是更直接的操作手册。

## 输入 / 输出

输入：

1. `string`: 想使用的模型代号
2. `string`: 视频或音频文件路径，常见如 `.mp4`, `.mkv`, `.wav`, `.mp3`

输出：

1. 一份 `.srt` 字幕文件

## Quick Start：新机器从零开始

```bash
git clone https://github.com/junchengl1/video-stt.git
cd video-stt

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e '.[test]'
python3 -m pytest tests -q
```

### 下载模型到项目目录

推荐先下载 faster-whisper 模型：

```bash
python3 -m pip install -U 'huggingface_hub[cli]'
hf download Systran/faster-whisper-large-v3 \
  --local-dir ./models/faster-whisper-large-v3
```

如果也要下载 OpenAI 原版权重：

```bash
hf download openai/whisper-large-v3 \
  model.safetensors config.json generation_config.json preprocessor_config.json \
  tokenizer.json tokenizer_config.json vocab.json merges.txt normalizer.json \
  special_tokens_map.json added_tokens.json \
  --local-dir ./models/whisper-large-v3
```

设置模型路径：

```bash
export VIDEO_STT_FASTER_LARGE_V3_DIR="$PWD/models/faster-whisper-large-v3"
export VIDEO_STT_OPENAI_LARGE_V3_DIR="$PWD/models/whisper-large-v3"
```

也可以不设置环境变量，运行时直接传模型路径：

```bash
video-stt faster-large-v3 /path/to/video.mp4 \
  --model-path ./models/faster-whisper-large-v3 \
  --language ja
```

模型路径优先级：

```text
--model-path > 环境变量 > 默认路径
```

> 模型目录已被 `.gitignore` 忽略，不要把 3GB+ 模型文件提交进 Git。

## 推荐使用

优先用 faster-whisper：

```bash
video-stt faster-large-v3 /path/to/video.mp4 --language ja
```

等价的模块调用方式：

```bash
PYTHONPATH=src python3 -m video_stt.cli faster-large-v3 /path/to/video.mp4 --language ja
```

默认会生成：

```text
/path/to/video.srt
```

指定输出路径：

```bash
video-stt faster-large-v3 /path/to/video.mp4 -o ./output.srt --language ja
```

如果不确定语言，可以不传 `--language`，让模型自动识别：

```bash
video-stt faster-large-v3 /path/to/video.mp4
```

## 模型代号

```text
faster-large-v3
openai-large-v3
```

其中 `faster-large-v3` 是当前更推荐的实际部署选项。

## 本机 Mac / CPU 建议参数

没有 NVIDIA CUDA 时，默认走 CPU：

```bash
video-stt faster-large-v3 /path/to/video.mp4 --device cpu --compute-type int8
```

如果以后在 NVIDIA 机器上跑：

```bash
video-stt faster-large-v3 /path/to/video.mp4 --device cuda --compute-type float16
```

## OpenAI 原版模型说明

`openai-large-v3` 需要额外依赖：

```bash
python3 -m pip install -e '.[openai]'
```

然后：

```bash
video-stt openai-large-v3 /path/to/video.mp4 --language ja
```

当前项目里保留这个入口，但实际使用优先建议 `faster-large-v3`。

## 开发测试

```bash
cd /Users/a123/workspace/video-stt
PYTHONPATH=src python3 -m pytest tests -q
```

## AI Agent Checklist

新 agent 拿到项目后按这个顺序做：

1. 读 `AGENTS.md`。
2. 跑 `python3 -m pip install -e '.[test]'`。
3. 跑 `PYTHONPATH=src python3 -m pytest tests -q`。
4. 确认模型路径：
   - 本机默认：`/Users/a123/models/...`
   - 其他机器：用 `VIDEO_STT_FASTER_LARGE_V3_DIR` / `VIDEO_STT_OPENAI_LARGE_V3_DIR` 覆盖。
5. 优先执行：`video-stt faster-large-v3 <media_path> --model-path <model_dir> --language <lang>`。
