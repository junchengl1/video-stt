# video-stt

把视频/音频文件转成 `.srt` 字幕文件。项目默认使用本机已下载的两个 Whisper large-v3 模型：

| 模型代号 | 引擎 | 本机模型路径 |
|---|---|---|
| `faster-large-v3` | faster-whisper / CTranslate2 | `/Users/a123/models/faster-whisper-large-v3` |
| `openai-large-v3` | Transformers + OpenAI Whisper 权重 | `/Users/a123/models/whisper-large-v3` |

## 输入 / 输出

输入：

1. `string`: 想使用的模型代号
2. `string`: 视频或音频文件路径，常见如 `.mp4`, `.mkv`, `.wav`, `.mp3`

输出：

1. 一份 `.srt` 字幕文件

## 推荐使用

优先用 faster-whisper：

```bash
cd /Users/a123/workspace/video-stt
PYTHONPATH=src python3 -m video_stt.cli faster-large-v3 /path/to/video.mp4 --language ja
```

默认会生成：

```text
/path/to/video.srt
```

指定输出路径：

```bash
PYTHONPATH=src python3 -m video_stt.cli faster-large-v3 /path/to/video.mp4 -o ./output.srt --language ja
```

如果不确定语言，可以不传 `--language`，让模型自动识别：

```bash
PYTHONPATH=src python3 -m video_stt.cli faster-large-v3 /path/to/video.mp4
```

## 模型代号

```text
faster-large-v3
openai-large-v3
```

其中 `faster-large-v3` 是当前更推荐的实际部署选项。

## 本机 Mac 建议参数

本机没有 NVIDIA CUDA 时，默认走 CPU：

```bash
PYTHONPATH=src python3 -m video_stt.cli faster-large-v3 /path/to/video.mp4 --device cpu --compute-type int8
```

如果以后在 NVIDIA 机器上跑：

```bash
PYTHONPATH=src python3 -m video_stt.cli faster-large-v3 /path/to/video.mp4 --device cuda --compute-type float16
```

## OpenAI 原版模型说明

`openai-large-v3` 需要额外依赖：

```bash
python3 -m pip install 'transformers>=4.35' torch accelerate
```

然后：

```bash
PYTHONPATH=src python3 -m video_stt.cli openai-large-v3 /path/to/video.mp4 --language ja
```

当前项目里保留这个入口，但实际使用优先建议 `faster-large-v3`。

## 开发测试

```bash
cd /Users/a123/workspace/video-stt
PYTHONPATH=src python3 -m pytest tests -q
```
