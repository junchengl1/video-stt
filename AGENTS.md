# AGENTS.md

This repository is intentionally small. It exists to turn a video/audio file into an `.srt` subtitle file with local Whisper large-v3 models.

## Goal

Input contract:

1. `model`: a string model code
2. `media_path`: a string path to a video/audio file, commonly `.mp4`, `.mkv`, `.wav`, `.mp3`

Output contract:

1. one `.srt` subtitle file

## Supported model codes

| Code | Engine | Default local path | Override env var | CLI override |
|---|---|---|---|---|
| `faster-large-v3` | `faster-whisper` / CTranslate2 | `./model/faster-whisper-large-v3` | `VIDEO_STT_FASTER_LARGE_V3_DIR` | `--model-path` |
| `openai-large-v3` | Transformers + OpenAI Whisper weights | `./model/whisper-large-v3` | `VIDEO_STT_OPENAI_LARGE_V3_DIR` | `--model-path` |

Prefer `faster-large-v3` for real work unless the user explicitly asks to test the OpenAI/Transformers path.

## First commands for a fresh agent

```bash
cd /path/to/video-stt
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e '.[test]'
python3 -m pytest tests -q
```

If model files are not present on the current machine, download them:

```bash
python3 -m pip install -U 'huggingface_hub[cli]'
hf download Systran/faster-whisper-large-v3 --local-dir ./model/faster-whisper-large-v3
hf download openai/whisper-large-v3 \
  model.safetensors config.json generation_config.json preprocessor_config.json \
  tokenizer.json tokenizer_config.json vocab.json merges.txt normalizer.json \
  special_tokens_map.json added_tokens.json \
  --local-dir ./model/whisper-large-v3
export VIDEO_STT_FASTER_LARGE_V3_DIR="$PWD/model/faster-whisper-large-v3"
export VIDEO_STT_OPENAI_LARGE_V3_DIR="$PWD/model/whisper-large-v3"
```

## Common run commands

Recommended path:

```bash
video-stt faster-large-v3 /path/to/video.mp4 --language ja
```

Use an explicit model path when the model is not in the default location:

```bash
video-stt faster-large-v3 /path/to/video.mp4 \
  --model-path ./model/faster-whisper-large-v3 \
  --language ja
```

Equivalent module invocation:

```bash
PYTHONPATH=src python3 -m video_stt.cli faster-large-v3 /path/to/video.mp4 --language ja
```

Specify output path:

```bash
video-stt faster-large-v3 /path/to/video.mp4 -o ./output.srt --language ja
```

CPU-friendly mode:

```bash
video-stt faster-large-v3 /path/to/video.mp4 --device cpu --compute-type int8
```

NVIDIA CUDA mode:

```bash
video-stt faster-large-v3 /path/to/video.mp4 --device cuda --compute-type float16
```

## Development rules

- Keep the CLI contract stable: `video-stt <model> <media_path> [-o output.srt] [--model-path model_dir]`.
- Add tests before changing behavior.
- Run `PYTHONPATH=src python3 -m pytest tests -q` before committing.
- Do not commit downloaded model files, generated `.srt` files, virtualenvs, or caches.
- Keep model locations configurable through `--model-path` or environment variables when a machine does not use the default `./model/...` paths.

## Troubleshooting

- `Media file does not exist`: use an absolute path or check quoting for spaces.
- `Model path does not exist`: set `VIDEO_STT_FASTER_LARGE_V3_DIR` or `VIDEO_STT_OPENAI_LARGE_V3_DIR`, or download models as shown above.
- `openai-large-v3 needs transformers and torch`: install the optional dependencies with `python3 -m pip install -e '.[openai]'`.
- Slow CPU transcription is expected for large-v3. Use `faster-large-v3`, `--compute-type int8`, or run on a CUDA machine for speed.
