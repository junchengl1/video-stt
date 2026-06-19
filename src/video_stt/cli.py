from __future__ import annotations

import argparse
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ModelSpec:
    code: str
    engine: str
    path: Path
    env_var: str


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "model"


MODEL_SPECS = {
    "faster-large-v3": ModelSpec(
        code="faster-large-v3",
        engine="faster-whisper",
        path=DEFAULT_MODEL_ROOT / "faster-whisper-large-v3",
        env_var="VIDEO_STT_FASTER_LARGE_V3_DIR",
    ),
    "openai-large-v3": ModelSpec(
        code="openai-large-v3",
        engine="openai-whisper",
        path=DEFAULT_MODEL_ROOT / "whisper-large-v3",
        env_var="VIDEO_STT_OPENAI_LARGE_V3_DIR",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-stt",
        description="Generate an SRT subtitle file from a video/audio file using local Whisper large-v3 models.",
    )
    parser.add_argument(
        "model",
        help="Model code: faster-large-v3 or openai-large-v3",
    )
    parser.add_argument(
        "media_path",
        help="Path to a video/audio file, e.g. mp4, mkv, wav, mp3.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output subtitle path. Defaults to input file with .srt suffix.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to the local model directory. Overrides the model-specific environment variable and default path.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code, e.g. ja, zh, en. Auto-detect when omitted.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device. Default: auto.",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="faster-whisper compute type, e.g. int8, float16, int8_float16. Defaults to int8 on CPU, float16 on CUDA.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for transcription. Default: 5.",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD filtering for faster-whisper.",
    )
    return parser


def resolve_model(code: str, model_path: str | None = None) -> ModelSpec:
    try:
        spec = MODEL_SPECS[code]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unsupported model code: {code}. Supported: {supported}") from exc

    if model_path:
        return ModelSpec(
            code=spec.code,
            engine=spec.engine,
            path=Path(model_path).expanduser().resolve(),
            env_var=spec.env_var,
        )
    if override := os.environ.get(spec.env_var):
        return ModelSpec(
            code=spec.code,
            engine=spec.engine,
            path=Path(override).expanduser().resolve(),
            env_var=spec.env_var,
        )
    return spec


def make_output_path(media_path: Path, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return media_path.expanduser().resolve().with_suffix(".srt")


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(math.floor(seconds * 1000)))
    ms = milliseconds % 1000
    total_seconds = milliseconds // 1000
    sec = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{ms:03d}"


def write_srt(output_path: Path, segments: Iterable[tuple[float, float, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    for index, (start, end, text) in enumerate(segments, start=1):
        clean_text = " ".join(text.strip().split())
        if not clean_text:
            continue
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
            f"{clean_text}"
        )
    output_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    # On this Mac, faster-whisper supports CPU reliably. CUDA is for NVIDIA hosts.
    return "cpu"


def choose_compute_type(device: str, requested: str | None) -> str:
    if requested:
        return requested
    return "float16" if device == "cuda" else "int8"


def transcribe_with_faster_whisper(
    model_path: Path,
    media_path: Path,
    *,
    language: str | None,
    device: str,
    compute_type: str,
    beam_size: int,
    vad_filter: bool,
) -> list[tuple[float, float, str]]:
    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(media_path),
        language=language,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    return [(segment.start, segment.end, segment.text) for segment in segments]


def choose_openai_device_and_dtype(requested: str, torch_module) -> tuple[str, object]:
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
        return "cuda", torch_module.float16
    if requested == "cpu":
        return "cpu", torch_module.float32

    if torch_module.cuda.is_available():
        return "cuda", torch_module.float16
    if platform.system() == "Darwin" and torch_module.backends.mps.is_available():
        return "mps", torch_module.float16
    return "cpu", torch_module.float32


def transcribe_with_openai_whisper(
    model_path: Path,
    media_path: Path,
    *,
    language: str | None,
    device: str,
) -> list[tuple[float, float, str]]:
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "openai-large-v3 needs transformers and torch. Install them with: "
            "python3 -m pip install 'transformers>=4.35' torch accelerate"
        ) from exc

    resolved_device, dtype = choose_openai_device_and_dtype(device, torch)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_path),
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(resolved_device)
    processor = AutoProcessor.from_pretrained(str(model_path))
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=resolved_device,
        return_timestamps=True,
    )
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    result = pipe(str(media_path), generate_kwargs=generate_kwargs)
    chunks = result.get("chunks") or []
    converted: list[tuple[float, float, str]] = []
    for chunk in chunks:
        timestamp = chunk.get("timestamp") or (0.0, 0.0)
        start = float(timestamp[0] or 0.0)
        end = float(timestamp[1] or start)
        converted.append((start, end, chunk.get("text", "")))
    if not converted and result.get("text"):
        converted.append((0.0, 0.0, result["text"]))
    return converted


def run(argv: Sequence[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = resolve_model(args.model, args.model_path)
    except ValueError as exc:
        parser.error(str(exc))

    media_path = Path(args.media_path).expanduser().resolve()
    if not media_path.exists():
        parser.error(f"Media file does not exist: {media_path}")
    if not spec.path.exists():
        parser.error(f"Model path does not exist: {spec.path}")

    output_path = make_output_path(media_path, args.output)
    if spec.engine == "faster-whisper":
        device = choose_device(args.device)
        compute_type = choose_compute_type(device, args.compute_type)
        segments = transcribe_with_faster_whisper(
            spec.path,
            media_path,
            language=args.language,
            device=device,
            compute_type=compute_type,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
        )
    elif spec.engine == "openai-whisper":
        segments = transcribe_with_openai_whisper(
            spec.path,
            media_path,
            language=args.language,
            device=args.device,
        )
    else:
        raise AssertionError(f"Unhandled engine: {spec.engine}")

    write_srt(output_path, segments)
    print(output_path)
    return output_path


def main() -> None:
    run()


if __name__ == "__main__":
    main()
