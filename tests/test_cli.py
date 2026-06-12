import argparse
from pathlib import Path

import pytest

from video_stt.cli import build_parser, resolve_model, make_output_path, format_timestamp, write_srt


def test_parser_accepts_model_and_media_path():
    parser = build_parser()
    args = parser.parse_args(["faster-large-v3", "sample.mp4"])

    assert args.model == "faster-large-v3"
    assert args.media_path == "sample.mp4"


def test_resolve_model_maps_supported_codes_to_local_paths():
    spec = resolve_model("faster-large-v3")

    assert spec.engine == "faster-whisper"
    assert spec.path == Path("/Users/a123/models/faster-whisper-large-v3")


def test_resolve_model_allows_env_path_override(monkeypatch, tmp_path):
    custom_model_dir = tmp_path / "models" / "faster-whisper-large-v3"
    monkeypatch.setenv("VIDEO_STT_FASTER_LARGE_V3_DIR", str(custom_model_dir))

    spec = resolve_model("faster-large-v3")

    assert spec.path == custom_model_dir


def test_resolve_model_cli_override_wins_over_env(monkeypatch, tmp_path):
    env_model_dir = tmp_path / "env" / "faster-whisper-large-v3"
    cli_model_dir = tmp_path / "cli" / "faster-whisper-large-v3"
    monkeypatch.setenv("VIDEO_STT_FASTER_LARGE_V3_DIR", str(env_model_dir))

    spec = resolve_model("faster-large-v3", str(cli_model_dir))

    assert spec.path == cli_model_dir


def test_parser_accepts_model_path_option():
    parser = build_parser()
    args = parser.parse_args([
        "faster-large-v3",
        "sample.mp4",
        "--model-path",
        "./models/faster-whisper-large-v3",
    ])

    assert args.model_path == "./models/faster-whisper-large-v3"


def test_resolve_model_rejects_unknown_code():
    with pytest.raises(ValueError, match="Unsupported model code"):
        resolve_model("unknown")


def test_make_output_path_defaults_to_srt_next_to_input(tmp_path):
    media = tmp_path / "clip.mp4"

    assert make_output_path(media, None) == tmp_path / "clip.srt"


def test_format_timestamp_uses_srt_comma_milliseconds():
    assert format_timestamp(3661.2345) == "01:01:01,234"


def test_write_srt_writes_indexed_segments(tmp_path):
    output = tmp_path / "out.srt"

    write_srt(output, [(0.0, 1.5, " hello "), (61.2, 62.0, "world")])

    assert output.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "hello\n\n"
        "2\n"
        "00:01:01,200 --> 00:01:02,000\n"
        "world\n"
    )
