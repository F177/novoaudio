import shutil
import subprocess

import numpy as np
import pytest

from packages.pipeline.assembly import (
    CLIPPING_THRESHOLD,
    TARGET_LUFS,
    TimedAudio,
    has_clipping,
    mix_with_background,
    normalize_loudness,
    place_segments_on_timeline,
    remux_with_video,
)

SAMPLE_RATE = 16000


def _tone(seconds: float, amplitude: float = 0.5, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_place_segments_on_timeline_matches_total_duration() -> None:
    segments = [TimedAudio(start_seconds=0.0, audio=_tone(1.0))]
    timeline = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    assert len(timeline) == int(5.0 * SAMPLE_RATE)


def test_place_segments_on_timeline_positions_at_correct_offset() -> None:
    tone = _tone(0.5, amplitude=0.9)
    segments = [TimedAudio(start_seconds=2.0, audio=tone)]
    timeline = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )

    start = int(2.0 * SAMPLE_RATE)
    assert np.allclose(timeline[start : start + len(tone)], tone)
    assert np.allclose(timeline[:start], 0.0)
    assert np.allclose(timeline[start + len(tone) :], 0.0)


def test_place_segments_on_timeline_fills_gaps_with_silence() -> None:
    segments = [
        TimedAudio(start_seconds=0.0, audio=_tone(0.2)),
        TimedAudio(start_seconds=3.0, audio=_tone(0.2)),
    ]
    timeline = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    gap_start = int(0.5 * SAMPLE_RATE)
    gap_end = int(2.5 * SAMPLE_RATE)
    assert np.allclose(timeline[gap_start:gap_end], 0.0)


def test_place_segments_on_timeline_truncates_beyond_total_duration() -> None:
    segments = [TimedAudio(start_seconds=4.5, audio=_tone(2.0))]
    timeline = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    assert len(timeline) == int(5.0 * SAMPLE_RATE)


def test_place_segments_on_timeline_ignores_segment_starting_after_end() -> None:
    segments = [TimedAudio(start_seconds=10.0, audio=_tone(1.0))]
    timeline = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    assert np.allclose(timeline, 0.0)


def test_mix_with_background_sums_signals() -> None:
    vocals = np.full(100, 0.3, dtype=np.float32)
    background = np.full(100, 0.1, dtype=np.float32)
    mixed = mix_with_background(vocals, background)
    assert np.allclose(mixed, 0.4)


def test_mix_with_background_applies_gain() -> None:
    vocals = np.zeros(100, dtype=np.float32)
    background = np.full(100, 0.2, dtype=np.float32)
    mixed = mix_with_background(vocals, background, background_gain=0.5)
    assert np.allclose(mixed, 0.1)


def test_mix_with_background_handles_length_mismatch() -> None:
    vocals = np.full(100, 0.1, dtype=np.float32)
    background = np.full(80, 0.1, dtype=np.float32)
    mixed = mix_with_background(vocals, background)
    assert len(mixed) == 80


def test_has_clipping_detects_over_threshold() -> None:
    audio = np.array([0.1, 0.5, 1.5, -0.2], dtype=np.float32)
    assert has_clipping(audio) is True


def test_has_clipping_false_for_normal_audio() -> None:
    audio = _tone(1.0, amplitude=0.5)
    assert has_clipping(audio) is False


def test_has_clipping_respects_custom_threshold() -> None:
    audio = np.array([0.5, 0.6], dtype=np.float32)
    assert has_clipping(audio, threshold=0.4) is True
    assert has_clipping(audio, threshold=CLIPPING_THRESHOLD) is False


def test_normalize_loudness_reaches_target() -> None:
    audio = _tone(3.0, amplitude=0.1)
    normalized = normalize_loudness(audio, SAMPLE_RATE)

    import pyloudnorm as pyln

    meter = pyln.Meter(SAMPLE_RATE)
    result_loudness = meter.integrated_loudness(normalized)
    assert result_loudness == pytest.approx(TARGET_LUFS, abs=0.5)


def test_normalize_loudness_silent_audio_is_unchanged() -> None:
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    normalized = normalize_loudness(audio, SAMPLE_RATE)
    assert np.allclose(normalized, 0.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg não disponível")
def test_remux_with_video_preserves_duration(tmp_path) -> None:
    video_path = tmp_path / "in.mp4"
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "out.mp4"

    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2",
            "-c:v", "libx264", "-c:a", "aac", str(video_path),
        ],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=16000",
         str(audio_path)],
        check=True, capture_output=True,
    )

    remux_with_video(video_path, audio_path, output_path)

    assert output_path.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(output_path)],
        check=True, capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    assert duration == pytest.approx(2.0, abs=0.2)
