import shutil
import subprocess

import numpy as np
import pytest

from packages.pipeline.assembly import (
    CLIPPING_THRESHOLD,
    MAX_DISCARDED_SECONDS,
    TARGET_LUFS,
    TimedAudio,
    TimelineOverflowError,
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
    result = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    assert len(result.timeline) == int(5.0 * SAMPLE_RATE)
    assert result.discarded_samples == 0


def test_place_segments_on_timeline_positions_at_correct_offset() -> None:
    tone = _tone(0.5, amplitude=0.9)
    segments = [TimedAudio(start_seconds=2.0, audio=tone)]
    result = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    timeline = result.timeline

    start = int(2.0 * SAMPLE_RATE)
    assert np.allclose(timeline[start : start + len(tone)], tone)
    assert np.allclose(timeline[:start], 0.0)
    assert np.allclose(timeline[start + len(tone) :], 0.0)


def test_place_segments_on_timeline_fills_gaps_with_silence() -> None:
    segments = [
        TimedAudio(start_seconds=0.0, audio=_tone(0.2)),
        TimedAudio(start_seconds=3.0, audio=_tone(0.2)),
    ]
    result = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    gap_start = int(0.5 * SAMPLE_RATE)
    gap_end = int(2.5 * SAMPLE_RATE)
    assert np.allclose(result.timeline[gap_start:gap_end], 0.0)


def test_place_segments_on_timeline_truncates_beyond_total_duration() -> None:
    # T0.16: estoura o fim por só 0.1s (bem abaixo do limiar de 0.5s) —
    # ainda cabe no orçamento de "arredondamento", não deve levantar.
    segments = [TimedAudio(start_seconds=4.9, audio=_tone(0.2))]
    result = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    assert len(result.timeline) == int(5.0 * SAMPLE_RATE)
    assert result.discarded_samples == int(0.1 * SAMPLE_RATE)


def test_place_segments_on_timeline_ignores_segment_starting_after_end() -> None:
    # T0.16: critério de aceite explícito — um segmento que estoura o fim
    # da timeline é REPORTADO (discarded_samples > 0), não silenciosamente
    # ignorado. Ficando acima do limiar, levanta TimelineOverflowError.
    segments = [TimedAudio(start_seconds=10.0, audio=_tone(3.0))]
    with pytest.raises(TimelineOverflowError):
        place_segments_on_timeline(segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE)


def test_place_segments_on_timeline_raises_above_discard_threshold() -> None:
    # 3s inteiros descartados é bem acima do piso legítimo (2,5s, ver
    # docstring de MAX_DISCARDED_SECONDS) — sinal de bug, não caso do
    # último segmento curto colado no fim do vídeo.
    segments = [TimedAudio(start_seconds=4.9, audio=_tone(3.0))]
    with pytest.raises(TimelineOverflowError):
        place_segments_on_timeline(segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE)


def test_place_segments_on_timeline_allows_discard_within_custom_threshold() -> None:
    segments = [TimedAudio(start_seconds=4.9, audio=_tone(1.0))]
    result = place_segments_on_timeline(
        segments,
        total_duration_seconds=5.0,
        sample_rate=SAMPLE_RATE,
        max_discarded_seconds=2.0,
    )
    assert result.discarded_samples == int(0.9 * SAMPLE_RATE)


def test_mix_with_background_sums_signals() -> None:
    vocals = np.full(100, 0.3, dtype=np.float32)
    background = np.full(100, 0.1, dtype=np.float32)
    result = mix_with_background(vocals, background, SAMPLE_RATE)
    assert np.allclose(result.audio, 0.4)
    assert result.discarded_samples == 0


def test_mix_with_background_applies_gain() -> None:
    vocals = np.zeros(100, dtype=np.float32)
    background = np.full(100, 0.2, dtype=np.float32)
    result = mix_with_background(vocals, background, SAMPLE_RATE, background_gain=0.5)
    assert np.allclose(result.audio, 0.1)


def test_mix_with_background_handles_small_length_mismatch() -> None:
    # Diferença de poucas amostras (arredondamento) — não levanta.
    vocals = np.full(100, 0.1, dtype=np.float32)
    background = np.full(80, 0.1, dtype=np.float32)
    result = mix_with_background(vocals, background, SAMPLE_RATE)
    assert len(result.audio) == 80
    assert result.discarded_samples == 20


def test_mix_with_background_raises_above_discard_threshold() -> None:
    # T0.16: diferença bem acima do piso legítimo (2,5s) entre vocais e
    # fundo é sinal de fontes diferentes, não arredondamento — deve falhar alto.
    vocals = np.zeros(int(4.0 * SAMPLE_RATE), dtype=np.float32)
    background = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    with pytest.raises(TimelineOverflowError):
        mix_with_background(vocals, background, SAMPLE_RATE)


def test_mix_with_background_default_threshold_matches_module_constant() -> None:
    assert MAX_DISCARDED_SECONDS == pytest.approx(0.5)


def test_place_segments_on_timeline_allows_single_segment_at_floor_cap() -> None:
    # Caso legítimo real (beast.mp4): só o último segmento do vídeo, curto,
    # descartando até o teto físico (piso do MOSS-TTS / razão máxima de
    # compressão) não é bug — não deve levantar mesmo passando dos 0.5s
    # agregados.
    from packages.pipeline.assembly import MAX_SINGLE_SEGMENT_DISCARD_SECONDS

    audio_seconds = MAX_SINGLE_SEGMENT_DISCARD_SECONDS - 0.05
    segments = [TimedAudio(start_seconds=5.0, audio=_tone(audio_seconds))]
    result = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    assert result.discarded_samples == int(round(audio_seconds * SAMPLE_RATE))


def test_place_segments_on_timeline_raises_when_single_segment_exceeds_floor_cap() -> None:
    from packages.pipeline.assembly import MAX_SINGLE_SEGMENT_DISCARD_SECONDS

    audio_seconds = MAX_SINGLE_SEGMENT_DISCARD_SECONDS + 1.0
    segments = [TimedAudio(start_seconds=5.0, audio=_tone(audio_seconds))]
    with pytest.raises(TimelineOverflowError):
        place_segments_on_timeline(segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE)


def test_place_segments_on_timeline_clips_segment_that_overlaps_next_one() -> None:
    # Achado real e sério (beast.mp4, N_CANDIDATES=5): um segmento cuja
    # síntese saiu bem mais longa que o espaço até o PRÓXIMO segmento
    # começar se sobrepunha silenciosamente a ele (áudio somado, sem erro,
    # sem flag) — a verificação antiga só olhava o fim do VÍDEO, nunca o
    # próximo segmento. Agora corta no que vier primeiro.
    long_tone = _tone(3.0, amplitude=0.9)  # bem mais longo que o 1.0s de espaço até o próximo
    next_tone = _tone(0.5, amplitude=0.3, freq=880.0)
    segments = [
        TimedAudio(start_seconds=0.0, audio=long_tone),
        TimedAudio(start_seconds=1.0, audio=next_tone),
    ]
    result = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    next_start = int(1.0 * SAMPLE_RATE)
    # o primeiro segmento não pode ter vazado pro espaço do segundo
    assert np.allclose(result.timeline[next_start : next_start + len(next_tone)], next_tone)
    assert result.discarded_samples == len(long_tone) - int(1.0 * SAMPLE_RATE)


def test_place_segments_on_timeline_no_overlap_when_segments_fit() -> None:
    # Caso normal (a maioria): segmentos com espaço de sobra até o
    # próximo — nada é cortado.
    segments = [
        TimedAudio(start_seconds=0.0, audio=_tone(0.5)),
        TimedAudio(start_seconds=2.0, audio=_tone(0.5)),
    ]
    result = place_segments_on_timeline(
        segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
    )
    assert result.discarded_samples == 0


def test_place_segments_on_timeline_raises_when_discard_spread_across_segments() -> None:
    # Achado apontado pelo usuário: um threshold agregado alto esconderia
    # bug real espalhado por VÁRIOS segmentos, mesmo com cada um pequeno
    # (bem abaixo do teto físico de 1 segmento sozinho no piso). O caso
    # legítimo só pode acontecer com 1 segmento (o mais próximo do fim do
    # vídeo) — descarte em 3 segmentos diferentes é sinal de bug (ex.:
    # `total_duration_seconds` medido errado), mesmo que cada um pareça
    # pequeno isoladamente.
    segments = [
        TimedAudio(start_seconds=4.5, audio=_tone(1.0)),  # descarta 0.5s
        TimedAudio(start_seconds=6.0, audio=_tone(1.0)),  # começa após o fim, descarta 1.0s
        TimedAudio(start_seconds=7.0, audio=_tone(1.0)),  # começa após o fim, descarta 1.0s
    ]
    with pytest.raises(TimelineOverflowError):
        place_segments_on_timeline(
            segments, total_duration_seconds=5.0, sample_rate=SAMPLE_RATE
        )


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
