import json
from pathlib import Path

import numpy as np

from scripts.judge_audio import (
    ANALYSIS_SR,
    HOP_SECONDS,
    SegmentVoice,
    f0_track,
    final_peak_st,
    is_polar_question,
    load_language,
    load_segments,
    median_hz,
    question_start_regex,
    sparkline,
    speaker_summary,
)


def _voice(hz: float, seconds: float = 1.0) -> np.ndarray:
    """Tom harmônico (5 parciais), parecido o bastante com voz para o pyin."""
    t = np.arange(int(seconds * ANALYSIS_SR)) / ANALYSIS_SR
    return sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6)).astype(np.float32) * 0.3


def _f0(*pieces: tuple[float, float]) -> np.ndarray:
    """Trilha de F0 sintética: (Hz ou NaN, segundos) em sequência."""
    return np.concatenate([np.full(round(s / HOP_SECONDS), hz) for hz, s in pieces])


def test_f0_track_finds_pitch_of_synthetic_voice() -> None:
    assert abs(median_hz(f0_track(_voice(150.0))) - 150.0) < 5.0


def test_polar_vs_wh_questions_from_ptbr_profile() -> None:
    wh = question_start_regex(load_language("pt-BR"))
    assert is_polar_question("Seu pai paga pra você?", wh)
    assert is_polar_question("Não sei. Ela vai voltar amanhã cedo?", wh)
    assert not is_polar_question("O que cê tá comendo agora?", wh)
    assert not is_polar_question("Quanto cê tá devendo?", wh)
    assert not is_polar_question("Do que tá falando?", wh)
    assert not is_polar_question("Liga pro seu pai.", wh)


def test_language_without_intonation_rule_disables_check() -> None:
    assert question_start_regex({"codigo": "xx", "perguntas": {}}) is None


def test_final_peak_separates_rising_from_flat_ending() -> None:
    rising = _f0((120.0, 1.5), (np.nan, 0.1), (170.0, 0.4))
    flat = _f0((120.0, 1.5), (np.nan, 0.1), (115.0, 0.4))
    assert final_peak_st(rising) > 4.0
    assert final_peak_st(flat) < 1.0
    assert final_peak_st(np.full(10, np.nan)) is None


def test_sparkline_fixed_scale_and_unvoiced() -> None:
    assert sparkline(np.array([np.nan, -6.0, 6.0])) == "·▁█"
    assert len(sparkline(np.zeros(200))) == 40


def test_speaker_summary_flags_shared_voice_and_register_gap() -> None:
    segs = [
        SegmentVoice("seg0000", 0, 1, "a", speaker="SPEAKER_00", voice="h21",
                     source_hz=120.0, dub_hz=125.0),
        SegmentVoice("seg0001", 1, 2, "b", speaker="SPEAKER_01", voice="h21",
                     source_hz=230.0, dub_hz=118.0),
    ]
    lines = "\n".join(speaker_summary(segs, {"h21": "m"}))
    assert "voz h21 (m)" in lines
    assert "⚠ registro" in lines  # 230 Hz -> 118 Hz = -11.6 st
    assert "⚠ a voz h21 foi usada para SPEAKER_00, SPEAKER_01" in lines


def test_load_segments_reads_voice_speaker_and_emotion(tmp_path: Path) -> None:
    report = {"segments": [{
        "id": "seg0000", "t_inicio": 0.0, "t_fim": 2.0, "traducao": "Oi?",
        "pitch_std_source": 2.0, "pitch_std_dub": 0.8,
        "emotion_match": {"source_emotion": "happy", "dub_emotion": "neutral", "similarity": 0.3},
    }]}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "segments.json").write_text(
        json.dumps([{"id": "seg0000", "speaker": "SPEAKER_01"}]), encoding="utf-8"
    )
    (tmp_path / "synth_jobs.json").write_text(
        json.dumps([{"id": "seg0000", "reference_audio_path": "voices/bank/h27.wav"}]),
        encoding="utf-8",
    )
    seg = load_segments(tmp_path)[0]
    assert (seg.speaker, seg.voice, seg.emotion["dub_emotion"]) == ("SPEAKER_01", "h27", "neutral")
