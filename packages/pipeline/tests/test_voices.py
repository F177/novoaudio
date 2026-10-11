import json
from pathlib import Path

import numpy as np
import pytest

from packages.pipeline.voices import Voice, assign_voices, load_bank, median_f0_hz

BANK = [
    Voice("h30", "m", "voices/bank/h30.wav"),
    Voice("h42", "m", "voices/bank/h42.wav"),
    Voice("m26", "f", "voices/bank/m26.wav"),
    Voice("m38", "f", "voices/bank/m38.wav"),
]


def test_two_speakers_get_distinct_voices_of_their_gender() -> None:
    voices = assign_voices(["A", "B"], {"A": 120.0, "B": 210.0}, BANK, seed="video1")
    assert voices["A"].gender == "m" and voices["B"].gender == "f"
    two_men = assign_voices(["A", "B"], {"A": 110.0, "B": 130.0}, BANK, seed="video1")
    assert two_men["A"].id != two_men["B"].id
    assert {two_men["A"].gender, two_men["B"].gender} == {"m"}


def test_assignment_is_stable_per_video() -> None:
    f0 = {"A": 120.0, "B": 210.0}
    assert assign_voices(["A", "B"], f0, BANK, "v") == assign_voices(["A", "B"], f0, BANK, "v")


def test_reuses_voices_when_bank_runs_out_and_handles_unknown_f0() -> None:
    voices = assign_voices(["A", "B", "C"], {"A": 100.0, "B": 110.0, "C": 120.0}, BANK, "v")
    assert all(v.gender == "m" for v in voices.values())
    assert assign_voices(["A"], {"A": None}, BANK, "v")["A"] in BANK
    assert assign_voices(["A"], {"A": 100.0}, [], "v") == {}


def test_median_f0_of_a_tone() -> None:
    sr = 16000
    t = np.arange(sr * 2) / sr
    assert median_f0_hz(0.5 * np.sin(2 * np.pi * 140 * t), sr) == pytest.approx(140, rel=0.03)
    assert median_f0_hz(np.zeros(100), sr) is None


def test_load_bank(tmp_path: Path) -> None:
    manifest = tmp_path / "voices.json"
    manifest.write_text(json.dumps([{"id": "h30", "gender": "m", "path": "x.wav"}]))
    assert load_bank(manifest) == [Voice("h30", "m", "x.wav")]
    assert load_bank(tmp_path / "missing.json") == []
