import json
from pathlib import Path

from scripts.dub import STAGE_TIMINGS_FILE, _timed_stage


def test_timed_stage_records_last_real_run_per_stage(tmp_path: Path) -> None:
    with _timed_stage("translate", tmp_path):
        pass
    with _timed_stage("synthesize", tmp_path):
        pass
    with _timed_stage("translate", tmp_path):
        pass
    timings = json.loads((tmp_path / STAGE_TIMINGS_FILE).read_text())
    assert set(timings) == {"translate", "synthesize"}
    assert all(v >= 0 for v in timings.values())


def test_timed_stage_without_cache_dir_writes_nothing(tmp_path: Path) -> None:
    with _timed_stage("translate"):
        pass
    assert not (tmp_path / STAGE_TIMINGS_FILE).exists()
