import pytest

from packages.pipeline.budget import (
    TOLERANCE_OFF_SCREEN,
    TOLERANCE_ON_SCREEN,
    Calibration,
    load_calibration,
    seconds_needed,
    syllables_that_fit,
    tolerance_range,
)

CALIB = Calibration(a=0.5, b=0.2, c=5.0)


def test_load_calibration_reads_the_real_file() -> None:
    calibration = load_calibration()
    assert calibration.a > 0
    assert calibration.b > 0
    assert calibration.c > 0


def test_seconds_needed_matches_the_formula() -> None:
    assert seconds_needed(10, CALIB) == pytest.approx(0.5 + 0.2 * 10)


def test_seconds_needed_accounts_for_pauses() -> None:
    base = seconds_needed(10, CALIB)
    with_pause = seconds_needed(10, CALIB, pause_seconds=1.0)
    assert with_pause == pytest.approx(base + CALIB.c)


def test_higher_speed_needs_less_time() -> None:
    normal = seconds_needed(20, CALIB, speed=1.0)
    faster = seconds_needed(20, CALIB, speed=1.15)
    assert faster < normal


def test_syllables_that_fit_zero_when_target_too_small() -> None:
    assert syllables_that_fit(0.1, CALIB) == 0


def test_syllables_that_fit_matches_the_inverse_formula() -> None:
    target = 5.0
    expected = int((target - CALIB.a) / CALIB.b)
    assert syllables_that_fit(target, CALIB) == expected


def test_syllables_that_fit_is_monotonic_in_duration() -> None:
    durations = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    values = [syllables_that_fit(d, CALIB) for d in durations]
    assert values == sorted(values)


def test_round_trip_closes_within_one_syllable() -> None:
    for syllables in [1, 3, 5, 10, 20, 40, 68]:
        for speed in [0.85, 1.0, 1.15]:
            for pause in [0.0, 0.5, 1.2]:
                duration = seconds_needed(syllables, CALIB, speed=speed, pause_seconds=pause)
                recovered = syllables_that_fit(duration, CALIB, speed=speed, pause_seconds=pause)
                assert abs(recovered - syllables) <= 1, (syllables, speed, pause, recovered)


def test_speed_must_be_positive() -> None:
    with pytest.raises(ValueError, match="speed"):
        seconds_needed(10, CALIB, speed=0)
    with pytest.raises(ValueError, match="speed"):
        syllables_that_fit(10, CALIB, speed=-1)


def test_tolerance_range_on_screen_is_tighter_than_off_screen() -> None:
    on_lo, on_hi = tolerance_range(10.0, on_screen=True)
    off_lo, off_hi = tolerance_range(10.0, on_screen=False)

    assert on_hi - on_lo < off_hi - off_lo
    assert on_lo == pytest.approx(10.0 * (1 - TOLERANCE_ON_SCREEN))
    assert off_hi == pytest.approx(10.0 * (1 + TOLERANCE_OFF_SCREEN))
