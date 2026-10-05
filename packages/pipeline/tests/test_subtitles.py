from packages.pipeline.subtitles import build_srt


def test_build_srt_single_entry_format() -> None:
    # SRT válido termina cada bloco (inclusive o último) com linha em branco.
    srt = build_srt([(1.0, 4.0, "Ola mundo.")])
    assert srt == "1\n00:00:01,000 --> 00:00:04,000\nOla mundo.\n"


def test_build_srt_multiple_entries_numbered_in_order() -> None:
    srt = build_srt(
        [
            (5.0, 8.0, "Segunda linha."),
            (1.0, 4.0, "Primeira linha."),
        ]
    )
    lines = srt.split("\n")
    assert lines[0] == "1"
    assert lines[2] == "Primeira linha."
    # bloco 2 começa depois de uma linha em branco separando os blocos
    assert "2" in lines
    second_block_index = lines.index("2")
    assert lines[second_block_index + 2] == "Segunda linha."


def test_build_srt_sorts_by_start_time_even_if_input_unsorted() -> None:
    srt = build_srt(
        [
            (10.0, 12.0, "Depois."),
            (0.0, 2.0, "Antes."),
        ]
    )
    assert srt.index("Antes.") < srt.index("Depois.")


def test_build_srt_formats_hours_minutes_correctly() -> None:
    srt = build_srt([(3661.5, 3665.25, "Depois de uma hora.")])
    assert "01:01:01,500 --> 01:01:05,250" in srt


def test_build_srt_skips_empty_text() -> None:
    srt = build_srt([(1.0, 2.0, "   "), (3.0, 4.0, "Texto real.")])
    assert "Texto real." in srt
    assert srt.count("-->") == 1


def test_build_srt_skips_zero_or_negative_duration() -> None:
    srt = build_srt([(2.0, 2.0, "Duração zero."), (2.0, 1.0, "Duração negativa.")])
    assert srt == ""


def test_build_srt_empty_input_returns_empty_string() -> None:
    assert build_srt([]) == ""


def test_build_srt_strips_surrounding_whitespace_from_text() -> None:
    srt = build_srt([(1.0, 2.0, "  com espaços  ")])
    assert "\ncom espaços\n" in srt
