import pytest

from packages.pipeline.segmentation import Word, segment_words


def test_empty_input_returns_no_segments() -> None:
    assert segment_words([]) == []


def test_tiny_gap_is_ignored_and_words_stay_together() -> None:
    words = [Word("Hello", 0.0, 0.5), Word("world", 0.55, 1.0)]
    segments = segment_words(words)

    assert len(segments) == 1
    assert segments[0].texto == "Hello world"
    assert segments[0].pausas_internas == []


def test_weak_pause_is_recorded_but_does_not_break() -> None:
    words = [Word("A", 0.0, 0.5), Word("B", 0.7, 1.2)]
    segments = segment_words(words)

    assert len(segments) == 1
    assert len(segments[0].pausas_internas) == 1
    pause = segments[0].pausas_internas[0]
    assert pause.after_word_index == 0
    assert pause.duration == pytest.approx(0.2)


def test_hard_pause_breaks_into_two_segments() -> None:
    words = [Word("A", 0.0, 0.5), Word("B", 1.0, 1.5)]
    segments = segment_words(words)

    assert len(segments) == 2
    assert segments[0].texto == "A"
    assert segments[1].texto == "B"
    assert segments[0].t_fim <= segments[1].t_inicio


def test_speaker_change_breaks_even_with_a_tiny_gap() -> None:
    words = [
        Word("A", 0.0, 0.5, speaker="S1"),
        Word("B", 0.55, 1.0, speaker="S2"),
    ]
    segments = segment_words(words)

    assert len(segments) == 2
    assert segments[0].speaker == "S1"
    assert segments[1].speaker == "S2"


def test_unassigned_speaker_does_not_force_a_break() -> None:
    words = [
        Word("A", 0.0, 0.5, speaker="S1"),
        Word("B", 0.55, 1.0, speaker=None),
        Word("C", 1.05, 1.5, speaker="S1"),
    ]
    segments = segment_words(words)

    assert len(segments) == 1
    assert segments[0].speaker == "S1"


def test_no_segment_exceeds_max_duration() -> None:
    # 30 palavras de 0.5s, espaçadas por uma pausa fraca de 0.15s — nunca
    # há âncora dura nem troca de locutor, só o limite de duração força quebra.
    words = [Word(f"w{i}", i * 0.65, i * 0.65 + 0.5) for i in range(30)]
    segments = segment_words(words)

    assert len(segments) > 1
    for seg in segments:
        assert seg.t_fim - seg.t_inicio <= 12.0


def test_short_isolated_segment_is_merged_into_a_neighbor() -> None:
    words = [
        Word("A", 0.0, 0.5),
        # isolada por âncora dura dos dois lados, duraria só 0.15s sozinha
        Word("B", 1.0, 1.15),
        Word("C", 1.75, 3.0),
    ]
    segments = segment_words(words)

    assert len(segments) == 2
    for seg in segments:
        assert seg.t_fim - seg.t_inicio >= 0.3
    # a pausa real entre B e C não desaparece, vira pausa interna do segmento fundido
    merged = segments[1]
    assert merged.texto == "B C"
    assert len(merged.pausas_internas) == 1
    assert merged.pausas_internas[0].duration == pytest.approx(0.6)


def test_segments_do_not_overlap_and_cover_every_word() -> None:
    words = [
        Word("A", 0.0, 0.4),
        Word("B", 0.45, 0.9),  # gap 0.05: ignorado
        Word("C", 1.2, 1.6),  # gap 0.3: fraco
        Word("D", 2.2, 2.6, speaker="S1"),  # gap 0.6: âncora dura
        Word("E", 2.65, 3.0, speaker="S2"),  # troca de locutor, gap ínfimo
    ]
    segments = segment_words(words)

    for prev, curr in zip(segments, segments[1:], strict=False):
        assert prev.t_fim <= curr.t_inicio

    assert segments[0].t_inicio == words[0].start
    assert segments[-1].t_fim == words[-1].end
    total_words = sum(len(seg.texto.split()) for seg in segments)
    assert total_words == len(words)
