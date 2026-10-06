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
    # min_word_count=1: teste é sobre a âncora dura, não sobre a fusão por
    # poucas palavras (ver test_*_merged_into_a_neighbor_by_word_count).
    words = [Word("A", 0.0, 0.5), Word("B", 1.0, 1.5)]
    segments = segment_words(words, min_word_count=1)

    assert len(segments) == 2
    assert segments[0].texto == "A"
    assert segments[1].texto == "B"
    assert segments[0].t_fim <= segments[1].t_inicio


def test_speaker_change_breaks_even_with_a_tiny_gap() -> None:
    # min_word_count=1: teste é sobre troca de locutor, não sobre fusão por
    # poucas palavras.
    words = [
        Word("A", 0.0, 0.5, speaker="S1"),
        Word("B", 0.55, 1.0, speaker="S2"),
    ]
    segments = segment_words(words, min_word_count=1)

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
    # min_word_count=1: teste é sobre fusão por duração curta, não por poucas
    # palavras (todas as palavras aqui são isoladas de propósito).
    words = [
        Word("A", 0.0, 0.5),
        # isolada por âncora dura dos dois lados, duraria só 0.15s sozinha
        Word("B", 1.0, 1.15),
        Word("C", 1.75, 3.0),
    ]
    segments = segment_words(words, min_word_count=1)

    assert len(segments) == 2
    for seg in segments:
        assert seg.t_fim - seg.t_inicio >= 0.3
    # a pausa real entre B e C não desaparece, vira pausa interna do segmento fundido
    merged = segments[1]
    assert merged.texto == "B C"
    assert len(merged.pausas_internas) == 1
    assert merged.pausas_internas[0].duration == pytest.approx(0.6)


def test_isolated_short_phrase_is_merged_by_word_count_even_with_enough_duration() -> None:
    # "Ok" sozinha dura 0.3s (>= min_duration, não seria fundida só por
    # duração) mas tem 1 palavra só (< min_word_count=4 default) — achado
    # real: MOSS-TTS falha em texto isolado curto mesmo com orçamento de
    # tempo grande, o problema é falta de contexto, não duração.
    words = [
        Word("Primeiro", 0.0, 0.4),
        Word("teste", 0.45, 0.9),
        Word("de", 0.95, 1.2),
        Word("verdade", 1.25, 1.8),
        Word("Ok", 2.5, 2.8),  # âncora dura dos dois lados, isolada, 1 palavra
        Word("Segunda", 3.5, 3.9),
        Word("parte", 3.95, 4.4),
        Word("do", 4.45, 4.7),
        Word("teste", 4.75, 5.2),
    ]
    segments = segment_words(words)

    assert len(segments) == 2
    assert segments[0].texto == "Primeiro teste de verdade"
    assert segments[1].texto == "Ok Segunda parte do teste"
    # a âncora dura real entre "Ok" e "Segunda" vira pausa interna do
    # segmento fundido, não desaparece
    assert len(segments[1].pausas_internas) == 1
    assert segments[1].pausas_internas[0].duration == pytest.approx(0.7)


def test_segment_with_exactly_min_word_count_is_not_merged_by_word_count() -> None:
    words = [
        Word("Um", 0.0, 0.2),
        Word("dois", 0.25, 0.45),
        Word("tres", 0.5, 0.7),
        Word("quatro", 0.75, 0.95),  # 4 palavras == min_word_count default
        Word("cinco", 2.0, 2.2, speaker=None),
        Word("seis", 2.25, 2.45),
        Word("sete", 2.5, 2.7),
        Word("oito", 2.75, 2.95),
    ]
    segments = segment_words(words, min_word_count=4)

    assert len(segments) == 2
    assert segments[0].texto == "Um dois tres quatro"


def test_min_word_count_can_be_disabled() -> None:
    words = [Word("A", 0.0, 0.5), Word("B", 1.0, 1.5), Word("C", 2.0, 2.5)]
    segments = segment_words(words, min_word_count=1)

    assert len(segments) == 3


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


def test_short_segment_never_merges_across_speaker_change() -> None:
    # T0.21: diálogo A/B/A do critério de aceite — "Yes." (B) é curto
    # demais (0,3s, 1 palavra) mas os únicos vizinhos são do locutor A,
    # então não funde: fica como está e vem marcado `curto=True`. As duas
    # falas de A duram >= 3,2s cada, então elas NÃO ficam marcadas — só o
    # "Yes." isolado entre elas.
    words = [
        Word("Are", 0.0, 1.0, speaker="A"),
        Word("you", 1.0, 2.0, speaker="A"),
        Word("ready?", 2.0, 3.3, speaker="A"),  # 3,3s, dura >= min_duration
        Word("Yes.", 4.0, 4.3, speaker="B"),  # 0,3s, 1 palavra: curto demais
        Word("Then", 5.0, 5.8, speaker="A"),
        Word("let's", 5.8, 6.6, speaker="A"),
        Word("go", 6.6, 7.4, speaker="A"),
        Word("now", 7.4, 8.3, speaker="A"),  # 3,3s, dura >= min_duration
    ]
    segments = segment_words(words, min_duration=3.2, min_word_count=1)

    assert len(segments) == 3
    assert segments[0].texto == "Are you ready?"
    assert segments[0].speaker == "A"
    assert segments[0].curto is False

    assert segments[1].texto == "Yes."
    assert segments[1].speaker == "B"
    assert segments[1].curto is True

    assert segments[2].texto == "Then let's go now"
    assert segments[2].speaker == "A"
    assert segments[2].curto is False


def test_short_segment_at_start_with_no_same_speaker_neighbor_is_marked() -> None:
    # sem vizinho ALGUM do mesmo locutor (nem antes, nem depois) — tem que
    # ficar como está, marcado, não só no caso do meio do diálogo.
    words = [
        Word("Oi.", 0.0, 0.3, speaker="A"),
        Word("Tudo", 1.0, 1.4, speaker="B"),
        Word("bem", 1.4, 1.8, speaker="B"),
        Word("com", 1.8, 2.2, speaker="B"),
        Word("voce", 2.2, 4.5, speaker="B"),
    ]
    segments = segment_words(words, min_duration=3.2, min_word_count=1)

    assert len(segments) == 2
    assert segments[0].texto == "Oi."
    assert segments[0].curto is True
    assert segments[1].curto is False


def test_short_groups_of_same_speaker_still_merge_as_before() -> None:
    # T0.21, critério de aceite: fusão de grupos curtos do MESMO locutor
    # continua funcionando igual — mesmo cenário de
    # test_short_isolated_segment_is_merged_into_a_neighbor, agora com
    # locutor explícito nos dois lados (em vez de None) pra confirmar que
    # a checagem de compatibilidade não bloqueia esse caso.
    words = [
        Word("A", 0.0, 0.5, speaker="S1"),
        Word("B", 1.0, 1.15, speaker="S1"),
        Word("C", 1.75, 3.0, speaker="S1"),
    ]
    segments = segment_words(words, min_word_count=1)

    assert len(segments) == 2
    for seg in segments:
        assert seg.t_fim - seg.t_inicio >= 0.3
        assert seg.curto is False
    assert segments[1].texto == "B C"


def test_dominant_speaker_picks_speaker_with_most_total_duration() -> None:
    # locutor B fala só uma palavra curta no início, A fala o resto — a
    # versão antiga (primeira palavra com speaker) erraria pra B. Palavra
    # ponte sem locutor (`bridge`) é necessária pra tudo ficar num grupo
    # só: `_break_into_groups` só detecta troca de locutor entre palavras
    # ADJACENTES com locutor conhecido nos dois lados — B direto ao lado
    # de A quebraria o grupo (comportamento correto e intencional,
    # inalterado por esta tarefa); com uma palavra sem locutor no meio,
    # a troca "atravessa" sem quebrar, criando o grupo misto que
    # `_dominant_speaker` precisa resolver escolhendo por duração.
    words = [
        Word("Hm", 0.0, 0.2, speaker="B"),
        Word("bridge", 0.22, 0.3, speaker=None),
        Word("Então", 0.32, 1.0, speaker="A"),
        Word("é", 1.0, 1.2, speaker="A"),
        Word("isso", 1.2, 2.0, speaker="A"),
    ]
    segments = segment_words(words, min_word_count=1)

    assert len(segments) == 1
    assert segments[0].speaker == "A"


def test_dominant_speaker_none_when_no_speaker_assigned() -> None:
    words = [Word("A", 0.0, 0.5), Word("B", 0.6, 1.0)]
    segments = segment_words(words)
    assert segments[0].speaker is None


def test_dominant_speaker_single_speaker_group() -> None:
    words = [Word("A", 0.0, 0.5, speaker="S1"), Word("B", 0.6, 1.0, speaker="S1")]
    segments = segment_words(words)
    assert segments[0].speaker == "S1"
