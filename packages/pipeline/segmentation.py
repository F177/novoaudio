"""Segmentação prosódica.

Recebe palavras com timestamps e locutor (já extraídas da saída do WhisperX
por quem chama — este módulo não conhece o formato do WhisperX, só listas de
`Word`) e devolve segmentos: blocos de fala delimitados por pausa real,
alinhados com `CLAUDE.md` ("Segmento — bloco de fala delimitado por pausa
real no áudio original").

Regras de quebra, nesta ordem de prioridade:
- Troca de locutor.
- Gap entre o fim de uma palavra e o início da próxima > `hard_pause_min`
  ("âncora dura").
- Duração do segmento ultrapassaria `max_duration`.

Um gap entre `weak_pause_min` e `hard_pause_min` ("pausa fraca") não quebra
o segmento, mas fica registrado em `pausas_internas` — é isso que
`packages/pipeline/synthesis.py` (T0.9) usa para decidir onde inserir
`[pause X.Ys]` no texto sintetizado. Gaps menores que `weak_pause_min` são
ignorados por completo (nem quebram, nem viram pausa registrada).

Depois de quebrar, segmentos mais curtos que `min_duration` são fundidos a
um vizinho (a pausa real que os isolava não desaparece — vira uma pausa
interna, possivelmente longa, dentro do segmento fundido). Isso é o que
garante o "nenhum segmento <0,3s" do critério de aceite mesmo quando uma
interjeição curta fica isolada por âncoras duras dos dois lados.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_WEAK_PAUSE_MIN = 0.12
DEFAULT_HARD_PAUSE_MIN = 0.35
DEFAULT_MAX_DURATION = 12.0
DEFAULT_MIN_DURATION = 0.3


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class InternalPause:
    after_word_index: int
    duration: float


@dataclass
class Segment:
    t_inicio: float
    t_fim: float
    texto: str
    speaker: str | None
    pausas_internas: list[InternalPause] = field(default_factory=list)


_WordGroup = tuple[list[Word], list[InternalPause]]


def segment_words(
    words: list[Word],
    *,
    weak_pause_min: float = DEFAULT_WEAK_PAUSE_MIN,
    hard_pause_min: float = DEFAULT_HARD_PAUSE_MIN,
    max_duration: float = DEFAULT_MAX_DURATION,
    min_duration: float = DEFAULT_MIN_DURATION,
) -> list[Segment]:
    if not words:
        return []

    groups = _break_into_groups(words, hard_pause_min=hard_pause_min, max_duration=max_duration)
    groups = _merge_short_groups(groups, min_duration=min_duration, max_duration=max_duration)
    groups = _mark_weak_pauses(groups, weak_pause_min=weak_pause_min)

    return [
        Segment(
            t_inicio=group_words[0].start,
            t_fim=group_words[-1].end,
            texto=" ".join(w.text for w in group_words),
            speaker=_dominant_speaker(group_words),
            pausas_internas=pauses,
        )
        for group_words, pauses in groups
    ]


def _break_into_groups(
    words: list[Word], *, hard_pause_min: float, max_duration: float
) -> list[list[Word]]:
    groups: list[list[Word]] = [[words[0]]]

    for prev, curr in zip(words, words[1:], strict=False):
        current = groups[-1]
        gap = curr.start - prev.end
        speaker_changed = (
            curr.speaker is not None and prev.speaker is not None and curr.speaker != prev.speaker
        )
        is_hard_anchor = gap > hard_pause_min
        would_exceed_max = (curr.end - current[0].start) > max_duration

        if is_hard_anchor or speaker_changed or would_exceed_max:
            groups.append([curr])
        else:
            current.append(curr)

    return groups


def _merge_short_groups(
    groups: list[list[Word]], *, min_duration: float, max_duration: float
) -> list[list[Word]]:
    def duration(group: list[Word]) -> float:
        return group[-1].end - group[0].start

    merged = list(groups)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, group in enumerate(merged):
            if duration(group) >= min_duration:
                continue
            if i + 1 < len(merged) and duration(group + merged[i + 1]) <= max_duration:
                merged[i : i + 2] = [group + merged[i + 1]]
            elif i > 0 and duration(merged[i - 1] + group) <= max_duration:
                merged[i - 1 : i + 1] = [merged[i - 1] + group]
            else:
                continue
            changed = True
            break

    return merged


def _mark_weak_pauses(groups: list[list[Word]], *, weak_pause_min: float) -> list[_WordGroup]:
    result: list[_WordGroup] = []
    for group in groups:
        pauses: list[InternalPause] = []
        for i, (prev, curr) in enumerate(zip(group, group[1:], strict=False)):
            gap = curr.start - prev.end
            if gap >= weak_pause_min:
                pauses.append(InternalPause(after_word_index=i, duration=gap))
        result.append((group, pauses))
    return result


def _dominant_speaker(words: list[Word]) -> str | None:
    for w in words:
        if w.speaker is not None:
            return w.speaker
    return None
