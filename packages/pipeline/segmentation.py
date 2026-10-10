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
um vizinho do MESMO locutor (a pausa real que os isolava não desaparece —
vira uma pausa interna, possivelmente longa, dentro do segmento fundido).
Isso é o que garante o "nenhum segmento <0,3s" do critério de aceite mesmo
quando uma interjeição curta fica isolada por âncoras duras dos dois
lados. Fusão NUNCA atravessa troca de locutor (T0.21) — misturaria a fala
de duas pessoas num segmento só, distorcendo a tradução. Um grupo curto
sem vizinho do mesmo locutor fica como está, marcado `curto=True`
(`Segment.curto`, exposto no editor como a flag `segmento_curto`).

Segmentos com poucas palavras (`min_word_count`) são fundidos do mesmo
jeito (mesma regra de locutor), mesmo quando já duram mais que
`min_duration` — achado real da investigação de qualidade do MOSS-TTS
(`docs/moss_tts_investigation.md`): uma frase/palavra isolada sem
contexto ao redor falha na síntese mesmo com orçamento de tempo grande
(testado até 8s pra uma palavra só, sem melhorar) — o problema é a falta
de CONTEXTO linguístico, não de tempo. Fundir aqui, antes da tradução,
garante que o texto traduzido carrega mais conteúdo real, em vez de só
esticar o áudio de uma palavra isolada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_WEAK_PAUSE_MIN = 0.12
DEFAULT_HARD_PAUSE_MIN = 0.35
DEFAULT_MAX_DURATION = 12.0
DEFAULT_MIN_DURATION = 0.3
DEFAULT_MIN_WORD_COUNT = 4


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
    # T0.21: True quando o segmento ficou mais curto que `min_duration`
    # (ou com menos palavras que `min_word_count`) mas NENHUM vizinho do
    # mesmo locutor estava disponível pra fundir — fica como está em vez
    # de fundir através de uma troca de locutor (o que distorceria a
    # tradução, misturando falas de pessoas diferentes). Visível no
    # editor como a flag `segmento_curto` (ver `quality.QualityFlags`).
    curto: bool = False


_WordGroup = tuple[list[Word], list[InternalPause], bool]


def segment_words(
    words: list[Word],
    *,
    weak_pause_min: float = DEFAULT_WEAK_PAUSE_MIN,
    hard_pause_min: float = DEFAULT_HARD_PAUSE_MIN,
    max_duration: float = DEFAULT_MAX_DURATION,
    min_duration: float = DEFAULT_MIN_DURATION,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
) -> list[Segment]:
    if not words:
        return []

    groups = _break_into_groups(words, hard_pause_min=hard_pause_min, max_duration=max_duration)
    groups = _merge_short_groups(
        groups, min_duration=min_duration, max_duration=max_duration, min_word_count=min_word_count
    )
    groups = _mark_weak_pauses(groups, weak_pause_min=weak_pause_min)

    return [
        Segment(
            t_inicio=group_words[0].start,
            t_fim=group_words[-1].end,
            texto=" ".join(w.text for w in group_words),
            speaker=_dominant_speaker(group_words),
            pausas_internas=pauses,
            curto=curto,
        )
        for group_words, pauses, curto in groups
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


def _group_speaker(group: list[Word]) -> str | None:
    for w in group:
        if w.speaker is not None:
            return w.speaker
    return None


def _mergeable(a: list[Word], b: list[Word]) -> bool:
    """Dois grupos podem se fundir se não tiverem locutores DIFERENTES e
    conhecidos — `None` (sem diarização) nunca bloqueia, mesmo critério
    já usado em `_break_into_groups` pra decidir troca de locutor."""
    speaker_a, speaker_b = _group_speaker(a), _group_speaker(b)
    return speaker_a is None or speaker_b is None or speaker_a == speaker_b


def _merge_short_groups(
    groups: list[list[Word]], *, min_duration: float, max_duration: float, min_word_count: int = 1
) -> list[tuple[list[Word], bool]]:
    def duration(group: list[Word]) -> float:
        return group[-1].end - group[0].start

    def is_too_short(group: list[Word]) -> bool:
        return duration(group) < min_duration or len(group) < min_word_count

    merged = list(groups)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, group in enumerate(merged):
            if not is_too_short(group):
                continue
            # T0.21: fusão nunca atravessa troca de locutor — um grupo
            # curto sem vizinho do MESMO locutor fica como está (marcado
            # abaixo), em vez de fundir com o vizinho errado e misturar
            # a fala de duas pessoas num segmento só.
            if (
                i + 1 < len(merged)
                and _mergeable(group, merged[i + 1])
                and duration(group + merged[i + 1]) <= max_duration
            ):
                merged[i : i + 2] = [group + merged[i + 1]]
            elif (
                i > 0
                and _mergeable(merged[i - 1], group)
                and duration(merged[i - 1] + group) <= max_duration
            ):
                merged[i - 1 : i + 1] = [merged[i - 1] + group]
            else:
                continue
            changed = True
            break

    return [(group, is_too_short(group)) for group in merged]


def _mark_weak_pauses(
    groups: list[tuple[list[Word], bool]], *, weak_pause_min: float
) -> list[_WordGroup]:
    result: list[_WordGroup] = []
    for group, curto in groups:
        pauses: list[InternalPause] = []
        for i, (prev, curr) in enumerate(zip(group, group[1:], strict=False)):
            gap = curr.start - prev.end
            if gap >= weak_pause_min:
                pauses.append(InternalPause(after_word_index=i, duration=gap))
        result.append((group, pauses, curto))
    return result


def _dominant_speaker(words: list[Word]) -> str | None:
    """Locutor com mais tempo de fala no grupo (T0.21) — não o da
    primeira palavra, que erra quando um locutor fala só uma palavra
    curta no início e o resto do grupo é de outro locutor."""
    totals: dict[str, float] = {}
    for w in words:
        if w.speaker is None:
            continue
        totals[w.speaker] = totals.get(w.speaker, 0.0) + (w.end - w.start)
    if not totals:
        return None
    return max(totals, key=lambda speaker: totals[speaker])


# --- Pontuação restaurada --------------------------------------------------
# Achado real (2026-10-10): o WhisperX às vezes devolve trechos longos sem
# pontuação nenhuma, e o tradutor erra a estrutura ("but you're not you're
# calm" virou "mas cê não tá calmo"). Pedir pontuação ao próprio Whisper
# (initial_prompt) fez ele PULAR uma frase inteira num Short — descartado. Um
# modelo de pontuação só acrescenta sinal às palavras já transcritas; nunca
# tira nem troca palavra.

# Sequência de palavras sem nenhum sinal a partir da qual o trecho é
# repontuado. Ponto de partida não calibrado.
PUNCTUATION_RUN_WORDS = 12

_ENDS_WITH_PUNCTUATION = re.compile(r"[.,;:!?…\-—][\"')\]]*$")


def needs_punctuation(words: list[str], run: int = PUNCTUATION_RUN_WORDS) -> bool:
    streak = 0
    for word in words:
        streak = 0 if _ENDS_WITH_PUNCTUATION.search(word) else streak + 1
        if streak >= run:
            return True
    return False


def apply_punctuation(words: list[str], marks: list[str]) -> list[str]:
    """`marks[i]` é o sinal previsto depois da palavra i ("." "," "?" ":" ou
    nada). Só acrescenta onde a palavra ainda não tem sinal, e põe maiúscula
    depois de fim de frase; o texto das palavras nunca muda."""
    out: list[str] = []
    sentence_start = False
    for word, mark in zip(words, marks, strict=True):
        if sentence_start and word[:1].islower():
            word = word[:1].upper() + word[1:]
        if mark in {".", ",", "?", ":"} and not _ENDS_WITH_PUNCTUATION.search(word):
            word += mark
        sentence_start = word.endswith((".", "?", "!"))
        out.append(word)
    return out

