"""Síntese com duração e pausa.

Converte um orçamento de duração em `tokens` pro MOSS-TTS, injeta os
marcadores `[pause X.Ys]` nas posições certas do texto traduzido, aplica
IPA do glossário do canal, e (via `runpod/synthesize/handler.py`, que
importa este módulo) decide o ajuste de uma segunda tentativa quando a
duração real sai da tolerância.

**Achado confirmado experimentalmente (T0.9) — pausa soma ou conta
dentro do orçamento?** A pausa `[pause X.Ys]` CONTA DENTRO do orçamento
de `tokens`, não soma por cima. Testado direto no MOSS-TTS: sintetizar o
mesmo texto com `tokens=100` (8,0s nominal), uma vez sem pausa e outra
com "[pause 2.0s]" no meio, deu exatamente a mesma duração real (8,08s
nos dois casos) — o modelo comprime a fala pra abrir espaço pra pausa
dentro do mesmo orçamento total, em vez de estender a duração. Conferido
também que a pausa é real (silêncio digital de ~-180dBFS na posição
esperada da forma de onda — bem mais limpo que o chão de ruído de
~-45dBFS que aparece em pausas *naturais* entre frases sem marcador
explícito, achado do T0.6).

Implicação prática: `tokens = round(target_seconds * 12.5)` usa a duração
alvo TOTAL do segmento, já incluindo o tempo que as pausas vão ocupar —
não subtrair as pausas antes de converter. É consistente com a
calibração do T0.6 (`calibration.json`), que já foi medida sobre textos
com marcadores de pausa embutidos.

**Validação real (36 segmentos, `translation_data/synth_validation_results.json`
— gitignored, dados locais):** 86,1% dos segmentos ficaram dentro de ±8%
após no máximo 1 ajuste (meta do T0.9: ≥85%). As falhas são quase todas
segmentos com alvo muito curto (0,3-0,8s) — parece existir um **piso de
duração mínima viável** do MOSS-TTS: pedir `tokens` correspondentes a
menos de ~1s às vezes ainda produz 1,7-3,3s de áudio, não importa o
ajuste na segunda tentativa. `adjust_tokens_for_retry` assume que a
relação segundos/token é aproximadamente linear e escalável — isso quebra
perto desse piso. Não investigado mais a fundo ainda; se segmentos muito
curtos continuarem sendo um problema recorrente em produção, vale medir
onde fica esse piso exatamente e tratar segmentos abaixo dele como um
caso especial (ex.: fundir com o vizinho antes de sintetizar, como
`segmentation.py` já faz pra segmentos <0,3s).
"""

from __future__ import annotations

import re

from packages.pipeline.segmentation import InternalPause

TOKENS_PER_SECOND = 12.5


def duration_to_tokens(target_seconds: float) -> int:
    """Converte segundos em `tokens` pro MOSS-TTS (12,5 Hz, granularidade 80ms).

    `target_seconds` é a duração TOTAL alvo do segmento, já incluindo
    qualquer pausa interna que for injetada no texto — ver docstring do
    módulo sobre por que pausas não somam por cima desse orçamento.
    """
    return round(target_seconds * TOKENS_PER_SECOND)


def inject_pauses(
    translated_text: str, pausas: list[InternalPause], original_word_count: int
) -> str:
    """Insere `[pause X.Ys]` no texto traduzido, nas posições relativas às
    pausas detectadas no áudio original (T0.5).

    Não há alinhamento palavra-a-palavra entre o texto original e a
    tradução — a posição é aproximada pela fração do caminho onde a pausa
    ocorria no original (ex.: pausa depois de 60% das palavras originais
    vira uma pausa depois de ~60% das palavras traduzidas). É uma
    heurística, não uma garantia de alinhamento perfeito.
    """
    if not pausas or original_word_count <= 0:
        return translated_text

    words = translated_text.split()
    if not words:
        return translated_text

    insertions: list[tuple[int, float]] = []
    for pause in pausas:
        relative_position = (pause.after_word_index + 1) / original_word_count
        target_index = max(0, min(len(words) - 1, round(relative_position * len(words)) - 1))
        insertions.append((target_index, pause.duration))

    # aplica de trás pra frente pra não bagunçar os índices ainda não usados
    insertions.sort(key=lambda item: item[0], reverse=True)
    for index, duration in insertions:
        words.insert(index + 1, f"[pause {duration:.1f}s]")

    return " ".join(words)


def apply_ipa_overrides(text: str, glossary: dict[str, str]) -> str:
    """Troca termos do glossário do canal (T1.8) pela forma IPA fixa entre
    barras, ex. `{"OpenAI": "oʊˈpɛn aɪ"}` -> "...`/oʊˈpɛn aɪ/`...".

    Comparação sem diferenciar maiúsculas/minúsculas; troca a primeira
    ocorrência de cada termo (glossário por canal ainda não existe — T1.8 —
    então não há caso real de múltiplas ocorrências pra decidir política).
    """
    result = text
    for term, ipa in glossary.items():
        result = re.sub(re.escape(term), f"/{ipa}/", result, count=1, flags=re.IGNORECASE)
    return result


def is_within_tolerance(achieved_seconds: float, target_seconds: float, tolerance: float) -> bool:
    if target_seconds <= 0:
        return False
    return abs(achieved_seconds - target_seconds) / target_seconds <= tolerance


def adjust_tokens_for_retry(
    current_tokens: int, achieved_seconds: float, target_seconds: float
) -> int:
    """Calcula os `tokens` pra uma segunda tentativa, quando a primeira saiu
    da tolerância.

    Em vez de reaplicar 12,5 Hz nominal, usa a taxa real observada nesta
    síntese específica (`achieved_seconds / current_tokens`) — o modelo não
    bate exatamente 12,5 Hz sempre, e escalar pela taxa real corrige o erro
    sistemático daquele texto específico, não só arredondamento.
    """
    if current_tokens <= 0 or achieved_seconds <= 0:
        return duration_to_tokens(target_seconds)
    seconds_per_token = achieved_seconds / current_tokens
    return max(1, round(target_seconds / seconds_per_token))
