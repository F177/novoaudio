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
segmentos com alvo muito curto (0,3-0,8s) — havia um **piso de duração
mínima viável** do MOSS-TTS, na época não explicado: pedir `tokens`
correspondentes a menos de ~1s às vezes ainda produz 1,7-3,3s de áudio,
não importa o ajuste na segunda tentativa.

**Explicado e corrigido (investigação de 2026-09-14, ver
`docs/moss_tts_investigation.md`):** o MOSS-TTS-v1.5 usa `n_vq=32`
codebooks de áudio em padrão "delay pattern" (cada codebook começa
defasado do anterior, tipo MusicGen) — o modelo precisa de pelo menos
~`n_vq` frames só pra todos os canais entrarem em regime antes de
produzir qualquer coisa coerente. A 12,5 Hz isso é `32/12,5 = 2,56s`.
Pedir menos tokens que isso corta o ciclo pela metade: confirmado
experimentalmente que segmentos assim saem com hipótese de ASR vazia,
texto sem nenhuma relação com o pedido, ou a mesma palavra bugada em
várias grafias — não é falta de sorte no retry, é estrutural. Dando pelo
menos `MIN_SYNTHESIS_TOKENS` de orçamento, o mesmo texto sintetiza
correto (confirmado em 2 de 3 casos reais testados; o terceiro, uma
palavra isolada sem frase ao redor, ainda falhou mesmo com mais tokens —
parece precisar de mais contexto textual, não só mais tempo; não
resolvido ainda).

`duration_to_tokens`/`adjust_tokens_for_retry` agora nunca pedem menos que
o piso — o excedente de duração fica para a montagem tratar (comprimir ou
just aceitar a folga e deixar o gate `fora_duracao` sinalizar pra revisão
manual; time-stretch de verdade ainda não implementado, ver
`docs/moss_tts_investigation.md`).
"""

from __future__ import annotations

import re

import librosa
import numpy as np

from packages.pipeline.segmentation import InternalPause

TOKENS_PER_SECOND = 12.5

# MOSS-TTS-v1.5 usa n_vq=32 (ver configuration_moss_tts.py do modelo,
# confirmado no cache do Pod) — abaixo disso o "delay pattern" nunca
# completa um ciclo. +25% de margem sobre o piso teórico (2,56s) porque a
# validação real mostrou que só acertar exatamente n_vq/12.5 ainda deixa
# pouca folga pro conteúdo de fato caber antes do fim do ciclo.
MOSS_TTS_N_VQ = 32
MIN_SYNTHESIS_SECONDS = (MOSS_TTS_N_VQ / TOKENS_PER_SECOND) * 1.25
MIN_SYNTHESIS_TOKENS = round(MIN_SYNTHESIS_SECONDS * TOKENS_PER_SECOND)


def duration_to_tokens(target_seconds: float) -> int:
    """Converte segundos em `tokens` pro MOSS-TTS (12,5 Hz, granularidade 80ms).

    `target_seconds` é a duração TOTAL alvo do segmento, já incluindo
    qualquer pausa interna que for injetada no texto — ver docstring do
    módulo sobre por que pausas não somam por cima desse orçamento.

    Nunca devolve menos que `MIN_SYNTHESIS_TOKENS` — abaixo do piso do
    delay pattern do MOSS-TTS, ver docstring do módulo.
    """
    return max(MIN_SYNTHESIS_TOKENS, round(target_seconds * TOKENS_PER_SECOND))


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


def infer_delivery_instruction(text: str) -> str | None:
    """Deriva uma instrução de estilo de fala pro MOSS-TTS a partir da
    pontuação do texto já traduzido (`instruction` de
    `processor.build_user_message` — controla tom/emoção/ritmo, separado
    do texto a falar).

    Mesma lógica de `packages/pipeline/translation.py::_expressiveness_score`:
    "!"/"?" são o único sinal de tom que temos sem processar o áudio
    original (sem pitch/energia extraídos ainda — ver
    docs/moss_tts_investigation.md). `None` = deixa o modelo decidir
    sozinho, comportamento de antes desta função existir.

    Checa "!" antes de "?" de propósito: uma frase como "Corre, agora!?"
    tem os dois, e ênfase/urgência (exclamação) é o sinal mais forte dos
    dois pra decidir tom de voz.

    Não validado contra o modelo real ainda — hipótese, não medição.
    """
    if "!" in text:
        return "fale com ênfase e emoção genuína, não leia como narração neutra de audiobook"
    if "?" in text:
        return "fale com entonação de pergunta real, subindo o tom no final"
    return None


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

    Nunca devolve menos que `MIN_SYNTHESIS_TOKENS` (mesmo piso de
    `duration_to_tokens`) — sem isso, um segmento curto que já falhou por
    estar abaixo do piso pediria tokens ainda mais baixos na retry.
    """
    if current_tokens <= 0 or achieved_seconds <= 0:
        return duration_to_tokens(target_seconds)
    seconds_per_token = achieved_seconds / current_tokens
    return max(MIN_SYNTHESIS_TOKENS, round(target_seconds / seconds_per_token))


# Limite de compressão/estiramento que ainda soa como fala reconhecível
# (phase vocoder do librosa, preserva pitch). MEDIDO, não só citado: um
# áudio conhecido bom ("Ele precisa, decide agora.", transcrito perfeito
# sem stretch) virou "Ele precisa desse de agora..." em 2.0x — ilegível.
# Testando 1.2/1.3/1.5/1.7x no mesmo áudio, só 1.5x manteve o conteúdo
# correto e sem repetição real. Não é um estudo rigoroso (n=1 caso, poucos
# valores testados) — é o piso de segurança real que temos por enquanto.
# Acima disso, a fala fica rápida/lenta demais pra soar bem — melhor
# aceitar a duração fora do alvo e deixar o gate `fora_duracao` sinalizar
# pra revisão manual do que forçar um resultado provavelmente ilegível.
MAX_TIME_STRETCH_RATIO = 1.5


def time_stretch_to_duration(
    audio: np.ndarray,
    sample_rate: int,
    target_seconds: float,
    *,
    max_ratio: float = MAX_TIME_STRETCH_RATIO,
) -> np.ndarray:
    """Comprime ou estica `audio` (phase vocoder, preserva pitch) pra chegar
    o mais perto possível de `target_seconds`.

    Existe especificamente pro piso de `MIN_SYNTHESIS_TOKENS`: um segmento
    curto sintetizado no piso sai mais longo que o alvo real, e isso
    comprime de volta em vez de deixar o segmento estourar a timeline (ver
    `assembly.py` T0.16) ou tocar mais devagar que devia.

    A razão de compressão/estiramento é limitada a `max_ratio` (default
    `MAX_TIME_STRETCH_RATIO`) — acima disso a fala fica rápida/devagar
    demais pra soar natural. Nesse caso aplica só o `max_ratio` (chega o
    mais perto possível do alvo, não bate exato) — quem chama ainda precisa
    checar `is_within_tolerance` depois, isso não garante bater o alvo.
    """
    current_seconds = len(audio) / sample_rate
    if current_seconds <= 0 or target_seconds <= 0:
        return audio

    rate = current_seconds / target_seconds
    if abs(rate - 1.0) < 1e-3:
        return audio

    clamped_rate = max(1.0 / max_ratio, min(max_ratio, rate))
    stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=clamped_rate)
    return stretched
