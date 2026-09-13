"""Gates de qualidade.

Emite flags automáticas de suspeita de erro por segmento, pra alimentar o
editor (T2.x, ver CLAUDE.md glossário "Flag"): cada flag marca um jeito
específico de o pipeline ter "chutado" nesse segmento — não é veredito
final, o humano decide no editor.

Duas categorias, dependendo de onde mora o cálculo:

- **Calculável aqui** (`fora_duracao`, `clipping`, `silencio_anormal`): a
  métrica bruta é pura (duração/amplitude/RMS de um array numpy), então a
  função inteira mora neste módulo.
- **Calculada em GPU, só o limiar mora aqui** (`cer_alto`,
  `traducao_infiel`, `locutor_suspeito`): CER (`runpod/evaluate`),
  similaridade fonte/tradução (embeddings) e similaridade de locutor
  (WavLM-TDNN/ECAPA) exigem modelo — este módulo recebe o score já
  calculado por quem chama e só aplica o limiar. Não reimplementa nem
  chama modelo (regra de dependência de `packages/pipeline`, CLAUDE.md).

Limiares default são ponto de partida (não medidos em produção ainda) —
ajustar com dado real do editor (quantos segmentos com a flag o usuário
de fato corrige) assim que houver volume.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy as np

from packages.pipeline.budget import tolerance_range
from packages.ptbr.normalize import normalize as ptbr_normalize

# Piso medido em pares (texto sintetizado, transcrição ASR round-trip)
# julgados manualmente como corretos — sem repetição, sem corte, conteúdo
# batendo — de um vídeo de teste real do T0.12 (2026-09-13,
# scripts/_measure_cer_floor.py). SÓ 4 pares no cache atingiam esse
# critério com confiança (a maioria dos 16 segmentos daquela rodada tinha
# repetição/corte/conteúdo errado — não são "bons conhecidos", são
# exatamente o problema que T0.14 existe pra pegar). Com n=4: CER
# 0,00-0,089, média 0,0425. Amostra pequena demais pra uma medição
# definitiva (o ideal são uns 20, como a tarefa original pedia) — vale
# refazer com mais dado assim que houver uma rodada de síntese mais limpa.
# Mesmo assim, já confirma que CER_THRESHOLD=0,15 (valor original, mantido)
# tem margem real acima do piso observado (quase 2x o máximo medido) e
# ainda fica bem abaixo de onde erro de conteúdo real aparece nesses
# mesmos dados (~0,36+, ver docstring de `character_error_rate`).
CER_THRESHOLD = 0.15
TRANSLATION_FIDELITY_THRESHOLD = 0.75
SPEAKER_SIMILARITY_THRESHOLD = 0.75
CLIPPING_THRESHOLD = 0.99

# RMS abaixo disso conta como "silencioso" numa janela; achado do T0.6 é que
# o chão de ruído de pausas naturais fica em torno de -45dBFS (~0.0056 linear)
# — usar um limiar um pouco acima disso (~-40dBFS) evita marcar silêncio real
# de pausa natural como se fosse janela de fala.
SILENCE_RMS_THRESHOLD = 0.01
SILENCE_WINDOW_SECONDS = 0.05
SILENCE_MARGIN_SECONDS = 0.5


def fora_duracao(achieved_seconds: float, target_seconds: float, *, on_screen: bool) -> bool:
    """Duração real fora da tolerância de isocronia (±8% on-screen, ±20% off-screen)."""
    low, high = tolerance_range(target_seconds, on_screen=on_screen)
    return not (low <= achieved_seconds <= high)


def _normalize_for_cer(text: str) -> str:
    """Normaliza texto pra comparação de CER (T0.13): expande números por
    extenso via `packages.ptbr.normalize` (pra "2026" e "dois mil e vinte e
    seis" comparem iguais), minúsculas, sem pontuação, espaços colapsados.

    Sem isso, o CER cru comparava a tradução (com pontuação/caixa do
    tradutor) contra a transcrição do Whisper (pontuação/caixa própria, às
    vezes dígitos onde a referência tem número por extenso) — ruído que
    inflava o CER antes de existir qualquer erro real de conteúdo, achado
    revisando os primeiros resultados de T0.12 rodando em vídeo real.
    """
    text = ptbr_normalize(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER round-trip (CLAUDE.md glossário): distância de edição por caractere
    entre o texto que foi sintetizado (`reference`) e o que o ASR devolveu ao
    transcrever de volta o áudio sintetizado (`hypothesis`), normalizada pelo
    tamanho da referência, depois de `_normalize_for_cer` nos dois lados.
    0.0 = idêntico; pode passar de 1.0 se a hipótese tiver bem mais
    inserções que o tamanho da referência.

    **CER não pega truncamento** (T0.14 — achado real revisando por que
    frases cortadas escapavam do gate `fora_duracao`): o MOSS-TTS corta o
    texto pra caber na duração pedida, então o segmento truncado bate a
    duração alvo quase exatamente E, como CER é uma média sobre a frase
    inteira, perder as últimas palavras de uma frase longa dilui pouco o
    número (~0,1-0,15 numa frase de 25 palavras perdendo 3). Detecção de
    truncamento é lexical, não numérica — ver `truncado()`.
    """
    ref = _normalize_for_cer(reference)
    hyp = _normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


TRUNCADO_TAIL_FRACTION = 0.25
TRUNCADO_LENGTH_RATIO_THRESHOLD = 0.85
TRUNCADO_TAIL_COVERAGE_THRESHOLD = 0.5
TRUNCADO_MIN_REFERENCE_WORDS = 4


def truncado(
    reference: str,
    hypothesis: str,
    *,
    tail_fraction: float = TRUNCADO_TAIL_FRACTION,
    length_ratio_threshold: float = TRUNCADO_LENGTH_RATIO_THRESHOLD,
    tail_coverage_threshold: float = TRUNCADO_TAIL_COVERAGE_THRESHOLD,
) -> bool:
    """Detecta corte de texto (T0.14) — achado real: `fora_duracao` é cego a
    isso por construção (o MOSS-TTS corta a fala pra caber na duração
    pedida, então o segmento cortado bate o alvo de duração quase exato), e
    CER dilui pouco a perda quando é só o final de uma frase longa (ver
    docstring de `character_error_rate`). Detecção lexical, dois sinais:

    - **razão de palavras**: hipótese com bem menos palavras que a
      referência sugere que faltou conteúdo, não importa onde.
    - **cobertura do trecho final**: as últimas `tail_fraction` palavras da
      referência (arredondado pra cima, mínimo 1) precisam aparecer em
      algum lugar da hipótese — é exatamente essa cauda que o MOSS-TTS
      derruba quando corta.

    Referências curtas (<4 palavras) nunca disparam — não há trecho final
    que faça sentido isolar, e uma frase de 2-3 palavras naturalmente tem
    razão de palavras instável.
    """
    ref_words = _normalize_for_cer(reference).split()
    hyp_words = _normalize_for_cer(hypothesis).split()
    if len(ref_words) < TRUNCADO_MIN_REFERENCE_WORDS:
        return False

    length_ratio = len(hyp_words) / len(ref_words)

    tail_len = max(1, round(len(ref_words) * tail_fraction))
    tail = ref_words[-tail_len:]
    hyp_word_set = set(hyp_words)
    tail_coverage = sum(1 for w in tail if w in hyp_word_set) / len(tail)

    return length_ratio < length_ratio_threshold or tail_coverage < tail_coverage_threshold


def _levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


def cer_alto(cer: float, threshold: float = CER_THRESHOLD) -> bool:
    """CER round-trip (CLAUDE.md glossário) acima do limiar aceitável."""
    return cer > threshold


def has_repetition(text: str, min_window: int = 2, max_window: int = 12) -> bool:
    """Detecta se `text` tem uma sequência de N palavras repetida logo em
    seguida de si mesma (`palavra1 palavra2 ... palavra1 palavra2 ...`).

    Achado real do MOSS-TTS (T0.12, ver histórico da sessão): a síntese às
    vezes entra num loop e repete a mesma frase inteira, ou um trecho dela,
    em vez de terminar naturalmente. Isso aparece de forma confiável na
    transcrição ASR round-trip (`character_error_rate`'s `hypothesis`), não
    no canal de "texto" que o próprio MOSS-TTS devolve — esse canal só
    carrega marcadores de controle no modo de geração pura, não uma
    transcrição real (tentativa de usá-lo direto não funcionou, ver sessão).
    Comparação sem diferenciar maiúsculas/pontuação.
    """
    words = [re.sub(r"[^\w]", "", w.lower()) for w in text.split()]
    words = [w for w in words if w]
    n = len(words)
    for w in range(min_window, min(max_window, n // 2) + 1):
        for i in range(n - 2 * w + 1):
            if words[i : i + w] == words[i + w : i + 2 * w]:
                return True
    return False


def traducao_infiel(
    fidelity_score: float, threshold: float = TRANSLATION_FIDELITY_THRESHOLD
) -> bool:
    """Score de fidelidade (similaridade semântica fonte/tradução) abaixo do limiar.

    O score em si (embeddings ou LLM-juiz) é calculado fora deste módulo —
    ver docstring de `translation.py` sobre por que fidelidade factual não é
    algo que uma função pura consiga avaliar sozinha.
    """
    return fidelity_score < threshold


def clipping(audio: np.ndarray, threshold: float = CLIPPING_THRESHOLD) -> bool:
    return bool(np.any(np.abs(audio) > threshold))


def silencio_anormal(
    audio: np.ndarray,
    sample_rate: int,
    expected_pause_seconds: float,
    *,
    rms_threshold: float = SILENCE_RMS_THRESHOLD,
    margin_seconds: float = SILENCE_MARGIN_SECONDS,
    window_seconds: float = SILENCE_WINDOW_SECONDS,
) -> bool:
    """Sinaliza quando o segmento tem mais silêncio do que o esperado.

    `expected_pause_seconds` é a soma das pausas internas já conhecidas
    (`segmentation.InternalPause` + qualquer `[pause X.Ys]` injetada,
    ver `synthesis.py`). Mede silêncio real em janelas de RMS e compara
    contra esse orçamento mais uma margem — silêncio muito acima disso
    normalmente indica falha de síntese (ex.: o TTS "travou" e devolveu
    áudio maior com trecho morto), não pausa intencional.
    """
    window_samples = max(1, int(window_seconds * sample_rate))
    n_windows = len(audio) // window_samples
    if n_windows == 0:
        return False

    trimmed = audio[: n_windows * window_samples].reshape(n_windows, window_samples)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    silent_seconds = float(np.sum(rms < rms_threshold)) * window_seconds

    return silent_seconds > expected_pause_seconds + margin_seconds


def locutor_suspeito(
    speaker_similarity: float, threshold: float = SPEAKER_SIMILARITY_THRESHOLD
) -> bool:
    """Similaridade de locutor (WavLM-TDNN/ECAPA, CLAUDE.md) abaixo do limiar.

    Sinaliza quando a voz clonada usada na síntese não soa como o "perfil de
    voz" esperado do canal/locutor — o score em si vem de fora.
    """
    return speaker_similarity < threshold


@dataclass
class QualityFlags:
    fora_duracao: bool
    cer_alto: bool
    truncado: bool
    traducao_infiel: bool
    clipping: bool
    silencio_anormal: bool
    locutor_suspeito: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def evaluate_segment(
    *,
    achieved_seconds: float,
    target_seconds: float,
    on_screen: bool,
    cer: float,
    reference_text: str,
    hypothesis_text: str,
    fidelity_score: float,
    audio: np.ndarray,
    sample_rate: int,
    expected_pause_seconds: float,
    speaker_similarity: float,
) -> QualityFlags:
    """Roda todos os gates pra um segmento e monta o conjunto de flags."""
    return QualityFlags(
        fora_duracao=fora_duracao(achieved_seconds, target_seconds, on_screen=on_screen),
        cer_alto=cer_alto(cer),
        truncado=truncado(reference_text, hypothesis_text),
        traducao_infiel=traducao_infiel(fidelity_score),
        clipping=clipping(audio),
        silencio_anormal=silencio_anormal(audio, sample_rate, expected_pause_seconds),
        locutor_suspeito=locutor_suspeito(speaker_similarity),
    )
