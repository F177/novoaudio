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

# T0.19: limiar de RMS pra separar "rajada de fala" de silêncio entre
# rajadas — diferente de SILENCE_RMS_THRESHOLD (esse é "silêncio total",
# mais baixo). Validado contra o único dado real que temos: o áudio de
# `seg0009` citado em docs/moss_tts_investigation.md (1,84s, referência
# "Deixa as bobagens pra lá.", 5 palavras) tem, à mão, 5-6 rajadas reais.
# Com esse limiar e SILENCE_WINDOW_SECONDS, a contagem automática bate em
# 5 nesse áudio exato (testado em `.cache/moss_tts_src/
# rep_test_seg0009_base.wav`) — é o único ponto de calibração que existe,
# não uma média sobre volume.
BURST_RMS_THRESHOLD = 0.03
# Ponto de partida, não calibrado (mesmo aviso dos outros limiares deste
# módulo): quantas vezes mais rajadas do que PALAVRAS esperadas conta
# como loop de verdade, em vez de variação normal de ritmo de fala.
LOOP_BURST_MARGIN = 1.5


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


def count_energy_bursts(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float = SILENCE_WINDOW_SECONDS,
    rms_threshold: float = BURST_RMS_THRESHOLD,
) -> int:
    """Conta rajadas de energia no áudio — aproximação de "quantas vezes a
    fala soou", pelo mesmo método usado à mão no achado do `seg0009`
    (T0.19, ver docs/moss_tts_investigation.md): janela de RMS acima do
    limiar começando logo depois de uma janela abaixo é UMA rajada nova.

    Sinal independente do ASR, pra desempatar `has_repetition`: loop real
    no áudio soa fisicamente mais vezes que o esperado; alucinação do
    Whisper (texto repetido na TRANSCRIÇÃO, áudio intacto) não muda o
    número de rajadas reais. Correlaciona com contagem de PALAVRAS da
    referência, não de sílabas — ver `loop_confirmed_by_bursts`.
    """
    window = max(1, int(window_seconds * sample_rate))
    n_windows = len(audio) // window
    if n_windows == 0:
        return 0
    trimmed = audio[: n_windows * window].reshape(n_windows, window)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    voiced = rms >= rms_threshold
    return int(np.sum(voiced[1:] & ~voiced[:-1])) + int(voiced[0])


def loop_confirmed_by_bursts(
    burst_count: int, expected_words: int, *, margin: float = LOOP_BURST_MARGIN
) -> bool:
    """`burst_count` muito acima do esperado confirma loop real no áudio,
    em vez de alucinação do Whisper na transcrição (T0.19) — ver
    `count_energy_bursts`.

    Comparação é com a contagem de PALAVRAS da referência, não de
    sílabas — achado real desta tarefa: `count_energy_bursts` mede rajada
    no nível de palavra/frase curta, não núcleo fonético por sílaba.
    Confirmado em dois pontos de dado real: o `seg0009` documentado (5
    palavras, 5-6 rajadas) e o teste de mutação da T0.19 (10 palavras,
    ~12 rajadas numa cópia, ~24 em duas coladas) — comparar contra sílabas
    (23 nesse segundo caso) deixava até a frase colada 2x invisível ao
    limiar (24 não passa de 23×1,5).

    Referência sem nenhuma palavra esperada (texto vazio) não tem base de
    comparação — qualquer rajada conta como suspeita nesse caso raro.
    """
    if expected_words <= 0:
        return burst_count > 0
    return burst_count > expected_words * margin


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


# Achado real 2026-09-26: o usuário pediu uma avaliação automática de
# "entonação", não só ouvido. Em vez de um modelo de emoção (categoria
# "raiva"/"alegria"), a versão barata e sem modelo novo é comparar
# diretamente os contornos de energia/pitch que `prosody.py` já extrai
# (mesmo dado usado pra TRANSFERIR prosódia agora vira métrica de
# avaliação também) — correlação de Pearson entre o contorno do original e
# o do dublado.
#
# T0.20, achado real: essa correlação deixou de alimentar a flag
# `entonacao_desalinhada` (ver abaixo) por dois problemas medidos — (1)
# era calculada DEPOIS de `apply_energy_envelope` já ter imposto esse
# mesmo contorno no dublado (quando `--prosody-transfer` está ligado), e
# a melhora registrada (0,235→0,353) era em boa parte a correção sendo
# medida contra si mesma; (2) correlação quadro a quadro entre inglês e
# português tem pouco significado — sílabas e ordem de palavras não se
# alinham no tempo entre os dois idiomas. A função continua aqui como
# diagnóstico puro (relatório, não gate).
PROSODY_CORRELATION_THRESHOLD = 0.3


def prosody_correlation(source_contour: np.ndarray, dub_contour: np.ndarray) -> float:
    """Correlação de Pearson entre dois contornos JÁ do mesmo tamanho
    (energia OU pitch relativo, mesmo formato de `packages.pipeline.prosody`)
    — mede se a entonação/energia do áudio dublado acompanha a forma do
    original, número objetivo em vez de impressão de ouvido.

    1.0 = contornos com a mesma forma; 0.0 = sem relação linear; -1.0 =
    forma invertida. Contorno sem nenhuma variação (todo constante, ex.:
    segmento sem fala vozeada detectada) não tem correlação definida
    (desvio padrão zero) — devolve 0.0 nesse caso, não NaN: não há
    contorno de verdade pra comparar, mas isso não é evidência de
    desalinhamento, é ausência de sinal.
    """
    if len(source_contour) != len(dub_contour):
        raise ValueError(
            "os dois contornos precisam ter o mesmo tamanho — resample "
            "(ver prosody.resample_envelope) antes de comparar"
        )
    if len(source_contour) < 2:
        return 0.0
    if np.std(source_contour) == 0 or np.std(dub_contour) == 0:
        return 0.0
    return float(np.corrcoef(source_contour, dub_contour)[0, 1])


# T0.20: ponto de partida, não calibrado (mesmo aviso dos outros limiares
# deste módulo) — dublado com desvio de F0 abaixo dessa fração do desvio
# da fonte conta como entrega achatada/monótona.
PITCH_FLATNESS_RATIO = 0.5


def extract_pitch_stats(pitch_contour: np.ndarray) -> dict[str, float]:
    """Estatísticas agregadas de um contorno de pitch relativo (mesmo
    formato de `prosody.extract_pitch_contour`: semitons relativos à
    mediana vozeada do segmento, 0.0 em frame não-vozado) — faixa, desvio
    padrão e inclinação ao longo do tempo.

    T0.20: substitui correlação quadro a quadro como base da flag
    `entonacao_desalinhada` — correlação entre EN/PT quadro a quadro não
    significa muito (sílabas e ordem de palavras não se alinham no
    tempo), e essas estatísticas resumem a VARIAÇÃO real de entonação por
    segmento, que `apply_energy_envelope` (só muda ganho) nunca toca —
    diferente da correlação de energia antiga, não é mais tautológico com
    a transferência de prosódia.

    Frames não-vozados (0.0) são excluídos do cálculo — não contam como
    "pitch plano de verdade", só ausência de sinal. Menos de 2 frames
    vozados não tem desvio/inclinação definidos — devolve tudo 0.0.
    """
    voiced_idx = np.flatnonzero(pitch_contour != 0.0)
    if len(voiced_idx) < 2:
        return {"range": 0.0, "std": 0.0, "slope": 0.0}
    voiced = pitch_contour[voiced_idx]
    return {
        "range": float(np.max(voiced) - np.min(voiced)),
        "std": float(np.std(voiced)),
        "slope": float(np.polyfit(voiced_idx, voiced, 1)[0]),
    }


def entonacao_desalinhada(
    source_pitch_std: float, dub_pitch_std: float, *, min_ratio: float = PITCH_FLATNESS_RATIO
) -> bool:
    """Dublado com desvio de F0 (`extract_pitch_stats`) muito menor que a
    fonte sinaliza entrega achatada/monótona (T0.20) — mede variação real
    de entonação por segmento, não correlação quadro a quadro entre
    idiomas diferentes.

    Fonte sem variação de pitch detectável (`source_pitch_std<=0`, ex.:
    segmento todo não-vozado) não tem base de comparação — não sinaliza;
    ausência de sinal na própria fonte não é evidência de desalinhamento.
    """
    if source_pitch_std <= 0:
        return False
    return dub_pitch_std < source_pitch_std * min_ratio


# Achado real 2026-09-26, item 4 do plano de avaliação de qualidade pedido
# pelo usuário: emoção da entrega (raiva, alegria, neutro...) não é
# capturada por CER nem pela correlação de contorno de energia/pitch acima
# — dois áudios podem ter volume/entonação parecidos e ainda soar
# emocionalmente errados (ex.: original irritado, dublado neutro).
# `emotion2vec` (ACL 2024, `iic/emotion2vec_plus_large` via FunASR) roda no
# Pod (ver `scripts/pod_worker.py::cmd_match_emotion`) e devolve um vetor
# de scores por classe de emoção para um trecho de áudio; a comparação em
# si (esta função) é pura — só similaridade de cosseno entre dois vetores
# já calculados, mesmo padrão de `traducao_infiel`/`locutor_suspeito`
# (modelo fora deste módulo, limiar aqui).
EMOTION_SIMILARITY_THRESHOLD = 0.5  # ponto de partida, não calibrado


def emotion_similarity(source_scores: np.ndarray, dub_scores: np.ndarray) -> float:
    """Similaridade de cosseno entre dois vetores de score de emoção (mesma
    ordem de classes nos dois, ver `cmd_match_emotion`) — 1.0 = mesma
    distribuição emocional, 0.0 = ortogonal, -1.0 = oposta.

    Vetor nulo (score zero em toda classe — não deveria acontecer com o
    modelo real, mas pode acontecer em teste sintético) devolve 0.0 em vez
    de NaN: sem sinal de emoção pra comparar não é evidência de
    incompatibilidade.
    """
    if len(source_scores) != len(dub_scores):
        raise ValueError("os dois vetores de score precisam ter a mesma ordem de classes")
    source_norm = np.linalg.norm(source_scores)
    dub_norm = np.linalg.norm(dub_scores)
    if source_norm == 0 or dub_norm == 0:
        return 0.0
    return float(np.dot(source_scores, dub_scores) / (source_norm * dub_norm))


def emocao_incompativel(
    similarity: float, threshold: float = EMOTION_SIMILARITY_THRESHOLD
) -> bool:
    """Similaridade de emoção (fonte vs. dublado) abaixo do limiar."""
    return similarity < threshold


def fala_fragmentada(texto_original: str) -> bool:
    """Segmento cujo texto original (inglês) começa em minúscula — achado
    real (vídeo padrão, 2026-10-09): o WhisperX preserva a capitalização
    real da fala, então minúscula no início não é estilo, é o segmento
    continuando uma frase cortada por pausa/interrupção, não começando uma
    nova. Risco elevado de perda de fidelidade na tradução — o modelo só
    vê o fragmento + frases vizinhas como contexto, e dados reais mostram
    que ele às vezes omite uma cláusula inteira (`"for that."` sumiu de
    `"for that. What are you talking about?"`) ou resolve pronome errado
    (`"Here you"` -> `"Aqui vcs"`, plural, quando o original é singular).
    `translation.build_prompt(..., is_continuation=True)` reage a este
    mesmo sinal reforçando a instrução de não cortar conteúdo — esta
    função expõe o MESMO sinal como flag pro editor, não calculado de
    novo: ambos checam a mesma propriedade do texto original.

    Heurística direta (não calculada por amostragem/calibração) — se um
    dia o WhisperX normalizar capitalização por conta própria, isso para
    de ser confiável; não observado até agora.
    """
    for ch in texto_original:
        if ch.isalpha():
            return ch.islower()
    return False


@dataclass
class QualityFlags:
    fora_duracao: bool
    cer_alto: bool
    truncado: bool
    traducao_infiel: bool
    clipping: bool
    silencio_anormal: bool
    locutor_suspeito: bool
    entonacao_desalinhada: bool = False
    emocao_incompativel: bool = False
    cer_final_alto: bool = False
    loop_confirmado: bool = False
    segmento_curto: bool = False
    fala_fragmentada: bool = False

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
