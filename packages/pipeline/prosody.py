"""Transferência de prosódia (Layer 2 do roteiro de expressividade).

Achado real desta sessão (2026-09-19/25): a única alavanca de tom que
tínhamos (`synthesis.py::infer_delivery_instruction`, pedir "fale com mais
ênfase" pro MOSS-TTS) causava instabilidade catastrófica — pedir emoção na
hora de GERAR competia com o orçamento de duração, e o modelo reagia mal
(loop de repetição) quando o pedido era estruturalmente impossível de
caber no tempo (ver docs/moss_tts_investigation.md, seção 2026-09-19).

Este módulo resolve o mesmo problema (dublagem soando "narrada"/monótona)
por um caminho estruturalmente seguro: em vez de pedir pro MODELO soar
diferente, extrai o contorno de ENERGIA (volume relativo ao longo do
tempo) do áudio ORIGINAL em inglês, e aplica esse contorno como modulação
de volume no áudio JÁ SINTETIZADO em português — depois da síntese, não
durante. Isso nunca muda a duração (só reescala amplitude amostra a
amostra), então não tem como competir com o orçamento de tempo do jeito
que a instrução de texto competia.

Layer 3 (2026-09-26): mesmo princípio, agora pro CONTORNO DE PITCH
(entonação) — extrai o pitch relativo (semitons acima/abaixo da própria
mediana do segmento, não Hz absoluto, porque vozes diferentes têm faixas
de pitch diferentes) do áudio original, e aplica como pitch-shift
variável no áudio sintetizado, em blocos com crossfade curto pra evitar
estalos na transição entre blocos.

**DESLIGADO por padrão em `scripts/dub.py` desde o mesmo dia — achado
real, medido, não suposição.** Usuário reportou ouvindo "às vezes o áudio
fica sem qualidade" com Layer 3 ligado; medição objetiva confirmou:
comparando CER round-trip (ASR) antes/depois em 8 segmentos reais
(Avengers), pitch-shift quase DOBROU o CER médio (0,082→0,149) e chegou a
piorar 5,5x num segmento (0,051→0,282). Energia sozinha teve CER
IDÊNTICO ao original nos 8 segmentos — só essa parte é segura por
padrão. Pitch-shift por fase vocoder em blocos curtos é uma técnica
conhecida por introduzir artefato robótico/metálico; `strength` parcial e
deslocamento máximo limitado (igual ao espírito do
`MAX_TIME_STRETCH_RATIO` em synthesis.py) não foram suficientes pra
evitar isso na prática. Mantido disponível (`pitch_transfer=True`) só
pra quem quiser experimentar/melhorar o algoritmo — não é mais parte do
caminho recomendado.

Pura, sem I/O, sem GPU — testável com dado sintético como o resto de
`packages/pipeline` (só depende de `librosa`, já uma dependência do
projeto desde o time-stretch).
"""

from __future__ import annotations

import librosa
import numpy as np

DEFAULT_WINDOW_SECONDS = 0.05
DEFAULT_STRENGTH = 0.5
_MIN_GAIN = 0.3
_MAX_GAIN = 2.0

# Janela maior que a de energia (0,3s vs 0,05s) — pitch-shift por fase
# vocoder precisa de blocos com amostras suficientes pra ter resolução de
# frequência decente; blocos curtos demais (na escala de energia) soariam
# pior, não melhor. Entonação real também não muda tão rápido quanto
# volume — 0,3s (~sílaba) já captura a variação que importa.
DEFAULT_PITCH_WINDOW_SECONDS = 0.3
MAX_PITCH_SHIFT_SEMITONES = 4.0
_PITCH_FMIN_HZ = 65.0  # ~C2, grave o bastante pra voz masculina adulta
_PITCH_FMAX_HZ = 1000.0  # ~B5, agudo o bastante sem pegar ruído/harmônicos


def extract_energy_envelope(
    audio: np.ndarray, sample_rate: int, *, window_seconds: float = DEFAULT_WINDOW_SECONDS
) -> np.ndarray:
    """RMS por janela de `window_seconds`, normalizado pelo pico do próprio
    áudio (0-1) — representa o contorno RELATIVO de volume ao longo do
    tempo, não o volume absoluto (que depende de calibração de gravação,
    não da entrega emocional que queremos capturar).

    Áudio vazio ou totalmente silencioso devolve um envelope neutro
    (todos 1.0) — sem sinal real pra transferir, não inventa dinâmica.
    """
    window = max(1, int(window_seconds * sample_rate))
    n_windows = len(audio) // window
    if n_windows == 0:
        return np.ones(1, dtype=np.float32)

    trimmed = audio[: n_windows * window].reshape(n_windows, window)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return np.ones(n_windows, dtype=np.float32)
    return (rms / peak).astype(np.float32)


def resample_envelope(envelope: np.ndarray, target_length: int) -> np.ndarray:
    """Redimensiona `envelope` pra ter `target_length` pontos (interpolação
    linear) — o segmento sintetizado quase nunca tem exatamente a mesma
    duração do original (esse é o problema de isocronia de sempre), então
    o contorno precisa ser esticado/comprimido pra cobrir o áudio de
    destino inteiro, mantendo a FORMA relativa (onde ficam os picos/vales
    proporcionalmente), não os timestamps absolutos.
    """
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(envelope) == 0:
        return np.ones(target_length, dtype=np.float32)
    if len(envelope) == target_length:
        return envelope.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, len(envelope))
    x_new = np.linspace(0.0, 1.0, target_length)
    return np.interp(x_new, x_old, envelope).astype(np.float32)


def apply_energy_envelope(
    audio: np.ndarray,
    sample_rate: int,
    source_envelope: np.ndarray,
    *,
    strength: float = DEFAULT_STRENGTH,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> np.ndarray:
    """Modula o volume de `audio` (já sintetizado) seguindo o contorno
    relativo de energia de `source_envelope` (do áudio original) — nunca
    muda o número de amostras, só reescala amplitude.

    `strength` (0-1) controla quanto do contorno original se aplica: 0 =
    nenhuma mudança (áudio sintetizado intacto), 1 = segue o contorno
    original com força total. Parcial por padrão (0.5) de propósito — a
    voz sintetizada já tem dinâmica própria, o objetivo é REFORÇAR ênfase
    onde o original tinha, não substituir a entrega inteira por uma cópia
    mecânica do envelope (que soaria artificial se aplicado a 100%).

    Ganho por janela é limitado a [0.3, 2.0] — evita apagar o áudio pra
    quase silêncio ou estourar em clipping severo mesmo se o envelope de
    origem tiver contraste extremo.
    """
    if strength <= 0 or len(audio) == 0:
        return audio

    window = max(1, int(window_seconds * sample_rate))
    n_windows = len(audio) // window
    if n_windows == 0:
        return audio

    target_envelope = resample_envelope(source_envelope, n_windows)
    # Centraliza no ganho neutro (média 1.0) antes de escalar pela força —
    # sem isso, um envelope de origem geralmente "baixo" (média < 1)
    # abaixaria o volume geral em vez de só variar o CONTORNO relativo.
    centered = target_envelope - target_envelope.mean()
    gain_per_window = np.clip(1.0 + strength * centered, _MIN_GAIN, _MAX_GAIN)

    gain_full = np.repeat(gain_per_window, window)
    result = audio.copy()
    applied_len = min(len(gain_full), len(result))
    result[:applied_len] = result[:applied_len] * gain_full[:applied_len]
    return result


def extract_pitch_contour(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float = DEFAULT_PITCH_WINDOW_SECONDS,
    fmin: float = _PITCH_FMIN_HZ,
    fmax: float = _PITCH_FMAX_HZ,
) -> np.ndarray:
    """Pitch RELATIVO por janela de `window_seconds`, em semitons acima/
    abaixo da MEDIANA de pitch vozeado do próprio segmento (não Hz
    absoluto) — o que queremos transferir é a FORMA da entonação (sobe
    aqui, desce ali), não o tom exato, que depende da voz de cada
    locutor.

    Frames não-vozados (silêncio, consoante surda, pitch não detectado)
    viram 0.0 (na própria linha de base) — não inventa contorno de
    entonação onde não há voz de verdade. Segmento sem NENHUM frame
    vozado detectável (ex.: todo silêncio) devolve um contorno neutro
    (tudo 0.0) — sem sinal real pra transferir.

    Achado real: passar `hop_length`/`frame_length` grandes direto pro
    `librosa.pyin` (pra bater com `window_seconds`) quebra um limite
    interno da lib (a matriz de transição de estado do algoritmo HMM
    depende de `fmin`/`fmax`, e um `hop_length` grande demais em relação a
    isso levanta `ParameterError`). Em vez disso, roda o `pyin` com os
    parâmetros PADRÃO da lib (frame/hop pequenos, granularidade fina) e
    agrega o resultado fino em janelas de `window_seconds` depois — mais
    robusto, sem depender de uma combinação de parâmetros não testada
    pelos mantenedores do librosa.
    """
    _PYIN_FRAME_LENGTH = 2048
    _PYIN_HOP_LENGTH = _PYIN_FRAME_LENGTH // 4

    if len(audio) < _PYIN_FRAME_LENGTH:
        return np.zeros(1, dtype=np.float32)

    f0, voiced_flag, _voiced_prob = librosa.pyin(
        audio.astype(np.float32),
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=_PYIN_FRAME_LENGTH,
        hop_length=_PYIN_HOP_LENGTH,
    )
    if f0 is None or len(f0) == 0:
        return np.zeros(1, dtype=np.float32)

    valid = voiced_flag & ~np.isnan(f0)

    window_frames = max(1, round((window_seconds * sample_rate) / _PYIN_HOP_LENGTH))
    n_windows = max(1, int(np.ceil(len(f0) / window_frames)))

    if not np.any(valid):
        return np.zeros(n_windows, dtype=np.float32)

    median_hz = float(np.median(f0[valid]))
    semitones_per_frame = np.zeros(len(f0), dtype=np.float32)
    semitones_per_frame[valid] = (12.0 * np.log2(f0[valid] / median_hz)).astype(np.float32)

    contour = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        start = i * window_frames
        end = min(start + window_frames, len(f0))
        window_valid = valid[start:end]
        if np.any(window_valid):
            contour[i] = semitones_per_frame[start:end][window_valid].mean()
    return contour


def apply_pitch_contour(
    audio: np.ndarray,
    sample_rate: int,
    source_contour: np.ndarray,
    *,
    strength: float = DEFAULT_STRENGTH,
    window_seconds: float = DEFAULT_PITCH_WINDOW_SECONDS,
    max_shift_semitones: float = MAX_PITCH_SHIFT_SEMITONES,
) -> np.ndarray:
    """Aplica pitch-shift VARIÁVEL em `audio` (já sintetizado) seguindo o
    contorno relativo de `source_contour` (semitons, do áudio original) —
    em blocos de `window_seconds`, cada um deslocado por um valor
    diferente, com crossfade curto na transição entre blocos pra evitar
    estalo.

    `strength` (0-1) escala o quanto do contorno original se aplica antes
    de deslocar — igual ao espírito de `apply_energy_envelope`, parcial
    por padrão (a voz sintetizada já tem entonação própria, o objetivo é
    reforçar contornos que combinam, não substituir por uma cópia
    mecânica). Deslocamento por bloco é limitado a
    `±max_shift_semitones` — acima disso o pitch-shift por fase vocoder
    tende a soar artificial ("efeito Chipmunk/demônio"), pior que não
    transferir nada.
    """
    if strength <= 0 or len(audio) == 0 or len(source_contour) == 0:
        return audio

    window = max(1, int(window_seconds * sample_rate))
    n_windows = max(1, int(np.ceil(len(audio) / window)))
    target_contour = resample_envelope(source_contour, n_windows)
    shifts = np.clip(strength * target_contour, -max_shift_semitones, max_shift_semitones)

    crossfade_samples = max(1, min(window // 4, sample_rate // 20))
    result = np.zeros_like(audio)

    def shift_block(block: np.ndarray, shift: float) -> np.ndarray:
        if abs(shift) < 0.01:
            return block
        shifted = librosa.effects.pitch_shift(
            block.astype(np.float32), sr=sample_rate, n_steps=shift
        )
        # pitch_shift por fase vocoder às vezes devolve 1-2 amostras a mais/
        # menos por arredondamento interno — corta ou preenche com zero pra
        # manter o bloco no tamanho exato, sem acumular deriva ao longo dos
        # blocos.
        if len(shifted) > len(block):
            shifted = shifted[: len(block)]
        elif len(shifted) < len(block):
            shifted = np.pad(shifted, (0, len(block) - len(shifted)))
        return shifted

    # Achado real: blocos extraídos SEM sobreposição real não têm o que
    # misturar na transição — a versão anterior "cruzava" com posições
    # ainda não escritas de `result` (zero/lixo), corrompendo o início de
    # todo bloco depois do primeiro mesmo com deslocamento zero. Corrigido:
    # cada bloco (i>0) estende pra TRÁS por `crossfade_samples`, deslocando
    # a MESMA região que o bloco anterior já colocou em `result` — só assim
    # existe conteúdo de verdade dos dois lados pra misturar.
    for i in range(n_windows):
        start = i * window
        end = min(start + window, len(audio))
        if start >= end:
            continue

        if i == 0:
            shifted = shift_block(audio[start:end], float(shifts[i]))
            result[start:end] = shifted
            continue

        ext_start = max(0, start - crossfade_samples)
        fade_len = start - ext_start
        block = audio[ext_start:end]
        shifted = shift_block(block, float(shifts[i]))

        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = 1.0 - fade_in
            result[ext_start:start] = (
                result[ext_start:start] * fade_out + shifted[:fade_len] * fade_in
            )
        result[start:end] = shifted[fade_len:]

    return result
