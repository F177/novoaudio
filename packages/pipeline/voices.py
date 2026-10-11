"""Vozes de amostra por locutor (T0.31).

Short com dois locutores não pode soar como uma pessoa só. Cada locutor da
diarização recebe uma voz diferente do banco de vozes pt-BR sintéticas, do
mesmo gênero aparente da voz original (pela altura média, F0). Voz clonada
do criador fica para depois (T1.7).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

# Entre a faixa típica de F0 masculina (~85–155 Hz) e feminina (~165–255 Hz).
# Ponto de partida não calibrado.
GENDER_F0_SPLIT_HZ = 165.0
# F0 de um locutor medido em no máximo isso de fala (o pyin é lento).
MAX_F0_SECONDS = 20.0


@dataclass(frozen=True)
class Voice:
    id: str
    gender: str  # "m" ou "f"
    path: str  # caminho da referência no Pod, relativo ao repo


def load_bank(manifest: Path) -> list[Voice]:
    """`voices.json`: lista de {id, gender, path}. Sem arquivo, banco vazio."""
    if not manifest.exists():
        return []
    return [Voice(**v) for v in json.loads(manifest.read_text(encoding="utf-8"))]


def median_f0_hz(audio: np.ndarray, sample_rate: int) -> float | None:
    """F0 mediano dos trechos vozeados; `None` se não há voz detectável."""
    audio = audio[: int(MAX_F0_SECONDS * sample_rate)].astype(np.float32)
    if len(audio) < 2048:
        return None
    f0, voiced, _ = librosa.pyin(audio, fmin=60.0, fmax=400.0, sr=sample_rate)
    valid = voiced & ~np.isnan(f0)
    return float(np.median(f0[valid])) if np.any(valid) else None


def assign_voices(
    speakers: list[str],
    f0_by_speaker: dict[str, float | None],
    bank: list[Voice],
    seed: str,
) -> dict[str, Voice]:
    """Uma voz por locutor, na ordem de `speakers` (quem fala mais primeiro):
    mesmo gênero aparente, sem repetir voz enquanto houver outra livre.
    `seed` (o vídeo) deixa a escolha estável entre rodadas do mesmo vídeo e
    variada entre vídeos. Locutor sem F0 aceita qualquer gênero."""
    order = random.Random(seed).sample(bank, len(bank))
    used: set[str] = set()
    out: dict[str, Voice] = {}
    for speaker in speakers:
        f0 = f0_by_speaker.get(speaker)
        gender = None if f0 is None else ("m" if f0 < GENDER_F0_SPLIT_HZ else "f")
        matching = [v for v in order if gender is None or v.gender == gender] or order
        free = [v for v in matching if v.id not in used] or matching
        if free:
            out[speaker] = free[0]
            used.add(free[0].id)
    return out
