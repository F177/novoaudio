"""Placar de qualidade por vídeo (item 5 do plano de avaliação de
qualidade pedido pelo usuário, 2026-09-26: "um sistema inteligente que
detecte todos os aspectos de qualidade... que também traga métricas de
desempenho do produto").

Agrega os resultados por segmento que `scripts/dub.py` já grava em
`report.json` (usando os gates de `quality.py` — CER, duração, clipping,
silêncio, correlação de prosódia, emoção) num resumo por vídeo: taxa de
cada flag, médias das métricas objetivas, os piores segmentos, e um
veredito qualitativo — a leitura "de especialista" num relance, em vez de
vasculhar segmento por segmento.

Puro (só dicts/números/numpy), sem I/O de rede nem acesso a modelo —
mesma regra de dependência de `packages/pipeline` (CLAUDE.md): recebe os
dicts de segmento já prontos, devolve dados.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FLAG_NAMES = (
    "fora_duracao",
    "cer_alto",
    "truncado",
    "traducao_infiel",
    "clipping",
    "silencio_anormal",
    "locutor_suspeito",
    "entonacao_desalinhada",
    "emocao_incompativel",
    "cer_final_alto",
)

VERDICT_APROVADO = "aprovado"
VERDICT_REVISAR = "revisar"
VERDICT_PROBLEMATICO = "problematico"

# Ponto de partida, não calibrado com volume real de produção (mesmo aviso
# dos limiares em quality.py) — fração de segmentos com qualquer flag a
# partir da qual o veredito passa de "aprovado" pra "revisar", e depois
# pra "problemático".
REVIEW_RATE_THRESHOLD = 0.2
PROBLEMATIC_RATE_THRESHOLD = 0.5


@dataclass
class Scorecard:
    total_segments: int
    synthesized_segments: int
    flag_counts: dict[str, int]
    flag_rates: dict[str, float]
    any_flag_rate: float
    avg_cer: float | None
    avg_cer_final: float | None
    avg_prosody_correlation: float | None
    avg_emotion_similarity: float | None
    worst_segments: list[dict[str, Any]]
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_segments": self.total_segments,
            "synthesized_segments": self.synthesized_segments,
            "flag_counts": self.flag_counts,
            "flag_rates": self.flag_rates,
            "any_flag_rate": self.any_flag_rate,
            "avg_cer": self.avg_cer,
            "avg_cer_final": self.avg_cer_final,
            "avg_prosody_correlation": self.avg_prosody_correlation,
            "avg_emotion_similarity": self.avg_emotion_similarity,
            "worst_segments": self.worst_segments,
            "verdict": self.verdict,
        }


def _verdict(any_flag_rate: float) -> str:
    if any_flag_rate >= PROBLEMATIC_RATE_THRESHOLD:
        return VERDICT_PROBLEMATICO
    if any_flag_rate >= REVIEW_RATE_THRESHOLD:
        return VERDICT_REVISAR
    return VERDICT_APROVADO


def build_scorecard(segments: list[dict[str, Any]], *, worst_n: int = 5) -> dict[str, Any]:
    """Recebe `report["segments"]` (lista de dicts, mesmo formato que
    `scripts/dub.py` grava por segmento) e devolve o placar agregado.

    Segmento sem síntese (`synthesized_ok=False`) entra na taxa de
    `any_flag` (falha total é a pior flag possível) mas fica de fora das
    taxas por flag específica e das médias — não tem `flags`/`cer`/etc.
    calculados, e misturar um "buraco" numérico ali distorceria a média.
    """
    total = len(segments)
    synthesized = [s for s in segments if s.get("synthesized_ok")]
    n_synth = len(synthesized)

    flag_counts = {name: 0 for name in FLAG_NAMES}
    for s in synthesized:
        flags = s.get("flags", {})
        for name in FLAG_NAMES:
            if flags.get(name):
                flag_counts[name] += 1

    flag_rates = {
        name: (count / n_synth if n_synth else 0.0) for name, count in flag_counts.items()
    }

    any_flag_rate = (sum(1 for s in segments if s.get("any_flag")) / total) if total else 0.0

    cers = [s["cer"] for s in synthesized if s.get("evaluated_ok")]
    avg_cer = float(np.mean(cers)) if cers else None

    cers_final = [s["cer_final"] for s in synthesized if s.get("cer_final") is not None]
    avg_cer_final = float(np.mean(cers_final)) if cers_final else None

    correlations = [
        s["prosody_correlation"] for s in synthesized if s.get("prosody_correlation") is not None
    ]
    avg_prosody_correlation = float(np.mean(correlations)) if correlations else None

    similarities = [
        s["emotion_match"]["similarity"] for s in synthesized if s.get("emotion_match") is not None
    ]
    avg_emotion_similarity = float(np.mean(similarities)) if similarities else None

    flagged = [s for s in segments if s.get("any_flag")]
    flagged_sorted = sorted(
        flagged,
        key=lambda s: (
            len([name for name in FLAG_NAMES if s.get("flags", {}).get(name)]),
            s.get("cer") or 0.0,
        ),
        reverse=True,
    )
    worst_segments = [
        {
            "id": s["id"],
            "t_inicio": s.get("t_inicio"),
            "t_fim": s.get("t_fim"),
            "flags": [name for name in FLAG_NAMES if s.get("flags", {}).get(name)],
            "cer": s.get("cer"),
        }
        for s in flagged_sorted[:worst_n]
    ]

    return Scorecard(
        total_segments=total,
        synthesized_segments=n_synth,
        flag_counts=flag_counts,
        flag_rates=flag_rates,
        any_flag_rate=any_flag_rate,
        avg_cer=avg_cer,
        avg_cer_final=avg_cer_final,
        avg_prosody_correlation=avg_prosody_correlation,
        avg_emotion_similarity=avg_emotion_similarity,
        worst_segments=worst_segments,
        verdict=_verdict(any_flag_rate),
    ).to_dict()
