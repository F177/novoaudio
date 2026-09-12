"""Orçamento de duração.

Usa a calibração do T0.6 (`packages/pipeline/calibration.json`,
`duracao = a + b*silabas + c*segundos_de_pausa`) pra converter entre "quantas
sílabas cabem num tempo alvo" e "quanto tempo um número de sílabas ocupa",
dada uma velocidade de fala.

Ordem de alavancas quando um segmento não cabe no orçamento (T0.8/T0.9 usam
isso pra decidir o que tentar primeiro, antes de recorrer à próxima):
1. **texto** — escolher um candidato de tradução mais curto (T0.8).
2. **velocidade** — ajustar a velocidade de fala dentro da tolerância (±15%,
   ver T2.4).
3. **silêncio** — comprimir ou expandir pausas internas.
4. **time-stretch** — último recurso: esticar/comprimir o áudio já
   sintetizado.

Tolerâncias de isocronia (CLAUDE.md, glossário "On-screen"): ±8% para
locutor visível em quadro, ±20% fora de quadro.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TOLERANCE_ON_SCREEN = 0.08
TOLERANCE_OFF_SCREEN = 0.20

LEVER_ORDER = ("texto", "velocidade", "silencio", "time_stretch")

_DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"


@dataclass
class Calibration:
    a: float
    b: float
    c: float


def load_calibration(path: Path | None = None) -> Calibration:
    """Lê os coeficientes calibrados do `calibration.json` (T0.6)."""
    coef = json.loads((path or _DEFAULT_CALIBRATION_PATH).read_text(encoding="utf-8"))[
        "coefficients"
    ]
    return Calibration(a=coef["a"], b=coef["b"], c=coef["c"])


def seconds_needed(
    syllables: float,
    calibration: Calibration,
    *,
    speed: float = 1.0,
    pause_seconds: float = 0.0,
) -> float:
    """Duração estimada (s) pra sintetizar `syllables` sílabas numa dada velocidade."""
    if speed <= 0:
        raise ValueError("speed precisa ser > 0")
    return calibration.a + (calibration.b / speed) * syllables + calibration.c * pause_seconds


def syllables_that_fit(
    target_seconds: float,
    calibration: Calibration,
    *,
    speed: float = 1.0,
    pause_seconds: float = 0.0,
) -> int:
    """Quantas sílabas cabem em `target_seconds`, dada velocidade e pausas já reservadas.

    Arredonda pra baixo de propósito — preferimos sobrar tempo (o segmento
    cabe com folga) a estourar o orçamento por causa de arredondamento.
    """
    if speed <= 0:
        raise ValueError("speed precisa ser > 0")
    available = target_seconds - calibration.a - calibration.c * pause_seconds
    if available <= 0:
        return 0
    return int((available * speed) / calibration.b)


def tolerance_range(target_seconds: float, *, on_screen: bool) -> tuple[float, float]:
    """Faixa aceitável de duração (s) ao redor do alvo, dada a tolerância de isocronia."""
    tolerance = TOLERANCE_ON_SCREEN if on_screen else TOLERANCE_OFF_SCREEN
    return target_seconds * (1 - tolerance), target_seconds * (1 + tolerance)
