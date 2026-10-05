"""Geração de legendas (SRT) a partir dos segmentos já traduzidos.

Pura, sem I/O — os segmentos de dublagem já carregam exatamente o que uma
legenda precisa (`t_inicio`, `t_fim`, texto traduzido), então isso é
formatação de dado que já temos, não uma etapa nova de pipeline. O
embutimento no arquivo de vídeo (`ffmpeg`, faixa `mov_text` selecionável)
mora em `assembly.py::remux_with_video`, que é quem faz I/O de verdade.

Granularidade: os segmentos de dublagem são otimizados pra isocronia (até
`segmentation.DEFAULT_MAX_DURATION`, várias frases coladas), não pra
leitura confortável — uma legenda "de verdade" geralmente quer linhas
mais curtas. Fica como está por enquanto (usa os segmentos como são); se
o texto ficar denso demais pra ler no tempo disponível, quebrar em
pedaços menores só pra exibição é o próximo passo, sem mexer na
dublagem.
"""

from __future__ import annotations


def _format_srt_timestamp(seconds: float) -> str:
    """`HH:MM:SS,mmm`, formato exigido pelo SRT (vírgula nos milissegundos,
    não ponto)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt(entries: list[tuple[float, float, str]]) -> str:
    """Monta o conteúdo de um arquivo `.srt` a partir de `(t_inicio, t_fim,
    texto)` por segmento.

    Ordena por `t_inicio` (SRT exige ordem crescente, os segmentos de
    origem nem sempre garantem isso dependendo de como foram montados) e
    descarta entradas sem texto útil (vazio/só espaço) ou com duração
    zero/negativa (nada pra mostrar, evita legenda "fantasma").
    """
    valid_entries = [
        (start, end, text) for start, end, text in entries if text.strip() and end > start
    ]
    sorted_entries = sorted(valid_entries, key=lambda e: e[0])

    lines: list[str] = []
    for index, (start, end, text) in enumerate(sorted_entries, start=1):
        lines.append(str(index))
        lines.append(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}")
        lines.append(text.strip())
        lines.append("")
    return "\n".join(lines)
