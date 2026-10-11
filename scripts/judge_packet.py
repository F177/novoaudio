"""Pacote de revisão compacto para o subagente `dub-judge` (ferramenta de dev).

Lê o que uma execução de `scripts/dub.py` já deixou em disco e imprime, por
segmento, só o que um revisor de tradução precisa: tempo, locutor, inglês,
pt-BR usado, transcrição do áudio mixado e flags ligadas. Não chama modelo,
não toca no Pod e não faz parte do caminho de geração — é só leitura de
artefato, para o juiz não gastar contexto com o `report.json` inteiro.

Entradas aceitas (em qualquer combinação no modo A/B):
- diretório de cache (`.cache/dub/<video>/`): usa `report.json` e, se
  existir, `segments.json` (locutor);
- `report.json` direto;
- `videos/dublados/<video>__<variante>_traducao.json` (só texto).

Uso:
    python -m scripts.judge_packet .cache/dub/<video>
    python -m scripts.judge_packet A_traducao.json B_traducao.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Fração mínima de sobreposição (sobre o segmento mais curto) para parear
# segmentos de duas execuções cuja segmentação mudou.
MIN_OVERLAP_RATIO = 0.5


@dataclass(frozen=True)
class ReviewSegment:
    id: str
    start: float
    end: float
    source_text: str
    dub_text: str
    speaker: str | None = None
    asr_text: str | None = None
    cer_final: float | None = None
    target_seconds: float | None = None
    achieved_seconds: float | None = None
    flags: tuple[str, ...] = ()
    audio_sha: str | None = None

    @property
    def rev(self) -> str:
        return segment_revision(self.dub_text, self.audio_sha)


def segment_revision(text: str, audio_sha: str | None = None) -> str:
    """Hash curto do texto dublado e, quando houver, do áudio do segmento.

    Rótulo com `rev` diferente do segmento atual está velho. Segmento com o
    mesmo `rev` em duas execuções é idêntico: o ciclo herda o rótulo em vez
    de pedir ao juiz de novo (menos custo, menos ruído do juiz).
    """
    key = " ".join(text.split()) + (f"|{audio_sha}" if audio_sha else "")
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def text_revision(text: str) -> str:
    return segment_revision(text)


def run_revision(segments: list[ReviewSegment]) -> str:
    """Hash de uma execução inteira: o veredito do Short vale para este conjunto exato."""
    key = "|".join(f"{s.id}:{s.rev}" for s in segments)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _audio_sha(path: Path) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def active_flags(flags: dict[str, Any]) -> tuple[str, ...]:
    """Flags ligadas, sem os marcadores `*_not_computed` (stubs)."""
    return tuple(k for k, v in flags.items() if v is True and not k.endswith("_not_computed"))


def _from_report(
    report: dict[str, Any], speakers: dict[str, str], audio_dir: Path | None = None
) -> list[ReviewSegment]:
    return [
        ReviewSegment(
            id=s["id"],
            start=float(s["t_inicio"]),
            end=float(s["t_fim"]),
            source_text=s.get("texto_original") or "",
            dub_text=s.get("traducao") or "",
            speaker=speakers.get(s["id"]),
            asr_text=s.get("hipotese_final"),
            cer_final=s.get("cer_final"),
            target_seconds=s.get("target_seconds"),
            achieved_seconds=s.get("achieved_seconds"),
            flags=active_flags(s.get("flags") or {}),
            audio_sha=_audio_sha(audio_dir / f"{s['id']}.wav") if audio_dir else None,
        )
        for s in report["segments"]
    ]


def _from_traducao(items: list[dict[str, Any]]) -> list[ReviewSegment]:
    return [
        ReviewSegment(
            id=s["id"],
            start=float(s["t_inicio"]),
            end=float(s["t_fim"]),
            source_text=s.get("texto_original") or "",
            dub_text=s.get("traducao_usada") or "",
        )
        for s in items
    ]


def load_run(path: Path) -> tuple[str, list[ReviewSegment]]:
    """Carrega uma execução e devolve (nome, segmentos)."""
    if path.is_dir():
        speakers: dict[str, str] = {}
        seg_path = path / "segments.json"
        if seg_path.exists():
            for s in json.loads(seg_path.read_text(encoding="utf-8")):
                if s.get("speaker"):
                    speakers[s["id"]] = s["speaker"]
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        return path.name, _from_report(report, speakers, path / "final_segments")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "segments" in data:
        return path.parent.name, _from_report(data, {})
    return path.stem.removesuffix("_traducao"), _from_traducao(data)


def _overlap(a: ReviewSegment, b: ReviewSegment) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    shortest = min(a.end - a.start, b.end - b.start)
    return inter / shortest if inter > 0 and shortest > 0 else 0.0


def pair_segments(
    a: list[ReviewSegment], b: list[ReviewSegment]
) -> tuple[list[tuple[ReviewSegment, list[ReviewSegment]]], list[ReviewSegment]]:
    """Pareia por sobreposição de tempo (a segmentação pode ter mudado).

    Devolve os pares (segmento de A, segmentos de B) e os segmentos de B
    que não casaram com nenhum de A.
    """
    pairs = [(sa, [sb for sb in b if _overlap(sa, sb) >= MIN_OVERLAP_RATIO]) for sa in a]
    matched = {sb.id for _, sbs in pairs for sb in sbs}
    return pairs, [sb for sb in b if sb.id not in matched]


def _span(s: ReviewSegment) -> str:
    return f"{s.id} {s.start:.2f}–{s.end:.2f}"


def render_single(name: str, segments: list[ReviewSegment]) -> str:
    lines = [f"# {name} — {len(segments)} segmentos"]
    for s in segments:
        head = _span(s) + (f" {s.speaker}" if s.speaker else "") + f" | rev {s.rev}"
        lines += [head, f"  EN : {s.source_text}", f"  PT : {s.dub_text}"]
        if s.asr_text is not None:
            cer = f" (cer_final {s.cer_final:.2f})" if s.cer_final is not None else ""
            lines.append(f"  ASR: {s.asr_text}{cer}")
        if s.target_seconds is not None and s.achieved_seconds is not None:
            target, achieved = s.target_seconds, s.achieved_seconds
            lines.append(f"  dur: alvo {target:.2f} s · obtido {achieved:.2f} s")
        lines.append(f"  flags: {', '.join(s.flags) if s.flags else '—'}")
    return "\n".join(lines)


def render_pair(
    name_a: str, a: list[ReviewSegment], name_b: str, b: list[ReviewSegment]
) -> str:
    pairs, unmatched = pair_segments(a, b)
    lines = [f"# A = {name_a} · B = {name_b}"]
    changed = 0
    for sa, sbs in pairs:
        b_text = " ".join(sb.dub_text for sb in sbs)
        if len(sbs) == 1 and sbs[0].rev == sa.rev:
            lines.append(f"{_span(sa)} (igual) {sa.dub_text}")
            continue
        changed += 1
        b_ids = ", ".join(sb.id for sb in sbs) or "sem par"
        b_flags = sorted({f for sb in sbs for f in sb.flags})
        lines += [
            f"{_span(sa)} ↔ B[{b_ids}]",
            f"  EN: {sa.source_text}",
            f"  A : {sa.dub_text}" + (f"  [{', '.join(sa.flags)}]" if sa.flags else ""),
            f"  B : {b_text}" + (f"  [{', '.join(b_flags)}]" if b_flags else ""),
        ]
    for sb in unmatched:
        lines.append(f"B sem par: {_span(sb)} {sb.dub_text}")
    lines.insert(1, f"{changed} de {len(pairs)} segmentos de A mudaram")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", type=Path, nargs="+", help="uma execução, ou duas para A/B")
    args = parser.parse_args()
    if len(args.runs) > 2:
        parser.error("passe uma execução ou duas (A e B)")
    if len(args.runs) == 1:
        print(render_single(*load_run(args.runs[0])))
    else:
        print(render_pair(*load_run(args.runs[0]), *load_run(args.runs[1])))


if __name__ == "__main__":
    main()
