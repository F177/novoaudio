"""Servidor local de escuta e rotulagem humana (ferramenta de dev).

    python -m scripts.listen <ciclo> <rotulo>                 fila do juiz + o que mudou
    python -m scripts.listen <ciclo> <rotulo> --todos         todos os segmentos (T0.28)
    python -m scripts.listen <ciclo> <candidato> --ab <base>  A/B cego: original, 1 e 2

Abre em http://127.0.0.1:8765, só nesta máquina. Cada rótulo vai direto para
`evals/human_labels/<par>/<video>.jsonl`, com chave segmento + rev, no mesmo
formato do juiz: vale em todo ciclo enquanto o segmento não mudar e
substitui o rótulo do juiz no `eval_cycle`. O veredito do Short (publicaria
ou não) é a métrica do portão da Fase 0. No A/B o servidor guarda qual
versão é qual; a página só vê "1" e "2", e a preferência vai para
`evals/human_labels/<par>/preferencias.jsonl`.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import soundfile as sf

from scripts.eval_common import (
    CYCLES,
    HUMAN_LABELS,
    TAXONOMY,
    add_human,
    append_jsonl,
    cycle_dir,
    load_cycle,
    load_human,
    now,
    pair_name,
    read_json,
    read_pending,
    run_dir,
    shorts,
)
from scripts.judge_packet import ReviewSegment, load_run, run_revision

PAD_SECONDS = 0.3
SEGMENT_LABELS = ("ok", "corrigiria")
SHORT_LABELS = ("publicaria", "nao_publicaria")


@dataclass
class Version:
    key: str
    label: str
    segment: ReviewSegment


@dataclass
class Card:
    id: str
    video: str
    split: str
    start: float
    end: float
    source_text: str
    motivo: str = ""
    o_que_ouvir: str = ""
    versions: list[Version] = field(default_factory=list)


@dataclass
class State:
    cycle_id: str
    cdir: Path
    par: str
    label: str
    ab_base: str | None
    cards: dict[str, Card]
    videos: list[str]
    classes: list[str]
    human_root: Path


def _overlap(a: ReviewSegment, b: ReviewSegment) -> float:
    return min(a.end, b.end) - max(a.start, b.start)


def build_state(cycle_id: str, label: str, ab_base: str | None = None, todos: bool = False,
                root: Path = CYCLES, human_root: Path = HUMAN_LABELS,
                taxonomy_path: Path = TAXONOMY, seed: int | None = None) -> State:
    _, manifest = load_cycle(cycle_id, root)
    cdir = cycle_dir(cycle_id, root)
    rng = random.Random(seed)
    cards: dict[str, Card] = {}
    videos = []
    for short in shorts(manifest):
        video = short["video"]
        rdir = run_dir(cdir, label, video)
        if not (rdir / "report.json").exists():
            continue
        segs = load_run(rdir)[1]
        queue_path = cdir / label / "escuta" / f"{video}.json"
        queue = {q["segment_id"]: q for q in (read_json(queue_path) if queue_path.exists()
                                              else [])}
        wanted = {s.id for s in segs} if todos else set(queue) | set(
            read_pending(cdir, label, video))
        base_segs = load_run(run_dir(cdir, ab_base, video))[1] if ab_base else []
        selected = [s for s in segs if s.id in wanted]
        if selected:
            videos.append(video)
        for seg in selected:
            versions = [Version("dub", label, seg)]
            if ab_base:
                match = max(base_segs, key=lambda b, s=seg: _overlap(s, b), default=None)
                if match is not None and _overlap(seg, match) > 0:
                    pair = [(label, seg), (ab_base, match)]
                    rng.shuffle(pair)
                    versions = [Version(str(i + 1), lbl, sg) for i, (lbl, sg) in enumerate(pair)]
            card_id = f"c{len(cards)}"
            item = queue.get(seg.id, {})
            cards[card_id] = Card(
                id=card_id, video=video, split=short.get("split", ""),
                start=min(v.segment.start for v in versions),
                end=max(v.segment.end for v in versions), source_text=seg.source_text,
                motivo=item.get("motivo", ""), o_que_ouvir=item.get("o_que_ouvir", ""),
                versions=versions)
    classes = sorted(read_json(taxonomy_path)["classes"])
    return State(cycle_id, cdir, pair_name(manifest), label, ab_base, cards, videos, classes,
                 human_root)


# ---------------------------------------------------------------- ações


def clip_bytes(state: State, card_id: str, which: str) -> bytes:
    card = state.cards[card_id]
    if which == "src":
        path = run_dir(state.cdir, card.versions[0].label, card.video) / "audio.wav"
    else:
        version = next(v for v in card.versions if v.key == which)
        path = run_dir(state.cdir, version.label, card.video) / "final_audio.wav"
    info = sf.info(str(path))
    start = max(0, int((card.start - PAD_SECONDS) * info.samplerate))
    stop = min(info.frames, int((card.end + PAD_SECONDS) * info.samplerate))
    data, sr = sf.read(str(path), start=start, stop=stop, dtype="float32", always_2d=True)
    buffer = io.BytesIO()
    sf.write(buffer, data, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def save_label(state: State, payload: dict[str, Any]) -> dict[str, Any]:
    card = state.cards[payload["card"]]
    version = next(v for v in card.versions if v.key == payload["v"])
    rotulo, classe = payload.get("rotulo"), payload.get("classe") or None
    if rotulo not in SEGMENT_LABELS:
        raise ValueError(f"rótulo inválido: {rotulo}")
    if rotulo == "corrigiria" and classe not in state.classes:
        raise ValueError("escolha a classe do problema")
    row = add_human(state.par, {
        "video": card.video, "segment_id": version.segment.id, "rev": version.segment.rev,
        "rotulo": rotulo, "classe": classe if rotulo == "corrigiria" else None,
        "nota": str(payload.get("nota", ""))[:500], "ciclo": state.cycle_id,
    }, state.human_root)
    return {"ok": True, "em": row["em"]}


def save_pref(state: State, payload: dict[str, Any]) -> dict[str, Any]:
    card = state.cards[payload["card"]]
    if len(card.versions) != 2 or payload.get("melhor") not in ("1", "2", "igual"):
        raise ValueError("preferência só existe no A/B, com 1, 2 ou igual")
    by_label = {v.label: v for v in card.versions}
    cand, base = by_label[state.label], by_label[state.ab_base or ""]
    winner = payload["melhor"]
    preferred = "igual" if winner == "igual" else (
        "candidato" if next(v for v in card.versions if v.key == winner) is cand else "base")
    append_jsonl(state.human_root / state.par / "preferencias.jsonl", {
        "ciclo": state.cycle_id, "video": card.video, "preferido": preferred,
        "segmento_candidato": cand.segment.id, "rev_candidato": cand.segment.rev,
        "segmento_base": base.segment.id, "rev_base": base.segment.rev, "em": now(),
    })
    return {"ok": True}


def save_short(state: State, payload: dict[str, Any]) -> dict[str, Any]:
    video, verdict = payload.get("video"), payload.get("short")
    if video not in state.videos or verdict not in SHORT_LABELS:
        raise ValueError("vídeo ou veredito inválido")
    run_rev = run_revision(load_run(run_dir(state.cdir, state.label, video))[1])
    add_human(state.par, {"video": video, "short": verdict, "run_rev": run_rev,
                          "ciclo": state.cycle_id}, state.human_root)
    return {"ok": True}


def page_data(state: State) -> dict[str, Any]:
    """O que a página pode saber. No A/B, nada indica qual versão é qual."""
    cards, verdicts = [], {}
    human_cache = {v: load_human(state.par, v, state.human_root) for v in state.videos}
    for card in state.cards.values():
        segs, _ = human_cache[card.video]
        versions = []
        for v in card.versions:
            existing = segs.get((v.segment.id, v.segment.rev))
            versions.append({"key": v.key, "texto": v.segment.dub_text,
                             "existente": {k: existing.get(k) for k in ("rotulo", "classe", "nota")}
                             if existing else None})
        cards.append({"id": card.id, "video": card.video, "split": card.split,
                      "inicio": round(card.start, 2), "fim": round(card.end, 2),
                      "original": card.source_text, "motivo": card.motivo,
                      "o_que_ouvir": card.o_que_ouvir, "versoes": versions})
    for video in state.videos:
        run_rev = run_revision(load_run(run_dir(state.cdir, state.label, video))[1])
        verdict = human_cache[video][1].get(run_rev)
        verdicts[video] = verdict["short"] if verdict else None
    return {"ciclo": state.cycle_id, "ab": state.ab_base is not None, "classes": state.classes,
            "cartoes": cards, "shorts": verdicts}


PAGE = """<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Escuta</title>
<style>
:root{--bg:#fff;--fg:#1f1f1d;--mu:#6b6a65;--bd:#d9d7cf;--ok:#2e7d32;--bad:#b3261e;--card:#f7f6f2}
@media (prefers-color-scheme:dark){:root{--bg:#1d1d1b;--fg:#ecebe6;--mu:#a3a29c;--bd:#3a3936;
--ok:#81c784;--bad:#f28b82;--card:#262624}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:820px;
margin:0 auto;padding:1.5rem 1rem 4rem}
h1{font-size:20px;font-weight:500;margin:0 0 .25rem}.mu{color:var(--mu);font-size:13px}
section{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:1rem;
margin:1rem 0}audio{width:100%;height:36px}.row{display:flex;gap:.5rem;flex-wrap:wrap;
align-items:center;margin:.4rem 0}button{font:inherit;padding:.3rem .8rem;border-radius:8px;
border:1px solid var(--bd);background:var(--bg);color:var(--fg);cursor:pointer}
button.on.ok{border-color:var(--ok);color:var(--ok)}button.on.bad{border-color:var(--bad);
color:var(--bad)}button.on{font-weight:500}select,input{font:inherit;padding:.25rem;
border-radius:8px;border:1px solid var(--bd);background:var(--bg);color:var(--fg)}
input{flex:1;min-width:10rem}.err{color:var(--bad);font-size:13px}.ver{border-top:1px solid
var(--bd);margin-top:.6rem;padding-top:.6rem}.tx{margin:.2rem 0}
</style>
<h1>Escuta — ciclo <span id="ciclo"></span></h1>
<p class="mu" id="prog"></p><div id="shorts"></div><div id="cards"></div>
<script>
const D = __DATA__;
const el = (t, p = {}, ...kids) => { const e = document.createElement(t);
  for (const [k, v] of Object.entries(p)) { if (k === "cls") e.className = v;
    else if (k.startsWith("on")) e[k] = v; else e.setAttribute(k, v); }
  e.append(...kids); return e; };
async function post(path, body) {
  const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)});
  if (!r.ok) throw new Error(await r.text()); return r.json(); }
let done = new Set();
function progress() { const total = D.cartoes.reduce((n, c) => n + c.versoes.length, 0);
  document.getElementById("prog").textContent =
    `${done.size} de ${total} versões rotuladas` + (D.ab ? " · A/B cego" : ""); }
function shortBox(video, current) {
  const msg = el("span", {cls: "mu"});
  const btns = [["publicaria", "Publicaria", "ok"], ["nao_publicaria", "Não publicaria", "bad"]]
    .map(([v, t, c]) => el("button", {cls: c + (current === v ? " on" : ""), onclick: async () => {
      try { await post("/short", {video, short: v});
        btns.forEach(b => b.classList.remove("on")); b0[v].classList.add("on");
        msg.textContent = "salvo"; } catch (e) { msg.textContent = e.message; } }}, t));
  const b0 = {publicaria: btns[0], nao_publicaria: btns[1]};
  return el("div", {cls: "row"}, el("b", {}, video), ...btns, msg); }
function versionBox(card, v) {
  const sel = el("select", {}, el("option", {value: ""}, "classe…"),
    ...D.classes.map(c => el("option", {value: c}, c)));
  const nota = el("input", {placeholder: "nota (opcional)"});
  const err = el("span", {cls: "err"}); const ok = el("button", {cls: "ok"}, "OK");
  const bad = el("button", {cls: "bad"}, "Corrigiria");
  if (v.existente) { sel.value = v.existente.classe || ""; nota.value = v.existente.nota || "";
    (v.existente.rotulo === "ok" ? ok : bad).classList.add("on"); done.add(card.id + v.key); }
  async function save(rotulo) {
    err.textContent = "";
    if (rotulo === "corrigiria" && !sel.value) { err.textContent = "escolha a classe"; return; }
    try { await post("/label", {card: card.id, v: v.key, rotulo, classe: sel.value,
        nota: nota.value});
      ok.classList.toggle("on", rotulo === "ok"); bad.classList.toggle("on", rotulo !== "ok");
      done.add(card.id + v.key); progress(); } catch (e) { err.textContent = e.message; } }
  ok.onclick = () => save("ok"); bad.onclick = () => save("corrigiria");
  sel.onchange = () => { err.textContent = ""; };
  const title = v.key === "dub" ? "Dublado" : "Versão " + v.key;
  return el("div", {cls: "ver"}, el("div", {cls: "mu"}, title),
    el("audio", {controls: "", preload: "none", src: `/clip?card=${card.id}&v=${v.key}`}),
    el("p", {cls: "tx"}, v.texto), el("div", {cls: "row"}, ok, bad, sel, nota), err); }
function cardBox(c) {
  const head = el("div", {cls: "mu"},
    `${c.video} · ${c.split} · ${c.inicio.toFixed(1)}–${c.fim.toFixed(1)} s` +
    (c.motivo ? ` · ${c.motivo}` : ""));
  const kids = [head];
  if (c.o_que_ouvir) kids.push(el("p", {}, el("b", {}, c.o_que_ouvir)));
  kids.push(el("div", {cls: "mu"}, "Original"),
    el("audio", {controls: "", preload: "none", src: `/clip?card=${c.id}&v=src`}),
    el("p", {cls: "tx mu"}, c.original), ...c.versoes.map(v => versionBox(c, v)));
  if (c.versoes.length === 2) { const msg = el("span", {cls: "mu"});
    const prefs = [["1", "1 é melhor"], ["2", "2 é melhor"], ["igual", "Iguais"]]
      .map(([k, t]) => { const b = el("button", {}, t);
        b.onclick = async () => {
          try { await post("/pref", {card: c.id, melhor: k});
            prefs.forEach(x => x.classList.remove("on")); b.classList.add("on");
            msg.textContent = "salvo"; } catch (e) { msg.textContent = e.message; } };
        return b; });
    kids.push(el("div", {cls: "row ver"}, ...prefs, msg)); }
  return el("section", {}, ...kids); }
document.getElementById("ciclo").textContent = D.ciclo;
document.getElementById("shorts").append(el("section", {},
  el("div", {cls: "mu"}, "Veredito por Short (métrica do portão)"),
  ...Object.entries(D.shorts).map(([v, cur]) => shortBox(v, cur))));
document.getElementById("cards").append(...D.cartoes.map(cardBox)); progress();
</script></html>
"""


def render_page(state: State) -> str:
    data = json.dumps(page_data(state), ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__DATA__", data)


ACTIONS = {"/label": save_label, "/pref": save_pref, "/short": save_short}


def make_handler(state: State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (nome exigido pelo http.server)
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    self._send(200, render_page(state).encode("utf-8"), "text/html; charset=utf-8")
                elif url.path == "/clip":
                    q = parse_qs(url.query)
                    self._send(200, clip_bytes(state, q["card"][0], q["v"][0]), "audio/wav")
                else:
                    self._send(404, b"nao encontrado", "text/plain")
            except (KeyError, StopIteration, OSError) as exc:
                self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            action = ACTIONS.get(urlparse(self.path).path)
            if action is None:
                self._send(404, b"nao encontrado", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = action(state, payload)
                self._send(200, json.dumps(result).encode("utf-8"), "application/json")
            except (KeyError, ValueError, StopIteration) as exc:
                self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")

    return Handler


def serve(state: State, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))


def main() -> None:
    parser = argparse.ArgumentParser(description="Escuta e rotulagem humana do ciclo")
    parser.add_argument("cycle")
    parser.add_argument("label", help="execução a ouvir (no A/B, o candidato)")
    parser.add_argument("--ab", metavar="BASE", help="A/B cego contra este rótulo")
    parser.add_argument("--todos", action="store_true", help="todos os segmentos")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    state = build_state(args.cycle, args.label, args.ab, args.todos)
    if not state.cards:
        raise SystemExit("nada para ouvir: sem fila do juiz, sem segmentos que mudaram "
                         "(use --todos para rotular tudo)")
    server = serve(state, args.port)
    print(f"{len(state.cards)} trechos em {len(state.videos)} Shorts — "
          f"http://127.0.0.1:{args.port}  (Ctrl+C para sair)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
