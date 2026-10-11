import io
import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import soundfile as sf

from scripts.eval_common import load_human, read_jsonl, write_json
from scripts.eval_cycle import carry
from scripts.judge_packet import load_run, run_revision
from scripts.listen import (
    PAD_SECONDS,
    build_state,
    clip_bytes,
    render_page,
    save_label,
    save_pref,
    save_short,
    serve,
)
from tests.cycle_fixtures import judge_labels, make_cycle, make_run

TAXONOMY = Path(__file__).resolve().parents[1] / "evals" / "taxonomy.json"


def _setup(tmp_path: Path):
    root, cid = make_cycle(tmp_path, [("a", "dev", "x"), ("h", "holdout", "y")])
    cdir = root / cid
    for v in ("a", "h"):
        make_run(cdir / "base" / "dub" / v, ["um", "ruim", "tres"])
        make_run(cdir / "cand" / "dub" / v, ["um", "bom", "tres"])
        judge_labels(cdir, "base", "dev" if v == "a" else "holdout", v, {})
    carry(cid, "base", "cand", root)
    write_json(cdir / "cand" / "escuta" / "a.json",
               [{"segment_id": "seg0002", "motivo": "emocao.achatada", "o_que_ouvir": "sobe?"}])
    return root, cid, cdir


def _state(tmp_path, root, cid, **kw):
    return build_state(cid, "cand", root=root, human_root=tmp_path / "human",
                       taxonomy_path=TAXONOMY, seed=1, **kw)


def test_queue_is_judge_queue_plus_changed_segments(tmp_path: Path) -> None:
    root, cid, _ = _setup(tmp_path)
    state = _state(tmp_path, root, cid)
    picked = sorted((c.video, c.versions[0].segment.id) for c in state.cards.values())
    assert picked == [("a", "seg0001"), ("a", "seg0002"), ("h", "seg0001")]
    assert len(_state(tmp_path, root, cid, todos=True).cards) == 6


def test_label_goes_to_human_store_with_segment_rev(tmp_path: Path) -> None:
    root, cid, cdir = _setup(tmp_path)
    state = _state(tmp_path, root, cid)
    card = next(c for c in state.cards.values() if c.video == "a")
    with pytest.raises(ValueError):
        save_label(state, {"card": card.id, "v": "dub", "rotulo": "corrigiria"})
    save_label(state, {"card": card.id, "v": "dub", "rotulo": "corrigiria",
                       "classe": "sentido.literal", "nota": "literal"})
    seg = card.versions[0].segment
    human, _ = load_human("en-xx", "a", tmp_path / "human")
    assert human[(seg.id, seg.rev)]["classe"] == "sentido.literal"


def test_blind_ab_hides_versions_and_resolves_preference(tmp_path: Path) -> None:
    root, cid, cdir = _setup(tmp_path)
    state = _state(tmp_path, root, cid, ab_base="base")
    card = next(c for c in state.cards.values() if len(c.versions) == 2)
    assert {v.key for v in card.versions} == {"1", "2"}
    assert {v.label for v in card.versions} == {"cand", "base"}
    page = render_page(state)
    assert '"cand"' not in page and '"base"' not in page  # a página não sabe qual é qual
    cand_key = next(v.key for v in card.versions if v.label == "cand")
    save_pref(state, {"card": card.id, "melhor": cand_key})
    [pref] = read_jsonl(tmp_path / "human" / "en-xx" / "preferencias.jsonl")
    assert pref["preferido"] == "candidato" and pref["ciclo"] == cid


def test_short_verdict_is_keyed_to_this_exact_run(tmp_path: Path) -> None:
    root, cid, cdir = _setup(tmp_path)
    state = _state(tmp_path, root, cid)
    save_short(state, {"video": "a", "short": "nao_publicaria"})
    _, verdicts = load_human("en-xx", "a", tmp_path / "human")
    expected = run_revision(load_run(cdir / "cand" / "dub" / "a")[1])
    assert verdicts[expected]["short"] == "nao_publicaria"


def test_clip_covers_segment_with_padding(tmp_path: Path) -> None:
    root, cid, _ = _setup(tmp_path)
    state = _state(tmp_path, root, cid)
    card = next(c for c in state.cards.values() if c.versions[0].segment.id == "seg0001")
    data, sr = sf.read(io.BytesIO(clip_bytes(state, card.id, "dub")))
    assert abs(len(data) / sr - (1.0 + 2 * PAD_SECONDS)) < 0.01


@pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")
def test_page_javascript_parses(tmp_path: Path) -> None:
    root, cid, _ = _setup(tmp_path)
    script = re.search(r"<script>(.*)</script>", render_page(_state(tmp_path, root, cid)),
                       re.S).group(1)
    js = tmp_path / "page.js"
    js.write_text(script, encoding="utf-8")
    assert subprocess.run(["node", "--check", str(js)], capture_output=True).returncode == 0


def test_http_server_end_to_end(tmp_path: Path) -> None:
    root, cid, _ = _setup(tmp_path)
    state = _state(tmp_path, root, cid)
    server = serve(state, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "Escuta" in urllib.request.urlopen(base + "/").read().decode()
        card = next(iter(state.cards))
        body = json.dumps({"card": card, "v": "dub", "rotulo": "ok"}).encode()
        req = urllib.request.Request(base + "/label", body, {"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["ok"]
        bad = urllib.request.Request(base + "/label", b'{"card": "c0", "v": "dub"}',
                                     {"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 400
        audio = urllib.request.urlopen(f"{base}/clip?card={card}&v=src").read()
        assert audio[:4] == b"RIFF"
    finally:
        server.shutdown()
