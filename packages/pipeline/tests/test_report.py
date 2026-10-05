import json
import re

from packages.pipeline.report import build_history_entry, render_dashboard_html


def _report(**overrides) -> dict:
    base = {
        "video": "video.mp4",
        "output": "video.dub.mp4",
        "assembly": {"vocals_only": False, "prosody_transfer": True},
        "scorecard": {
            "total_segments": 2,
            "synthesized_segments": 2,
            "any_flag_rate": 0.0,
            "flag_counts": {"cer_alto": 0},
            "flag_rates": {"cer_alto": 0.0},
            "avg_cer": 0.05,
            "avg_cer_final": 0.05,
            "avg_prosody_correlation": 0.8,
            "avg_emotion_similarity": 0.9,
            "worst_segments": [],
            "verdict": "aprovado",
        },
        "diagnostics": [],
        "segments": [],
    }
    base.update(overrides)
    return base


def _entry(*, git_sha: str | None = "abc1234", git_dirty: bool = False, **overrides) -> dict:
    return build_history_entry(
        _report(**overrides),
        generated_at="2026-09-27T10:00:00",
        git_sha=git_sha,
        git_dirty=git_dirty,
    )


def test_build_history_entry_extracts_expected_fields() -> None:
    entry = build_history_entry(
        _report(), generated_at="2026-09-27T10:00:00", git_sha="abc1234", git_dirty=False
    )
    assert entry["generated_at"] == "2026-09-27T10:00:00"
    assert entry["video"] == "video.mp4"
    assert entry["scorecard"]["verdict"] == "aprovado"
    assert entry["diagnostics"] == []
    assert entry["segments"] == []


def test_build_history_entry_includes_git_sha_and_dirty_flag() -> None:
    entry = build_history_entry(
        _report(), generated_at="2026-09-27T10:00:00", git_sha="deadbeef", git_dirty=True
    )
    assert entry["git_sha"] == "deadbeef"
    assert entry["git_dirty"] is True


def test_build_history_entry_allows_unknown_git_sha() -> None:
    entry = build_history_entry(
        _report(), generated_at="2026-09-27T10:00:00", git_sha=None, git_dirty=False
    )
    assert entry["git_sha"] is None


def test_render_dashboard_html_returns_full_document() -> None:
    html = render_dashboard_html([_entry()])
    assert html.strip().startswith("<!doctype html>")
    assert "</html>" in html


def test_render_dashboard_html_empty_history_does_not_crash() -> None:
    html = render_dashboard_html([])
    assert "<!doctype html>" in html
    payload = _extract_payload(html)
    assert payload == []


def test_render_dashboard_html_embeds_all_runs_in_payload() -> None:
    history = [
        _entry(video="a.mp4"),
        _entry(video="b.mp4"),
        _entry(video="c.mp4"),
    ]
    html = render_dashboard_html(history)
    payload = _extract_payload(html)
    assert [e["video"] for e in payload] == ["a.mp4", "b.mp4", "c.mp4"]


def test_render_dashboard_html_payload_carries_dirty_flag_per_run() -> None:
    # T0.18: execução com working tree sujo precisa aparecer marcada no
    # dashboard — a marcação em si é JS client-side (não dá pra executar
    # num teste Python), então o teste garante o que é verificável aqui:
    # o dado chega certo no payload, e a lógica de marcação (classe
    # "dirty-badge" condicionada a `git_dirty`) está presente no HTML/JS
    # servido.
    history = [
        _entry(video="clean.mp4", git_dirty=False),
        _entry(video="dirty.mp4", git_dirty=True),
    ]
    html = render_dashboard_html(history)
    payload = _extract_payload(html)
    assert payload[0]["git_dirty"] is False
    assert payload[1]["git_dirty"] is True
    assert "dirty-badge" in html
    assert "entry.git_dirty" in html


def test_render_dashboard_html_never_inlines_segment_text_as_raw_html() -> None:
    # Texto de segmento (transcrição/tradução) vem de ASR/LLM sobre vídeo
    # de terceiro — não confiável. A página não deve nunca concatenar
    # esse texto direto no HTML server-side: ele só entra via o payload
    # JSON (escapado no client-side pela função `esc()` do JS antes de
    # qualquer innerHTML) — o teste garante que nada foi colado cru fora
    # do bloco de dados.
    dangerous = "<script>alert(1)</script>"
    history = [
        _entry(
            segments=[
                {
                    "id": "seg0000",
                    "t_inicio": 0.0,
                    "t_fim": 1.0,
                    "texto_original": dangerous,
                    "traducao": "fala & aspas",
                    "cer_final": 0.2,
                    "hipotese_final": "<b>bold</b>",
                }
            ]
        )
    ]
    html = render_dashboard_html(history)
    before_data, _, after_data = html.partition('<script id="report-data"')
    _, _, rest = after_data.partition("</script>")
    assert dangerous not in before_data
    assert dangerous not in rest

    payload = _extract_payload(html)
    assert payload[0]["segments"][0]["texto_original"] == dangerous


def test_render_dashboard_html_includes_flag_overlap_payload_and_wiring() -> None:
    # T0.20: scorecard reporta a matriz de sobreposição entre flags — o
    # dashboard precisa levar o dado e ter a lógica client-side que lê
    # `flag_overlap`/`segments_with_multiple_flags` (não dá pra executar
    # JS aqui, então confirma que o dado chega e que o código que lê
    # esses campos está presente no HTML servido).
    history = [
        _entry(
            scorecard={
                "total_segments": 2,
                "synthesized_segments": 2,
                "any_flag_rate": 0.5,
                "flag_counts": {"cer_alto": 1, "clipping": 1},
                "flag_rates": {"cer_alto": 0.5, "clipping": 0.5},
                "flag_overlap": {
                    "cer_alto": {"cer_alto": 1, "clipping": 1},
                    "clipping": {"cer_alto": 1, "clipping": 1},
                },
                "segments_with_multiple_flags": 1,
                "avg_cer": 0.1,
                "avg_cer_final": 0.1,
                "avg_prosody_correlation": None,
                "avg_emotion_similarity": None,
                "worst_segments": [],
                "verdict": "revisar",
            }
        )
    ]
    html = render_dashboard_html(history)
    payload = _extract_payload(html)
    assert payload[0]["scorecard"]["flag_overlap"]["cer_alto"]["clipping"] == 1
    assert payload[0]["scorecard"]["segments_with_multiple_flags"] == 1
    assert "sc.flag_overlap" in html
    assert "sc.segments_with_multiple_flags" in html


def _extract_payload(html: str) -> list[dict]:
    match = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    return json.loads(match.group(1))
