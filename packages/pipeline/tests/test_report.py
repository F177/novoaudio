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


def _entry(**overrides) -> dict:
    entry = build_history_entry(_report(**overrides), generated_at="2026-09-27T10:00:00")
    return entry


def test_build_history_entry_extracts_expected_fields() -> None:
    entry = build_history_entry(_report(), generated_at="2026-09-27T10:00:00")
    assert entry["generated_at"] == "2026-09-27T10:00:00"
    assert entry["video"] == "video.mp4"
    assert entry["scorecard"]["verdict"] == "aprovado"
    assert entry["diagnostics"] == []
    assert entry["segments"] == []


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


def _extract_payload(html: str) -> list[dict]:
    match = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    return json.loads(match.group(1))
