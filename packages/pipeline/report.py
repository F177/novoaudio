"""Gera o dashboard de qualidade como UM ÚNICO arquivo HTML local,
consolidado (item 5 pedido pelo usuário, 2026-09-26/27: primeiro "o
relatório quero que seja um arquivo local... e não um artefato do
Claude", depois "vamos manter um html de dashboard que fica salvo local
com todos os relatórios em vez de ter um arquivo pra cada").

Duas peças, ambas puras (sem I/O — mesma regra de dependência de
`packages/pipeline`, CLAUDE.md; quem chama, `scripts/dub.py`, decide onde
ler/escrever `reports/history.json` e `reports/dashboard.html`):

- `build_history_entry(report, generated_at=...)`: reduz um `report`
  completo de `scripts/dub.py` a um registro compacto pra guardar no
  histórico.
- `render_dashboard_html(history)`: devolve a página inteira — barra
  lateral com TODAS as execuções (mais recente primeiro), painel de
  detalhe da execução selecionada (o mesmo conteúdo que antes vivia num
  HTML por vídeo: métricas, flags, diagnóstico, auditoria final, piores
  segmentos). Tudo embutido como um JSON só; a troca de execução é só
  JS local, sem servidor.
"""

from __future__ import annotations

import json
from typing import Any

FLAG_LABELS = {
    "fora_duracao": "Fora da duração",
    "cer_alto": "CER alto (candidato)",
    "truncado": "Truncado",
    "traducao_infiel": "Tradução infiel",
    "clipping": "Clipping",
    "silencio_anormal": "Silêncio anormal",
    "locutor_suspeito": "Locutor suspeito",
    "entonacao_desalinhada": "Entonação desalinhada",
    "emocao_incompativel": "Emoção incompatível",
    "cer_final_alto": "CER alto (áudio final)",
}
NOT_COMPUTED_FLAGS = {"traducao_infiel", "locutor_suspeito"}
VERDICT_LABELS = {"aprovado": "Aprovado", "revisar": "Revisar", "problematico": "Problemático"}


def build_history_entry(report: dict[str, Any], *, generated_at: str) -> dict[str, Any]:
    """Reduz um `report` completo a um registro de histórico. `generated_at`
    (ISO 8601) vem de quem chama — mantém esta função determinística/pura,
    sem chamar relógio (mesma regra de `packages/pipeline`)."""
    return {
        "generated_at": generated_at,
        "video": report.get("video", ""),
        "output": report.get("output", ""),
        "assembly": report.get("assembly", {}),
        "scorecard": report.get("scorecard", {}),
        "diagnostics": report.get("diagnostics", []),
        "segments": report.get("segments", []),
    }


def render_dashboard_html(history: list[dict[str, Any]]) -> str:
    """Devolve a página HTML completa (string) — ver docstring do módulo.
    `history` mais recente por último (ordem de append); a página mostra
    o mais recente primeiro e o seleciona por padrão."""
    payload = json.dumps(history, ensure_ascii=False).replace("</script>", "<\\/script>")

    return f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Painel de Dublagem</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
{_STYLE}
</style>
</head>
<body>
<div class="page">
  <aside class="sidebar">
    <div class="sidebar-title">novoaudio &middot; execuções</div>
    <div class="run-list" id="run-list"></div>
  </aside>
  <main class="detail" id="detail"></main>
</div>

<script id="report-data" type="application/json">{payload}</script>
<script>
{_SCRIPT}
</script>
</body>
</html>
"""


_STYLE = """
  :root {
    --bg: #f6f8f7; --surface: #ffffff; --surface-2: #eef2f0;
    --ink: #12211d; --muted: #5b6c66; --border: #dde4e1;
    --accent: #0f766e; --accent-soft: #d7ece9;
    --good: #2f9e58; --good-soft: #e1f3e6;
    --warning: #c97a1f; --warning-soft: #faecd8;
    --critical: #c1432f; --critical-soft: #fbe4df;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0b1412; --surface: #101c19; --surface-2: #16221f;
      --ink: #e7efec; --muted: #93a8a1; --border: #223330;
      --accent: #2dd4bf; --accent-soft: #163330;
      --good: #4ade80; --good-soft: #16301f;
      --warning: #f2a94e; --warning-soft: #362510;
      --critical: #f2705c; --critical-soft: #35180f;
    }
  }
  * { box-sizing: border-box; }
  html { background: var(--bg); }
  body {
    background: var(--bg); color: var(--ink);
    font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    margin: 0;
  }
  h1, h2, h3 { font-family: "IBM Plex Sans Condensed", ui-sans-serif, system-ui, sans-serif; margin: 0; }
  .tabular { font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono", ui-monospace, monospace; }

  .page { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
  .sidebar { background: var(--surface); border-right: 1px solid var(--border); padding: 20px 14px; overflow-y: auto; }
  .sidebar-title { font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent); font-weight: 600; margin: 0 6px 14px; }
  .run-list { display: flex; flex-direction: column; gap: 6px; }
  .run-item { display: flex; flex-direction: column; gap: 4px; padding: 10px 12px; border-radius: 10px; cursor: pointer; border: 1px solid transparent; }
  .run-item:hover { background: var(--surface-2); }
  .run-item.selected { background: var(--accent-soft); border-color: var(--accent); }
  .run-item-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .run-item-name { font-size: 0.83rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 160px; }
  .run-item-meta { display: flex; align-items: center; justify-content: space-between; font-size: 0.72rem; color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .dot.aprovado { background: var(--good); }
  .dot.revisar { background: var(--warning); }
  .dot.problematico { background: var(--critical); }

  .detail { padding: 32px 28px 64px; max-width: 900px; }
  header.run-header { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 16px; padding-block-end: 20px; border-bottom: 1px solid var(--border); }
  .eyebrow { font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent); font-weight: 600; margin-block-end: 6px; }
  h1 { font-size: 1.7rem; font-weight: 700; }
  .video-name { color: var(--muted); font-size: 0.82rem; margin-block-start: 4px; max-width: 60ch; word-break: break-all; }
  .verdict { display: inline-flex; align-items: center; gap: 8px; padding: 9px 18px; border-radius: 999px; font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; font-size: 0.98rem; white-space: nowrap; }
  .verdict::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: currentColor; }
  .verdict.aprovado { background: var(--good-soft); color: var(--good); }
  .verdict.revisar { background: var(--warning-soft); color: var(--warning); }
  .verdict.problematico { background: var(--critical-soft); color: var(--critical); }
  section { margin-block-start: 34px; }
  .section-title { font-size: 1.05rem; font-weight: 600; margin-block-end: 4px; }
  .section-hint { color: var(--muted); font-size: 0.83rem; margin-block-end: 16px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
  .tile { background: var(--surface); padding: 16px 18px; display: flex; flex-direction: column; gap: 6px; }
  .tile-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
  .tile-value { font-size: 1.55rem; font-weight: 600; line-height: 1; }
  .tile-caption { font-size: 0.76rem; color: var(--muted); }
  .bars { display: flex; flex-direction: column; gap: 10px; }
  .bar-row { display: grid; grid-template-columns: 168px 1fr 84px; align-items: center; gap: 12px; }
  .bar-name { font-size: 0.85rem; }
  .bar-track { position: relative; height: 20px; background: var(--surface-2); border-radius: 5px; overflow: hidden; }
  .bar-fill { position: absolute; inset-block: 0; left: 0; border-radius: 5px; }
  .bar-count { font-size: 0.8rem; color: var(--muted); text-align: right; }
  .axis { display: grid; grid-template-columns: 168px 1fr 84px; gap: 12px; margin-block-start: 6px; }
  .axis-ticks { display: flex; justify-content: space-between; font-size: 0.68rem; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 12px; }
  th, td { text-align: left; padding: 11px 14px; border-bottom: 1px solid var(--border); vertical-align: top; }
  thead th { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; background: var(--surface-2); }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: var(--surface-2); }
  .time-cell { white-space: nowrap; }
  .cer-cell { font-weight: 600; }
  .cer-cell.high { color: var(--critical); }
  .chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip { font-size: 0.68rem; padding: 2px 8px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); font-weight: 500; white-space: nowrap; }
  .quote-original { color: var(--muted); font-style: italic; margin: 0 0 3px; }
  .quote-dub { margin: 0 0 3px; }
  .quote-heard { margin: 0; color: var(--accent); }
  .diag-list { display: flex; flex-direction: column; gap: 14px; }
  .diag-card { border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; background: var(--surface); }
  .diag-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-block-end: 8px; flex-wrap: wrap; }
  .diag-title { font-weight: 600; font-family: "IBM Plex Sans Condensed", sans-serif; font-size: 1.02rem; }
  .diag-rate { font-size: 0.78rem; color: var(--critical); font-weight: 600; font-family: "IBM Plex Mono", monospace; }
  .diag-cause { font-size: 0.87rem; margin: 0 0 8px; }
  .diag-card ul { margin: 0 0 8px; padding-inline-start: 20px; font-size: 0.85rem; display: flex; flex-direction: column; gap: 6px; }
  .diag-kpi { font-size: 0.8rem; color: var(--muted); margin: 0; }
  .diag-empty { color: var(--muted); font-size: 0.88rem; }
  .empty-state { color: var(--muted); font-size: 0.9rem; padding: 40px 0; }
  footer.run-footer { margin-block-start: 40px; padding-block-start: 16px; border-top: 1px solid var(--border); font-size: 0.78rem; color: var(--muted); }
  @media (max-width: 720px) {
    .page { grid-template-columns: 1fr; }
    .sidebar { border-right: none; border-bottom: 1px solid var(--border); max-height: 260px; }
    .detail { padding: 24px 16px 48px; }
  }
  @media (max-width: 560px) { .bar-row, .axis { grid-template-columns: 110px 1fr 60px; } .bar-name { font-size: 0.78rem; } }
"""

_SCRIPT = """
  const history = JSON.parse(document.getElementById('report-data').textContent);
  const FLAG_LABELS = __FLAG_LABELS__;
  const NOT_COMPUTED = new Set(__NOT_COMPUTED__);
  const VERDICT_LABELS = __VERDICT_LABELS__;

  function esc(value) {
    if (value == null) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  function fmtPct(x) { return (x * 100).toFixed(0) + "%"; }
  function fmtTime(s) {
    if (s == null) return "-";
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1).padStart(4, "0");
    return m + ":" + sec;
  }
  function fmtWhen(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    return d.toLocaleString("pt-BR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
  }
  function baseName(path) {
    if (!path) return "(sem vídeo)";
    return path.replace(/\\\\/g, "/").split("/").pop();
  }
  function severityColor(rate) {
    if (rate === 0) return "var(--muted)";
    if (rate < 0.25) return "var(--good)";
    if (rate < 0.5) return "var(--warning)";
    return "var(--critical)";
  }

  let selectedIndex = history.length - 1;

  function renderSidebar() {
    const list = document.getElementById("run-list");
    list.innerHTML = "";
    for (let i = history.length - 1; i >= 0; i--) {
      const entry = history[i];
      const sc = entry.scorecard || {};
      const item = document.createElement("div");
      item.className = "run-item" + (i === selectedIndex ? " selected" : "");
      item.innerHTML =
        '<div class="run-item-top"><span class="dot ' + esc(sc.verdict) + '"></span>' +
        '<span class="run-item-name">' + esc(baseName(entry.video)) + '</span></div>' +
        '<div class="run-item-meta"><span>' + esc(fmtWhen(entry.generated_at)) + '</span>' +
        '<span class="tabular">' + fmtPct(sc.any_flag_rate || 0) + '</span></div>';
      item.addEventListener("click", () => { selectedIndex = i; renderSidebar(); renderDetail(); });
      list.appendChild(item);
    }
  }

  function renderDiagnostics(diagnostics) {
    if (!diagnostics || !diagnostics.length) {
      return '<div class="diag-empty">Nenhuma flag acima do limiar de revisão nesta execução.</div>';
    }
    const cards = diagnostics.map((d) => {
      const steps = (d.como_resolver || []).map((s) => '<li>' + esc(s) + '</li>').join("");
      const stepsHtml = steps ? ('<ul>' + steps + '</ul>') : "";
      return '<div class="diag-card"><div class="diag-head"><span class="diag-title">' + esc(d.titulo) +
        '</span><span class="diag-rate">' + ((d.rate || 0) * 100).toFixed(0) + '% dos segmentos</span></div>' +
        '<p class="diag-cause"><strong>Causa raiz:</strong> ' + esc(d.causa_raiz) + '</p>' + stepsHtml +
        '<p class="diag-kpi"><strong>KPI afetado:</strong> ' + esc(d.kpi_impacto) + '</p></div>';
    });
    return '<div class="diag-list">' + cards.join("") + '</div>';
  }

  function renderDetail() {
    const detail = document.getElementById("detail");
    if (selectedIndex < 0 || !history.length) {
      detail.innerHTML = '<div class="empty-state">Nenhuma execução registrada ainda. Rode <code>scripts/dub.py</code> pra gerar a primeira.</div>';
      return;
    }
    const entry = history[selectedIndex];
    const sc = entry.scorecard || {};
    const assembly = entry.assembly || {};
    const flagsOn = Object.entries(assembly).filter(([, v]) => v === true).map(([k]) => k).join(", ") || "nenhuma (padrão)";

    detail.innerHTML =
      '<header class="run-header"><div><div class="eyebrow">novoaudio &middot; placar de qualidade</div>' +
      '<h1>Raio-X da Dublagem</h1>' +
      '<div class="video-name">' + esc(entry.video) + '</div>' +
      '<div class="video-name">' + esc(fmtWhen(entry.generated_at)) + ' &middot; etapas ligadas: ' + esc(flagsOn) + '</div></div>' +
      '<div class="verdict ' + esc(sc.verdict) + '">' + esc(VERDICT_LABELS[sc.verdict] || sc.verdict) + '</div></header>' +
      '<section><div class="tiles" id="tiles"></div></section>' +
      '<section><div class="section-title">Suspeitas por tipo</div>' +
      '<div class="section-hint">Fração dos segmentos sintetizados marcados com cada flag automática.</div>' +
      '<div class="bars" id="bars"></div>' +
      '<div class="axis"><div></div><div class="axis-ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div><div></div></div></section>' +
      '<section><div class="section-title">Diagnóstico automático</div>' +
      '<div class="section-hint">Pra cada suspeita com taxa acima do limiar de revisão: causa raiz provável e como corrigir (packages/pipeline/diagnostics.py).</div>' +
      renderDiagnostics(entry.diagnostics) + '</section>' +
      '<section><div class="section-title">Auditoria de transcrição final</div>' +
      '<div class="section-hint">Transcrição INDEPENDENTE do áudio realmente entregue comparada com o que era pra ser dito.</div>' +
      '<div class="table-wrap" id="audit-table-wrap"></div></section>' +
      '<section><div class="section-title">Piores segmentos</div>' +
      '<div class="section-hint">Ordenados por número de flags simultâneas, depois por CER.</div>' +
      '<div class="table-wrap" id="worst-table-wrap"></div></section>' +
      '<footer class="run-footer">CER via ASR round-trip, prosódia via correlação de contorno de energia, emoção via emotion2vec_plus_large.</footer>';

    const tiles = document.getElementById("tiles");
    const tileDefs = [
      { label: "Segmentos", value: (sc.synthesized_segments ?? 0) + " / " + (sc.total_segments ?? 0), caption: fmtPct(sc.any_flag_rate || 0) + " com alguma suspeita" },
      { label: "CER candidato", value: sc.avg_cer == null ? "—" : sc.avg_cer.toFixed(3), caption: "ASR round-trip, pré-mixagem" },
      { label: "CER final", value: sc.avg_cer_final == null ? "—" : sc.avg_cer_final.toFixed(3), caption: "ASR round-trip, áudio entregue" },
      { label: "Correlação de prosódia", value: sc.avg_prosody_correlation == null ? "—" : sc.avg_prosody_correlation.toFixed(2), caption: "energia: original × dublado" },
      { label: "Similaridade emocional", value: sc.avg_emotion_similarity == null ? "—" : sc.avg_emotion_similarity.toFixed(2), caption: "emotion2vec: original × dublado" },
    ];
    for (const t of tileDefs) {
      const el = document.createElement("div");
      el.className = "tile";
      el.innerHTML = '<div class="tile-label">' + t.label + '</div><div class="tile-value tabular">' + t.value + '</div><div class="tile-caption">' + t.caption + '</div>';
      tiles.appendChild(el);
    }

    const bars = document.getElementById("bars");
    const flagEntries = Object.entries(sc.flag_rates || {}).sort((a, b) => b[1] - a[1]);
    for (const [name, rate] of flagEntries) {
      const count = (sc.flag_counts || {})[name] || 0;
      const color = severityColor(rate);
      const row = document.createElement("div");
      row.className = "bar-row";
      const notComputed = NOT_COMPUTED.has(name);
      row.innerHTML = '<div class="bar-name">' + (FLAG_LABELS[name] || name) + (notComputed ? ' <span style="color:var(--muted)">(n/d)</span>' : '') + '</div><div class="bar-track"><div class="bar-fill" style="width:' + (rate * 100) + '%; background:' + color + '"></div></div><div class="bar-count tabular">' + count + ' · ' + fmtPct(rate) + '</div>';
      bars.appendChild(row);
    }

    const segments = entry.segments || [];
    const segmentsById = Object.fromEntries(segments.map((s) => [s.id, s]));

    const worstWrap = document.getElementById("worst-table-wrap");
    let worstRows = "";
    for (const w of (sc.worst_segments || [])) {
      const seg = segmentsById[w.id];
      const cerHigh = (w.cer || 0) > 0.15;
      const chips = (w.flags || []).map((f) => '<span class="chip">' + (FLAG_LABELS[f] || f) + '</span>').join("");
      worstRows += '<tr><td class="time-cell tabular">' + fmtTime(w.t_inicio) + '–' + fmtTime(w.t_fim) + '</td><td class="cer-cell tabular' + (cerHigh ? ' high' : '') + '">' + (w.cer ?? 0).toFixed(3) + '</td><td><div class="chips">' + chips + '</div></td><td>' + (seg ? '<p class="quote-original">' + esc(seg.texto_original) + '</p><p class="quote-dub">' + esc(seg.traducao) + '</p>' : '') + '</td></tr>';
    }
    worstWrap.innerHTML = worstRows ? ('<table><thead><tr><th>Tempo</th><th>CER</th><th>Flags</th><th>Fala</th></tr></thead><tbody>' + worstRows + '</tbody></table>') : '<div class="diag-empty" style="padding:14px 16px">Nenhum segmento com suspeita nesta execução.</div>';

    const auditWrap = document.getElementById("audit-table-wrap");
    let auditRows = "";
    const auditable = segments.filter((s) => s.cer_final != null).sort((a, b) => (b.cer_final || 0) - (a.cer_final || 0));
    for (const s of auditable) {
      const cerHigh = (s.cer_final || 0) > 0.15;
      auditRows += '<tr><td class="time-cell tabular">' + fmtTime(s.t_inicio) + '–' + fmtTime(s.t_fim) + '</td><td class="cer-cell tabular' + (cerHigh ? ' high' : '') + '">' + (s.cer_final).toFixed(3) + '</td><td><p class="quote-original">esperado: ' + esc(s.traducao) + '</p><p class="quote-heard">ouvido: ' + esc(s.hipotese_final) + '</p></td></tr>';
    }
    auditWrap.innerHTML = auditable.length ? ('<table><thead><tr><th>Tempo</th><th>CER final</th><th>Esperado × ouvido (ASR)</th></tr></thead><tbody>' + auditRows + '</tbody></table>') : '<div class="diag-empty" style="padding:14px 16px">Auditoria final não rodou nesta execução (relatório antigo, gerado antes desse check existir).</div>';
  }

  renderSidebar();
  renderDetail();
"""

_SCRIPT = (
    _SCRIPT.replace("__FLAG_LABELS__", json.dumps(FLAG_LABELS, ensure_ascii=False))
    .replace("__NOT_COMPUTED__", json.dumps(sorted(NOT_COMPUTED_FLAGS)))
    .replace("__VERDICT_LABELS__", json.dumps(VERDICT_LABELS, ensure_ascii=False))
)
