"""Diagnóstico automático por flag (extensão do item 5 pedido pelo
usuário, 2026-09-26: "para todo grande erro quero que você também faça
uma avaliação de como resolver o problema e aumentar os KPIs" — e depois
esclarecido: "isso tem que ser função do avaliador inteligente que
construímos", não uma análise manual pontual).

Cada vez que `scripts/dub.py` roda, `build_diagnostics(scorecard)` decide
sozinho (sem intervenção humana) quais flags viraram "grande erro" nesse
vídeo (taxa acima de `scorecard.REVIEW_RATE_THRESHOLD`) e anexa, pra cada
uma, uma hipótese de causa raiz e passos concretos pra resolver — extraído
de achados REAIS já documentados neste repositório (`docs/*.md`,
descobertas desta sessão) e de pesquisa externa (whitepapers de TTS
expressivo/dublagem, outros projetos), não de suposição genérica.

Puro (só dicts), sem I/O — mesma regra de dependência de
`packages/pipeline` (CLAUDE.md). O conteúdo em si (`DIAGNOSTICS`) é
mantido à mão como uma base de conhecimento — precisa ser atualizado
quando uma causa raiz for corrigida ou uma nova for descoberta, mas a
DECISÃO de quando incluir cada um no relatório é automática.
"""

from __future__ import annotations

from typing import Any

from packages.pipeline.scorecard import REVIEW_RATE_THRESHOLD

# Cada entrada: causa raiz mais provável (com a fonte real de onde veio —
# achado deste repo, ou pesquisa externa), passos concretos pra corrigir,
# e qual KPI do scorecard deve subir se a correção funcionar. Sem entrada
# aqui = flag ainda não investigada a fundo (`traducao_infiel` e
# `locutor_suspeito` nem são calculadas ainda, ver `quality.py`).
DIAGNOSTICS: dict[str, dict[str, Any]] = {
    "entonacao_desalinhada": {
        "titulo": "Entonação/energia do dublado não acompanha o original",
        "causa_raiz": (
            "Causa raiz real, já documentada em `docs/moss_tts_investigation.md` "
            "(2026-09-19): `infer_delivery_instruction` (packages/pipeline/synthesis.py) "
            "detecta ênfase por pontuação ('!'/'?') e MANDA o MOSS-TTS falar com "
            "'ênfase e emoção genuína' — mas está DESLIGADA (`instruction=None` sempre) "
            "desde essa data, porque aplicá-la incondicionalmente causou 2 casos reais "
            "de loop catastrófico: o modelo, sob instrução de ênfase, 'quer' uma duração "
            "natural de 7-10s pra um orçamento calibrado de ~4,6s, e trava tentando "
            "caber (10/10 amostras em loop com a instrução; 0/10 sem ela). A causa não é "
            "'o modelo é instável' — é a instrução de emoção brigando com o orçamento de "
            "tokens, sem nenhum jeito de sinalizar emoção que não implique falar mais devagar."
        ),
        "como_resolver": [
            "Re-habilitar `infer_delivery_instruction`, mas SÓ quando houver folga real "
            "de orçamento — ex.: pedir `tokens` com uma margem extra (ver "
            "`MIN_SYNTHESIS_SECONDS`/piso de tokens em `synthesis.py`) especificamente "
            "quando a instrução for aplicada, e comprimir o resultado de volta ao alvo "
            "com `time_stretch_to_duration` (já existe, já testado, já usado em produção "
            "pra outros casos de estouro — MAX_TIME_STRETCH_RATIO=1.5x preserva pitch via "
            "phase vocoder). Isso separa 'deixar o modelo falar expressivo' de 'caber no "
            "tempo', em vez de pedir os dois ao mesmo tempo pro mesmo mecanismo.",
            "Validar ANTES de reativar em produção: rodar os mesmos 2 segmentos "
            "catastróficos do achado de 2026-09-19 (seg0006/seg0011 do vídeo beast.mp4, "
            "citados no doc) com instrução+tokens extra+time-stretch, medindo taxa de "
            "loop e CER — mesma disciplina de medição usada pra confirmar que o Layer 3 "
            "de pitch-transfer era prejudicial (não reativar por suposição).",
            "Referência externa: IndexTTS2 (arXiv:2506.21619) resolve exatamente essa "
            "tensão desacoplando emoção de duração — dois modos de geração (contagem de "
            "tokens explícita vs. geração livre fiel à prosódia do prompt) em vez de um "
            "único mecanismo que tenta os dois ao mesmo tempo. TED-TTS (arXiv:2601.03170) "
            "propõe controle de emoção+duração 'training-free' sobre um TTS zero-shot já "
            "pronto — mesma categoria de modelo que o MOSS-TTS, sem precisar re-treinar.",
        ],
        "kpi_impacto": (
            "flag_rates.entonacao_desalinhada e avg_prosody_correlation do placar — "
            "hoje entonacao_desalinhada em 50% no teste do Avengers."
        ),
    },
    "emocao_incompativel": {
        "titulo": "Emoção do dublado (emotion2vec) não bate com a do original",
        "causa_raiz": (
            "Mesma causa raiz de `entonacao_desalinhada`: sem `infer_delivery_instruction` "
            "ativa, todo segmento sai com a entrega neutra/default do MOSS-TTS, "
            "independente da emoção real da fala original (raiva, tristeza, etc.) — o "
            "pipeline não tem NENHUM sinal de emoção chegando na síntese hoje. A tentativa "
            "desta sessão de resolver via pós-processamento de áudio (Layer 3: extrair "
            "pitch/energia do original e impor no dublado, `packages/pipeline/prosody.py`) "
            "foi medida e É PREJUDICIAL — quase dobra o CER round-trip (0,082→0,149, até "
            "5,5x pior num segmento) porque perturba o pitch de um áudio já sintetizado "
            "coerentemente, em vez de pedir a emoção certa NA síntese."
        ),
        "como_resolver": [
            "Não insistir em pós-processamento de sinal (Layer 3 já provou ser o caminho "
            "errado, com número real). O caminho correto é o mesmo do item acima: a "
            "emoção precisa entrar como CONDICIONAMENTO da síntese (instruction do "
            "MOSS-TTS), não como filtro de áudio depois — resolver "
            "`entonacao_desalinhada` (acima) resolve a maior parte deste também, já que "
            "são a mesma causa raiz.",
            "Fonte de sinal de emoção pra alimentar a instrução: hoje "
            "`infer_delivery_instruction` só olha pontuação ('!'/'?') do texto traduzido. "
            "Com `packages/pipeline/quality.py::emotion_similarity` e "
            "`scripts/pod_worker.py::cmd_match_emotion` já existindo (rodam "
            "`emotion2vec_plus_large` no áudio ORIGINAL), dá pra classificar a emoção do "
            "segmento fonte e mapear pra uma instrução mais específica que 'ênfase e "
            "emoção genuína' (ex.: instrução diferente pra 'sad' vs. 'angry' vs. "
            "'neutral') — mais preciso que inferir só da pontuação.",
            "Referência externa: IndexTTS2 usa um módulo Text-to-Emotion (T2E) que prediz "
            "a distribuição de emoção a partir do texto/contexto e combina com embeddings "
            "de emoção pré-computados — mesma ideia de usar um classificador de emoção "
            "pra CONDICIONAR a síntese, não pra corrigir o áudio depois de pronto.",
        ],
        "kpi_impacto": (
            "flag_rates.emocao_incompativel e avg_emotion_similarity do placar — hoje "
            "emocao_incompativel em 50% no teste do Avengers."
        ),
    },
    "cer_final_alto": {
        "titulo": "Áudio final (pós-mixagem/transferência) diverge do texto pretendido",
        "causa_raiz": (
            "`cer_final` é um check novo (2026-09-26, pedido do usuário) que retranscreve "
            "o áudio REALMENTE entregue por segmento — já com energia/pitch transferidos, "
            "se ligados — e recalcula CER, independente do CER medido na escolha do "
            "candidato (que é do áudio seco, antes de qualquer pós-processamento). Se "
            "`cer_final_alto` estiver alto mas `cer_alto` (candidato) não, a divergência "
            "foi introduzida DEPOIS da escolha do candidato — os suspeitos mais prováveis, "
            "em ordem: `--pitch-transfer` (já medido como prejudicial ao CER, Layer 3), "
            "`--prosody-transfer`/energia (medido como seguro, mas revalidar se o áudio de "
            "origem mudou), ou o próprio mix com o fundo original abafando a voz."
        ),
        "como_resolver": [
            "Primeiro passo, automatizável: comparar `cer_final` com `cer` (candidato) "
            "por segmento no próprio relatório — se a diferença for grande e "
            "`--pitch-transfer`/`--prosody-transfer` estiverem ligados nesse run "
            "(`report['assembly']`), a causa é quase certamente a transferência de "
            "prosódia, não a síntese em si.",
            "Se `cer_final_alto` aparecer mesmo SEM nenhuma transferência de prosódia "
            "ligada, o suspeito passa a ser a mixagem com o fundo original "
            "(`packages/pipeline/assembly.py::mix_with_background`) — validar ganho "
            "relativo voz/fundo nesses segmentos especificamente.",
        ],
        "kpi_impacto": "flag_rates.cer_final_alto e avg_cer_final do placar.",
    },
    "silencio_anormal": {
        "titulo": "Segmento com mais silêncio do que o esperado",
        "causa_raiz": (
            "Historicamente (T0.12-T0.17, `docs/moss_tts_investigation.md`) associado a "
            "duas causas raiz reais já corrigidas parcialmente: alucinação de repetição do "
            "faster-whisper em clipes curtos (mitigado com "
            "`condition_on_previous_text=False` + `repetition_penalty`), e o MOSS-TTS "
            "'travando' sob orçamento de tokens apertado. Um segmento ainda marcado hoje "
            "provavelmente é um caso residual não coberto por essas correções."
        ),
        "como_resolver": [
            "Inspecionar o áudio do segmento flagrado diretamente (o relatório local "
            "aponta o intervalo de tempo) — confirmar se é silêncio real do TTS ou "
            "corte por mixagem antes de investir em mudança de código.",
        ],
        "kpi_impacto": "flag_rates.silencio_anormal do placar.",
    },
}


def build_diagnostics(
    scorecard: dict[str, Any], *, threshold: float = REVIEW_RATE_THRESHOLD
) -> list[dict[str, Any]]:
    """Decide automaticamente, a partir do placar, quais flags viraram
    "grande erro" (taxa >= `threshold`) neste vídeo e devolve o
    diagnóstico correspondente pra cada uma — ordenado da taxa mais alta
    pra mais baixa, pra priorizar o que mais vale corrigir primeiro.

    Flag acima do limiar sem entrada em `DIAGNOSTICS` ainda aparece no
    resultado, com uma nota explícita de que a causa raiz não foi
    investigada a fundo ainda — não silencia o problema só por falta de
    pesquisa prévia.
    """
    flag_rates = scorecard.get("flag_rates", {})
    big_problems = sorted(
        ((name, rate) for name, rate in flag_rates.items() if rate >= threshold),
        key=lambda item: item[1],
        reverse=True,
    )

    diagnostics = []
    for name, rate in big_problems:
        entry = DIAGNOSTICS.get(name)
        diagnostics.append(
            {
                "flag": name,
                "rate": rate,
                "titulo": entry["titulo"] if entry else name,
                "causa_raiz": entry["causa_raiz"]
                if entry
                else "Causa raiz ainda não investigada a fundo neste repositório.",
                "como_resolver": entry["como_resolver"] if entry else [],
                "kpi_impacto": entry["kpi_impacto"] if entry else f"flag_rates.{name}",
            }
        )
    return diagnostics
