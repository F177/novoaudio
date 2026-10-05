"""Tradução com restrição de comprimento.

Lógica pura: monta o prompt que pede vários candidatos de tradução pt-BR
pra um orçamento de sílabas, e faz o parsing/pontuação local das respostas.
A chamada de rede pro LLM (Qwen2.5-7B-Instruct, self-hosted — ver
CLAUDE.md) não mora aqui; vive em `runpod/translate/handler.py`, que
importa este módulo pras partes puras.

A pontuação local cobre **aderência ao orçamento** (calculável) e uma
heurística fraca de **naturalidade** (presença de marcadores de registro
falado). **Fidelidade factual não é pontuada aqui** — o critério de
aceite do T0.8 é explícito que isso é checagem manual (comparar 20
candidatos contra o original), não algo que uma função pura consiga
avaliar sem outra chamada de LLM/embeddings.

**Validação real (36 segmentos de um vídeo real, `translation_data/` —
gitignored, dados locais):** o melhor candidato bateu a tolerância de
±20% em 81% dos segmentos com orçamento ≥10 sílabas (17/21) — dentro da
meta de 80% do T0.8. Mas caiu pra 12,5% (2/16) em segmentos com orçamento
<10 sílabas: numa tolerância relativa, um alvo de poucas sílabas só
aceita erro de fração de sílaba, o que nenhuma tradução consegue
garantir palavra por palavra. Isso não é falha deste módulo — é
exatamente o motivo de existir a ordem de alavancas (texto → velocidade
→ silêncio → time-stretch, ver `budget.py`): segmentos curtos dependem
das próximas alavancas (ajuste automático no T0.9, slider manual no
T2.4) pra fechar esse resíduo de 1-2 sílabas, não só do texto. A
checagem manual de fidelidade factual (20 candidatos) ainda não foi
feita — pendente.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from packages.ptbr.syllables import count_syllables

N_CANDIDATES = 5

PROMPT_TEMPLATE = """Traduza a frase abaixo do inglês para português brasileiro falado, \
natural, como um dublador de vídeo do YouTube diria. Gere {n_candidates} candidatos com \
comprimentos diferentes, do mais compacto ao mais completo, todos fiéis ao sentido original \
— sem perder nenhuma informação factual do texto de origem.

Aproveite as alavancas do português falado sempre que fizer sentido: pro-drop (omitir sujeito \
quando óbvio), voz ativa em vez de passiva, contrações do registro falado ("tá", "pra", "cê", \
"tô", "né").

Preserve a pontuação de ênfase do original ("!", "?", "...") — se a frase original exclama ou \
pergunta, o candidato traduzido deve exclamar ou perguntar também, não virar uma afirmação \
neutra. O sintetizador de voz não recebe indicação de tom além da pontuação do texto — perder \
"!" nesse ponto é a única forma que ele tem de saber que a fala é emocionada, não narrada.

Frase original (inglês): "{source_text}"
Orçamento aproximado: {target_syllables} sílabas — o candidato ideal fica perto disso, mas \
prefira fidelidade e naturalidade a bater o número exato.

Responda APENAS em JSON, sem markdown: {{"candidates": ["...", ...]}}"""


def build_prompt(
    source_text: str, target_syllables: int, *, n_candidates: int = N_CANDIDATES
) -> str:
    return PROMPT_TEMPLATE.format(
        source_text=source_text, target_syllables=target_syllables, n_candidates=n_candidates
    )


def _extract_json_object(text: str) -> str:
    """Acha o primeiro objeto JSON bem-formado em `text` (tolera markdown/
    texto ao redor), contando chaves em vez de regex guloso.

    Achado real 2026-09-18 (beast.mp4, segmentos maiores após o merge do
    piso de duração): o regex antigo (`\\{.*\\}` guloso com DOTALL) pega do
    PRIMEIRO `{` até o ÚLTIMO `}` da resposta inteira — se o LLM escrever
    qualquer coisa extra com chaves (rascunho, correção, exemplo) depois do
    JSON de verdade, o match engole tudo isso junto e `json.loads` tanto
    pode falhar quanto (pior) "conseguir" parsear uma estrutura garantida
    errada. Contar profundidade de chaves acha o par certo do primeiro `{`,
    não importa o que vier depois na resposta.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("resposta do LLM não contém um objeto JSON")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("resposta do LLM tem um objeto JSON incompleto (chave não fechada)")


def parse_candidates(llm_response: str) -> list[str]:
    """Extrai a lista de candidatos do JSON devolvido pelo LLM (tolera markdown ao redor).

    Achado real 2026-09-18 (beast.mp4): pra segmentos maiores (depois do
    merge pelo piso de duração de síntese), o Qwen às vezes aninhou cada
    candidato numa lista de 1 elemento (`[["texto"]]` em vez de
    `["texto"]`) — a versão antiga fazia `str(c)` cegamente, e isso virava
    o texto literal `"['texto']"` sendo mandado pro MOSS-TTS sintetizar
    (com colchetes/aspas falados e sem qualquer validação). O sintoma era
    duração de síntese completamente imprevisível (até 2,3x o alvo, em
    TODOS os candidatos) — parecia instabilidade do MOSS-TTS, mas a raiz
    era texto de entrada corrompido, não o modelo de voz.

    Achado seguinte, mesmo dia: pra pelo menos 1 segmento em 13, o Qwen
    repetiu o MESMO aninhamento nas 3 tentativas de retry seguidas — não é
    sorte de amostragem, é um hábito de formatação dele pra esse tipo de
    conteúdo. Rejeitar e torcer pro retry sortear diferente desperdiça
    tentativas à toa quando o formato do erro é conhecido e recuperável:
    lista de 1 elemento string vira o próprio elemento, sem gastar retry.
    Qualquer outro formato (lista com mais de 1 item, não-string dentro)
    continua sendo rejeitado — só o padrão exato e conhecido é recuperado.
    """
    json_text = _extract_json_object(llm_response)
    data = json.loads(json_text)
    candidates = data.get("candidates") or data.get("candidatos")
    if not candidates:
        raise ValueError("JSON não tem a chave 'candidates'")
    result: list[str] = []
    for c in candidates:
        if isinstance(c, list) and len(c) == 1 and isinstance(c[0], str):
            c = c[0]
        if not isinstance(c, str):
            raise ValueError(
                f"candidato não é uma string (veio {type(c).__name__}: {c!r:.80}) — "
                "resposta do LLM provavelmente aninhou uma lista a mais"
            )
        result.append(c)
    return result


@dataclass
class ScoredCandidate:
    text: str
    syllables: int
    budget_score: float
    naturalness_score: float
    expressiveness_score: float
    score: float


_BUDGET_WEIGHT = 0.6
_NATURALNESS_WEIGHT = 0.2
_EXPRESSIVENESS_WEIGHT = 0.2

_SPOKEN_MARKERS = ("tá", "pra", "né", " cê ", "tô", "vc")

_EMPHASIS_CHARS = "!?"


def score_candidate(text: str, target_syllables: int, source_text: str) -> ScoredCandidate:
    syllables = count_syllables(text)
    budget_score = _budget_score(syllables, target_syllables)
    naturalness_score = _naturalness_score(text)
    expressiveness_score = _expressiveness_score(source_text, text)
    score = (
        budget_score * _BUDGET_WEIGHT
        + naturalness_score * _NATURALNESS_WEIGHT
        + expressiveness_score * _EXPRESSIVENESS_WEIGHT
    )
    return ScoredCandidate(
        text=text,
        syllables=syllables,
        budget_score=budget_score,
        naturalness_score=naturalness_score,
        expressiveness_score=expressiveness_score,
        score=score,
    )


def rank_candidates(
    candidates: list[str], target_syllables: int, source_text: str
) -> list[ScoredCandidate]:
    """Ordena os candidatos do melhor pro pior pela pontuação local.

    Não decide sozinho qual usar — só ordena. A escolha final no editor
    (T2.3) mostra os 5 com a contagem de sílabas, o humano troca com 1 clique.
    """
    scored = [score_candidate(c, target_syllables, source_text) for c in candidates]
    return sorted(scored, key=lambda c: c.score, reverse=True)


def _budget_score(syllables: int, target_syllables: int) -> float:
    if target_syllables <= 0:
        return 0.0
    relative_error = abs(syllables - target_syllables) / target_syllables
    return max(0.0, 1.0 - relative_error)


def _naturalness_score(text: str) -> float:
    """Heurística fraca (presença de marcadores de fala) — não substitui ouvir."""
    lower = f" {text.lower()} "
    hits = sum(1 for marker in _SPOKEN_MARKERS if marker in lower)
    return min(1.0, hits / 2)


def _emphasis_signature(text: str) -> str:
    """Quais marcadores de ênfase (!, ?) aparecem no texto, em qualquer
    posição — não só no final, porque a tradução pode reordenar a frase.
    """
    return "".join(sorted({c for c in text if c in _EMPHASIS_CHARS}))


def _expressiveness_score(source_text: str, candidate_text: str) -> float:
    """Recompensa preservar a pontuação de ênfase (!, ?) do original.

    Achado documentado em `packages/pipeline/calibration.json`
    (`known_issues.flat_delivery_on_concatenated_clauses`): o MOSS-TTS não
    tem parâmetro explícito de tom/emoção — pontuação como "!" parece ser
    a única alavanca implícita de entonação que ele usa de fato. Se a
    tradução perde o "!"/"?" do original, a síntese perde o único sinal
    que tinha pra soar menos "narrado" — daí o pedido explícito no prompt
    (`PROMPT_TEMPLATE`) e essa pontuação reforçando quando ele funciona.

    Não validado contra o modelo real ainda — hipótese com base num
    achado anterior, não medição nova desta mudança específica. Ver
    `docs/moss_tts_investigation.md`.
    """
    return 1.0 if _emphasis_signature(source_text) == _emphasis_signature(candidate_text) else 0.0
