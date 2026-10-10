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
import re
import unicodedata
from dataclasses import dataclass

from packages.ptbr.syllables import count_syllables

WINDOW_SIZE = 12
WINDOW_CONTEXT_LINES = 3


@dataclass(frozen=True)
class TargetLanguage:
    """Tudo que a tradução sabe sobre o idioma de destino.

    O produto só faz EN->pt-BR hoje, mas outros idiomas vêm depois
    (decisão de 2026-10-10): regra específica de idioma mora aqui, como
    dado, e não no prompt nem na validação compartilhados. Um idioma novo é
    um perfil novo em `LANGUAGES`, sem mexer na lógica.
    """

    code: str
    name: str
    scripts: tuple[str, ...]
    extra_letters: frozenset[str]
    style_rules: tuple[str, ...]
    # Faixa aceitável de len(tradução)/len(original), em caracteres — muda
    # muito por idioma (EN->pt fica perto de 1; EN->japonês bem abaixo).
    char_ratio_range: tuple[float, float]


PT_BR = TargetLanguage(
    code="pt-BR",
    name="Brazilian Portuguese",
    scripts=("LATIN",),
    extra_letters=frozenset("ªº"),
    style_rules=(
        'Spoken register: use contractions such as "tá", "pra", "cê", "tô", "né" when they '
        "sound natural.",
        'Never use chat abbreviations ("vc", "vcs", "q", "pq", "tb", "tbm", "blz"); write the '
        "words in full.",
        '"you" addressing one person is "você" or "cê"; addressing several people is "vocês".',
    ),
    # Medido em 25 falas boas de 5 Shorts: 0,77–1,34. Margem larga de propósito.
    char_ratio_range=(0.5, 1.8),
)

LANGUAGES: dict[str, TargetLanguage] = {PT_BR.code: PT_BR}

# T0.30: o segmento deixou de ser a unidade de TRADUÇÃO (continua sendo a de
# síntese e edição). Traduzido sozinho, mesmo com as frases vizinhas como
# contexto, um 7B errava pronome ("Here you" -> "Aqui vcs"), perdia cláusula
# de frase cortada entre segmentos ("for that." sumia) e deixava palavra em
# inglês ("Job?") — ~6 de 13 falas erradas no Short padrão. Uma janela de
# diálogo inteira por chamada deixa o modelo resolver tudo isso junto, e
# custa menos: as instruções vão uma vez por janela, não uma por segmento.
# Instruções em inglês e genéricas; o que é do idioma vem do `TargetLanguage`.
WINDOW_PROMPT_TEMPLATE = """You dub YouTube Shorts from English into spoken {language}. Translate \
every line below the way a native {language} dubbing actor would say it: natural and direct.

Rules:
- One translation per line, with the same "id". Do not merge or split lines. If a sentence is \
cut across two lines, translate each piece in its own place, so that read in order they form \
the complete sentence. Never split a word between two lines: every line holds whole words.
- Translate ALL the content of every line. Do not summarize and do not add anything.
- Keep profanity, slang, irony and intensity at exactly the original level. Do not soften, \
censor or change the content.
- Use the whole conversation to resolve pronouns and references (who is talking to whom).
- Keep the original emphasis punctuation ("!", "?", "...").
- Write only in {language}, in its own writing system. Do not leave English words, except \
proper nouns and the terms listed below.
{language_rules}{preserve_block}{context_block}
Lines to translate (JSON):
{lines_json}

Reply ONLY with JSON, no markdown: {{"translations": [{{"id": "...", "text": "..."}}, ...]}}"""

_WINDOW_PRESERVE_BLOCK = """
Proper nouns and terms to keep exactly as written: {terms}
"""

_WINDOW_CONTEXT_BLOCK = """
Previous lines, for conversation context only — do NOT translate these:
{context_json}
"""


@dataclass
class WindowLine:
    id: str
    speaker: str | None
    text: str


def build_windows(
    lines: list[WindowLine],
    *,
    size: int = WINDOW_SIZE,
    context_lines: int = WINDOW_CONTEXT_LINES,
) -> list[tuple[list[WindowLine], list[WindowLine]]]:
    """Fatia o vídeo em janelas de `size` falas, cada uma com as
    `context_lines` falas anteriores (texto ORIGINAL) como contexto.

    Contexto em inglês, não as traduções da janela anterior: assim as
    janelas não dependem umas das outras e todas vão no mesmo lote de GPU —
    um Short cabe em 1–2 janelas, então a perda de consistência é pequena.
    """
    return [
        (lines[max(0, start - context_lines) : start], lines[start : start + size])
        for start in range(0, len(lines), size)
    ]


def split_window(
    context_before: list[WindowLine],
    lines: list[WindowLine],
    *,
    context_lines: int = WINDOW_CONTEXT_LINES,
) -> list[tuple[list[WindowLine], list[WindowLine]]]:
    """Divide uma janela que falhou em duas metades, cada uma com contexto.

    Achado real (NFL, fala rápida cheia de frase cortada): numa janela de
    12 falas o modelo às vezes junta dois fragmentos numa fala só (um id
    some) ou quebra o JSON — e repete o erro nas 3 tentativas, deixando 12
    segmentos sem tradução. Janela menor erra menos; no limite, 1 fala com
    contexto não tem como ser fundida com a vizinha.
    """
    half = len(lines) // 2
    first, second = lines[:half], lines[half:]
    second_context = (context_before + first)[-context_lines:] if context_lines else []
    return [(context_before, first), (second_context, second)]


def _lines_json(lines: list[WindowLine]) -> str:
    return json.dumps(
        [{"id": line.id, "speaker": line.speaker or "?", "text": line.text} for line in lines],
        ensure_ascii=False,
        indent=1,
    )


def build_window_prompt(
    lines: list[WindowLine],
    language: TargetLanguage,
    *,
    context_before: list[WindowLine] | None = None,
    preserve_terms: list[str] | None = None,
) -> str:
    language_rules = "".join(f"- {rule}\n" for rule in language.style_rules)
    preserve_block = (
        _WINDOW_PRESERVE_BLOCK.format(terms=", ".join(preserve_terms)) if preserve_terms else ""
    )
    context_block = (
        _WINDOW_CONTEXT_BLOCK.format(context_json=_lines_json(context_before))
        if context_before
        else ""
    )
    return WINDOW_PROMPT_TEMPLATE.format(
        language=language.name,
        language_rules=language_rules,
        preserve_block=preserve_block,
        context_block=context_block,
        lines_json=_lines_json(lines),
    )


# Folga absoluta pra fala curta, onde a razão de caracteres é ruidosa.
_RATIO_SLACK_CHARS = 12


def parse_window_response(
    llm_response: str, lines: list[WindowLine], language: TargetLanguage
) -> dict[str, str]:
    """Extrai {id: tradução} e exige exatamente uma tradução por fala pedida.

    Qualquer desvio (id faltando, sobrando, texto vazio, resíduo de
    estrutura, alfabeto que não é o do idioma, tamanho incompatível com o
    original) levanta `ValueError` — a janela inteira é refeita, nunca
    aceita parcialmente: uma fala faltando deixaria o segmento com o texto
    em inglês, e uma sobrando desalinharia os ids.

    O tamanho pega o pior caso real (Short do Jay Shetty, frases paralelas
    cortadas no meio): todos os ids presentes, mas o conteúdo deslocado —
    frases inteiras numa fala, repetidas na seguinte, e a última com só o
    rabo. Razões de 3,07 e 0,20 contra 0,77–1,34 nas falas boas.
    """
    expected_ids = [line.id for line in lines]
    source_len = {line.id: len(line.text) for line in lines}
    low, high = language.char_ratio_range
    data = json.loads(_extract_json_object(llm_response))
    items = data.get("translations")
    if not isinstance(items, list):
        raise ValueError("JSON não tem a lista 'translations'")
    result: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"item de tradução não é objeto: {item!r:.80}")
        seg_id, text = item.get("id"), item.get("text")
        if not isinstance(seg_id, str) or not isinstance(text, str) or not text.strip():
            raise ValueError(f"item sem id ou texto: {item!r:.80}")
        if seg_id in result:
            raise ValueError(f"id repetido na resposta: {seg_id}")
        if text.lstrip()[:1] in ("[", "]", "{", "}"):
            raise ValueError(f"texto com resíduo de estrutura JSON: {text!r:.80}")
        if has_disallowed_script(text, language):
            raise ValueError(f"texto fora do alfabeto de {language.code}: {text!r:.80}")
        n_src, n_out = source_len.get(seg_id, 0), len(text.strip())
        if n_src and not (low * n_src - _RATIO_SLACK_CHARS <= n_out
                          <= high * n_src + _RATIO_SLACK_CHARS):
            raise ValueError(
                f"tamanho incompatível em {seg_id} ({n_out} vs {n_src} caracteres no "
                "original): conteúdo provavelmente deslocado entre falas"
            )
        result[seg_id] = text.strip()
    if set(result) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(result))
        extra = sorted(set(result) - set(expected_ids))
        raise ValueError(
            f"ids não batem com as falas pedidas (faltando {missing}, sobrando {extra})"
        )
    return result


# T0.23: palavras capitalizadas comuns em início de frase no inglês falado de
# trailer/vídeo — sem essa lista, a regra "sentence-initial repetido em >=2
# segmentos" (ver `extract_proper_nouns`) falsamente marcaria "The"/"Now"/"I"
# como nome próprio em qualquer vídeo com mais de uma frase. Achado real
# testando contra `phase1test/` (NFL, 90 segmentos de narração esportiva):
# a lista original (artigos/pronomes) não bastava — verbos auxiliares/modais
# ("Has", "Can") e interjeições de narração ("Yeah", "Look") também abrem
# frase repetidamente sem serem nome próprio. Ampliada, não específica da
# NFL (são palavras comuns do inglês falado em geral, não termos do esporte).
_SENTENCE_STARTER_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "we", "you", "he", "she",
    "they", "it", "i", "my", "your", "his", "her", "our", "their", "yes", "no",
    "ok", "okay", "and", "but", "so", "now", "well", "oh", "hey", "here", "there",
    "what", "who", "when", "where", "why", "how", "let's", "lets", "don't", "dont",
    "can't", "cant", "just", "if", "then", "there's", "theres", "it's", "its",
    "we're", "were", "you're", "youre", "i'm", "im", "all", "come", "go",
    "that's", "thats", "he's", "hes", "she's", "shes", "can", "could", "should",
    "would", "will", "may", "might", "must", "has", "have", "had", "is", "was",
    "does", "did", "do", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "look", "watch", "see", "wow", "boom", "runs", "run",
    "for", "to", "with", "on", "in", "at", "yeah", "yep", "nope", "alright",
}

_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z']*")
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")


def extract_proper_nouns(texts: list[str]) -> list[str]:
    """Heurística pra achar nomes próprios/termos recorrentes num vídeo inteiro.

    Não é NER real — não há modelo de reconhecimento de entidade no pipeline
    (invariante de simplicidade/custo do CLAUDE.md). Um nome próprio é
    sinalizado quando a palavra capitalizada:
    - aparece FORA do início de frase pelo menos uma vez (sentence-initial é
      ambíguo: qualquer palavra comum capitaliza ali); ou
    - é toda em maiúsculas com 2+ letras (sigla/acrônimo, ex. "NFL", "GT3"); ou
    - aparece capitalizada no início de frase em 2+ segmentos DIFERENTES — uma
      palavra comum não teria motivo pra se repetir assim entre falas
      distintas, mas um nome de jogador/personagem citado várias vezes sim.

    Palavras da lista `_SENTENCE_STARTER_STOPWORDS` (artigos, pronomes,
    conectivos comuns de início de frase) nunca são sinalizadas, mesmo se
    baterem a regra de repetição — do contrário qualquer vídeo com >1 frase
    marcaria "The"/"Now"/"I" como nome próprio.

    Critério de aceite da T0.23 é checagem MANUAL (comparar 30 segmentos
    reais) — esta função é um ponto de partida pro prompt, não a validação.
    """
    sentence_initial_counts: dict[str, set[int]] = {}
    non_initial_hit: set[str] = set()
    first_seen_spelling: dict[str, str] = {}

    for seg_index, text in enumerate(texts):
        for sentence in _SENTENCE_SPLIT_PATTERN.split(text):
            words = _WORD_PATTERN.findall(sentence)
            for position, word in enumerate(words):
                # len<2 descarta letra solta colada a número por tokenização
                # (achado real: "6X gloves" -> ['X'] isolado, "X" não é nome).
                if len(word) < 2 or not word[0].isupper():
                    continue
                key = word.lower()
                if key not in first_seen_spelling:
                    first_seen_spelling[key] = word
                if position == 0:
                    sentence_initial_counts.setdefault(key, set()).add(seg_index)
                else:
                    non_initial_hit.add(key)

    result: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(text):
            for word in _WORD_PATTERN.findall(sentence):
                key = word.lower()
                if key in seen or key in _SENTENCE_STARTER_STOPWORDS:
                    continue
                if len(word) < 2 or not word[0].isupper():
                    continue
                is_acronym = word.isupper() and len(word) >= 2
                is_repeated_sentence_initial = len(sentence_initial_counts.get(key, ())) >= 2
                if key in non_initial_hit or is_acronym or is_repeated_sentence_initial:
                    seen.add(key)
                    result.append(first_seen_spelling[key])
    return result


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
        # T0.22, achado real (lego_batman seg0002): mesmo depois do
        # desaninhamento acima, às vezes sobra resíduo de estrutura de
        # lista/JSON NO TEXTO do candidato (`['Alguma fala`) — não é o
        # padrão `[["texto"]]` conhecido e recuperável, é a string em si
        # já corrompida. Chegou a ser mandado pro MOSS-TTS sintetizar
        # colchete/aspas falados. Diferente do caso acima, aqui não tem
        # recuperação óbvia — rejeita e deixa o retry tentar de novo.
        if c.lstrip()[:1] in ("[", "]", "'", '"'):
            raise ValueError(
                f"candidato começa com resíduo de estrutura de lista/JSON "
                f"(colchete ou aspas): {c!r:.80} — resposta do LLM corrompida"
            )
        # Achado real (NFL seg0022, "but evidently"): o Qwen trocou de idioma
        # no meio da frase ("Ele foi derrubado, mas显而易见") nos 5 candidatos.
        # Descarta só o candidato contaminado; se nenhum sobrar, o retry tenta
        # de novo — nunca deixa outro alfabeto chegar no MOSS-TTS.
        if has_disallowed_script(c, PT_BR):
            continue
        result.append(c)
    if not result:
        raise ValueError(
            "todos os candidatos têm letras fora do alfabeto latino (troca de "
            "idioma do LLM no meio da frase)"
        )
    return result


def has_disallowed_script(text: str, language: TargetLanguage) -> bool:
    """Alguma letra fora do(s) sistema(s) de escrita do idioma de destino.

    Usada em dois pontos: aqui, pra rejeitar tradução contaminada, e em
    `pod_worker.cmd_translate`, pra listar os tokens do vocabulário do Qwen
    que nunca podem ser gerados (`suppress_tokens`) — só o filtro depois da
    geração não bastava: no NFL seg0022 os 3 retries × 5 candidatos vazaram
    chinês, mesmo com instrução explícita no prompt.
    """
    return any(
        ch.isalpha()
        and ch not in language.extra_letters
        and not unicodedata.name(ch, "").startswith(language.scripts)
        for ch in text
    )


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
    (`WINDOW_PROMPT_TEMPLATE`) e essa pontuação reforçando quando ele funciona.

    Não validado contra o modelo real ainda — hipótese com base num
    achado anterior, não medição nova desta mudança específica. Ver
    `docs/moss_tts_investigation.md`.
    """
    return 1.0 if _emphasis_signature(source_text) == _emphasis_signature(candidate_text) else 0.0
