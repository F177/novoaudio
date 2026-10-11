"""Tradução EN -> idioma de destino para dublagem de Shorts.

Lógica pura (sem rede, sem GPU). O modelo roda em `scripts/pod_worker.py`.

T0.30 (2026-10-10): tradutor dedicado OPUS-MT em vez de LLM. Benchmark em 5
Shorts (Qwen2.5-7B por janela, OPUS-MT, OPUS-MT + Qwen encurtando): o Qwen
errava palavra e sentido ("Quanto débito", "não conhece eu", "muita poder")
e cortava conteúdo; o encurtamento por LLM deixava falas telegráficas. O
OPUS acertou mais, saiu igual a cada rodada e custa ~6 s por Short. Ele
traduz a frase inteira, devolve 8 alternativas (n-best) e a escolha da que
cabe no orçamento de sílabas é feita aqui, sem gerar texto novo.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from packages.ptbr.syllables import count_syllables


@dataclass(frozen=True)
class TargetLanguage:
    """Tudo que a tradução sabe sobre o idioma de destino.

    O produto só faz EN->pt-BR hoje, mas outros idiomas vêm depois
    (decisão de 2026-10-10): regra específica de idioma mora aqui, como
    dado, e não na lógica compartilhada. Um idioma novo é um perfil novo em
    `LANGUAGES`.
    """

    code: str
    count_syllables: Callable[[str], int]
    # Radicais de palavrão: nenhuma alternativa pode ter menos palavrão que
    # as outras (decisão de 2026-10-10: palavrão sai como no original).
    profanity_stems: tuple[str, ...]
    # Tradutor dedicado (OPUS-MT, Marian) e o prefixo de idioma que ele pede.
    mt_model: str
    mt_prefix: str
    # Contrações da fala, (regex, troca): só entram como alternativa mais
    # curta, nunca na tradução que já cabe.
    contractions: tuple[tuple[str, str], ...] = ()
    # Reescritas sempre aplicadas à saída do tradutor, (regex, troca).
    mt_rewrites: tuple[tuple[str, str], ...] = ()
    # Formas de outra variante do idioma (pt-PT pra pt-BR): a alternativa
    # que tem uma delas sai, se sobrar alguma sem.
    mt_avoid: tuple[str, ...] = ()


PT_BR = TargetLanguage(
    code="pt-BR",
    count_syllables=count_syllables,
    mt_model="Helsinki-NLP/opus-mt-tc-big-en-pt",
    mt_prefix=">>pob<< ",  # pob = pt-BR; ">>por<<" sai pt-PT
    # Achado real: o OPUS-MT mistura pt-PT ("mande-me", "tiveres", "Vais",
    # "está bem") no meio das alternativas pt-BR.
    mt_rewrites=((r"\b(\w+)-(me|te|nos|lhe|lhes)\b", r"\2 \1"),),
    mt_avoid=(
        r"\btu\b",
        r"\b(vais|estás|tens|podes|queres|sabes|tiveres|fazes|dizes|és|connosco)\b",
        r"\bestá bem\b",
        r"\bestou\?",  # "Hello?" ao telefone em pt-PT
        r"\b(estou|está|estás|estamos|estão|estava) a \w+r\b",
        r"\b(autocarro|comboio|equipa|telemóvel|fixe|facto|miúd\w*|rapariga\w*)\b",
    ),
    contractions=(
        (r"\bestou\b", "tô"),
        (r"\bestá\b", "tá"),
        (r"\bestava\b", "tava"),
        (r"\bestão\b", "tão"),
        (r"\bestamos\b", "tamos"),
        # "para" verbo ("Para!") fica: só a preposição antes de outra palavra.
        # ponytail: "ele para o carro" vira "pra"; raro, aceito.
        (r"\bpara o\b", "pro"),
        (r"\bpara\b(?=\s+\w)", "pra"),
        (r"\bvocê\b", "cê"),
    ),
    profanity_stems=(
        "porra", "merda", "caralh", "foda", "fod", "puta", "puto", "cacete", "bosta",
        "cuzão", "arrombad", "desgraç", "viad", "babaca",
    ),
)

LANGUAGES: dict[str, TargetLanguage] = {PT_BR.code: PT_BR}


def profanity_count(text: str, language: TargetLanguage) -> int:
    return sum(
        word.startswith(language.profanity_stems) for word in re.findall(r"\w+", text.lower())
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


# --- Tradutor dedicado (OPUS-MT) -------------------------------------------
# Alternativa ao LLM em teste (2026-10-10): um modelo só de tradução, pequeno
# (roda em CPU), devolve várias traduções por frase (n-best do beam search) e
# a escolha da mais curta que cabe é feita aqui, sem gerar texto novo.

# Queda máxima na log-prob média por token (nota do beam) pra uma alternativa
# mais curta ainda valer. Ponto de partida não calibrado.
MT_MAX_SCORE_DROP = 0.25
# Alternativas por frase (beam search, n-best).
MT_BEAMS = 8

_TERMINAL = re.compile(r"[.!?…][\"')\]]*$")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


# Fala sem nenhuma pontuação e mais longa que isso não é emendada: o ASR
# perdeu a pontuação, e juntar vira um corrido sem fim de frase. Caso real
# ("but you're not you're calm" virou "mas cê não tá calmo").
_UNPUNCTUATED_JOIN_MAX_WORDS = 4


def _continues(text: str) -> bool:
    text = text.strip()
    if _TERMINAL.search(text):
        return False
    return bool(re.search(r"[.,;:!?…]", text)) or len(text.split()) <= _UNPUNCTUATED_JOIN_MAX_WORDS


def sentence_groups(texts: list[str]) -> list[list[int]]:
    """Índices de falas consecutivas que formam frases inteiras: uma fala que
    não termina em pontuação final continua na próxima ("Here you" + "go.").
    O tradutor vê a frase inteira, nunca o pedaço."""
    groups: list[list[int]] = []
    for i in range(len(texts)):
        if groups and _continues(texts[i - 1]):
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def contracted(text: str, language: TargetLanguage) -> str:
    def keep_case(match: re.Match[str], repl: str) -> str:
        return repl[:1].upper() + repl[1:] if match.group(0)[:1].isupper() else repl

    for pattern, repl in language.contractions:
        text = re.sub(pattern, lambda m, r=repl: keep_case(m, r), text, flags=re.IGNORECASE)
    return text


def mt_options(
    alternatives: list[tuple[str, float]], language: TargetLanguage
) -> list[tuple[str, float]]:
    """Alternativas do tradutor (melhor primeiro) prontas pra escolha:
    reescritas do idioma aplicadas, variante errada descartada (se sobrar
    alguma certa) e a versão contraída de cada uma, com a mesma nota."""

    def rewrite(text: str) -> str:
        for pattern, repl in language.mt_rewrites:
            text = re.sub(
                pattern,
                lambda m, r=repl: (
                    m.expand(r).capitalize() if m.group(0)[:1].isupper() else m.expand(r)
                ),
                text,
                flags=re.IGNORECASE,
            )
        return text

    alts = [(rewrite(text), score) for text, score in alternatives]
    clean = [
        a for a in alts if not any(re.search(p, a[0], re.IGNORECASE) for p in language.mt_avoid)
    ]
    alts = clean or alts
    # Palavrão sai como no original (decisão de 2026-10-10). Caso real: a
    # alternativa mais curta de "Her bullshit spending" era "Seus gastos".
    swearing = max(profanity_count(text, language) for text, _ in alts)
    alts = [a for a in alts if profanity_count(a[0], language) >= swearing]
    alts += [(contracted(text, language), score) for text, score in alts]
    return list(dict.fromkeys(alts))


def choose_alternatives(
    options: list[list[tuple[str, float]]],
    budget: float,
    count: Callable[[str], int],
    *,
    max_drop: float = MT_MAX_SCORE_DROP,
) -> list[str]:
    """Uma tradução por frase. `options[i]` são as alternativas da frase i,
    `(texto, nota)`, a melhor primeiro. Começa da melhor de cada frase e,
    enquanto o total de sílabas passar de `budget`, troca pela alternativa
    que mais economiza sílaba por ponto de nota perdido — só entre as que
    perdem no máximo `max_drop` de nota. Se nada mais serve, fica como está.
    """
    pick = [0] * len(options)
    while sum(count(options[i][pick[i]][0]) for i in range(len(options))) > budget:
        best: tuple[float, int, int] | None = None
        for i, alts in enumerate(options):
            top, now = alts[0][1], alts[pick[i]]
            for j, (text, score) in enumerate(alts):
                saved = count(now[0]) - count(text)
                if saved <= 0 or top - score > max_drop:
                    continue
                gain = saved / (max(now[1] - score, 0.0) + 1e-3)
                if best is None or gain > best[0]:
                    best = (gain, i, j)
        if best is None:
            break
        pick[best[1]] = best[2]
    return [options[i][pick[i]][0] for i in range(len(options))]


def split_by_source(translation: str, sources: list[str]) -> list[str]:
    """Reparte a tradução de uma frase cortada entre as falas de origem, na
    proporção de caracteres de cada pedaço do original, sempre em fronteira
    de palavra. ponytail: proporção, não alinhamento de palavra; trocar por
    alinhador se a divisão errar muito."""
    words = translation.split()
    if len(sources) == 1:
        return [translation]
    total = sum(len(s) for s in sources) or 1
    parts, start, acc = [], 0, 0
    for k, source in enumerate(sources[:-1]):
        acc += len(source)
        remaining = len(sources) - k - 1
        cut = round(len(words) * acc / total)
        cut = min(max(cut, start + 1), len(words) - remaining)
        parts.append(" ".join(words[start:cut]))
        start = cut
    parts.append(" ".join(words[start:]))
    return parts


def split_by_alignment(translation: str, word_segments: list[int], n: int) -> list[str] | None:
    """Reparte a tradução entre `n` falas seguindo o alinhamento: cada
    palavra traduzida traz o índice da fala de origem com que ela se alinha
    (atenção do tradutor). Os cortes são os que mais concordam com isso,
    mantendo a ordem e pelo menos uma palavra por fala. `None` quando não dá
    (contagem de palavras não bate ou faltam palavras) — aí vale a proporção.
    """
    words = translation.split()
    if len(words) != len(word_segments) or len(words) < n:
        return None
    inf = float("inf")
    # cost[j] = menor erro com a palavra atual na fala j; back guarda a origem
    cost = [0.0 if word_segments[0] == 0 else 1.0] + [inf] * (n - 1)
    back: list[list[int]] = [[0] * n]
    for label in word_segments[1:]:
        new, came = [inf] * n, [0] * n
        for j in range(n):
            stay, step = cost[j], cost[j - 1] if j else inf
            new[j], came[j] = (stay, j) if stay <= step else (step, j - 1)
            new[j] += label != j
        cost, back = new, [*back, came]
    j, owner = n - 1, []
    for i in range(len(words) - 1, -1, -1):
        owner.append(j)
        j = back[i][j]
    owner.reverse()
    return [" ".join(w for w, o in zip(words, owner, strict=True) if o == k) for k in range(n)]

