"""Contador de sílabas para português brasileiro.

A contagem de sílabas depende só dos NÚCLEOS silábicos (vogais e
semivogais), nunca das consoantes que ficam entre eles. Por isso o
algoritmo não precisa tratar dígrafos (`lh`, `nh`, `ch`, `rr`, `ss`) nem
encontros consonantais (`tr`, `bl`, `pr`...) de forma especial: eles
nunca contêm vogal e nunca formam núcleo, então só atuam como
separadores entre sequências de vogais.

O trabalho real é decidir, para cada sequência contígua de letras
vocálicas ("encontro vocálico"), quantos núcleos ela representa:

- Uma vogal plena (a, e, i, o, u tônicos ou átonos, e suas formas
  acentuadas) sempre é o centro de um núcleo.
- Uma semivogal (`i`/`u` átonos, sem acento) gruda no núcleo vizinho
  como ditongo (crescente: semivogal + vogal, ex. "quase"; decrescente:
  vogal + semivogal, ex. "pai").
- Duas vogais plenas seguidas nunca dividem o mesmo núcleo: viram
  hiato (ex. "mo-e-da", "vo-o"). `í`/`ú` acentuados contam como plenos
  propositalmente — é para isso que o acento existe (ex. "sa-ú-de",
  "pa-ís").
- `ão`, `ãe`, `ãi`, `õe` grafam um ditongo nasal: contam como um único
  núcleo pleno, inclusive formando tritongo com uma semivogal antes
  (ex. "sa-guão", onde o `u` de "gu" é sempre mudo/consonantal).
- Exceção: uma semivogal que abre a palavra e vem seguida de um
  ditongo nasal grafado forma hiato com ele, a menos que essa
  semivogal seja o `u` de um dígrafo `gu`/`qu` (que nunca é silábico).
  É o padrão de "a-vi-ão" vs. "sa-guão".

Limitação conhecida: a distinção ditongo/hiato depende, em alguns
casos, da tonicidade específica da palavra e não só da grafia (ex. a
família "-uação": "gra-du-a-ção" tem hiato onde a regra geral aqui
preveria ditongo). Esses casos não são tratados; revisar se o harness
de avaliação (T3.1) apontar erro sistemático.
"""

from __future__ import annotations

import re

_GLIDES = {"i", "u", "y"}
_FULL_VOWELS = {
    "a", "á", "à", "â", "ã",
    "e", "é", "ê",
    "o", "ó", "ô", "õ",
    "í", "ú",
}
_NASAL_DIPHTHONGS = {"ão", "ãe", "ãi", "õe"}

_VOWEL_RUN = re.compile(f"[{''.join(_GLIDES | _FULL_VOWELS)}]+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def count_syllables(text: str) -> int:
    """Conta o total de sílabas das palavras em `text`.

    Ignora números, pontuação e espaços — quem chama já deve ter
    normalizado o texto (ver `packages/ptbr/normalize.py`).
    """
    return sum(_count_syllables_in_word(word.lower()) for word in _WORD.findall(text))


def _count_syllables_in_word(word: str) -> int:
    total = 0
    for match in _VOWEL_RUN.finditer(word):
        run = match.group()
        onset_after_gu_qu = (
            run[0] == "u" and match.start() >= 1 and word[match.start() - 1] in ("g", "q")
        )
        total += _count_nuclei_in_run(run, onset_after_gu_qu)
    return total


def _tokenize_vowel_run(run: str) -> list[tuple[bool, bool]]:
    """Quebra um encontro vocálico em tokens `(é_pleno, é_ditongo_nasal)`."""
    tokens: list[tuple[bool, bool]] = []
    i = 0
    n = len(run)
    while i < n:
        pair = run[i : i + 2]
        if pair in _NASAL_DIPHTHONGS:
            tokens.append((True, True))
            i += 2
            continue
        ch = run[i]
        tokens.append((ch in _FULL_VOWELS, False))
        i += 1
    return tokens


def _count_nuclei_in_run(run: str, onset_after_gu_qu: bool) -> int:
    tokens = _tokenize_vowel_run(run)

    nuclei = 0
    open_nucleus = False
    has_full = False
    has_trailing_glide = False

    for idx, (is_full, is_nasal) in enumerate(tokens):
        if not open_nucleus:
            open_nucleus = True
            has_full = is_full
            has_trailing_glide = False
            continue

        if is_full:
            if has_full:
                # duas vogais plenas seguidas: hiato, fecha a anterior
                nuclei += 1
                has_full = True
                has_trailing_glide = False
                continue
            if is_nasal and idx == 1 and not onset_after_gu_qu:
                # semivogal isolada antes de ditongo nasal: hiato (ex. avião)
                nuclei += 1
                has_full = True
                has_trailing_glide = False
                continue
            has_full = True
            continue

        # token é semivogal
        if not has_full:
            # duas semivogais seguidas (ex. "ui" em "gratuito"): 1 núcleo
            nuclei += 1
            open_nucleus = False
            continue
        if not has_trailing_glide:
            has_trailing_glide = True
            nuclei += 1
            open_nucleus = False
            continue

    if open_nucleus:
        nuclei += 1
    return nuclei
