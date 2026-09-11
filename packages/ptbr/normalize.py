"""Normalização de texto pt-BR para o front-end do TTS.

Expande números, moeda, datas, horas, porcentagem, siglas, abreviações
e unidades para a forma por extenso que o modelo de síntese deve
pronunciar. Escala curta (bilhão = 10^9), como usado no Brasil.

O texto de entrada não é todo minusculizado: só os trechos reconhecidos
(números, siglas, abreviações) são substituídos in-place, preservando
o resto da grafia original.

Limitações conhecidas:
- Moeda em `US$` assume formatação brasileira (`.` milhar, `,` decimal),
  não a formatação americana.
- Datas só cobrem `DD/MM/AAAA` com ano de 4 dígitos.
- Siglas desconhecidas (fora da tabela de exceções) são sempre
  soletradas; não há como saber, só pela grafia, quais siglas o
  público já lê como palavra.
- Números negativos (`cardinal_to_words` com n<0) não são detectados
  automaticamente no texto, porque "-" é ambíguo com hífen/marcador de
  lista em prosa comum.
"""

from __future__ import annotations

import re

_UNITS = ["zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove"]
_TEENS = [
    "dez", "onze", "doze", "treze", "catorze", "quinze",
    "dezesseis", "dezessete", "dezoito", "dezenove",
]
_TENS = [
    "", "", "vinte", "trinta", "quarenta", "cinquenta",
    "sessenta", "setenta", "oitenta", "noventa",
]
_HUNDREDS = [
    "", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos",
    "seiscentos", "setecentos", "oitocentos", "novecentos",
]
_SCALES = [
    (10**12, "trilhão", "trilhões"),
    (10**9, "bilhão", "bilhões"),
    (10**6, "milhão", "milhões"),
    (10**3, "mil", "mil"),
]

_ORDINAL_UNITS = [
    "", "primeiro", "segundo", "terceiro", "quarto", "quinto",
    "sexto", "sétimo", "oitavo", "nono",
]
_ORDINAL_TENS = [
    "", "décimo", "vigésimo", "trigésimo", "quadragésimo", "quinquagésimo",
    "sexagésimo", "septuagésimo", "octogésimo", "nonagésimo",
]
_ORDINAL_HUNDREDS = [
    "", "centésimo", "ducentésimo", "trecentésimo", "quadringentésimo", "quingentésimo",
    "sexcentésimo", "setingentésimo", "octingentésimo", "noningentésimo",
]

_MONTHS = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]

_CURRENCY_UNITS = {
    "R$": ("real", "reais"),
    "US$": ("dólar", "dólares"),
}

_UNIT_WORDS = {
    "km/h": ("quilômetro por hora", "quilômetros por hora"),
    "km": ("quilômetro", "quilômetros"),
    "kg": ("quilo", "quilos"),
    "cm": ("centímetro", "centímetros"),
    "mm": ("milímetro", "milímetros"),
    "ml": ("mililitro", "mililitros"),
    "m": ("metro", "metros"),
    "g": ("grama", "gramas"),
    "l": ("litro", "litros"),
}

_ACRONYM_AS_WORD = {
    "OTAN": "otan",
    "ONU": "onu",
    "UNESCO": "unesco",
    "UNICEF": "unicef",
    "SUS": "sus",
}
_LETTER_NAMES = {
    "A": "á", "B": "bê", "C": "cê", "D": "dê", "E": "é", "F": "éfe",
    "G": "gê", "H": "agá", "I": "i", "J": "jota", "K": "cá", "L": "éle",
    "M": "ême", "N": "ene", "O": "ó", "P": "pê", "Q": "quê", "R": "érre",
    "S": "ésse", "T": "tê", "U": "u", "V": "vê", "W": "dáblio", "X": "xis",
    "Y": "ípsilon", "Z": "zê",
}

_ABBREVIATIONS = {
    "Dr.": "doutor",
    "Dra.": "doutora",
    "Sr.": "senhor",
    "Sra.": "senhora",
    "Sr.ª": "senhora",
    "Srta.": "senhorita",
    "Prof.": "professor",
    "Profa.": "professora",
    "Av.": "avenida",
    "av.": "avenida",
    "R.": "rua",
    "n.º": "número",
    "nº": "número",
    "etc.": "etecétera",
}


def cardinal_to_words(n: int) -> str:
    """Converte um inteiro para a leitura por extenso em pt-BR (escala curta)."""
    if n < 0:
        return "menos " + cardinal_to_words(-n)
    if n == 0:
        return "zero"

    groups: list[tuple[int, str | None, str | None, int]] = []
    remaining = n
    for scale_value, sing, plur in _SCALES:
        if remaining >= scale_value:
            count, remaining = divmod(remaining, scale_value)
            groups.append((count, sing, plur, scale_value))
    if remaining > 0:
        groups.append((remaining, None, None, 1))

    parts: list[tuple[str, int]] = []
    for count, sing, plur, scale_value in groups:
        if scale_value == 1:
            word = _group_to_words(count)
        elif scale_value == 1000 and count == 1:
            word = "mil"
        else:
            word = f"{_group_to_words(count)} {sing if count == 1 else plur}"
        parts.append((word, count))

    if len(parts) == 1:
        return parts[0][0]

    head = " ".join(word for word, _ in parts[:-1])
    last_word, last_count = parts[-1]
    joiner = " e " if last_count < 100 or last_count % 100 == 0 else " "
    return head + joiner + last_word


def _group_to_words(n: int) -> str:
    """Escreve um valor de 0 a 999 por extenso."""
    if n == 0:
        return ""
    if n < 10:
        return _UNITS[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        tens, units = divmod(n, 10)
        word = _TENS[tens]
        if units:
            word += " e " + _UNITS[units]
        return word

    hundreds, rest = divmod(n, 100)
    word = "cem" if hundreds == 1 and rest == 0 else _HUNDREDS[hundreds]
    if rest:
        word += " e " + _group_to_words(rest)
    return word


def ordinal_to_words(n: int, feminine: bool = False) -> str:
    """Converte um inteiro (1 a 1000) para o ordinal por extenso em pt-BR."""
    if n < 1 or n > 1000:
        raise ValueError("ordinal_to_words só cobre de 1 a 1000")
    if n == 1000:
        return _feminize("milésimo", feminine)

    hundreds, rest = divmod(n, 100)
    tens, units = divmod(rest, 10)
    words = []
    if hundreds:
        words.append(_ORDINAL_HUNDREDS[hundreds])
    if tens:
        words.append(_ORDINAL_TENS[tens])
    if units:
        words.append(_ORDINAL_UNITS[units])
    return _feminize(" ".join(words), feminine)


def _feminize(text: str, feminine: bool) -> str:
    if not feminine:
        return text
    return " ".join(w[:-1] + "a" if w.endswith("o") else w for w in text.split(" "))


def _parse_ptbr_int(num_str: str) -> int:
    return int(num_str.replace(".", ""))


def _format_decimal(int_part: int, frac_str: str) -> str:
    if frac_str[0] == "0":
        frac_words = " ".join(_UNITS[int(d)] for d in frac_str)
    else:
        frac_words = cardinal_to_words(int(frac_str))
    return f"{cardinal_to_words(int_part)} vírgula {frac_words}"


_DATE_PATTERN = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_TIME_COLON_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_TIME_H_PATTERN = re.compile(r"\b([01]?\d|2[0-3])h([0-5]\d)?\b")
_CURRENCY_PATTERN = re.compile(r"(R\$|US\$)\s?(\d{1,3}(?:\.\d{3})*)(?:,(\d{2}))?")
_PERCENT_PATTERN = re.compile(r"\b(\d+)(?:,(\d+))?%")
_CELSIUS_PATTERN = re.compile(r"\b(\d+)\s?°C\b")
_UNIT_PATTERN = re.compile(
    r"(\d+(?:,\d+)?)\s?("
    + "|".join(re.escape(u) for u in sorted(_UNIT_WORDS, key=len, reverse=True))
    + r")\b"
)
_ORDINAL_PATTERN = re.compile(r"\b(\d+)([ºª])")
_ACRONYM_PATTERN = re.compile(r"\b[A-ZÀÂÃÉÊÍÓÔÕÚÇ]{2,}\b")
_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")"
)
_PLAIN_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d{3})*(?:,\d+)?")


def _sub_date(match: re.Match[str]) -> str:
    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return match.group()
    day_word = "primeiro" if day == 1 else cardinal_to_words(day)
    return f"{day_word} de {_MONTHS[month - 1]} de {cardinal_to_words(year)}"


def _sub_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    hour_word = "uma" if hour == 1 else cardinal_to_words(hour)
    hour_unit = "hora" if hour == 1 else "horas"
    if minute == 0:
        return f"{hour_word} {hour_unit}"
    minute_word = cardinal_to_words(minute)
    minute_unit = "minuto" if minute == 1 else "minutos"
    return f"{hour_word} {hour_unit} e {minute_word} {minute_unit}"


def _sub_currency(match: re.Match[str]) -> str:
    symbol, int_str, dec_str = match.group(1), match.group(2), match.group(3)
    integer = _parse_ptbr_int(int_str)
    decimal = int(dec_str) if dec_str else 0
    unit_sing, unit_plur = _CURRENCY_UNITS[symbol]

    parts = []
    if integer > 0 or decimal == 0:
        parts.append(f"{cardinal_to_words(integer)} {unit_sing if integer == 1 else unit_plur}")
    if decimal > 0:
        sub_word = "centavo" if decimal == 1 else "centavos"
        parts.append(f"{cardinal_to_words(decimal)} {sub_word}")
    return " e ".join(parts)


def _sub_percent(match: re.Match[str]) -> str:
    int_part, frac_str = int(match.group(1)), match.group(2)
    words = _format_decimal(int_part, frac_str) if frac_str else cardinal_to_words(int_part)
    return f"{words} por cento"


def _sub_celsius(match: re.Match[str]) -> str:
    value = int(match.group(1))
    word = "grau" if value == 1 else "graus"
    return f"{cardinal_to_words(value)} {word} Celsius"


def _sub_unit(match: re.Match[str]) -> str:
    num_str, unit = match.group(1), match.group(2)
    if "," in num_str:
        int_part, frac_part = num_str.split(",", 1)
        words = _format_decimal(int(int_part), frac_part)
        singular = False
    else:
        value = int(num_str)
        words = cardinal_to_words(value)
        singular = value == 1
    sing, plur = _UNIT_WORDS[unit]
    return f"{words} {sing if singular else plur}"


def _sub_ordinal(match: re.Match[str]) -> str:
    return ordinal_to_words(int(match.group(1)), feminine=match.group(2) == "ª")


def _sub_acronym(match: re.Match[str]) -> str:
    word = match.group()
    if word in _ACRONYM_AS_WORD:
        return _ACRONYM_AS_WORD[word]
    return "-".join(_LETTER_NAMES.get(ch, ch) for ch in word)


def _sub_plain_number(match: re.Match[str]) -> str:
    text = match.group()
    if "," in text:
        int_str, frac_str = text.split(",", 1)
        return _format_decimal(_parse_ptbr_int(int_str), frac_str)
    return cardinal_to_words(_parse_ptbr_int(text))


def normalize(text: str) -> str:
    """Expande números, moeda, datas, horas, porcentagem, siglas, abreviações e unidades."""
    text = _DATE_PATTERN.sub(_sub_date, text)
    text = _TIME_COLON_PATTERN.sub(_sub_time, text)
    text = _TIME_H_PATTERN.sub(_sub_time, text)
    text = _CURRENCY_PATTERN.sub(_sub_currency, text)
    text = _PERCENT_PATTERN.sub(_sub_percent, text)
    text = _CELSIUS_PATTERN.sub(_sub_celsius, text)
    text = _UNIT_PATTERN.sub(_sub_unit, text)
    text = _ORDINAL_PATTERN.sub(_sub_ordinal, text)
    text = _ACRONYM_PATTERN.sub(_sub_acronym, text)
    text = _ABBREV_PATTERN.sub(lambda m: _ABBREVIATIONS[m.group(1)], text)
    text = _PLAIN_NUMBER_PATTERN.sub(_sub_plain_number, text)
    return text
