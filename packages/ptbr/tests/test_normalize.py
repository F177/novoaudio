import pytest

from packages.ptbr.normalize import cardinal_to_words, normalize, ordinal_to_words

CARDINAL_CASES = {
    0: "zero",
    1: "um",
    2: "dois",
    5: "cinco",
    7: "sete",
    9: "nove",
    10: "dez",
    11: "onze",
    14: "catorze",
    15: "quinze",
    16: "dezesseis",
    19: "dezenove",
    20: "vinte",
    21: "vinte e um",
    34: "trinta e quatro",
    50: "cinquenta",
    67: "sessenta e sete",
    99: "noventa e nove",
    100: "cem",
    101: "cento e um",
    120: "cento e vinte",
    150: "cento e cinquenta",
    199: "cento e noventa e nove",
    200: "duzentos",
    234: "duzentos e trinta e quatro",
    300: "trezentos",
    500: "quinhentos",
    999: "novecentos e noventa e nove",
    1000: "mil",
    1001: "mil e um",
    1100: "mil e cem",
    1200: "mil e duzentos",
    1234: "mil duzentos e trinta e quatro",
    2026: "dois mil e vinte e seis",
    5000: "cinco mil",
    10000: "dez mil",
    100000: "cem mil",
    101000: "cento e um mil",
    999999: "novecentos e noventa e nove mil novecentos e noventa e nove",
    1000000: "um milhão",
    2000000: "dois milhões",
    1000001: "um milhão e um",
    1500000: "um milhão e quinhentos mil",
    1000000000: "um bilhão",
    2000000000: "dois bilhões",
    1500000000: "um bilhão e quinhentos milhões",
}


@pytest.mark.parametrize("value,expected", list(CARDINAL_CASES.items()))
def test_cardinal_to_words(value: int, expected: str) -> None:
    assert cardinal_to_words(value) == expected


ORDINAL_CASES = {
    (1, False): "primeiro",
    (1, True): "primeira",
    (2, False): "segundo",
    (3, False): "terceiro",
    (4, False): "quarto",
    (5, False): "quinto",
    (9, False): "nono",
    (10, False): "décimo",
    (11, False): "décimo primeiro",
    (20, False): "vigésimo",
    (21, False): "vigésimo primeiro",
    (21, True): "vigésima primeira",
    (30, False): "trigésimo",
    (40, False): "quadragésimo",
    (50, False): "quinquagésimo",
    (60, False): "sexagésimo",
    (70, False): "septuagésimo",
    (80, False): "octogésimo",
    (90, False): "nonagésimo",
    (100, False): "centésimo",
    (101, False): "centésimo primeiro",
    (121, False): "centésimo vigésimo primeiro",
    (200, False): "ducentésimo",
    (1000, False): "milésimo",
}


@pytest.mark.parametrize("args,expected", list(ORDINAL_CASES.items()))
def test_ordinal_to_words(args: tuple[int, bool], expected: str) -> None:
    n, feminine = args
    assert ordinal_to_words(n, feminine=feminine) == expected


NORMALIZE_CASES = {
    # Exigidos pelo critério de aceite da T0.3
    "R$ 1.234,56": "mil duzentos e trinta e quatro reais e cinquenta e seis centavos",
    "10/09/2026": "dez de setembro de dois mil e vinte e seis",
    # Moeda
    "R$ 10": "dez reais",
    "R$ 1": "um real",
    "R$ 0,50": "cinquenta centavos",
    "R$ 0,01": "um centavo",
    "R$ 1.500": "mil e quinhentos reais",
    "US$ 100": "cem dólares",
    "US$ 1": "um dólar",
    "US$ 2.500,75": "dois mil e quinhentos dólares e setenta e cinco centavos",
    # Datas
    "01/01/2000": "primeiro de janeiro de dois mil",
    "25/12/1999": "vinte e cinco de dezembro de mil novecentos e noventa e nove",
    "03/07/2024": "três de julho de dois mil e vinte e quatro",
    # Horas
    "14:30": "catorze horas e trinta minutos",
    "09:00": "nove horas",
    "01:00": "uma hora",
    "01:01": "uma hora e um minuto",
    "12h": "doze horas",
    "14h30": "catorze horas e trinta minutos",
    # Porcentagem
    "50%": "cinquenta por cento",
    "100%": "cem por cento",
    "12,5%": "doze vírgula cinco por cento",
    "3,05%": "três vírgula zero cinco por cento",
    "0,5%": "zero vírgula cinco por cento",
    "7%": "sete por cento",
    # Unidades
    "10km": "dez quilômetros",
    "1km": "um quilômetro",
    "5kg": "cinco quilos",
    "1kg": "um quilo",
    "3m": "três metros",
    "1m": "um metro",
    "2cm": "dois centímetros",
    "500ml": "quinhentos mililitros",
    "1l": "um litro",
    "80km/h": "oitenta quilômetros por hora",
    "30°C": "trinta graus Celsius",
    "1°C": "um grau Celsius",
    "2,5km": "dois vírgula cinco quilômetros",
    # Siglas
    "OTAN": "otan",
    "IBGE": "i-bê-gê-é",
    "CPF": "cê-pê-éfe",
    "ONU": "onu",
    # Abreviações
    "Dr. Silva": "doutor Silva",
    "Sr.ª Maria": "senhora Maria",
    "av. Paulista": "avenida Paulista",
    "Prof. João": "professor João",
    # Ordinais no texto
    "1º lugar": "primeiro lugar",
    "2ª posição": "segunda posição",
    "10º andar": "décimo andar",
    # Números soltos em frase
    "Ele tem 25 anos": "Ele tem vinte e cinco anos",
    "Comprei 3 livros": "Comprei três livros",
    "O total é 1000000": "O total é um milhão",
    "São 2024 habitantes": "São dois mil e vinte e quatro habitantes",
}


@pytest.mark.parametrize("text,expected", list(NORMALIZE_CASES.items()))
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


def test_total_case_count_meets_acceptance_bar() -> None:
    total = len(CARDINAL_CASES) + len(ORDINAL_CASES) + len(NORMALIZE_CASES)
    assert total >= 80
