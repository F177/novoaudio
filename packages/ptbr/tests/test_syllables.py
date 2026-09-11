from packages.ptbr.syllables import count_syllables

# Palavras exigidas pelo critério de aceite da T0.2 — têm que bater exato.
REQUIRED_CASES = {
    "saúde": 3,
    "pai": 1,
    "saguão": 2,
    "pneu": 1,
    "transporte": 3,
    "carro": 2,
    "aninha": 3,
}

# Conjunto mais amplo cobrindo ditongos (crescentes e decrescentes),
# hiatos, dígrafos, encontros consonantais e ditongos/tritongos nasais.
OTHER_CASES = {
    "casa": 2,
    "bola": 2,
    "gato": 2,
    "mesa": 2,
    "porta": 2,
    "livro": 2,
    "tempo": 2,
    "cidade": 3,
    "felicidade": 5,
    "universidade": 6,
    "computador": 4,
    "telefone": 4,
    "cachorro": 3,
    "banheiro": 3,
    "trabalho": 3,
    "escola": 3,
    "problema": 3,
    "abraço": 3,
    "flor": 1,
    "claro": 2,
    "grande": 2,
    "prato": 2,
    "quatro": 2,
    "mais": 1,
    "rei": 1,
    "touro": 2,
    "europeu": 3,
    "saudade": 3,
    "hoje": 2,
    "série": 2,
    "água": 2,
    "igual": 2,
    "quase": 2,
    "história": 3,
    "gratuito": 3,
    "saída": 3,
    "país": 2,
    "baú": 2,
    "egoísta": 4,
    "moeda": 3,
    "poema": 3,
    "voo": 2,
    "coordenar": 4,
    "ideia": 3,
    "não": 1,
    "coração": 3,
    "mãe": 1,
    "põe": 1,
    "aviões": 3,
    "avião": 3,
    "milho": 2,
    "filho": 2,
    "calor": 2,
    "animal": 3,
    "importante": 4,
    "exemplo": 3,
    "também": 2,
    "depois": 2,
    "muito": 2,
    "cuidado": 3,
    "saiu": 2,
    "amor": 2,
    "feliz": 2,
    "difícil": 3,
    "rápido": 3,
    "lápis": 2,
    "útil": 2,
}

ALL_CASES = {**REQUIRED_CASES, **OTHER_CASES}


def test_required_examples_match_exactly() -> None:
    for word, expected in REQUIRED_CASES.items():
        assert count_syllables(word) == expected, word


def test_accuracy_at_least_95_percent() -> None:
    failures = [
        (word, expected, count_syllables(word))
        for word, expected in ALL_CASES.items()
        if count_syllables(word) != expected
    ]
    accuracy = 1 - len(failures) / len(ALL_CASES)
    assert accuracy >= 0.95, f"acurácia {accuracy:.1%}, falhas: {failures}"


def test_ignores_punctuation_and_counts_full_phrase() -> None:
    assert count_syllables("Bom dia, tudo bem?") == count_syllables("Bom dia tudo bem")


def test_empty_and_non_alphabetic_input() -> None:
    assert count_syllables("") == 0
    assert count_syllables("123 !!! ...") == 0
