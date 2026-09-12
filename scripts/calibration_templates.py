"""Gera frases pt-BR por template, cobrindo uma faixa de sílabas controlada.

Uso: `python scripts/calibration_templates.py caminho/de/saida.json`

Parte do T0.6 (calibração de duração do TTS). Só gera texto — não sintetiza
nada, não precisa de GPU. A metade "natural" do conjunto de 300 frases vem
de um LLM à parte (rodado no pod, junto do MOSS-TTS).

As frases são montadas concatenando cláusulas curtas e completas (cada uma
vira sua própria sentença, separada por ". "), em vez de tentar costurar
conectores no meio — mais simples e sempre gramatical, e ainda assim
representativo: um segmento real do pipeline (T0.5) já é um bloco de fala
curto, não uma frase longa com múltiplas orações encadeadas.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.ptbr.syllables import count_syllables

CLAUSES = [
    "Isso é sério",
    "Ela chegou",
    "O carro parou",
    "Ninguém sabia",
    "Choveu muito",
    "Ele confirmou",
    "A porta abriu",
    "Tudo mudou",
    "Ninguém se importou",
    "O som sumiu",
    "O prefeito anunciou a obra",
    "A polícia investiga o caso",
    "Os moradores reclamaram do barulho",
    "A empresa demitiu funcionários",
    "O trânsito ficou intenso hoje",
    "O hospital recebeu mais pacientes",
    "A escola cancelou as aulas",
    "O banco fechou mais cedo",
    "O time perdeu o campeonato",
    "A vizinha chamou os bombeiros",
    "De acordo com as autoridades locais, o incêndio começou de madrugada",
    "Segundo testemunhas, o suspeito fugiu correndo pela rua principal",
    "Um grupo de moradores organizou um protesto em frente à prefeitura",
    "A vítima foi socorrida e levada com urgência ao hospital mais próximo",
    "Os investigadores ainda não descartam nenhuma linha de investigação",
    "A prefeitura prometeu responder oficialmente até o fim da semana",
    "Especialistas alertam que o problema pode se agravar nos próximos meses",
    "A reportagem apurou que o contrato foi assinado sem licitação pública",
    "Familiares das vítimas cobram explicações da direção do hospital",
    "O governo anunciou um pacote de medidas para conter a crise econômica",
    "Isso gerou grande repercussão nas redes sociais",
    "A diretoria já tinha recebido diversas advertências antes",
    "Os bombeiros trabalharam a noite inteira para conter as chamas",
    "A assessoria de imprensa não quis comentar o caso",
    "O boletim foi divulgado logo pela manhã",
    "A equipe de resgate enfrentou dificuldades no local",
    "Nossa reportagem apurou detalhes ainda não divulgados",
    "A cidade vive um clima de tensão generalizada",
    "A negociação levou meses até ser concluída",
    "Fontes ligadas à investigação pediram sigilo",
]


TARGET_SYLLABLE_COUNTS = [3, 5, 8, 12, 16, 20, 25, 30, 36, 42, 48, 55, 62, 68]
PHRASES_PER_TARGET = 15
PAUSE_INJECTION_RATE = 0.3


def build_phrase(target: int, rng: random.Random) -> str:
    pool = CLAUSES.copy()
    rng.shuffle(pool)
    chosen: list[str] = []
    remaining = target
    while remaining > 2 and pool:
        pool.sort(key=lambda frag: abs(count_syllables(frag) - remaining))
        # entre os mais próximos, sorteia pra não repetir sempre a mesma combinação
        frag = rng.choice(pool[: min(4, len(pool))])
        pool.remove(frag)
        chosen.append(frag)
        remaining -= count_syllables(frag)
        if remaining <= -4:
            break

    if not chosen:
        chosen = [rng.choice(CLAUSES)]

    return ". ".join(chosen) + "."


def maybe_inject_pause(text: str, rng: random.Random) -> tuple[str, int, float]:
    if rng.random() > PAUSE_INJECTION_RATE:
        return text, 0, 0.0
    parts = text[:-1].split(". ")
    if len(parts) < 2:
        return text, 0, 0.0
    duration = round(rng.uniform(0.2, 1.2), 1)
    idx = rng.randrange(len(parts) - 1)
    parts[idx] = parts[idx] + f". [pause {duration}s]"
    return ". ".join(parts) + ".", 1, duration


def main() -> None:
    rng = random.Random(42)
    phrases = []
    for target in TARGET_SYLLABLE_COUNTS:
        for _ in range(PHRASES_PER_TARGET):
            text = build_phrase(target, rng)
            text, num_pauses, pause_seconds = maybe_inject_pause(text, rng)
            phrases.append(
                {
                    "text": text,
                    "source": "template",
                    "target_syllables": target,
                    "measured_syllables": count_syllables(text.split("[pause")[0]),
                    "num_pauses": num_pauses,
                    "requested_pause_seconds": pause_seconds,
                }
            )

    out_path = sys.argv[1]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
