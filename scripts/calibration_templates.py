"""Gera frases pt-BR por template, cobrindo uma faixa de sílabas controlada
E uma faixa de registros de fala (não só telejornal — o produto dubla
qualquer vídeo do YouTube: vlog, tutorial, gameplay, review, comédia...).

Uso: `python scripts/calibration_templates.py caminho/de/saida.json`

Parte do T0.6 (calibração de duração do TTS). Só gera texto — não sintetiza
nada, não precisa de GPU. A metade "natural" do conjunto de 300 frases vem
de um LLM à parte (rodado no pod, junto do MOSS-TTS) — ver
`calibration_generate_qwen.py`, que também cobre os mesmos registros.

Cada frase usa cláusulas de um único registro (não mistura vlog com
notícia na mesma frase) — cada uma vira sua própria sentença, separada por
". ", o que é sempre gramatical e representativo de como um segmento real
do pipeline (T0.5) se parece: um bloco de fala curto, não uma frase longa
com múltiplas orações encadeadas.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from packages.ptbr.syllables import count_syllables

REGISTERS: dict[str, list[str]] = {
    "noticia": [
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
    ],
    "vlog": [
        "Gente, eu não acredito",
        "Cara, isso foi muito louco",
        "Eu tava pensando aqui",
        "Sério, vocês não têm ideia",
        "Isso me deixou super feliz",
        "Não vou mentir pra vocês",
        "Foi surreal, gente",
        "Eu quase não acreditei",
        "Vocês sabem que eu amo isso",
        "Já fazia tempo que eu queria fazer esse vídeo",
        "Hoje o dia começou bem corrido",
        "Eu decidi tentar algo diferente",
        "Isso mudou completamente minha rotina",
        "Vou contar rapidinho o que aconteceu",
        "Prometo que vale a pena assistir até o final",
        "Eu realmente não esperava por essa reação",
        "Isso aqui virou minha parte favorita do dia",
        "Deixa eu explicar direitinho o que rolou",
    ],
    "tutorial": [
        "Primeiro, você precisa abrir o programa",
        "Depois disso, clique em configurações",
        "Repare que esse botão muda tudo",
        "Se você seguir esses passos",
        "Isso vai facilitar bastante o processo",
        "Não esqueça de salvar antes de sair",
        "Aqui embaixo tem mais detalhes",
        "Essa parte é a mais importante",
        "Agora vamos para o próximo passo",
        "Preste atenção nesse detalhe",
        "Você vai perceber a diferença na hora",
        "Esse ajuste resolve o problema rapidinho",
        "Confira se está tudo configurado direito",
        "No final, é só confirmar a alteração",
        "Isso economiza bastante tempo depois",
    ],
    "gameplay": [
        "Nossa, que jogada incrível",
        "Eu não vi isso chegando",
        "Olha o tamanho desse inimigo",
        "Isso quase me pegou de surpresa",
        "Vamos tentar de novo",
        "Que sorte a minha, hein",
        "Consegui, gente, eu consegui",
        "Isso foi por pouco",
        "Cuidado que vem outro ali",
        "Eu não tinha visto esse item",
        "Isso vai ser mais difícil do que pensei",
        "Preciso trocar de estratégia agora",
        "Essa fase está brutal",
        "Vamos ver até onde dá pra chegar",
        "Isso foi loucura demais",
    ],
    "review": [
        "Esse produto realmente surpreendeu",
        "Eu esperava mais dessa marca",
        "Vale muito a pena o investimento",
        "Não recomendo de jeito nenhum",
        "A qualidade deixou a desejar",
        "O acabamento é melhor do que eu imaginava",
        "O preço não condiz com a qualidade",
        "Testei por semanas antes de opinar",
        "Isso resolveu um problema antigo meu",
        "Prefiro o modelo anterior, sinceramente",
        "A experiência de uso é bem melhor agora",
        "Ainda tem alguns detalhes pra ajustar",
        "Comparado com o concorrente, vale mais a pena",
        "Fiquei impressionado com o resultado final",
    ],
    "comedia": [
        "Isso é a coisa mais engraçada que eu já vi",
        "Juro que não é mentira",
        "Vocês vão rir muito com isso",
        "Vem comigo que a história é boa",
        "Isso aconteceu de verdade comigo",
        "Ninguém ia acreditar se eu contasse",
        "Foi um dos momentos mais vergonhosos da minha vida",
        "Eu ainda rio toda vez que lembro disso",
        "Essa situação saiu completamente do controle",
        "Vocês precisavam ver a cara dele",
        "Foi cômico e trágico ao mesmo tempo",
        "Até hoje ninguém sabe explicar o que rolou",
    ],
}

TARGET_SYLLABLE_COUNTS = [3, 5, 8, 12, 16, 20, 25, 30, 36, 42, 48, 55, 62, 68]
PHRASES_PER_TARGET = 15
PAUSE_INJECTION_RATE = 0.3


def build_phrase(target: int, clauses: list[str], rng: random.Random) -> str:
    pool = clauses.copy()
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
        chosen = [rng.choice(clauses)]

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
    registers = list(REGISTERS.items())
    phrases = []
    for target in TARGET_SYLLABLE_COUNTS:
        for i in range(PHRASES_PER_TARGET):
            register_name, clauses = registers[i % len(registers)]
            text = build_phrase(target, clauses, rng)
            text, num_pauses, pause_seconds = maybe_inject_pause(text, rng)
            phrases.append(
                {
                    "text": text,
                    "source": "template",
                    "register": register_name,
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
