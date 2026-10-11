# Perfil de avaliação: português brasileiro (pt-BR)

Critérios específicos do idioma-alvo para o `dub-judge` e o `dub-analyst`. A lógica dos agentes é a mesma para qualquer idioma; o que muda está aqui e em `pt-BR.json`. Para um idioma novo, copie este arquivo e o JSON e reescreva o conteúdo; não mude os agentes.

## Alvo

Português brasileiro **falado**, como um criador brasileiro falaria no vídeo. Não é legenda, não é texto escrito.

## Exemplos por classe da taxonomia

Todos os exemplos abaixo são inventados. Nunca copie para cá frases do conjunto de avaliação: o perfil é lido por todos os agentes, e exemplo real do conjunto acopla o juiz ao dev e vaza texto para quem implementa.

- `sentido.pessoa_sujeito`: o pt-BR omite sujeito e usa "seu" ambíguo, e a tradução frase a frase não sabe com quem se fala. "Does your mom cook for you?" → "Sua mãe te cozinha?" (inverte o sentido); "She bought it" → "Cê falou que comprou" (vira "você comprou"); "His idea", dito a outra pessoa → "Sua ideia" (vira a ideia de quem ouve).
- `sentido.literal`: "Give your sister a call" → "Dê uma chamada à sua irmã" (o certo é "Liga pra sua irmã"); "That's the catch" → "Essa é a pegada"; "room" → "quarto" quando é "espaço".
- `registro.formal_demais`: "lhe", "conosco", ênclise ("dá-me"), "Ouça", imperativo formal ("Telefone") na fala casual. Mistura de registro também é erro: "tô lhe falando".
- `registro.variante`: formas de pt-PT ("estás", "a fazer", "rapariga", "casa de banho"). No áudio, sotaque lusitano do TTS conta como `pronuncia.termo` ou `registro.variante`, conforme a evidência.
- `tom.suavizado`: palavrão sai como no original (decisão de 2026-10-10). "This is bullshit" não vira "isso é besteira". Vale também para as suas sugestões.
- `concordancia.genero`: "Tô cansada" para locutor homem; adjetivo concordando com o gênero errado.
- `pronuncia.termo`: siglas, marcas, números e valores ("NBA", "iPhone", "R$ 3.500"), nomes em inglês.

## Fala natural (para sugestões)

Use: "cê/você", "tá", "pra", "a gente", "né", próclise ("me dá"), imperativo do dia a dia ("liga", "olha", "fala").
Evite: "lhe", "conosco", ênclise, "Ouça", "Aproximadamente" onde se diria "quase" ou "mais ou menos".
Mantenha o número de sílabas parecido com o da tradução atual (o orçamento de duração é por sílaba).

## Entonação

Pergunta sim/não em pt-BR só se distingue da afirmação pela entonação: "Sua irmã vem com a gente?" e "Sua irmã vem com a gente." têm as mesmas palavras. O `judge_audio` checa o pico final de F0 nessas perguntas. Pergunta com palavra interrogativa (lista em `pt-BR.json`) costuma descer no fim e não é checada.

## Particularidades conhecidas do pipeline

- O MOSS-TTS tende ao sotaque de Portugal (CLAUDE.md, `docs/ptbr.md`).
- A instrução de estilo do MOSS está desligada e a voz de referência puxa a entrega para o neutro: achatamento é esperado. Aponte onde ele estraga a fala, não todo neutro.
- O banco de vozes atual só tem vozes masculinas.
