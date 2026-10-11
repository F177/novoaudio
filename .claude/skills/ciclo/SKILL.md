---
name: ciclo
description: 'Ciclo de qualidade da dublagem, com aprovação humana entre as fases. Uso: /ciclo <fase> [args]. Fases: avaliar [manifesto], calibrar <ciclo>, planejar <ciclo>, implementar <ciclo> <tarefa>, revisar <ciclo> <tarefa>, verificar <ciclo> <tarefa>, marco [manifesto], status.'
disable-model-invocation: true
---

Você orquestra o ciclo de qualidade. Argumentos: `$ARGUMENTS` (a primeira palavra é a fase).

## Princípios

- **Produto, não vídeo.** As defesas são estruturais e você não contorna nenhuma: splits dev/holdout/exame, elegibilidade por evidência espalhada, sondas antes de propor, tarefa sem dados do conjunto, revisor adversarial, guarda de diff (`check_generalization`), A/B cego, aceite que exige o holdout, renovação do holdout e o registro do que foi tentado (`evals/ledger.md`).
- **O juiz é proxy e precisa de calibração.** Seus rótulos (`evals/human_labels/`) valem mais que os do juiz e o substituem onde existem. O portão da Fase 0 é só rótulo humano. Classe em que o juiz não está calibrado só é aceita com seu rótulo nos segmentos que mudaram.
- **O humano decide em quatro pontos:** aprovar o plano, gastar GPU, aceitar a mudança e usar o exame selado. Use AskUserQuestion; subagentes não conseguem perguntar, por isso essas paradas são suas.
- **Idioma é dado:** tudo que é específico vem do manifesto (`evals/golden/<par>.json`) e do perfil (`evals/languages/<idioma>.*`). Passe o idioma-alvo do manifesto a cada subagente.
- **Custo:** cada Short julgado custa tokens; cada Short dublado custa ~6 min de GPU até a T0.32. Diga o custo antes de gastar.
- **Servidor de escuta:** `python -m scripts.listen ...` roda em primeiro plano. Rode-o em background e passe a URL ao humano; encerre quando ele disser que terminou. Agentes não podem rodá-lo.
- Se um subagente reportar bloqueio do guarda, não contorne: leia o motivo e ajuste a delegação ou pergunte ao humano.

## `avaliar [manifesto]` (padrão `evals/golden/en-pt-BR.json`)

1. Checagens de conjunto. Avise e pergunte antes de seguir se:
   - há Shorts sem `categoria`, menos de 3 categorias, ou holdout vazio (a generalização não pode ser verificada);
   - `python -m scripts.eval_cycle usage --manifest <m>` mostra o holdout acima de `higiene.max_usos_holdout`. Proponha renovar: o humano traz Shorts novos, de preferência amostrados do que clippers realmente sobem, que entram no holdout; os holdout mais usados passam para o dev. Nunca mova um Short de dev para holdout ou exame.
2. Execuções: confira `.cache/dub/<video>/report.json` de cada Short dev e holdout, mais novo que o último commit em `packages/` ou `scripts/`. Para faltantes ou velhos, mostre o custo e pergunte: você roda `python -m scripts.dub <fonte>` (se o SSH falhar, pare e avise), o humano roda, ou cancela. Lembre de desligar o Pod.
3. `python -m scripts.eval_cycle new --manifest <m> --slug <tema>` e `python -m scripts.eval_cycle snapshot <ciclo> base`.
4. Se houver ciclo anterior aceito: `python -m scripts.eval_cycle carry <ciclo> <ciclo-anterior>:candidato base`.
5. Um `dub-judge` por Short (dev e holdout), modo normal, rótulo `base`, até 5 em paralelo.
6. Ruído e calibração: se `evals/calibration/<par>.json` não existe, é de outro modelo de juiz, ou não tem `ruido`, julgue de novo os Shorts de dev no rótulo `base-r2` (modo normal; o juiz lê a execução `base`, rotula tudo e não olha a primeira leitura) e proponha `/ciclo calibrar <ciclo>` antes de planejar.
7. `python -m scripts.eval_cycle summary <ciclo> base` e grave em `.cache/cycles/<ciclo>/base/resumo.md`. Relate: Shorts publicáveis (proxy), classes elegíveis do dev, estado da calibração. Próximo passo: `calibrar` ou `planejar`.

## `calibrar <ciclo>`

1. Suba `python -m scripts.listen <ciclo> base --todos` e peça ao humano para rotular Shorts inteiros (segmentos e veredito do Short). Isso também é a rotulagem da T0.28. Sugira começar pelos Shorts com mais problemas segundo o juiz e cobrir pelo menos 5 casos de cada classe frequente.
2. Quando ele terminar: `python -m scripts.eval_cycle calibrate <ciclo> base --repeat base-r2` (sem `--repeat` se o ruído já foi medido com este modelo).
3. Mostre a tabela (concordância, ruído, classes confiáveis). Commite `evals/human_labels/` e `evals/calibration/`: `evals: add human labels and judge calibration (<ciclo>)`.

## `planejar <ciclo>`

1. Delegue o `dub-analyst` (ciclo, idioma-alvo). Se ele terminar "aguardando sonda de GPU", mostre o comando e o critério de corte de cada sonda, peça aprovação do gasto, rode-a, grave a saída em `.cache/cycles/<ciclo>/sondas/` e retome o analista com o resultado.
2. Mostre as propostas de `.cache/cycles/<ciclo>/propostas.md` (a parte de evidência é para o humano). Pergunte quais aprovar.
3. Para cada aprovada: pergunte o id e a posição; escreva no `TASKS.md` **só** a parte "Tarefa para o TASKS.md"; rode `python -m scripts.check_generalization`. Se houver bloqueio, a tarefa vazou dado do conjunto: reescreva os exemplos (ou devolva ao analista). Acrescente "planejado" ao `evals/ledger.md` e commite: `docs: add <id> from quality cycle <ciclo>`.

## `implementar <ciclo> <tarefa>`

1. Exija árvore limpa. Base = branch atual. Crie `ciclo/<ciclo>-<tarefa>`.
2. Delegue o `dub-implementer` com o texto da tarefa do `TASKS.md`, o branch e a base.
3. Verifique por conta própria: `make lint && make test`, `python -m scripts.check_generalization --base <base>`, `git diff --stat <base>`. Siga para `revisar`.

## `revisar <ciclo> <tarefa>`

1. Delegue o `dub-reviewer` com a tarefa, o branch e a base.
2. Se **PEDIR MUDANÇAS**: retome o `dub-implementer` uma vez com os achados e os testes adversariais; depois rode o revisor de novo. Se continuar pedindo mudanças, ou se o implementador contestar um teste, pare e mostre os dois lados ao humano.
3. Com **APROVAR** e lint, testes e guarda verdes: pergunte se pode gastar GPU em `verificar`.

## `verificar <ciclo> <tarefa>`

1. No branch, com o gasto aprovado (N Shorts × ~6 min), rode `python -m scripts.dub <fonte>` para os Shorts de dev e holdout. Se o SSH falhar, pare e avise.
2. `python -m scripts.eval_cycle snapshot <ciclo> candidato`, depois `carry <ciclo> base candidato`, depois `blind <ciclo> base candidato`.
3. Um `dub-judge` em **modo cego** por Short listado em `.cache/cycles/<ciclo>/blind/` (ciclo, vídeo, idioma). Não diga ao juiz qual versão é qual.
4. `python -m scripts.eval_cycle unblind <ciclo> base candidato` e `python -m scripts.eval_cycle compare <ciclo> base candidato --alvo <classes da tarefa>`.
5. Escuta, sempre: suba `python -m scripts.listen <ciclo> candidato --ab base`. O humano ouve cego, rotula as versões e marca a preferência. Rode o `compare` de novo, porque os rótulos dele substituem os do juiz. Com veredito PRECISA DE RÓTULO HUMANO, a escuta é obrigatória antes de qualquer decisão.
6. Mostre o veredito final, as regressões e as preferências (juiz e humano) e pergunte: aceitar (merge), rejeitar ou iterar. ACEITAR do script não dispensa a decisão dele.
7. Escreva a entrada do ciclo no `evals/ledger.md` com os números do `compare` e a decisão. O analista lê o ledger: use só números e classes, nunca nome de Short ou trecho do holdout e do exame. Commite com `evals/human_labels/`: `docs: record quality cycle <ciclo>`. Lembre de desligar o Pod.

## `marco [manifesto]`

Só para decisões de marco (portão de fase, antes de mostrar o produto a alguém). O exame selado perde valor a cada uso.
1. Pergunte se é mesmo um marco. Mostre quantas vezes o exame já foi usado (`eval_cycle usage`).
2. Rode o pipeline atual nos Shorts de `exame` (gasto aprovado), crie um ciclo, faça `snapshot <ciclo> marco --marco` e suba `python -m scripts.listen <ciclo> marco --todos`. O humano dá o veredito de cada Short. O juiz não participa: a métrica do portão é só humana.
3. `python -m scripts.eval_cycle summary <ciclo> marco --marco` registra o uso e dá a taxa de publicação. Registre no ledger.

## `status`

Liste `.cache/cycles/*/` com a fase em que cada ciclo parou e o próximo comando, o estado da calibração (modelo, ruído, classes confiáveis) e o uso do holdout e do exame.
