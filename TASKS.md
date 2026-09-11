# TASKS.md

Backlog em ordem de execução. Cada tarefa é uma sessão de Claude Code e um PR.

**Legenda de execução:**
- 🤖 = Claude Code faz bem sozinho
- 🤝 = Claude Code escreve, você valida ouvindo/medindo
- 👤 = faça à mão; é julgamento ou trabalho de ML, não de código

**Regra:** não comece a próxima tarefa antes dos critérios de aceite da atual passarem. Se uma tarefa crescer além de ~400 linhas de diff, ela estava mal fatiada — pare e quebre.

---

## Fase 0 — Pipeline na linha de comando

Objetivo: provar que dá para dublar um vídeo real antes de construir produto. Nada de web, nada de banco.

### T0.1 — Scaffold do monorepo 🤖
Estrutura de pastas conforme `CLAUDE.md`, `pyproject.toml` com ruff e pytest, `Makefile` com os comandos, `.env.example`, CI mínimo no GitHub Actions rodando lint e test.
**Aceite:** `make lint && make test` passa num repo vazio.

### T0.2 — Contador de sílabas pt-BR 🤖
`packages/ptbr/syllables.py`. Separação silábica por regras: ditongos, hiatos, encontros consonantais, dígrafos. Função `count_syllables(text: str) -> int`.
**Aceite:** teste com ≥60 palavras cobrindo `sa-ú-de`, `pai`, `sa-guão`, `pneu`, `trans-por-te`, `car-ro`, `a-ni-nha`. Acurácia ≥95% no conjunto de teste.

### T0.3 — Normalização de texto pt-BR 🤖
`packages/ptbr/normalize.py`. Expande números cardinais e ordinais, moeda (R$, US$), datas, horas, porcentagem, siglas (com tabela de exceções para as que viram palavra: OTAN, IBGE soletrado), abreviações (Dr., Sr.ª, av.), unidades. Escala curta (bilhão = 10⁹).
**Aceite:** ≥80 casos de teste. `R$ 1.234,56` → `mil duzentos e trinta e quatro reais e cinquenta e seis centavos`. `10/09/2026` → `dez de setembro de dois mil e vinte e seis`.

### T0.4 — Wrappers de GPU no Modal 🤝
`modal/functions.py`. Quatro funções serverless: `separate_stems` (Demucs), `transcribe` (WhisperX com timestamps por palavra + diarização), `synthesize` (MOSS-TTS), `evaluate` (Whisper para CER). Cada uma recebe e devolve caminhos em R2. Container com modelo pré-baixado na imagem.
**Aceite:** cada função roda num arquivo de teste e devolve saída válida. Cold start medido e anotado no README.

### T0.5 — Segmentação prosódica 🤖
`packages/pipeline/segmentation.py`. Recebe a saída do WhisperX (palavras com timestamps + locutor), devolve segmentos. Classifica gaps: <120ms ignora, 120–350ms pausa fraca, >350ms âncora dura. Quebra também em troca de locutor e em duração >12s. Emite `[t_inicio, t_fim, texto, speaker, pausas_internas]`.
**Aceite:** testes com fixtures JSON sintéticas. Nenhum segmento >12s, nenhum <0,3s, soma das durações cobre o áudio sem sobreposição.

### T0.6 — Calibração de duração do TTS 👤
Gere ~300 frases pt-BR de comprimentos variados, sintetize, meça a duração real. Ajuste a regressão `duração = a + b·sílabas + c·pausas`. Salve os coeficientes em `packages/pipeline/calibration.json`.
**Por que à mão:** exige ouvir, inspecionar outliers e julgar. Claude Code pode escrever o script de coleta, mas a análise é sua.
**Aceite:** R² ≥0,93. Coeficientes versionados. Documente o erro em segmentos <5 sílabas.

### T0.7 — Orçamento de duração 🤖
`packages/pipeline/budget.py`. Usa a calibração para `syllables_that_fit(target_seconds, speed) -> int` e a inversa. Define tolerâncias (±8% on-screen, ±20% off-screen) e a ordem de alavancas: texto → velocidade → silêncio → time-stretch.
**Aceite:** testes de propriedade — o orçamento é monotônico na duração; inverter ida e volta fecha dentro de 1 sílaba.

### T0.8 — Tradução com restrição 🤝
`packages/pipeline/translation.py`. Prompt que recebe orçamento em sílabas e gera 5 candidatos de comprimentos diferentes em JSON, numa só chamada. Pontuação local: fidelidade × aderência ao orçamento × naturalidade. Codifica as alavancas pt-BR no prompt (pro-drop, voz ativa, contrações do registro falado).
**Aceite:** em 50 segmentos de teste, ≥80% têm ao menos um candidato dentro da tolerância. Nenhum candidato perde informação factual (checagem manual em 20).

### T0.9 — Síntese com duração e pausa 🤝
`packages/pipeline/synthesis.py`. Converte duração alvo em `tokens` (×12,5), injeta `[pause X.Ys]` nas posições das pausas internas, aplica IPA do glossário. Sintetiza, mede a duração real, ajusta uma vez se estiver fora da tolerância.
**Aceite:** ≥85% dos segmentos dentro da tolerância após no máximo uma iteração.
**Investigue e documente:** a pausa `[pause X.Ys]` conta dentro do orçamento de `tokens` ou soma por cima? Isso muda o cálculo.

### T0.10 — Montagem e mix 🤖
`packages/pipeline/assembly.py`. Posiciona cada segmento na timeline, mixa com o stem de fundo, normaliza loudness (EBU R128, alvo −14 LUFS), remuxa com o vídeo via ffmpeg.
**Aceite:** vídeo de saída tem a mesma duração do original, áudio sem clipping, fundo audível.

### T0.11 — Gates de qualidade 🤖
`packages/pipeline/quality.py`. Para cada segmento emite flags: `fora_duracao`, `cer_alto`, `traducao_infiel`, `clipping`, `silencio_anormal`, `locutor_suspeito`.
**Aceite:** cada flag tem teste com caso positivo e negativo. Flags serializam em JSON.

### T0.12 — CLI end-to-end 🤖
`scripts/dub.py`. Um comando: vídeo entra, vídeo dublado sai, com relatório JSON de segmentos e flags. Cacheia resultados intermediários em disco.
**Aceite:** roda em 10 vídeos reais de YouTube. Relatório com taxa de sucesso por etapa.

### 🚦 Portão de decisão
Rode nos 10 vídeos. **Se mais de 30% dos segmentos precisam de correção manual, pare e conserte o pipeline antes de tocar em produto.** Não construa web em cima de um pipeline que não funciona.

---

## Fase 1 — MVP web

### T1.1 — Schema e migrations 🤖
`org`, `user`, `channel`, `project`, `asset`, `speaker`, `voice_profile`, `glossary_entry`, `segment`, `segment_revision`, `job`. `org_id` em todas. Índices para as queries do editor (segmentos por projeto, ordenados por tempo).
**Aceite:** migrations sobem e descem limpo. Seed de desenvolvimento.

### T1.2 — Fila em Postgres 🤖
`apps/api/queue.py`. `SELECT ... FOR UPDATE SKIP LOCKED`, com retry, backoff, dead-letter e jobs idempotentes por `(project_id, stage, input_hash)`.
**Aceite:** teste de concorrência com 10 workers não processa o mesmo job duas vezes. Job interrompido é retomado.

### T1.3 — Orquestração do pipeline 🤖
Máquina de estados por projeto: `uploaded → separated → transcribed → segmented → translated → synthesized → assembled → ready`. Cada transição é um job. Falha numa etapa não perde o trabalho das anteriores.
**Aceite:** matar o worker no meio e reiniciar retoma do ponto certo.

### T1.4 — Auth e upload 🤖
Supabase Auth. Upload direto para R2 via URL pré-assinada (o vídeo não passa pela sua API). Limite de 10 minutos de duração e 2 GB. Declaração de direitos sobre as vozes, persistida com timestamp.
**Aceite:** upload de 1 GB funciona sem estourar memória da API. Registro de consentimento gravado.

### T1.5 — Endpoints REST 🤖
`POST /projects`, `GET /projects/:id`, `GET /projects/:id/segments`, `PATCH /segments/:id`, `POST /segments/:id/resynth`, `GET /projects/:id/export`. Paginação nos segmentos.
**Aceite:** testes de integração cobrindo cada rota, incluindo isolamento por `org_id`.

### T1.6 — Dashboard mínimo 🤖
Lista de projetos com status e progresso, upload, download do resultado. Sem editor ainda. Feio é aceitável.
**Aceite:** três pessoas de fora conseguem dublar um vídeo sozinhas, sem instrução.

### T1.7 — Perfil de voz por canal 🤝
Entidade `voice_profile`: áudio de referência do criador, embedding, reutilizado em todos os vídeos do canal. Extração automática do segmento mais limpo e longo do primeiro vídeo, com opção de substituir.
**Aceite:** dois vídeos do mesmo canal produzem SECS ≥0,75 entre si.

### T1.8 — Glossário por canal 🤖
Entidade `glossary_entry`: termo de origem, tradução fixa, pronúncia IPA opcional. Aplicado na tradução e na síntese. UI de CRUD simples.
**Aceite:** termo no glossário sempre sai com a tradução e a pronúncia definidas, em 10 vídeos de teste.

---

## Fase 2 — Editor

### T2.1 — Timeline de segmentos 🤖
Lista virtualizada (pode ter centenas). Cada linha: texto original, tradução, duração alvo vs. real, flags destacadas. Player sincronizado que destaca o segmento em reprodução.
**Aceite:** 500 segmentos rolam a 60fps. Clicar num segmento salta o player.

### T2.2 — Edição e ressíntese 🤖
Editar a tradução, disparar ressíntese só daquele segmento, substituir o áudio sem reprocessar o vídeo. Indicador de carregamento. Nova revisão em `segment_revision`.
**Aceite:** ressíntese completa em <5s (container Modal quente). Histórico preservado. Desfazer funciona.

### T2.3 — Candidatos de tradução 🤖
Mostrar os 5 candidatos já gerados na T0.8, com contagem de sílabas e aderência ao orçamento. Um clique troca.
**Aceite:** trocar candidato não chama o LLM de novo (já estão persistidos).

### T2.4 — Controles de duração 🤖
Slider de velocidade por segmento (±15%), arrastar borda de pausa, indicador visual de estouro de orçamento.
**Aceite:** ajuste reflete no áudio. Estouro fica visualmente óbvio.

### T2.5 — Mesclar, dividir, silenciar 🤖
Operações estruturais na timeline, preservando o alinhamento temporal.
**Aceite:** mesclar e dividir não criam buracos nem sobreposições.

### T2.6 — Troca de voz e correção de locutor 🤖
Reatribuir segmento a outro locutor; trocar a voz de um locutor em todo o projeto.
**Aceite:** mudança em massa atualiza todos os segmentos do locutor.

### T2.7 — Correção de pronúncia inline 🤖
Selecionar palavra, definir IPA, escolher entre "só aqui" e "salvar no glossário do canal".
**Aceite:** salvar no glossário afeta vídeos futuros do canal.

### T2.8 — Instrumentação de edições 🤖
Registrar toda edição: segmento, flags que tinha, o que mudou, antes e depois. Painel interno com "edições por minuto de vídeo" e distribuição por tipo.
**Aceite:** o painel responde: qual flag gera mais correção? Esse número é o seu roadmap.

### T2.9 — Watermark e export 🤖
Watermark de conteúdo sintético no áudio. Export com opções de qualidade. Registro de exportações.
**Aceite:** watermark detectável, inaudível em teste cego com 5 pessoas.

---

## Fase 3 — Qualidade pt-BR

### T3.1 — Harness de avaliação 🤝
`evals/`. 150 frases pt-BR com casos difíceis (números, siglas, nomes, code-switching), 3 seeds cada, CER via Whisper large-v3, SECS com WavLM e ECAPA, correlação de F0. Relatório comparativo entre versões.
**Aceite:** `make eval` produz relatório reproduzível. Piso de CER do ASR no ground-truth documentado.

### T3.2 — G2P pt-BR 🤝
`packages/ptbr/g2p.py`. TugaPhone com dialeto pt-BR, fallback por regras, dicionário de exceções. Detecta palavras de alto risco (estrangeirismos, nomes) e as marca para IPA explícito.
**Aceite:** melhora mensurável de CER no harness contra o baseline sem G2P.

### T3.3 — Fine-tune LoRA 👤
Fora do repo. CML-TTS pt + TTS-Portuguese, limpeza, LoRA usando a configuração da LoRA norueguesa do repo MOSS-TTS como template. Monitore CER held-out para detectar esquecimento.
**Por que à mão:** é trabalho de ML com julgamento perceptual, não de engenharia de software.
**Aceite:** ΔCER e ΔSECS positivos no harness, e CMOS não-negativo com 15–20 ouvintes brasileiros.

---

## Fase 4 — Negócio

### T4.1 — Planos e cobrança 🤖
Integração Asaas, assinatura com franquia de minutos, medidor de uso, bloqueio no limite com upgrade.
### T4.2 — Onboarding 🤖
Fluxo do cadastro ao primeiro vídeo dublado, com vídeo de exemplo pronto para quem quer testar antes de subir o seu.
### T4.3 — Landing page 🤖
Antes e depois em áudio, preço, prova social. O demo de áudio é o argumento de venda inteiro.

---

## Como conduzir as sessões

**Abertura de sessão:** aponte a tarefa por id, peça para reler os critérios de aceite, e peça um plano curto antes de codar.

**Uma tarefa por PR.** Diff acima de ~400 linhas é sinal de fatiamento ruim.

**Peça o teste antes da implementação** nas tarefas de `packages/pipeline` e `packages/ptbr`. São puras — dão testes limpos e é onde bugs custam caro.

**Nas tarefas 🤝, ouça o áudio.** Teste verde não significa que soa bem. Nenhuma métrica substitui você escutando 10 amostras.

**Não deixe refatorar fora do escopo.** Se aparecer "aproveitei para melhorar X", peça para reverter X.

**Atualize o `CLAUDE.md`** sempre que uma decisão nova for tomada — especialmente os achados da T0.6 e T0.9, que afetam todo o resto.
