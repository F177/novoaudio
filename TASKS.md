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

### T0.4 — Wrappers de GPU no RunPod 🤝
`runpod/` (um subdiretório por endpoint, com `Dockerfile` + `handler.py`). Cinco workers serverless: `separate_stems` (Demucs), `transcribe` (WhisperX com timestamps por palavra + diarização), `translate` (Qwen2.5-7B-Instruct), `synthesize` (MOSS-TTS), `evaluate` (Whisper para CER). Cada um recebe e devolve caminhos em R2. Imagem com modelo pré-baixado.
**Aceite:** cada worker roda num arquivo de teste e devolve saída válida. Cold start medido e anotado no README.
**Status:** os cinco handlers foram escritos em 2026-09-12, mas nunca validados em Serverless e hoje estão desatualizados em relação ao `scripts/pod_worker.py`, que é o que roda de fato (harness via SSH num Pod A40; ver CLAUDE.md). Volta na T1.2a, antes de produção.

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
**Status:** o CLI funciona (orquestrando o Pod via SSH), mas o aceite está pendente: rodou em 5 vídeos, mas com scorecard completo só num clipe de 8 segmentos. Fecha junto com a T0.28.

### T0.13 — CER confiável 🤝
`packages/pipeline/quality.py`. Hoje `character_error_rate` compara o texto cru com a saída do Whisper: pontuação, caixa e dígitos contra números por extenso inflam o CER antes de existir erro real. Normalize os dois lados antes de medir — minúsculas, sem pontuação, espaços colapsados — passando por `packages/ptbr/normalize.py`, para que "2026" e "dois mil e vinte e seis" comparem iguais. Mantenha a função pura e determinística. Peça os testes antes da implementação: é módulo puro.
**Aceite:** testes cobrindo cada classe de normalização. Piso de CER medido em 20 segmentos sabidamente bons e documentado no docstring. `CER_THRESHOLD` redefinido em relação a esse piso, com justificativa escrita.

### T0.14 — Flag de truncamento 🤖
`packages/pipeline/quality.py`. Truncamento é invisível para `fora_duracao` por construção: o MOSS-TTS corta o texto justamente para caber na duração pedida, então o segmento truncado bate o alvo. A detecção tem que ser lexical. Adicione `truncado(reference, hypothesis)` — razão de palavras e cobertura do trecho final da referência na hipótese — e inclua em `QualityFlags`. Em `scripts/dub.py`, troque o predicado `_is_bad_synthesis`: hoje é `has_repetition or CER >= 1.0`, o que só pega falha catastrófica; passe a usar `has_repetition or truncado or cer_alto(cer)`.
**Aceite:** teste com caso positivo e negativo para `truncado`, incluindo referência curta (<4 palavras) que não deve disparar. Teste de que o predicado de retry dispara em CER intermediário.

### T0.15 — Corrigir o termo de pausa no orçamento 🤝
`packages/pipeline/budget.py` + recalibração. O coeficiente `c = 5.6865` de `calibration.json` foi ajustado em geração livre (`max_new_tokens=4096`, ver `scripts/calibration_measure_durations.py`), mas é aplicado em regime de orçamento fechado, onde o achado da T0.9 diz que a pausa cabe dentro do orçamento. Efeito: `syllables_that_fit` com alvo de 4,0 s e `pause_seconds=1.0` devolve 0 sílabas. Refaça o ajuste em três variantes — termo atual, sem termo de pausa, e pausa como contagem em vez de segundos — e compare o R².
**Aceite:** R² documentado nas três variantes; variante escolhida com justificativa escrita em `known_issues`. Teste que trava o comportamento contra números esperados reais, não só a consistência ida e volta do teste atual (que passa com qualquer `c`, porque usa o mesmo coeficiente dos dois lados).

### T0.16 — Contabilizar cortes na montagem 🤖
`packages/pipeline/assembly.py`. `place_segments_on_timeline` e `mix_with_background` cortam em silêncio (`min(...)` nos dois). Devolva quantas amostras foram descartadas e falhe acima de um limiar: algumas amostras é arredondamento, meio segundo é bug.
**Aceite:** teste em que um segmento estourando o fim da timeline é reportado, não cortado silenciosamente.

### T0.17 — Sistema de avaliação de qualidade 🤝 (feito; scorecard completo só num clipe)
Registro do que foi construído fora do backlog original, para este arquivo continuar fiel ao repo:
- `quality.py`: flags novas `entonacao_desalinhada`, `emocao_incompativel` e `cer_final_alto`. `traducao_infiel` e `locutor_suspeito` continuam stubs.
- `prosody.py`: transferência de envelope de energia (ligada, `--prosody-transfer`, neutra em CER) e de contorno de pitch (desligada, `--pitch-transfer`, prejudicial).
- `scripts/pod_worker.py::cmd_match_emotion`: emotion2vec_plus_large, opt-in (`--emotion-match`).
- `scorecard.py`, `diagnostics.py`, `report.py`: agregação por vídeo, base de diagnóstico por flag, dashboard local em `reports/`.
- `subtitles.py`: legenda pt-BR embutida como faixa selecionável (`--embed-subtitles`).

**Estado:** 346 testes passando. Scorecard completo só no trailer de 8 segmentos; outros 4 vídeos (beast, clairo, lego_batman, nfl — 197 segmentos) foram dublados, com transcrição e tradução salvas em `phase1test/`, mas sem scorecard. Limiares não calibrados. `diagnostics.py` congelado até a T0.28.

### T0.18 — Versionar e rastrear execuções 🤖
O trabalho pendente foi commitado (`783a872`), mas com mensagem "a" e tudo num commit só. Falta:
- `make lint` falha com o ruff atual: B905 em `scripts/pod_worker.py:236` (`zip` sem `strict=`). O CI está vermelho.
- Daqui para a frente, um commit por tarefa, mensagem no padrão do CLAUDE.md.
- Toda entrada nova em `reports/history.json` grava `git_sha` e `git_dirty`. O SHA é lido em `scripts/dub.py` e passado como dado; `packages/pipeline` continua puro.

**Aceite:** CI verde. Teste de que a entrada do histórico inclui os dois campos. Execução com working tree sujo aparece marcada no dashboard.

### T0.19 — Instrumento de avaliação confiável 🤝
Três problemas no instrumento que mede todo o resto:

1. **O "CER final" não é pós-mix.** `scripts/dub.py` transcreve `final_segments/<seg>.wav`, que é o segmento depois da transferência de envelope, mas antes de entrar na timeline, ser mixado com o fundo e normalizado. Como a transferência só mexe em ganho e o decoder é determinístico, a transcrição sai igual à do candidato — por isso `avg_cer_final == avg_cer`. Sem `--prosody-transfer` é literalmente o mesmo áudio transcrito duas vezes. Recorte cada segmento do `final_audio.wav` mixado, na janela em que ele foi posicionado: é ali que fundo alto, sobreposição com vizinho e corte no fim da timeline aparecem.
2. **O decoder do `evaluate` não enxerga loop.** `repetition_penalty=1.3` e `no_repeat_ngram_size=3` foram postos para conter alucinação do Whisper, mas proíbem o ASR de transcrever qualquer repetição real de 3+ tokens. Com isso `has_repetition` fica praticamente cego para loop do TTS, e palavras que se repetem legitimamente (artigos, preposições) são penalizadas. Decodifique sem essas restrições e, quando aparecer repetição, desempate com um sinal independente do ASR: contagem de núcleos silábicos pela energia contra as sílabas esperadas do texto (o método usado à mão no seg0009, em `docs/moss_tts_investigation.md`). Loop real tem muito mais núcleos que o esperado; alucinação do Whisper, não.
3. **Teste de mutação.** Corrompa de propósito o áudio final de um segmento (cortar o meio, ruído forte, frase colada duas vezes) e confirme que as flags certas disparam.

**Aceite:** `cer_final` medido sobre o áudio mixado. Os três tipos de corrupção disparam `cer_final_alto` ou `has_repetition`. Piso de CER da T0.13 remedido com o decoder novo, n ≥ 20.

### T0.20 — Métricas de expressividade confiáveis 🤝
1. `entonacao_desalinhada` mede a correlação de contorno de energia entre fonte e dublagem, e é calculada depois de `apply_energy_envelope` ter imposto esse contorno (com `strength=0.5`). A melhora de 0,235 para 0,353 é em boa parte tautológica. Além disso, correlação quadro a quadro entre inglês e português significa pouco, porque sílabas e ordem de palavras não se alinham. Troque por estatísticas de F0 agregadas por segmento (faixa, desvio, inclinação), que a transferência de energia não toca. A energia sai da flag e fica só como diagnóstico. Se a transferência de envelope vale a pena — inclusive se desloca ênfase para a sílaba errada — passa a ser decidido pelos rótulos da T0.28.
2. `emocao_incompativel`: a fonte já usa o stem de voz (conferido em `cmd_match_emotion`). Falta verificar se o cosseno é dominado pela classe neutra e, se o piso de 3,2 s cair na T0.26, definir uma duração mínima abaixo da qual a flag não é emitida.
3. O scorecard passa a reportar a sobreposição entre flags (quantos segmentos têm duas ou mais).

**Aceite:** teste com sinal sintético mostrando que a transferência de envelope não altera a nova métrica de entonação. Checagem do emotion2vec documentada com números. Scorecard mostra a matriz de sobreposição.

### T0.21 — Segmentação respeita troca de locutor 🤖
`packages/pipeline/segmentation.py`. `_break_into_groups` separa por locutor, mas `_merge_short_groups` funde grupos curtos com o vizinho sem olhar o locutor. Com `min_duration=3,2 s` (como `dub.py` chama), o diálogo "Are you ready?" (A) / "Yes." (B) / "Then let's go now" (A) vira um segmento só, atribuído a A. E `_dominant_speaker` devolve o locutor da primeira palavra, não o dominante. Hoje isso fica escondido porque toda síntese usa a mesma voz, mas já distorce a tradução (falas de duas pessoas chegam como uma frase só) e vira bug de produto assim que houver voz por locutor (T1.7, T2.6).

Fusão nunca atravessa troca de locutor. Grupo curto sem vizinho do mesmo locutor fica como está e recebe uma flag nova, `segmento_curto`, visível no editor, em vez de ser fundido em silêncio. `_dominant_speaker` passa a escolher por duração.
**Aceite:** o diálogo A/B/A acima produz três segmentos, com "Yes." atribuído a B e marcado. Teste de `_dominant_speaker` por duração. Fusão de grupos curtos do mesmo locutor continua igual.

### T0.22 — Texto que chega ao TTS 🤖
`packages/ptbr/normalize.py` (T0.3) só é usado dentro do cálculo de CER; nunca é aplicado ao texto enviado ao MOSS-TTS. Nos vídeos de `phase1test/`, o TTS recebeu "q", "vcs", "6X" e "D6" crus — e o CER, que normaliza os dois lados, esconde isso.
- Aplique `normalize` no caminho de síntese, em ordem compatível com o glossário IPA.
- Acrescente abreviações de chat ao normalizador (q, vc, vcs, pq, tb, tbm).
- Em `parse_candidates`, rejeite candidato com resíduo de estrutura (colchetes, aspas de lista): `lego_batman` seg0002 chegou ao TTS começando com `['`.

**Aceite:** testes do normalizador para as abreviações. Teste de que o texto do job de síntese sai normalizado, que termo do glossário continua sendo aplicado e que IPA entre barras não é alterado. Regressão do caso `['...` rejeitado.

### T0.23 — Tradução com contexto 🤝
`packages/pipeline/translation.py`. Cada segmento é traduzido isolado: o prompt só recebe a frase. Consequências visíveis em `phase1test/`: nome próprio traduzido como palavra comum ("Hurts has time" → "Dói no instante!" — Hurts é o quarterback), frase partida entre segmentos traduzida em pedaços ("you" | "made it!"), termo inconsistente entre segmentos ("guitar" e "guitarra" no mesmo vídeo).
- Passe no prompt o original do segmento anterior e do seguinte, só como contexto, sem traduzir.
- Extraia uma vez por vídeo uma lista de nomes próprios e termos, e instrua o modelo a mantê-los.
- Proíba abreviação de chat no prompt, mantendo as contrações faladas ("tá", "pra", "cê").

**Aceite:** re-traduzir e comparar à mão 30 segmentos de `phase1test/` antes e depois. Zero nome próprio traduzido nessa amostra. Aderência ao orçamento de sílabas (T0.8) não piora.

### T0.24 — Seleção de candidato e teto de geração 🤖
1. `pick_best_synthesis` devolve o **primeiro** candidato que passa, não o melhor. Entre os que passam, escolha pelo menor CER e, em empate, pela duração mais próxima do alvo. Não custa nada: os 3 candidatos já foram gerados e avaliados.
2. Em `cmd_synthesize`, `max_new_tokens = max(1024, 6 × tokens)`. Com segmentos de até 12 s (150 tokens), 6× dá no máximo 900, então o piso de 1024 sempre vence e o multiplicador é código morto. Um candidato em loop segura o lote inteiro por 1024 passos. Registre o número de passos de cada candidato no `synth_meta.json` e calibre o teto pela distribuição real (o maior caso legítimo registrado é 446 passos para ~150 tokens, ~3×).

**Aceite:** teste em que, entre dois candidatos que passam, vence o de menor CER. Teto calibrado com a distribuição registrada, sem cortar nenhum candidato legítimo num vídeo de teste.

### T0.25 — Cache por hash de entrada 🤖
`scripts/dub.py` decide se refaz uma etapa só pela existência do arquivo (`needs()`). Os segmentos são recalculados a cada execução, mas `translations.json` e `synth/` são reaproveitados por posição (`seg0007`). Se a segmentação mudar com o mesmo `--cache-dir`, a tradução antiga do seg0007 é aplicada ao novo seg0007, sem erro nenhum. A única saída hoje é `--force`, que refaz tudo, inclusive Demucs e ASR.

Chaveie cada etapa pelo hash das entradas e dos parâmetros relevantes — o mesmo princípio de `(project_id, stage, input_hash)` da T1.2 — e permita forçar uma etapa só.
**Aceite:** teste em que mudar a segmentação invalida tradução e síntese, mas não separação nem ASR. `--force-stage synthesize` refaz só a síntese.

### T0.26 — Isolar a causa do piso de 3,2 s 🤝
`MIN_SYNTHESIS_SECONDS = 3,2 s` decide a segmentação do vídeo inteiro: tudo abaixo disso é fundido com o vizinho. A explicação registrada (delay pattern com `n_vq=32`) foi confirmada em 2 de 3 casos, e o próprio log mostra um segundo fator misturado: texto curto sem contexto ("decide" falha com 100 tokens, mas funciona dentro de uma frase com 40). Em conteúdo de diálogo rápido essa regra pesa muito: em `phase1test/`, 20 de 33 segmentos do Lego Batman e 47 de 88 do NFL estão abaixo de 3,2 s.

Experimento em duas curvas, com ~10 amostras por ponto, medindo CER e duração:
- (a) texto fixo de várias palavras, variando `tokens` (13, 19, 25, 32, 40);
- (b) `tokens` fixos, variando o tamanho do texto (1, 2, 4, 8 palavras).

**Aceite:** as duas curvas documentadas. Piso revisado ou mantido, com justificativa. Se a causa for falta de contexto e não duração, proponha uma alternativa à fusão.

### T0.27 — Latência de ressíntese no Pod 🤝
Decisão: até entrar em produção, todo trabalho de GPU roda no Pod; Serverless fica para a T1.2a. O que não precisa esperar é saber se a invariante 3 é viável com o modelo carregado, e isso dá para medir no Pod. Cronometre a ressíntese completa de um segmento isolado: lote de candidatos, retry quando a mediana sai da tolerância e avaliação — não só a geração (o lote de 3 sozinho já leva ~3,5 s). Se passar de 5 s, proponha a separação entre caminho interativo (menos candidatos, avaliador menor) e caminho em lote.

Na mesma tarefa, congele `runpod/*/handler.py`: uma nota no topo do `runpod/README.md` dizendo que os handlers estão desatualizados e que a fonte de verdade é `scripts/pod_worker.py`, para ninguém editar ou fazer deploy deles por engano.
**Aceite:** tempos (mediana e pior caso, em ~20 segmentos) anotados no `runpod/README.md`, separados por etapa. Nota de congelamento no README.

### T0.28 — Rotulagem humana e calibração das flags 👤 + 🤖
Escolha 10 vídeos do conteúdo-alvo, de gêneros variados, com pelo menos metade de fala contínua de um locutor (review, educação, vlog). Os 5 vídeos de `phase1test/` servem de ponto de partida, mas precisam ser re-rodados depois das T0.19 a T0.25: os rótulos devem medir o pipeline atual.

Rode o pipeline sem mudanças e rotule cada segmento como `ok` ou `corrigiria`, com o motivo em uma palavra. Salve em JSONL indexado por `segment_id` + revisão. Claude Code escreve o formato e o cálculo; o julgamento é seu. Se possível, outra pessoa brasileira rotula uma parte, para medir concordância.

O scorecard passa a calcular, contra os rótulos, a **taxa de correção** real (a métrica do portão) e a precisão e o recall de cada flag. Os limiares são escolhidos por ponto de operação nesses dados, nunca num vídeo só.
**Aceite:** 10 vídeos rotulados (fecha também o aceite da T0.12). Taxa de correção por vídeo e agregada. Tabela de precisão e recall por flag. Limiares novos com justificativa escrita.

### 🚦 Portão de decisão
Rode nos 10 vídeos da T0.28. **Se a taxa de correção — medida por rótulo humano, não por `any_flag_rate` — passar de 30%, pare e conserte o pipeline antes de tocar em produto.** Não construa web em cima de um pipeline que não funciona. Metas intermediárias medidas num clipe só (como os ≤40% no trailer de 8 segmentos) não contam para o portão.

**Se o portão falhar:** ataque a causa que mais aparece nos motivos dos rótulos, não a que a métrica automática aponta. Se for emoção ou entonação, o caminho é a T0.29.

### T0.29 — Expressividade desacoplada da duração, no modelo do IndexTTS2 🤝 (condicional)
Só entra se a T0.28 mostrar emoção ou entonação entre as principais causas de correção.

**Base: a lógica que já está no repo.** `diagnostics.py` e `synthesis.py::infer_delivery_instruction` já definem a direção:
- emoção entra como condicionamento da síntese, nunca como filtro de áudio depois (o pitch transfer provou isso);
- a instrução só é reativada onde houver folga de orçamento, gerando com mais tokens e comprimindo de volta com `time_stretch_to_duration`;
- a emoção da fonte vem do emotion2vec, não só da pontuação.

**O exemplo do IndexTTS2** (arXiv:2506.21619) separa três coisas que o MOSS mistura:
1. duração exata por contagem de tokens;
2. geração livre, com a duração natural da fala;
3. timbre e emoção vindos de prompts separados (timbre prompt e style prompt).

Ele não serve como substituto direto — pelas fontes disponíveis, só fala inglês e chinês, e a licença precisa ser conferida. O plano reproduz essas três separações com o MOSS, sem re-treinar.

**Passos, nesta ordem:**

1. **Medir o fator de alongamento da instrução em geração livre** (o "modo livre" do IndexTTS2). Em ~30 segmentos atuais com emoção clara na fonte, gere com e sem instrução, sem orçamento apertado (`max_new_tokens` largo), e meça a razão de duração `k`. O 1,7–2,4x registrado foi medido sob orçamento apertado e com loop; é esse `k` que decide quantos segmentos comportam instrução. Registre também se há loop em geração livre.

2. **Instrução só onde cabe** (a regra que já está em `diagnostics.py`, agora explícita). Para cada segmento, `r` = duração natural estimada pela calibração ÷ tempo disponível. Aplique a instrução só se `r × k ≤ MAX_TIME_STRETCH_RATIO`, pedindo `tokens` para a duração instruída natural e comprimindo de volta ao alvo. Nos demais, `instruction=None`. Estimativa sobre `phase1test/`, sem contar pausas internas: com `k = 1,7`, 44% dos segmentos cabem (20% sem precisar de stretch); com `k = 2,4`, só 15%. Valide nos segmentos catastróficos de 2026-09-19 (seg0006/seg0011): a regra precisa recusá-los, não deixá-los entrar em loop.

3. **Emoção da fonte, não da pontuação** (ideia já registrada em `diagnostics.py`, equivalente ao Text-to-Emotion do IndexTTS2). Troque o `!`/`?` de `infer_delivery_instruction` pela classe do emotion2vec no stem de voz do segmento original, com uma instrução específica por classe (raiva, tristeza, alegria, surpresa; neutro → sem instrução). Atenção: usar o mesmo modelo para controlar e para medir torna `emotion_similarity` otimista por construção. O julgamento final é o rótulo humano da T0.28.

4. **Timbre e emoção separados** (o style prompt do IndexTTS2). Para os segmentos em que a instrução não cabe, a emoção vem pela referência de áudio, que não custa tempo:
   - Monte um banco de referências **na mesma voz pt-BR**, uma por classe de emoção, geradas com o próprio MOSS em modo livre com instrução (onde não há loop) e escolhidas de ouvido.
   - Na síntese com orçamento fechado, passe a referência da classe da fonte com `instruction=None`.

   Assim a duração fica com os `tokens` e a emoção com o áudio de referência — a separação que o IndexTTS2 faz no treino. Compare com usar o trecho do stem original como referência, que traz o tom certo mas arrisca sotaque inglês e mudança de timbre.

5. **Proteção:** teto de passos da T0.24, rejeição de candidato fora da duração, stretch nunca acima de 1,5x. Compare phase vocoder com WSOLA ou Rubber Band na mesma razão, por CER e escuta.

**Aceite:** `k` medido e documentado. Nenhum loop chega ao áudio entregue. CER round-trip não piora além do piso da T0.19 mais uma margem documentada. A taxa de correção por emoção e entonação cai nos vídeos da T0.28 (re-rotulando só os segmentos afetados). Caminho quente continua < 5 s, medido como na T0.27 — segmento instruído gera mais tokens, então meça os dois casos.

---

## Fase 1 — MVP web

### T1.1 — Schema e migrations 🤖
`org`, `user`, `channel`, `project`, `asset`, `speaker`, `voice_profile`, `glossary_entry`, `segment`, `segment_revision`, `job`. `org_id` em todas. Índices para as queries do editor (segmentos por projeto, ordenados por tempo).
**Aceite:** migrations sobem e descem limpo. Seed de desenvolvimento.

### T1.2 — Fila em Postgres 🤖
`apps/api/queue.py`. `SELECT ... FOR UPDATE SKIP LOCKED`, com retry, backoff, dead-letter e jobs idempotentes por `(project_id, stage, input_hash)`.
**Aceite:** teste de concorrência com 10 workers não processa o mesmo job duas vezes. Job interrompido é retomado.

### T1.2a — Migração para RunPod Serverless 🤝
Antes de qualquer usuário externo, já que o Pod via SSH é só para desenvolvimento.
1. Extraia a lógica de cada estágio de `scripts/pod_worker.py` para um módulo compartilhado (por exemplo `gpu_stages/synthesize.py` com `load()` e `run(jobs)`; evite o nome `runpod`, que colide com o SDK). `pod_worker.py` e os handlers viram cascas finas de I/O.
2. Refaça as imagens a partir dessa lógica e suba os cinco endpoints.
3. Meça cold start e latência quente de cada um. Para o `synthesize`, decida como a invariante 3 convive com o cold start: worker quente durante sessão de editor ou aviso na UX.

**Aceite:** nenhuma lógica de modelo duplicada entre `pod_worker.py` e os handlers. Os cinco endpoints rodam o pipeline de um vídeo de teste de ponta a ponta. Cold start e latência quente anotados no `runpod/README.md`. Decisão sobre a invariante 3 registrada no CLAUDE.md.

### T1.3 — Orquestração do pipeline 🤖
Máquina de estados por projeto: `uploaded → separated → transcribed → segmented → translated → synthesized → assembled → ready`. Cada transição é um job. Falha numa etapa não perde o trabalho das anteriores.
**Aceite:** matar o worker no meio e reiniciar retoma do ponto certo.

### T1.4 — Auth e upload 🤖
Supabase Auth. Upload direto para R2 via URL pré-assinada (o vídeo não passa pela sua API). Limite de 10 minutos de duração e 2 GB. Declaração de direitos sobre as vozes, persistida com timestamp.
**Aceite:** upload de 1 GB funciona sem estourar memória da API. Registro de consentimento gravado.

### T1.5 — Endpoints REST 🤖
`POST /projects`, `GET /projects/:id`, `GET /projects/:id/segments`, `PATCH /segments/:id`, `POST /segments/:id/resynth`, `GET /projects/:id/export`. Paginação nos segmentos.
**Aceite:** testes de integração cobrindo cada rota, incluindo isolamento por `org_id`.

### T1.5a — Watermark de conteúdo sintético 🤝
Marca d'água em todo áudio gerado, aplicada na montagem, antes de qualquer download. Precisa existir antes da T1.6, que é o primeiro momento em que áudio sai para pessoas de fora (obrigação de conformidade do CLAUDE.md). Veio da antiga T2.9.
**Aceite:** watermark detectável no áudio exportado, inclusive depois da recompressão AAC do remux. Inaudível em teste cego com 5 pessoas.

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
**Aceite:** ressíntese completa em <5s com worker RunPod quente (o cold start é tratado na UX, conforme o medido na T1.2a). Histórico preservado. Desfazer funciona.

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

### T2.9 — Export 🤖
Export com opções de qualidade, sempre passando pelo watermark da T1.5a. Registro de exportações.
**Aceite:** teste garantindo que nenhum caminho de export pula o watermark. Registro de exportação gravado com `org_id`.

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

**Commit ao fim de cada sessão.** Trabalho de vários dias fora do git é trabalho que pode sumir.

**Desligue o Pod ao fim de cada sessão de GPU**, enquanto o harness da Fase 0 existir.

**Peça o teste antes da implementação** nas tarefas de `packages/pipeline` e `packages/ptbr`. São puras — dão testes limpos e é onde bugs custam caro.

**Nas tarefas 🤝, ouça o áudio.** Teste verde não significa que soa bem. Nenhuma métrica substitui você escutando 10 amostras.

**Não deixe refatorar fora do escopo.** Se aparecer "aproveitei para melhorar X", peça para reverter X.

**Atualize o `CLAUDE.md`** sempre que uma decisão nova for tomada — especialmente os achados da T0.6, T0.9, T0.26, T0.27 e T1.2a, que afetam todo o resto.
