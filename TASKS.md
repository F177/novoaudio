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

**Foco (2026-10-10):** o produto é dublagem em massa de YouTube Shorts para clippers (ver CLAUDE.md). Fala rápida com troca de locutor é o caso principal; o Short tem que sair bom sem edição; custo de GPU por Short importa.

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

### T0.19 — Instrumento de avaliação confiável 🤝 (feito, 2026-10-05)
Três problemas no instrumento que mede todo o resto:

1. **O "CER final" não é pós-mix.** `scripts/dub.py` transcreve `final_segments/<seg>.wav`, que é o segmento depois da transferência de envelope, mas antes de entrar na timeline, ser mixado com o fundo e normalizado. Como a transferência só mexe em ganho e o decoder é determinístico, a transcrição sai igual à do candidato — por isso `avg_cer_final == avg_cer`. Sem `--prosody-transfer` é literalmente o mesmo áudio transcrito duas vezes. Recorte cada segmento do `final_audio.wav` mixado, na janela em que ele foi posicionado: é ali que fundo alto, sobreposição com vizinho e corte no fim da timeline aparecem.
2. **O decoder do `evaluate` não enxerga loop.** `repetition_penalty=1.3` e `no_repeat_ngram_size=3` foram postos para conter alucinação do Whisper, mas proíbem o ASR de transcrever qualquer repetição real de 3+ tokens. Com isso `has_repetition` fica praticamente cego para loop do TTS, e palavras que se repetem legitimamente (artigos, preposições) são penalizadas. Decodifique sem essas restrições e, quando aparecer repetição, desempate com um sinal independente do ASR: contagem de núcleos silábicos pela energia contra as sílabas esperadas do texto (o método usado à mão no seg0009, em `docs/moss_tts_investigation.md`). Loop real tem muito mais núcleos que o esperado; alucinação do Whisper, não.
3. **Teste de mutação.** Corrompa de propósito o áudio final de um segmento (cortar o meio, ruído forte, frase colada duas vezes) e confirme que as flags certas disparam.

**Aceite:** `cer_final` medido sobre o áudio mixado. Os três tipos de corrupção disparam `cer_final_alto` ou `has_repetition`. Piso de CER da T0.13 remedido com o decoder novo, n ≥ 20.

### T0.20 — Métricas de expressividade confiáveis 🤝 (feito, 2026-10-05)
1. `entonacao_desalinhada` mede a correlação de contorno de energia entre fonte e dublagem, e é calculada depois de `apply_energy_envelope` ter imposto esse contorno (com `strength=0.5`). A melhora de 0,235 para 0,353 é em boa parte tautológica. Além disso, correlação quadro a quadro entre inglês e português significa pouco, porque sílabas e ordem de palavras não se alinham. Troque por estatísticas de F0 agregadas por segmento (faixa, desvio, inclinação), que a transferência de energia não toca. A energia sai da flag e fica só como diagnóstico. Se a transferência de envelope vale a pena — inclusive se desloca ênfase para a sílaba errada — passa a ser decidido pelos rótulos da T0.28.
2. `emocao_incompativel`: a fonte já usa o stem de voz (conferido em `cmd_match_emotion`). Falta verificar se o cosseno é dominado pela classe neutra e, se o piso de 3,2 s cair na T0.26, definir uma duração mínima abaixo da qual a flag não é emitida.
3. O scorecard passa a reportar a sobreposição entre flags (quantos segmentos têm duas ou mais).

**Aceite:** teste com sinal sintético mostrando que a transferência de envelope não altera a nova métrica de entonação. Checagem do emotion2vec documentada com números. Scorecard mostra a matriz de sobreposição.

### T0.21 — Segmentação respeita troca de locutor 🤖 (feito, 2026-10-05)
`packages/pipeline/segmentation.py`. `_break_into_groups` separa por locutor, mas `_merge_short_groups` funde grupos curtos com o vizinho sem olhar o locutor. Com `min_duration=3,2 s` (como `dub.py` chama), o diálogo "Are you ready?" (A) / "Yes." (B) / "Then let's go now" (A) vira um segmento só, atribuído a A. E `_dominant_speaker` devolve o locutor da primeira palavra, não o dominante. Hoje isso fica escondido porque toda síntese usa a mesma voz, mas já distorce a tradução (falas de duas pessoas chegam como uma frase só) e vira bug de produto assim que houver voz por locutor (T1.7, T2.6).

Fusão nunca atravessa troca de locutor. Grupo curto sem vizinho do mesmo locutor fica como está e recebe uma flag nova, `segmento_curto`, visível no editor, em vez de ser fundido em silêncio. `_dominant_speaker` passa a escolher por duração.
**Aceite:** o diálogo A/B/A acima produz três segmentos, com "Yes." atribuído a B e marcado. Teste de `_dominant_speaker` por duração. Fusão de grupos curtos do mesmo locutor continua igual.

### T0.22 — Texto que chega ao TTS 🤖 (feito, 2026-10-05)
`packages/ptbr/normalize.py` (T0.3) só é usado dentro do cálculo de CER; nunca é aplicado ao texto enviado ao MOSS-TTS. Nos vídeos de `phase1test/`, o TTS recebeu "q", "vcs", "6X" e "D6" crus — e o CER, que normaliza os dois lados, esconde isso.
- Aplique `normalize` no caminho de síntese, em ordem compatível com o glossário IPA.
- Acrescente abreviações de chat ao normalizador (q, vc, vcs, pq, tb, tbm).
- Em `parse_candidates`, rejeite candidato com resíduo de estrutura (colchetes, aspas de lista): `lego_batman` seg0002 chegou ao TTS começando com `['`.

**Aceite:** testes do normalizador para as abreviações. Teste de que o texto do job de síntese sai normalizado, que termo do glossário continua sendo aplicado e que IPA entre barras não é alterado. Regressão do caso `['...` rejeitado.

### T0.23 — Tradução com contexto 🤝 (feito, 2026-10-09)
`packages/pipeline/translation.py`. Cada segmento é traduzido isolado: o prompt só recebe a frase. Consequências visíveis em `phase1test/`: nome próprio traduzido como palavra comum ("Hurts has time" → "Dói no instante!" — Hurts é o quarterback), frase partida entre segmentos traduzida em pedaços ("you" | "made it!"), termo inconsistente entre segmentos ("guitar" e "guitarra" no mesmo vídeo).
- Passe no prompt o original do segmento anterior e do seguinte, só como contexto, sem traduzir.
- Extraia uma vez por vídeo uma lista de nomes próprios e termos, e instrua o modelo a mantê-los.
- Proíba abreviação de chat no prompt, mantendo as contrações faladas ("tá", "pra", "cê").

**Validação real (30 segmentos do NFL de `phase1test/` — `.cache/dub_nfl/`, antes/depois, Qwen2.5-7B no Pod):** inconsistência de nome entre segmentos era real e sistemática — "Buda Baker" (seg0009) saía "Buda" sozinho em TODOS os 5 candidatos antes do contexto/preserve_terms, "Manning" (seg0024) desaparecia em 1 de 5. Depois: "Buda Baker" completo nos 5 candidatos de ambos os segmentos onde aparece, "Manning" presente nos 5. "Hurts"/"Barkley" (seg0001) já saíam corretos nos 5 candidatos antes E depois nesta amostra — o erro "Dói no instante!" do exemplo acima não reproduziu aqui (amostragem `do_sample=True`, não determinístico). Aderência ao orçamento de sílabas: score médio do candidato top em 30 segmentos, 0,838 antes vs 0,837 depois — sem regressão (diferença dentro do ruído de amostragem). `extract_proper_nouns` é heurística (capitalização fora de início de frase, sigla, ou repetição de início de frase em 2+ segmentos — sem lista de stopwords ficaria poluída por "Can"/"Has"/"Yeah" etc., sem heurística de repetição perderia nome próprio citado uma única vez, caso conhecido e documentado em teste). Achado paralelo, fora do escopo desta tarefa: seg0022 ("but evidently") produziu caractere chinês (`显而易见`) nos dois lados, antes e depois — vazamento de idioma do Qwen2.5, não relacionado a contexto/preserve_terms, não corrigido aqui.

**Aceite:** re-traduzir e comparar à mão 30 segmentos de `phase1test/` antes e depois. Zero nome próprio traduzido nessa amostra. Aderência ao orçamento de sílabas (T0.8) não piora.

### T0.24 — Seleção de candidato e teto de geração 🤖 (feito, 2026-10-09)
1. `pick_best_synthesis` devolve o **primeiro** candidato que passa, não o melhor. Entre os que passam, escolha pelo menor CER e, em empate, pela duração mais próxima do alvo. Não custa nada: os 3 candidatos já foram gerados e avaliados.
2. Em `cmd_synthesize`, `max_new_tokens = max(1024, 6 × tokens)`. Com segmentos de até 12 s (150 tokens), 6× dá no máximo 900, então o piso de 1024 sempre vence e o multiplicador é código morto. Um candidato em loop segura o lote inteiro por 1024 passos. Registre o número de passos de cada candidato no `synth_meta.json` e calibre o teto pela distribuição real (o maior caso legítimo registrado é 446 passos para ~150 tokens, ~3×).

**Aceite:** teste em que, entre dois candidatos que passam, vence o de menor CER. Teto calibrado com a distribuição registrada, sem cortar nenhum candidato legítimo num vídeo de teste.

**Achado real de implementação:** o comentário antigo assumia que `generate()` em lote devolve um tensor único padded (passos por CHAMADA). Testado direto contra o modelo: `generate()` do MOSS-TTS devolve uma lista de `(escalar, tensor[n_passos, n_canais])`, um por candidato, já cortado no tamanho real — "passos usados" é por CANDIDATO, não por chamada. `steps_used` agora vai dentro de cada entrada de `candidates` em `synth_meta.json`. Teto recalibrado pra `MAX_NEW_TOKENS_MULTIPLIER=4`, `MIN_MAX_NEW_TOKENS=256` (piso baixo o bastante pro multiplicador não ficar morto de novo acima de tokens≈64, alto o bastante pro piso de síntese tokens≈40). **Validação real (NFL completo, 90 segmentos/270 candidatos, Pod):** `steps_used` máximo observado foi 173 (mínimo 75) — nenhum candidato chegou perto do novo teto (256 pro menor orçamento do vídeo), razão passos/tokens no máximo 1,9× (não os ~3× do pior caso histórico, nem precisou chegar lá) — zero candidato cortado. 39/90 jobs precisaram de retry, 0 precisaram de escalação, 17/90 saíram fora de tolerância mesmo após retry+stretch (não é regressão desta tarefa, consistente com limitações já conhecidas de calibração de duração).

### T0.25 — Cache por hash de entrada 🤖 (feito, 2026-10-09)
`scripts/dub.py` decide se refaz uma etapa só pela existência do arquivo (`needs()`). Os segmentos são recalculados a cada execução, mas `translations.json` e `synth/` são reaproveitados por posição (`seg0007`). Se a segmentação mudar com o mesmo `--cache-dir`, a tradução antiga do seg0007 é aplicada ao novo seg0007, sem erro nenhum. A única saída hoje é `--force`, que refaz tudo, inclusive Demucs e ASR.

Chaveie cada etapa pelo hash das entradas e dos parâmetros relevantes — o mesmo princípio de `(project_id, stage, input_hash)` da T1.2 — e permita forçar uma etapa só.
**Aceite:** teste em que mudar a segmentação invalida tradução e síntese, mas não separação nem ASR. `--force-stage synthesize` refaz só a síntese.

Aplicado às 5 etapas que tinham o bug de reúso por existência de arquivo: `separate_stems`, `transcribe`, `translate`, `synthesize`, `evaluate` (CER do candidato). Hash de entrada = conteúdo real que a etapa recebe (bytes do áudio/vocals pra `separate_stems`/`transcribe`; os `*_jobs.json` já montados, agora montados SEMPRE, não só quando a etapa vai rodar, pra poder calcular o hash mesmo com cache existente) — registrado num sidecar `<etapa>.hash` no `cache_dir`. Sem sidecar (cache de antes desta tarefa, ou primeira rodada) conta como obsoleto — mais seguro que confiar num arquivo de saída sem prova de que veio da mesma entrada. `final_cer`/`match_emotion` não tinham ESSE bug (já rodavam sem cache nenhum, sempre do zero) — fora do escopo, não tocados. `--force-stage {separate_stems,transcribe,translate,synthesize,evaluate}` força só a etapa nomeada.

**Validação real:** rodando o vídeo padrão (`videos/`) duas vezes seguidas sem mudar nada — a 2ª rodada não chamou `separate_stems`/`transcribe`/`translate`/`synthesize`/`evaluate` no Pod (ausentes do log de timing), só as etapas já sem cache de antes (`evaluate_final`, `match_emotion`, uploads). Confirma que o hash não gera reprocessamento espúrio quando a entrada não mudou — tão importante quanto invalidar quando muda.

### T0.26 — Isolar a causa do piso de 3,2 s 🤝 (feito, 2026-10-09)
`MIN_SYNTHESIS_SECONDS = 3,2 s` decide a segmentação do vídeo inteiro: tudo abaixo disso é fundido com o vizinho. A explicação registrada (delay pattern com `n_vq=32`) foi confirmada em 2 de 3 casos, e o próprio log mostra um segundo fator misturado: texto curto sem contexto ("decide" falha com 100 tokens, mas funciona dentro de uma frase com 40). Em conteúdo de diálogo rápido essa regra pesa muito: em `phase1test/`, 20 de 33 segmentos do Lego Batman e 47 de 88 do NFL estão abaixo de 3,2 s.

Experimento em duas curvas, com ~10 amostras por ponto, medindo CER e duração:
- (a) texto fixo de várias palavras, variando `tokens` (13, 19, 25, 32, 40);
- (b) `tokens` fixos, variando o tamanho do texto (1, 2, 4, 8 palavras).

**Aceite:** as duas curvas documentadas. Piso revisado ou mantido, com justificativa. Se a causa for falta de contexto e não duração, proponha uma alternativa à fusão.

**Resultado real (`scripts/experiment_t026_synthesis_floor.py`, 90 amostras, N=10/ponto — números completos em `docs/moss_tts_investigation.md`, 2026-10-09):** são DUAS causas independentes, não uma. (1) Duração é determinística e governada quase só por `tokens` — zero variância entre amostras, mesmo texto com 4 tamanhos diferentes e mesmo `tokens` produziu a MESMA duração exata nas 40 amostras. Confirma o delay pattern (n_vq) como causa física real da duração mínima. (2) O colapso de qualidade (CER catastrófico, 1,167 de média) não vem de duração curta — vem de contagem de palavras: 1 palavra forçada a preencher 2,08s degenera em muleta de fala ("ã...", "Uhum,"), sem repetição/loop; a partir de 2 palavras CER cai pra ~0,02-0,03 e não piora mais indo pra 4 ou 8. **Piso MANTIDO em 3,2s** — nenhuma evidência coletada (não testei tokens<13, só uma frase) justifica baixá-lo com segurança, mesmo princípio de nunca calibrar limiar de produto num experimento só.

**Investigação do problema real de produto (decisão tomada com o usuário, 2026-10-09):** a proposta inicial (fundir só <2 palavras, deixar segmentos de 2+ palavras sintetizarem isolados) foi checada contra o vídeo padrão real (`videos/`) ANTES de implementar e se mostrou incompleta — ela resolveria qualidade de conteúdo mas não resolveria o `TimelineOverflowError` real que motivou a investigação, porque o piso de DURAÇÃO (independente de palavras) ainda faria o segmento isolado estourar o espaço até o próximo. Inspecionando os `segments.json` reais do vídeo padrão: as 4 ocorrências do erro já tinham tentado fundir (`curto=True`) e falharam porque `_mergeable` (T0.21) bloqueou corretamente — diálogo rápido com troca de locutor dos dois lados, não falta de regra de fusão por palavra. A causa real não era a regra de fusão, era o gate de `assembly.place_segments_on_timeline` assumir que só 1 segmento legítimo no piso físico pode existir por vídeo — falso, como os 4 casos reais provam. **Implementado em `packages/pipeline/assembly.py`**: `MAX_SEGMENTS_WITH_DISCARD` removido; qualquer número de segmentos pode descartar contra o início do próximo, desde que CADA UM fique dentro de `MAX_SINGLE_SEGMENT_DISCARD_SECONDS`. Novo gate `MAX_SEGMENTS_STARTING_BEYOND_VIDEO=1`, escopo estrito e separado: um segmento cujo PRÓPRIO início já passa de `total_duration_seconds` nunca é o piso de síntese, é `total_duration_seconds`/timestamp errado — mantido limitado a 1, igual antes. `segmentation.py` não foi tocado (a fusão por palavra, T0.21, continua igual).

**Validação real:** vídeo padrão (`videos/`) rodado sem `--allow-timeline-overflow` (removido o workaround) — completou sem erro, 2,837s descartados nos mesmos segmentos de antes (mesma magnitude, agora tratado como legítimo, não bug). Durante a validação, dois bugs SEPARADOS (não deste gate) apareceram e foram corrigidos: (1) cache de `translations.json`/`synth/` de ANTES da T0.25 existir, inconsistente com a segmentação atual — resolvido limpando o cache obsoleto, não é falha da T0.25 (o mecanismo de hash funciona corretamente pra dados gerados depois dele existir); (2) `scripts/pod_runner.py::upload_to_pod` não limpava o diretório remoto antes de subir um diretório — reupload pra um `remote_job_dir` já usado aninhava o conteúdo novo dentro do antigo (`final_mixed_segments/final_mixed_segments/`), fazendo `cmd_evaluate` não achar o segmento esperado. Corrigido: `upload_to_pod` agora remove o destino remoto antes de subir quando `local_path` é diretório.

## Reorientação para Shorts em massa (2026-10-10)

As tarefas T0.30 a T0.33 entram **antes** da T0.27 e da T0.28, nesta ordem. Motivo: o Short padrão (`videos/`, streamer, 53 s) mostrou que a tradução é o elo mais fraco (cerca de 6 de 13 falas claramente erradas), que ~2,8 s de fala são cortados em diálogo rápido e que um Short custa ~5–6 min de GPU.

### T0.30 — Tradução 🤝 (feito, 2026-10-10)
Hoje cada segmento era traduzido sozinho pelo Qwen2.5-7B. Erros reais no Short padrão: "How much debt do you have?" → "Tá valendo quantos débitos tu tá com aí?"; "Here you" → "Aqui vcs"; "What are you talking about?" → "Vamos ver."; "job?" ficou em inglês.

**O que foi tentado e medido (5 Shorts, comparação lado a lado e escuta do usuário):**
- Qwen por janela de diálogo de 12 falas, JSON validado 1:1, encurtamento por LLM com juiz de fidelidade. Melhorou sobre o segmento isolado, mas continuou errando palavra e sentido ("Quanto débito", "não conhece eu", "muita poder"), cortando conteúdo ("e riem" sumiu) e variando a cada rodada. O encurtamento deixava fala telegráfica ("Pai paga?"). Regra de registro falado mais forte fez o 7B trocar "você" por "tu" — revertida.
- **Escolhido: OPUS-MT** (`Helsinki-NLP/opus-mt-tc-big-en-pt`, 240M, tag `>>pob<<` = pt-BR; com `>>por<<` sai pt-PT). Traduz a frase inteira (falas cortadas são juntadas), devolve 8 alternativas e a mais curta que cabe no orçamento (com perda de nota limitada) vence; contração (está→tá) é alternativa extra. A tradução volta para os segmentos pelo alinhamento de atenção do próprio modelo. Filtros do perfil pt-BR: formas pt-PT descartadas, ênclise vira próclise, palavrão nunca diminui. ~6 s por Short na GPU, determinístico.
- Encurtar com Qwen depois do OPUS piorou as falas — descartado. Todo o caminho de tradução por LLM foi removido.

**Achados no caminho, corrigidos:** cache da avaliação ignorava o áudio novo (escolhia o candidato pela transcrição de outro áudio — causa de palavra engolida); WhisperX às vezes devolve trecho sem pontuação, e o tradutor invertia o sentido — resolvido com modelo de restauração de pontuação (pedir pontuação ao Whisper com `initial_prompt` fez ele pular uma frase inteira, descartado); volume diferente por fala do MOSS — nível de voz igualado ao da fala original.

**Ainda erra (aceito, vai para a revisão rápida):** expressão idiomática ("the kicker"), alguma escolha literal ("room" → "quarto", "ahead" → "antes"), gênero ocasional ("Tá relaxada").

### T0.31 — Vozes de amostra por locutor 🤝
Voz do criador clonada não é prioridade. Mas Short com dois locutores não pode soar como uma pessoa só. Banco pequeno de vozes pt-BR de amostra (ex.: 2 masculinas, 2 femininas), geradas com o próprio MOSS e escolhidas de ouvido — com energia, já que referência neutra puxa a entrega pro neutro. Cada locutor da diarização recebe uma voz diferente, escolhida pelo registro da voz original (F0 médio).
**Aceite:** o Short padrão (2 locutores) sai com 2 vozes distintas e estáveis em todos os segmentos de cada locutor. CER round-trip não piora.

### T0.32 — Vazão em lote 🤖
Um Short de 53 s gasta ~5–6 min de GPU, e cada etapa recarrega o modelo para cada Short. `dub.py` passa a aceitar uma pasta de Shorts e roda cada etapa para N Shorts por carregamento de modelo. Avaliação final e comparação de emoção ficam opcionais ou por amostragem.
**Aceite:** tempo de GPU por Short, por etapa, num lote de 10 Shorts, antes e depois. Saída idêntica à do processamento individual.

### T0.33 — Fala curta em diálogo rápido 🤝
Todo segmento abaixo de ~2 s sai com ~2,19 s (piso físico, T0.26) e é cortado contra o início do próximo — ~2,8 s de fala perdidos no Short padrão. Opções a medir: usar o silêncio entre segmentos antes de cortar; deslocar levemente o próximo segmento; sintetizar falas curtas seguidas juntas e redistribuir; acelerar até o teto de 1,5x antes de cortar.
**Aceite:** fala cortada por Short, medida nos Shorts da T0.28, cai. Qualquer relaxamento de invariante tem decisão explícita do usuário.

### T0.27 — Latência de ressíntese no Pod 🤝 (adiada para depois do portão)
Com revisão rápida no lugar do editor detalhado, a ressíntese interativa de um segmento importa menos que a vazão em lote (T0.32). Continua valendo antes do editor.
Decisão: até entrar em produção, todo trabalho de GPU roda no Pod; Serverless fica para a T1.2a. O que não precisa esperar é saber se a invariante 3 é viável com o modelo carregado, e isso dá para medir no Pod. Cronometre a ressíntese completa de um segmento isolado: lote de candidatos, retry quando a mediana sai da tolerância e avaliação — não só a geração (o lote de 3 sozinho já leva ~3,5 s). Se passar de 5 s, proponha a separação entre caminho interativo (menos candidatos, avaliador menor) e caminho em lote.

Na mesma tarefa, congele `runpod/*/handler.py`: uma nota no topo do `runpod/README.md` dizendo que os handlers estão desatualizados e que a fonte de verdade é `scripts/pod_worker.py`, para ninguém editar ou fazer deploy deles por engano.
**Aceite:** tempos (mediana e pior caso, em ~20 segmentos) anotados no `runpod/README.md`, separados por etapa. Nota de congelamento no README.

### T0.28 — Rotulagem humana e calibração das flags 👤 + 🤖
Escolha **20 Shorts** do conteúdo-alvo, variados: streamer, podcast, reação, comentário — com fala rápida e troca de locutor, que é o caso principal. Os vídeos de `phase1test/` e o Short padrão servem de ponto de partida, re-rodados com o pipeline atual.

Rode o pipeline sem mudanças e rotule **cada Short** como `publicaria` ou `não publicaria`, e cada segmento como `ok` ou `corrigiria`, com o motivo em uma palavra (tradução, corte, voz, emoção, pronúncia, timing). Salve em JSONL indexado por `segment_id` + revisão. Claude Code escreve o formato e o cálculo; o julgamento é seu. Se possível, outra pessoa brasileira rotula uma parte, para medir concordância.

O scorecard passa a calcular, contra os rótulos, a **taxa de correção** real (a métrica do portão) e a precisão e o recall de cada flag. Os limiares são escolhidos por ponto de operação nesses dados, nunca num vídeo só.
**Aceite:** 20 Shorts rotulados (fecha também o aceite da T0.12). Taxa de publicação agregada; taxa de correção por segmento, por Short e agregada, com a contagem por motivo. Tabela de precisão e recall por flag. Limiares novos com justificativa escrita.

### 🚦 Portão de decisão
Rode nos 20 Shorts da T0.28. **Se menos de 70% dos Shorts saírem publicáveis sem edição — medido por rótulo humano, não por `any_flag_rate` — pare e conserte o pipeline antes de tocar em produto.** (70% confirmado pelo usuário em 2026-10-10; antes o portão era 30% de segmentos corrigidos.) Não construa web em cima de um pipeline que não funciona. Metas intermediárias medidas num clipe só (como os ≤40% no trailer de 8 segmentos) não contam para o portão.

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

### T0.34 — Mixagem 🤝 (último estágio da Fase 0, pedido do usuário em 2026-10-10)
A mixagem é muito importante para o Short soar publicável. Hoje ela só soma voz e fundo e normaliza o áudio inteiro (T0.10). Já feito: nível de voz por fala igualado ao da fala original (`assembly.match_speech_levels`, 2026-10-10), porque o MOSS gera cada fala num volume diferente.

Falta:
1. **Ducking:** abaixar o fundo enquanto a voz dublada fala (sidechain com ataque e soltura suaves) e devolver nas pausas. O fundo alto às vezes cobre a voz (achado de 2026-09-14, ainda aberto).
2. **Vazamento da voz original:** o stem de fundo do Demucs às vezes traz resto da fala em inglês por baixo da dublagem. Medir (ASR no fundo, nas janelas com dublagem) e atenuar só nessas janelas.
3. **Bordas das falas:** fade curto no início e no fim de cada fala, sem clique nem corte seco.
4. **Ambiência:** a voz do TTS sai seca e o original tem sala. Medir se uma reverberação leve, igual à do original, soa mais natural.
5. **Entrega para o YouTube:** −14 LUFS integrado e pico real ≤ −1 dBTP.

**Aceite:** nos Shorts da T0.28, a voz fica X dB acima do fundo durante a fala (X a definir por escuta) e nenhum segmento sai com fala em inglês audível no fundo. Sem clique nas bordas. Loudness e pico dentro do alvo. Comparação de ouvido antes e depois, feita pelo usuário.

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

### T1.7 — Perfil de voz por canal 🤝 (baixa prioridade, 2026-10-10)
Voz do criador é detalhe extra; vozes de amostra (T0.31) bastam. Só depois do produto sair bom, e com consentimento do dono da voz resolvido.

Entidade `voice_profile`: áudio de referência do criador, embedding, reutilizado em todos os vídeos do canal. Extração automática do segmento mais limpo e longo do primeiro vídeo, com opção de substituir.
**Aceite:** dois vídeos do mesmo canal produzem SECS ≥0,75 entre si.

### T1.8 — Glossário por canal 🤖
Entidade `glossary_entry`: termo de origem, tradução fixa, pronúncia IPA opcional. Aplicado na tradução e na síntese. UI de CRUD simples.
**Aceite:** termo no glossário sempre sai com a tradução e a pronúncia definidas, em 10 vídeos de teste.

---

## Fase 2 — Editor

**Reorientação (2026-10-10):** para clippers com volume, o primeiro passo é uma **revisão rápida** por Short — fila, prévia, aceitar / regenerar / rejeitar, trocar a tradução de uma fala. O editor detalhado abaixo vem depois, na ordem em que a T2.8 mostrar necessidade.

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
