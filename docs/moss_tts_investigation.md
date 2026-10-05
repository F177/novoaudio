# Investigação: qualidade do MOSS-TTS-v1.5 no pipeline de dublagem

Log vivo da investigação de por que a síntese de voz (T0.12) sai com qualidade
ruim, e do que já foi tentado. Objetivo: entender e corrigir a causa raiz,
inclusive mexendo no código de geração do modelo se precisar (decisão
explícita do usuário em 2026-09-14, depois de esgotar engenharia de pipeline
— ver seção "Decisão" no fim).

**Não editar o cache do HF (`transformers_modules`) diretamente.** Qualquer
patch precisa virar monkey-patch versionado no nosso repo (ver
`scripts/pod_worker.py`), nunca uma edição no arquivo baixado — isso se perde
a qualquer redownload/limpeza de cache.

## Linha do tempo

### Sessão anterior (2026-09-12/13) — achados herdados

- Primeiro teste E2E real (trailer Avengers Doomsday) avaliado pelo usuário
  em 1/10 — "só entendo 2 palavras".
- MOSS-TTS-v1.5 tem um bug de repetição conhecido e documentado pelo próprio
  time do SGLang ("a small fraction of utterances loop and generate up to
  max_new_tokens"), classificado como raro por eles mas dominante no nosso
  caso de uso (frases curtas, duração fechada).
- Achado real, não relacionado ao MOSS-TTS: `faster-whisper` com
  `condition_on_previous_text=True` (default) alucina texto repetido em
  áudio curto e limpo — confirmado num áudio real de 4,2s que virou a mesma
  frase 16x. Corrigido com `condition_on_previous_text=False`
  (`scripts/pod_worker.py::cmd_evaluate`). Isso quer dizer que os números de
  CER/repetição medidos ANTES desse fix estavam inflados.
- Tentado e descartado como fix pro MOSS-TTS: parâmetros de geração
  (`text_temperature`, `audio_repetition_penalty`, `top_p`/`top_k` —
  inclusive a combinação que a própria documentação do SGLang recomenda),
  dar mais tempo do que o orçamento de isocronia pede (piorou — mais espaço
  pra entrar em loop).
- LoRA fine-tuning (`scripts/pod_lora_train.py`, adapter `ptbr_v2_3150ex`):
  melhorou naturalidade/sotaque de forma real e confirmada pelo usuário
  ouvindo. **Estabilidade/repetição não melhorou claramente** com esse
  primeiro LoRA (corpus genérico, não frases curtas estilo dublagem).
  Decisão da sessão de 2026-09-14: LoRA fica pausado, focar primeiro nos
  problemas estruturais do pipeline (T0.13-T0.16).

Detalhe completo em memória de sessão anterior (fora do repo).

### 2026-09-14 — T0.13-T0.16 concluídos, depois retomada da investigação de qualidade

T0.13 (CER normalizado), T0.14 (flag de truncamento), T0.15 (termo de pausa
no orçamento), T0.16 (contabilizar cortes na montagem) — todos commitados,
ver `git log`. Depois disso, rodou o pipeline completo (T0.12) de novo,
Pod novo (A40, 46GB, `ca-mtl-1`) — ver `.cache/dub_v5/report.json`:

```
"within_duration_tolerance_rate": 0.6875
"segments_with_any_flag_rate": 1.0
"segments_needing_synth_retry_rate": 0.9375
```

93,75% dos segmentos precisando de retry — MUITO acima do portão de decisão
de 30% do T0.12. Inspeção manual de `translations.json`/`eval.json` mostrou
dois padrões: (a) segmentos muito curtos saindo com hipótese de ASR vazia,
(b) loop de repetição clássico (ex.: `"Tá, só isso mesmo."` → `"Tá, tá, tá,
tá..."` ~20x).

Usuário decidiu (consciente, ver "Decisão" abaixo): mexer no código de
geração do MOSS-TTS em si, em vez de só engenharia de pipeline/LoRA.

#### Tentativa 1: `text_repetition_penalty` no canal semântico (canal 0) — FALHOU, revertida

Lendo `modeling_moss_tts.py::MossTTSDelayModel.generate()` (baixado do cache
do Pod pra `.cache/moss_tts_src/`, não versionado): o `repetition_penalty` só
é aplicado nos canais de áudio (VQ), nunca no canal de texto/semântico
(canal 0) — `inference_utils.py::sample_token` já suporta genericamente esse
caso ("Case 1: regular [N, V] (text layer)"), só nunca foi conectado no
`generate()`. Hipótese: o loop de frase inteira nasce nesse canal, não nos
codebooks de áudio — bateria com o padrão observado (repete FRASE, não
artefato de áudio) e explicaria por que `audio_repetition_penalty` (sessão
anterior) nunca ajudou.

**Testado e refutado.** Aplicar `repetition_penalty` no histórico COMPLETO
do canal 0 (prompt inteiro, incluindo tokens estruturais como
`audio_start_token_id`, `im_end_token_id`, `audio_assistant_delay_slot_token_id`)
quebra a máquina de estados do delay pattern — o modelo nunca mais reemite o
marcador estrutural certo, e a saída vira ~0.16s de áudio (praticamente
nada) em 4/4 casos testados. Corrigido restringindo a penalidade a um
histórico separado, só de tokens de conteúdo REALMENTE sampleados (não o
prompt, não os tokens estruturais forçados) — não crashou mais, mas o
resultado real ainda não foi validado como correto (ver "Estado atual").

**Conclusão provisória**: repetição no canal 0 pode ser uma parte NORMAL e
NECESSÁRIA da arquitetura (sustentar um token semântico por várias frames de
áudio, alinhado ao delay pattern), não um bug em si — penalizar isso sem
muito cuidado é perigoso. Ver "Próximos passos".

#### Achado real nº 2, não esperado: um SEGUNDO bug de alucinação do faster-whisper

Investigando os casos de "repetição", medi as rajadas de energia do áudio
real (não confiei só na transcrição) — ex.: `seg0009` (referência "Deixa as
bobagens pra lá.", 5 palavras) tinha só 5-6 rajadas de fala num clipe de
1,84s, fisicamente incompatível com a frase repetida 28x que o Whisper
transcreveu. Ou seja: **o Whisper, mesmo já com
`condition_on_previous_text=False`, ainda aluciona repetição de frase em
alguns clipes curtos** — um bug diferente do já corrigido antes, não do
MOSS-TTS.

Testados sistematicamente no mesmo áudio (`seg0009`, retido pelo pipeline
real): `no_speech_threshold`, `compression_ratio_threshold`,
`hallucination_silence_threshold`, `temperature=0.0`, `vad_filter=True` —
**nenhum mudou nada** (todos continuaram com a frase 28x). `repetition_penalty`
e `no_repeat_ngram_size` (parâmetros padrão de decodificação anti-loop de
LLM de texto, aplicados aqui no decoder do Whisper) **funcionaram**:

| segmento | antes do fix | com `repetition_penalty=1.3, no_repeat_ngram_size=3` |
|---|---|---|
| seg0003 | "Morreram... né?" repetido 2x | limpo, 1x, bate com a referência |
| seg0007 | várias frases repetidas | limpo, 1x, bate com a referência |
| seg0009 | frase 28x | melhorou muito, ainda ecoa parcial (não 100% limpo mesmo com `repetition_penalty=2.0`) |

Corrigido em `scripts/pod_worker.py::cmd_evaluate` (commit `1777cc9`).
2 de 3 casos reais resolvidos por completo, sem tocar em nada do MOSS-TTS.

#### Depois do fix do Whisper: o quadro muda

Rodada nova (`.cache/dub_v6`, ver `_analysis.json` nessa pasta — não
versionado, é dado de teste) com o fix do Whisper: `has_repetition` só
disparou em 1 de 17 segmentos (era 6+ antes). **Boa parte do que parecia
"MOSS-TTS repetindo frase" era alucinação da avaliação, não da síntese.**

Mas a taxa de retry continua alta (14/17). O padrão dominante agora é
diferente:
- Hipótese de ASR **vazia** (4 segmentos: `"é isso aí"`, `"Volte se puder."`,
  `"se não ficarmos juntos"`, `"Eu já dei muita luta pra warrior"`).
- Saída **sem nexo nenhum** com o texto de entrada (`"Me assustou bem
  menos"` → transcrito como `"A CIDADE NO BRASIL"`; `"Tá na hora de encarar
  algo inimaginável"` → pontuação solta e fragmentos).
- **Palavra única "explodindo" em variações**: `"decide"` (1 palavra, alvo
  0,68s) → `"Dessidio Decídio Decígio Decide Decisivo Decida Decido
  Decidi..."` — o modelo parece hesitar entre pronúncias em vez de decidir
  uma.
- Segmentos com fala normal (várias palavras, duração >1-2s) continuam
  saindo bem (CER baixo, sem repetição).

Isso bate com o gap já documentado em `packages/pipeline/calibration.json`
(`known_issues.short_segments_below_5_syllables`) — segmento curto parece
ser onde o MOSS-TTS realmente quebra, não repetição de frase longa.

### 2026-09-14 (continuação) — piso de tokens implementado e validado

Confirmado o mecanismo: `n_vq=32` (config do MOSS-TTS-v1.5, arquivo
`configuration_moss_tts.py`) — a 12,5 Hz isso é **2,56s**, batendo quase
exato com o limiar visto nos dados reais (tudo ≤2,5s ruim, tudo ≥3,8s
limpo na rodada v6). Testado forçando 40 tokens (3,2s) em 3 segmentos
curtos que falhavam na produção: 2 de 3 viraram fala correta e limpa (uma
delas com match quase perfeito). O terceiro (`"decide"`, uma palavra
isolada, sem frase ao redor) continuou falhando mesmo com até 100 tokens
— mas colocando a MESMA palavra dentro de uma frase completa
(`"Ele precisa decide agora."`) o resultado saiu quase perfeito com só 40
tokens. Ou seja: existem dois problemas distintos, não um só —
**duração curta** (resolvido) e **texto isolado sem contexto** (não
resolvido, parece precisar de mais contexto linguístico, não mais tempo).

Implementado em `packages/pipeline/synthesis.py` (commit `af0fc69`):
`MIN_SYNTHESIS_TOKENS` (piso derivado de `n_vq` + 25% de margem, ~40
tokens/3,2s) — `duration_to_tokens`/`adjust_tokens_for_retry` nunca pedem
menos que isso.

**Resultado real, rodada v7 completa** (`.cache/dub_v7`, mesmo vídeo de
teste): comparando a hipótese de ASR do primeiro attempt (`eval.json`)
segmento a segmento contra a referência, **7 de 16 segmentos saíram
limpos (CER baixo, sem repetição, sem truncamento)** já na primeira
tentativa — incluindo segmentos de 0,54s, 1,74s e 2,08s que antes do fix
falhavam sempre. Antes (v6, mesmo vídeo, sem o piso): só 1-2 de 17
limpos. Ganho real, não só teórico.

**Efeito colateral esperado, ainda sem solução**: forçar mais tokens pra
segmentos curtos produz áudio mais longo que o alvo real —
`within_duration_tolerance_rate` caiu de 0,69 (v5) pra 0,19 (v7). O gate
`fora_duracao` pega isso corretamente (não é falha silenciosa), mas ainda
não existe nada que COMPRIMA o áudio de volta pro alvo (time-stretch,
alavanca já prevista em `budget.py::LEVER_ORDER` mas nunca implementada).
Até isso existir, segmento curto vai continuar marcado pra revisão manual
por duração, mesmo com conteúdo correto — melhor que conteúdo errado
E duração errada, mas não é o estado final.

**Achado colateral, não investigado ainda**: alguns segmentos com
conteúdo claramente limpo no attempt 1 (ex. `seg0006`, `seg0009` — CER
0,0, sem repetição) ainda assim aparecem em `synth_retry_2.json`/`3.json`
e o texto final pós-retry é BYTE-IDÊNTICO ao do attempt 1 — sugere um
bug de bookkeeping no loop de retry do `dub.py` (talvez o texto de
referência usado por `_is_bad_synthesis` não é exatamente o mesmo que foi
usado pra calcular o CER que eu chequei manualmente — possível
discrepância em qual candidato de tradução é tratado como "o melhor").
Gasta tempo de GPU à toa, mas não parece corromper o resultado final.
Vale investigar separadamente se sobrar tempo.

## Estado atual (não concluído)

- Fix do Whisper (2 modos de alucinação): commitado, validado em produção
  real, funcionando.
- Piso de tokens (`MIN_SYNTHESIS_TOKENS`): commitado, validado em produção
  real (rodada v7) — melhora clara e medida no conteúdo (7/16 limpos vs
  1-2/17 antes). **Não precisou mexer no `generate()`/código de geração do
  MOSS-TTS** — a causa raiz era como o NOSSO pipeline calculava o orçamento
  de tokens, não a arquitetura do modelo em si.
- Patch do canal de texto (`text_repetition_penalty`): protótipo em
  `scratchpad` da sessão (não commitado, não é parte do repo) — abandonado
  por ora. Depois do fix do Whisper + piso de tokens, a maior parte do que
  parecia repetição já está explicada por outras causas; não há mais um
  caso claro e isolado que precise dessa intervenção especificamente.
- Dois problemas reais ainda sem solução:
  1. **Excedente de duração** — segmento curto forçado ao piso gera áudio
     mais longo que o alvo real; precisa de time-stretch (não implementado)
     ou de aceitar a folga e confiar no gate `fora_duracao` pra revisão
     manual (implementado, mas não é o estado final desejado).
  2. **Texto isolado sem contexto** (ex. uma palavra solta como linha de
     diálogo) — piso de tokens não resolve sozinho; parece precisar de mais
     contexto linguístico ao redor do texto, não mais tempo de áudio.
- Achado colateral não resolvido: bug de bookkeeping no loop de retry do
  `dub.py` fazendo segmentos já limpos serem re-sintetizados à toa (ver
  seção acima).

### 2026-09-14 (continuação) — time-stretch implementado, e uma correção real no limite

Implementado `time_stretch_to_duration` (`packages/pipeline/synthesis.py`,
commit `abdc7d5`) usando `librosa.effects.time_stretch` (phase vocoder,
preserva pitch — nova dependência do projeto). Limite inicial de
`MAX_TIME_STRETCH_RATIO=2.0` foi copiado de uma referência genérica ("faixa
geralmente citada como natural pra voz"), sem medir pra este caso
específico.

**Rodada v8 real** (piso de tokens + stretch): `within_duration_tolerance_rate`
subiu de 0,19 (v7, sem stretch) pra **0,53** — quase triplicou. Mas ao
conferir o CONTEÚDO pós-stretch, vários segmentos vieram vazios/estranhos
no Whisper que antes (sem stretch) provavelmente estariam ok.

**Teste de controle isolado**: peguei um áudio já confirmado limpo
("Ele precisa, decide agora.", transcrito perfeito) e apliquei SÓ o
stretch, sem re-sintetizar nada. Em 2.0x (o limite que eu tinha posto)
virou "Ele precisa desse de agora... desci-de agoram" — ilegível.
Testando 1.2/1.3/1.5/1.7x no mesmo áudio, só **1.5x** manteve o conteúdo
correto e sem repetição real (`"Ele precisa, ele decide agora."`) — 1.7x e
2.0x já degradam. **`MAX_TIME_STRETCH_RATIO` corrigido pra 1.5** (commit
seguinte) — o número antigo era chute de referência genérica, esse é
medido, mesmo que num teste pequeno (n=1 caso, poucos valores).

Isso reforça um padrão desta investigação: sempre medir antes de confiar
num "limite geralmente aceito" — o phase vocoder do librosa em fala curta
sintética se comporta pior que a intuição de "stretch de música" sugere.

**Rodada v9 real** (piso + stretch com o limite corrigido de 1.5x):
`within_duration_tolerance_rate` = 0,375 (era 0,53 com o limite de 2.0x que
degradava conteúdo, e 0,19 sem stretch nenhum). Quadro honesto, por faixa
de alvo:
- Alvo entre ~2,1s e o piso (3,2s): stretch de até 1.5x alcança o alvo
  real E preserva conteúdo — funciona como esperado (ex.: segmentos com
  CER baixo e duração dentro da tolerância).
- Alvo bem abaixo de ~2,1s (a maioria dos casos problemáticos reais, tipo
  0,4-1,1s): precisaria de mais de 1.5x de compressão pra bater o alvo
  exato — trava no limite seguro (2,1867s = 3,2/1,5), fica mais perto mas
  não bate, e `fora_duracao` continua sinalizando corretamente. **Esse não
  é um bug do time-stretch** — é o time-stretch reconhecendo seu próprio
  limite em vez de forçar um resultado ruim.
- Pra esses mesmos casos bem curtos, o CONTEÚDO (independente de duração)
  também segue com problema — vazio, alucinado, ou garbled. Isso é o
  problema separado de "texto isolado sem contexto" (próxima seção),
  não algo que o time-stretch deveria resolver.

### 2026-09-14 (continuação) — fusão por poucas palavras, resultado real (v10)

Usuário confirmou explicitamente: fundir segmento com o vizinho quando tem
poucas palavras, mesmo que dure mais que o piso de 0,3s de
`min_duration`. Implementado em `packages/pipeline/segmentation.py`
(commit `76fb531`): `_merge_short_groups` agora funde também por
`min_word_count` (default 4), reusando o mesmo mecanismo (e o mesmo
tratamento de pausa interna) já usado pra fusão por duração curta.

**Rodada v10 real** (piso de tokens + stretch + fusão por poucas
palavras, tudo junto): 16 segmentos originais viraram **11** (fusão
funcionando — nenhum ficou com menos de 5 palavras). Resultado, primeira
tentativa de síntese, sem contar retry:
- `within_duration_tolerance_rate` = **0,818** (era 0,375 sem a fusão —
  mais que dobrou)
- `segments_needing_synth_retry_rate` = **0,455** (era 0,8125)
- `segments_with_any_flag_rate` = **0,818** (era 1,0)
- **7 de 11 segmentos (63,6%) saíram limpos** (CER baixo, sem repetição,
  sem truncamento) já na primeira tentativa — melhor número desta
  investigação inteira (era 43,75% no melhor resultado anterior, v7).

Os 4 restantes ainda têm problema real (2 vazios, 1 garbled, 1 truncado) —
mas são uma fração muito menor do total, e continuam entrando no loop de
retry já existente (T0.14) pra segunda/terceira tentativa.

**Combinando os três fixes de hoje** (piso de tokens, time-stretch, fusão
por poucas palavras) — nenhum exigiu tocar no código de geração do
MOSS-TTS. A causa raiz real estava em como o NOSSO pipeline decidia
quanto tempo pedir e como segmentava o texto antes de mandar pro modelo,
não na arquitetura do modelo em si.

## Próximos passos (em ordem)

1. ~~Excedente de duração~~ — **feito** (`time_stretch_to_duration`).
2. ~~Texto isolado sem contexto~~ — **feito** (fusão por `min_word_count`,
   confirmado com o usuário, resultado real medido acima).
3. ~~Bookkeeping de retry~~ — **obsoleto**: o mecanismo de retry por
   rodadas que tinha esse bug foi inteiramente substituído (ver próxima
   seção), o código com o bug nem existe mais.
4. Rodar os vídeos do portão de decisão do T0.12 pra medir a taxa de
   correção manual de verdade com todos os fixes juntos (piso de tokens,
   time-stretch, fusão por palavra, lote de candidatos, voz masculina) —
   4 de 5 vídeos de teste feitos (Avengers, LEGO Batman, NFL, Clairo);
   falta o F1 (o maior, `Drivers-React-After-Qualifying`), pausado a
   pedido do usuário pra priorizar o fix do retry primeiro.

### 2026-09-15 — retry por rodadas substituído por lote de candidatos paralelos

Reclamação real do usuário: o retry por rodadas "consome muito, triplica
tempo e custo, ruim pra prototipar". Achado que resolveu: o `generate()`
do MOSS-TTS já trata `batch_size` como dimensão real em toda a máquina de
estados do delay pattern (confirmado lendo o código, não suposição) — 3
amostras independentes da MESMA frase numa chamada em lote levaram
**3,52s**, quase o mesmo tempo de 1 amostra sozinha (2-4s). Isso significa
que gerar várias tentativas em paralelo é essencialmente de graça em GPU —
o custo do retry antigo era quase todo overhead de processo/rede (rodada =
processo novo no Pod + round-trip de avaliação), não geração em si.

Reescrito (`scripts/pod_worker.py::cmd_synthesize`, `scripts/dub.py`):
cada segmento gera `N_CANDIDATES=3` amostras numa chamada só; avalia todas
via Whisper numa única leva; escolhe a melhor localmente
(`pick_best_synthesis`, reusa `_is_bad_synthesis` do T0.14) sem round-trip
extra. **Resultado real** (Avengers, mesmo vídeo testado o dia todo):
pipeline completo caiu de ~20-25min pra **~6 minutos**. Novo campo no
relatório, `segments_needing_non_first_candidate_rate` (45% no teste) —
mostra quantas vezes o candidato #0 sozinho teria saído ruim, tornando
visível o que antes só aparecia como "precisou de retry caro".

Também nesta sessão: `--vocals-only` (CLI, bypass do remix com fundo, só
pra avaliação — não é comportamento padrão), voz de referência trocada
pra um homem adulto real (áudio de 13,84s fornecido pelo usuário,
`312_84sil_0pause.wav`, ~118Hz — ver [[tts_lora_pending_fixes]]), e
`scripts/dub.py` agora salva `<video>_transcript.json` e
`<video>_traducao.json` ao lado de todo vídeo dublado.

### 2026-09-15 — "soa como audiobook": dois fixes sem GPU, não validados ainda

Usuário reportou que a dublagem soa como narração de audiobook, não como
a entrega natural do áudio original. Sem GPU disponível nesta sessão pra
validar contra o modelo real — implementado e testado só com dado
sintético, **pendente de validação real na próxima sessão**:

1. `packages/pipeline/translation.py`: prompt agora pede explicitamente
   pra preservar pontuação de ênfase ("!", "?", "...") do original: novo
   `expressiveness_score` em `rank_candidates` recompensa candidatos que
   preservam essa pontuação (pesos rebalanceados: orçamento 0,7→0,6,
   naturalidade 0,3→0,2, expressividade 0,2 novo).
2. `packages/pipeline/synthesis.py::infer_delivery_instruction`: deriva
   uma instrução de estilo pro campo `instruction` do MOSS-TTS (existe na
   API, nunca tinha sido usado) a partir da mesma pontuação.

Base para os dois: achado já documentado em
`calibration.json.known_issues.flat_delivery_on_concatenated_clauses` —
o MOSS-TTS não tem parâmetro explícito de tom, "!" parece ser o único
sinal implícito de emoção que ele usa. **Não medido se isso realmente
muda o áudio gerado** — só a lógica de seleção/prompt está testada.
Próximo passo real: rodar contra o modelo de verdade e ouvir a diferença.

Não implementado ainda (mais esforço, precisa do Pod): extrair
energia/pitch do áudio original (stem de voz já separado pelo Demucs) e
mapear pra `instruction` ou pra pós-processar o volume do segmento
sintetizado seguindo o contorno original.

### 2026-09-18 — naturalidade: validado contra o modelo real (não só sintético)

As duas correções de 2026-09-15 (`expressiveness_score` na tradução,
`infer_delivery_instruction` na síntese) tinham só teste unitário com
dado sintético — nunca tinham rodado contra o MOSS-TTS de verdade.
Testado agora com 3 jobs curtos (texto neutro, com "!", com "?") direto
via `pod_worker.py synthesize`: os 3 rodaram sem erro, todos dentro da
tolerância de duração. Confirma que passar `instruction` pro
`processor.build_user_message` não quebra o `generate()` real (a única
coisa que era, de fato, incerta sem GPU — a lógica de seleção de texto
já estava testada). **Ainda não confirmado por ouvido humano se o tom
realmente muda** — isso não dá pra validar sem alguém ouvir o áudio.

### 2026-09-18 — Pod novo (3º da sessão), rebuild validado, achado novo de torchcodec

Gap de sessão de alguns dias; Pod anterior fechado pelo usuário. Toda vez
que isso acontece o volume de rede vem vazio de novo (region-locked, não
persiste entre Pods) — reconstruído do zero seguindo a mesma receita já
validada (venv `asr` com `numpy<2` + pins de `pyannote.*`, venv `tts` com
`librosa` incluído desde o início desta vez).

**Achado novo:** `pip install torchcodec --index-url .../cu128` sem
versão pinada resolveu pra `torchcodec==0.11.1+cu128` — essa versão falha
ao carregar (`Could not load libtorchcodec`, tenta ffmpeg 5-8 que não
existem no sistema, e a variante ffmpeg4 dá
`undefined symbol: torch_dtype_float4_e2m1fn_x2`, incompatível com torch
2.9.1). Confirmado que `torchcodec==0.8.1` (a mesma versão já validada
numa sessão anterior, mesmo par torch/cu128) funciona normalmente. Pino
explícito na receita de rebuild a partir de agora: `pip install
torchcodec==0.8.1 --index-url https://download.pytorch.org/whl/cu128` —
não confiar no resolver do pip pra essa lib, ela não é retrocompatível
entre versões torch como o normal do PyPI.

Smoke test rodado após o fix (`pod_worker.py synthesize` com 1 job curto,
3 candidatos): pipeline completo funcionou — geração em lote, retry por
tolerância, compressão por time-stretch no piso de duração, tudo
disparou como esperado (`tokens_used=40` = piso, `attempts=2`,
`stretched=true`, `within_tolerance=true` nos 3 candidatos). Ambiente
confirmado pronto para retomar o trabalho pendente (F1, validação das
duas correções de naturalidade contra o modelo real, etc.).

### 2026-09-18 — sessão de "empurrar pra perto de 100%": achados reais, um sério

Usuário pediu pra investigar/testar de verdade em cima do resultado do
beast.mp4 (63,2% dos segmentos com alguma flag), pensando fora da caixa,
e alertou explicitamente: qualquer constante calibrada num único vídeo de
teste pode não generalizar pro público geral (todo tipo de vídeo).

1. **N_CANDIDATES 3→5** (`pod_worker.py`): mudança de baixo risco (mais
   amostra independente só ajuda), pedida e testada primeiro.

2. **Erro de design corrigido, apontado pelo usuário**: a primeira reação
   ao achar o caso legítimo de overflow (último segmento curto colado no
   fim do vídeo) foi subir `assembly.MAX_DISCARDED_SECONDS` de 0,5s pra
   2,5s no agregado. Isso esconderia bug real espalhado por vários
   segmentos pequenos (ex.: 3 segmentos com 0,8s de erro cada). Corrigido
   pra dois gates ortogonais: teto por segmento individual (derivado do
   piso físico do modelo, `MIN_SYNTHESIS_SECONDS/MAX_TIME_STRETCH_RATIO`),
   e no máximo 1 segmento pode ter qualquer descarte — o caso legítimo só
   pode afetar um. Ver `[[feedback_generalize_constants]]` na memória.

3. **`segmentation.py` ganhou o piso real de síntese**: `dub.py` agora
   chama `segment_words(words, min_duration=MIN_SYNTHESIS_SECONDS)` em vez
   do default antigo (0,3s) — os dois valores tinham ficado fora de
   sincronia desde que o piso foi medido (T0.9). Elimina os `fora_duracao`
   estruturais (segmento curto que nunca cabe no piso do MOSS-TTS, não
   importa o candidato).

4. **Achado sério: bug de parsing de tradução, não instabilidade do
   MOSS-TTS.** Testando com 5 candidatos + segmentos maiores (efeito do
   merge do item 3), 4 segmentos saíram com TODOS os 5 candidatos de
   síntese consistentemente ~1,9-2,3x mais longos que o alvo — parecia
   instabilidade do modelo, mas tinha variância baixa demais pra ser
   estocástico. Investigando a tradução: `parse_candidates()` fazia
   `str(c) for c in candidates` sem checar o tipo — quando o Qwen aninhava
   cada candidato numa lista de 1 elemento (`[["texto"]]` em vez de
   `["texto"]`), isso virava o texto LITERAL `"['texto']"` sendo mandado
   pro MOSS-TTS sintetizar (colchetes, aspas, e às vezes cortado no meio
   da palavra). O MOSS-TTS não tinha culpa nenhuma — estava sintetizando
   lixo de entrada corretamente. Corrigido: `parse_candidates` agora
   valida que cada candidato é `str` antes de aceitar (levanta `ValueError`
   se não for, o retry existente em `cmd_translate` resolve reamostrando).
   De brinde, trocado o regex guloso `\{.*\}` (que pega do primeiro `{` até
   o ÚLTIMO `}` da resposta inteira, vulnerável a qualquer chave extra
   depois do JSON de verdade) por uma extração por contagem de chaves, que
   para no par certo do primeiro `{`.

   **Lição maior que a correção em si**: um sintoma que parece
   "instabilidade estocástica do modelo de voz" pode ser, na real, um bug
   de parsing silencioso rio acima — vale sempre checar o TEXTO de entrada
   antes de suspeitar do modelo, especialmente quando o padrão é
   consistente demais pra ser aleatório (todos os 5 candidatos concordando
   de perto não é cara de estocástico).

5. **Overlap entre segmentos na timeline, nunca detectado antes**:
   `place_segments_on_timeline` só verificava estouro no FIM do vídeo;
   nunca verificava se a síntese de um segmento (mais longa que o
   esperado) invadia o espaço do PRÓXIMO segmento. No caso real do item 4
   (antes do fix), um segmento de 10,3s saiu com 52,9s e teria se
   sobreposto a CINCO segmentos seguintes — silenciosamente, sem erro, sem
   flag, corrompendo quase metade do vídeo com áudio somado/ininteligível.
   Corrigido: cada segmento agora é cortado no que vier primeiro (início
   do próximo segmento ou fim do vídeo), sujeito aos mesmos dois gates do
   item 2. Validado em produção nesta mesma sessão: com o bug de tradução
   ainda ativo, esse gate disparou corretamente (`TimelineOverflowError`,
   46,4s descartados em 6 segmentos) em vez de deixar passar áudio
   corrompido — a rede de segurança funcionou exatamente como desenhada.

### 2026-09-19 — causa raiz real do "MOSS-TTS instável" achada e corrigida

Pedido explícito do usuário: parar de contornar e achar a causa raiz de
verdade do bug de repetição/instabilidade, essencial pra produção. Sessão
de investigação controlada direto no Pod, lendo o `generate()` real do
modelo (`modeling_moss_tts.py`, baixado e lido por completo) em vez de só
tentar parâmetros às cegas.

**Mecanismo real do "loop"**: o modelo só para quando amostra
`im_end_token_id` no canal de texto (canal 0). Esse canal é amostrado com
`text_temperature=1.5` por padrão, independente de `audio_temperature=1.7`
(controla o conteúdo/voz). `im_end_token_id` fica banido de amostragem nos
primeiros `n_vq` (32) passos — daí o piso de duração já conhecido.

**Hipótese 1 (testada, refutada): temperatura da decisão de parar.**
Isolei `text_temperature` de `audio_temperature` (nunca tinha sido testado
separado antes, achados anteriores só mexiam nos dois juntos ou em
parâmetros de áudio). Resultado no texto real que falhava em produção
(seg0011, beast.mp4, 10 amostras por condição):
- Padrão (text_temp=1,5): 10/10 loop, 9,3-36,5s pra alvo de 4,6s.
- text_temp=0,5: 10/10 loop, PIOR (até 79,4s).
- text_temp=0 (greedy): 10/10 loop, AINDA PIOR (6/10 bateram exatamente no
  teto de 1024 passos = 79,36s).

Conclusão: baixar a temperatura da decisão de parar piora, não ajuda — o
argmax do modelo nesse ponto genuinamente "quer" continuar, então tirar a
aleatoriedade só remove a única chance de escapar por sorte.

**Hipótese 2 (testada, parcialmente confirmada): orçamento de tokens
impossível.** O job real usava `tokens=40` (piso, depois do retry ter
encolhido de 57 pra 40 baseado numa tentativa 1 que provavelmente já
tinha alucinado). Testei o MESMO texto com tokens=40 vs 57 vs 100:
- tokens=40: 10/10 "loop" (>1,5x alvo), 9,3-36,5s — variância enorme.
- tokens=57: 7/10 "loop", mas MUITO mais estável (6,1-10,2s).
- tokens=100 (~8s): 10/10 tecnicamente ">1,5x" do alvo de 4,6s, mas
  variância baixíssima (7,7-11,5s) — não é mais "loop" instável, é o
  modelo convergindo pra uma duração natural PRÓPRIA dele pro texto, que
  não muda muito com o orçamento pedido.

Achado: o modelo parece ter uma duração "natural" pra esse texto
(~7-10s) bem acima do orçamento pedido (3,2-4,6s) — mais tokens não
"conserta" a duração, só estabiliza a variância. Isso apontou pra: por
que o texto "quer" 7-10s pra um orçamento calibrado pra ~4,6s?

**Hipótese 3 (testada, CONFIRMADA — causa raiz real):
`infer_delivery_instruction`.** Os 2 casos catastróficos reais da sessão
(seg0006, seg0011) tinham em comum texto cheio de "!", disparando a
instrução "fale com ênfase e emoção genuína, não leia como narração
neutra de audiobook" (adicionada em 2026-09-15, T0.12c, explicitamente
marcada como "não validada contra o modelo real ainda"). Teste direto,
mesmo texto/tokens=40, só removendo a instrução:
- COM instrução: 10/10 loop, 9,3-36,5s.
- SEM instrução: **0/10 loop**, todas as 10 amostras em 2,88-3,44s — quase
  exatamente o orçamento pedido (3,2s nominal pra tokens=40).

Resultado limpo e 100% reprodutível. O modelo interpreta "fale com ênfase
e emoção genuína" como licença pra falar bem mais devagar/expansivo — sob
orçamento apertado (comum: frases de ênfase tendem a ser ditas mais
rápido na fala real, não mais devagar), isso vira um pedido
estruturalmente impossível de caber no tempo, e a "instabilidade do
MOSS-TTS" caçada a sessão inteira era o modelo reagindo mal a um pedido
impossível — não um bug aleatório do modelo em si.

**Correção aplicada**: `pod_worker.py::cmd_synthesize` não chama mais
`infer_delivery_instruction` (`instruction = None` sempre, com comentário
explicando o achado). A função continua existindo e testada em
`synthesis.py` (a lógica de detecção de pontuação está correta, só a
APLICAÇÃO incondicional era o problema) — uma versão futura pode
redesenhar isso validando o custo de tempo real antes de aplicar
qualquer instrução de estilo, em vez de aplicar sempre que houver "!"/"?".

**Nota**: as duas outras correções desta sessão (compressão universal
pós-retry, escalação de candidatos extra) continuam válidas e úteis —
ajudam em casos de instabilidade genuína do modelo (rara, documentada
pelo SGLang) que não têm a ver com a instrução de ênfase. A diferença é
que agora sabemos que a MAIORIA dos casos catastróficos observados nesta
sessão tinha uma causa raiz identificável e corrigível, não eram só "sorte
ruim" do modelo.

## Decisão (2026-09-14, confirmada explicitamente com o usuário)

- Mexer no código de geração do MOSS-TTS em si é uma escolha consciente,
  sabendo que foge do escopo normal de engenharia de pipeline, pode levar
  dias, e tem risco real de não funcionar ou piorar casos que hoje
  funcionam. Prioridade: raiz do problema > velocidade.
- Escopo do produto continua EN→pt-BR. "Diversos idiomas" foi motivação
  pra investir na correção de raiz, não um novo requisito de multi-idioma
  agora.
