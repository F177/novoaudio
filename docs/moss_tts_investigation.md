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

## Próximos passos (em ordem)

1. ~~Decidir e implementar o que fazer com o excedente de duração~~ —
   **feito** (`time_stretch_to_duration`, commits `abdc7d5`/`313c870`).
   Funciona bem pra alvo entre ~2,1s e o piso; abaixo disso, trava no
   limite seguro e deixa `fora_duracao` sinalizar — comportamento
   pretendido, não pendência.
2. **Atual**: "texto isolado sem contexto" (uma palavra/frase muito curta
   solta, ex. `"decide"`, `"milagre."`). `packages/pipeline/segmentation.py`
   já tem um mecanismo de fundir segmento curto com vizinho
   (`_merge_short_groups`, `DEFAULT_MIN_DURATION=0.3`) — mas o piso dele
   (0,3s) é sobre TIMING do vídeo original, não sobre o que o MOSS-TTS
   consegue sintetizar bem; os casos problemáticos de hoje (0,4-1,5s) estão
   todos ACIMA desse piso e passam direto como segmento isolado. Decisão
   de arquitetura pendente (não é só engenharia): fundir segmento baseado
   em limite de SÍNTESE, não de timing, muda o significado de "segmento"
   pro resto do pipeline (afeta T0.5, isocronia por segmento, e o
   invariante de "segmento é a unidade atômica" do CLAUDE.md) — não decidir
   isso sozinho sem confirmar com o usuário.
3. Investigar o achado colateral do bookkeeping de retry (segmentos limpos
   sendo re-sintetizados sem necessidade) — desperdício de GPU, não parece
   corromper o resultado mas vale entender.
4. Rodar os 10 vídeos do portão de decisão do T0.12 só depois do item 2,
   pra medir a taxa de correção manual de verdade.

## Decisão (2026-09-14, confirmada explicitamente com o usuário)

- Mexer no código de geração do MOSS-TTS em si é uma escolha consciente,
  sabendo que foge do escopo normal de engenharia de pipeline, pode levar
  dias, e tem risco real de não funcionar ou piorar casos que hoje
  funcionam. Prioridade: raiz do problema > velocidade.
- Escopo do produto continua EN→pt-BR. "Diversos idiomas" foi motivação
  pra investir na correção de raiz, não um novo requisito de multi-idioma
  agora.
