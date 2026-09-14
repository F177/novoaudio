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

## Estado atual (não concluído)

- Fix do Whisper: commitado, validado em produção real, funcionando.
- Patch do canal de texto (`text_repetition_penalty`): protótipo em
  `scratchpad` da sessão (não commitado, não é parte do repo ainda) — corrigido
  pra não quebrar a máquina de estados, mas ainda não validado se o RESULTADO
  é correto (áudio soa bem? resolve os casos reais de repetição real, se
  ainda existirem depois do fix do Whisper?).
- Hipótese corrente: o problema real agora é **segmento curto** (poucas
  palavras, <1-2s de alvo), não repetição de frase — precisa de investigação
  própria, provavelmente olhando como o delay pattern se comporta quando o
  orçamento de tokens é muito pequeno (poucos frames pra completar o ciclo
  de delay entre os `n_vq` canais de áudio).

## Próximos passos (em ordem)

1. Confirmar quantos dos 14/17 "ruins" da rodada v6 são de fato segmento
   curto (correlacionar `target_seconds`/contagem de palavras com o tipo de
   falha) — separar sinal de ruído antes de mais uma rodada de patch.
2. Investigar a mecânica do delay pattern especificamente pra alvo de
   token curto — ver se `n_vq` frames de delay cabem no orçamento, ou se o
   modelo é forçado a "atropelar" o ciclo normal quando `tokens` é pequeno.
3. Só depois disso, decidir se meche mais no `generate()` (e com qual
   intervenção específica, orientada pelo achado acima) ou se um LoRA
   direcionado a frases curtas (ver memória da sessão anterior) resolve
   sem precisar de cirurgia na arquitetura.

## Decisão (2026-09-14, confirmada explicitamente com o usuário)

- Mexer no código de geração do MOSS-TTS em si é uma escolha consciente,
  sabendo que foge do escopo normal de engenharia de pipeline, pode levar
  dias, e tem risco real de não funcionar ou piorar casos que hoje
  funcionam. Prioridade: raiz do problema > velocidade.
- Escopo do produto continua EN→pt-BR. "Diversos idiomas" foi motivação
  pra investir na correção de raiz, não um novo requisito de multi-idioma
  agora.
