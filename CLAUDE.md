# CLAUDE.md

Contexto permanente do projeto. Leia antes de qualquer tarefa.

## O que é este projeto

SaaS de dublagem automática de vídeo, inglês → português brasileiro, para YouTubers de médio e grande porte que querem alcançar o público brasileiro.

O usuário faz upload de um vídeo, o sistema dubla automaticamente, e um **editor web** permite corrigir segmento a segmento antes de exportar. O produto não promete perfeição automática — promete erro visível e corrigível em dois cliques.

Fundador solo. Prioridade absoluta: **simplicidade operacional e custo baixo em ociosidade**.

## Invariantes de arquitetura (não viole sem me perguntar)

1. **O segmento é a unidade atômica.** Toda operação de dublagem acontece no nível do segmento. Nunca reprocesse um vídeo inteiro para corrigir uma parte.

2. **Toda etapa do pipeline é um job idempotente e retomável.** Separação de fontes, ASR e diarização são caros — rode uma vez, persista o resultado, nunca refaça sem flag explícita `force=True`.

3. **Ressíntese de um segmento isolado deve levar < 5 segundos com o worker quente.** O cold start é medido e reportado à parte (T1.2a), nunca embutido escondido nessa meta. Se uma mudança tornar o caminho quente mais lento, pare e me avise.

4. **Nunca destrua a versão anterior de um segmento.** Edições criam uma nova revisão; o histórico é preservado.

5. **Edição de um segmento não altera os vizinhos automaticamente.** Mostre o impacto na timeline, mas não "conserte" nada sozinho. O usuário perde confiança quando coisas mudam sem ele pedir.

6. **`org_id` em toda tabela de domínio**, desde o primeiro dia. Toda query filtra por ele. Não construa gestão de times ainda, mas não pinte o multi-tenant num canto.

7. **Nada de GPU ociosa fora de uso ativo.** Todo trabalho de GPU roda em RunPod Serverless (cobrança por segundo, escala a zero). Não introduza serviços que exijam GPU ligada esperando requisição.
   - Única exceção de produto: o endpoint `synthesize` pode manter um worker quente **enquanto houver uma sessão de editor aberta**, com desligamento por ociosidade. Sem isso, a invariante 3 é impossível logo depois de um cold start.
   - Exceção de desenvolvimento: até entrar em produção, todo trabalho de GPU roda num Pod via SSH (ver "Ambiente de GPU atual"). Desligue o Pod ao fim de cada sessão. A migração para Serverless é a T1.2a, antes de qualquer usuário externo.

## Stack (travada — não substitua sem me perguntar)

| Camada | Tecnologia |
|---|---|
| Frontend | Next.js (App Router) + TypeScript + Tailwind, na Vercel |
| API | FastAPI (Python 3.12), em Railway |
| Banco / Auth | Supabase (Postgres + Auth) |
| Storage de mídia | Cloudflare R2 (S3-compatible, egresso grátis) |
| GPU / workers de ML | RunPod Serverless (containers Docker, um endpoint por função) |
| Fila | Postgres, `SELECT ... FOR UPDATE SKIP LOCKED` |
| Pagamento | Asaas (PIX e boleto) |
| Erros | Sentry |

**Não introduza:** Kubernetes, Airflow, Temporal, Celery, RabbitMQ, Kafka, microserviços, GraphQL, ORM pesado além do SQLAlchemy Core, ou qualquer AWS service com taxa de egresso.

## Modelos de ML usados

| Etapa | Modelo | Onde roda |
|---|---|---|
| Separação voz/fundo | Demucs (htdemucs) | RunPod |
| ASR + timestamps por palavra | WhisperX (large-v3) | RunPod |
| Diarização | pyannote (via WhisperX) | RunPod |
| Tradução | Qwen2.5-7B-Instruct (self-hosted) | RunPod |
| TTS | MOSS-TTS-v1.5 (OpenMOSS-Team, 8B, Apache-2.0) | RunPod |
| Avaliação (CER) | Whisper large-v3 | RunPod |
| Comparação de emoção | emotion2vec_plus_large (FunASR) — opt-in, não calibrado | RunPod |
| Similaridade de locutor | WavLM-TDNN / ECAPA — **ainda não implementado** (`locutor_suspeito` é stub) | RunPod |

Hoje tudo isso roda num Pod de desenvolvimento via SSH, não em Serverless. Ver "Ambiente de GPU atual".

**Detalhes do MOSS-TTS que importam para o código:**
- Controle de duração: parâmetro `tokens` em `processor.build_user_message(text=..., tokens=N)`.
- Frame rate do tokenizer: 12,5 Hz → `tokens = round(segundos * 12.5)`. Granularidade de 80 ms.
- Pausas explícitas: marcador inline `[pause 3.2s]` no texto. Elas contam **dentro** do orçamento de `tokens` (achado da T0.9); não some a duração da pausa por cima.
- Pronúncia forçada: aceita IPA entre barras, ex. `/oʊˈpɛn aɪ/`.
- Tag de idioma: `language="Portuguese"` — atenção, o modelo tende ao pt-PT; ver `docs/ptbr.md`.
- Sob orçamento apertado o modelo **trunca o texto** para caber. O segmento truncado bate a duração, então `fora_duracao` não detecta truncamento; use a flag lexical `truncado` (T0.14).
- Campo `instruction` (estilo/emoção): **sempre `None`**. Com orçamento de `tokens` apertado, uma instrução de ênfase faz o modelo alongar a fala e pode entrar em loop de geração: 10/10 amostras em loop de 9–36 s para um alvo de ~4,6 s, contra 0/10 sem instrução (ver `docs/moss_tts_investigation.md`, 2026-09-19). Fala instruída sai naturalmente 1,7–2,4x mais longa que o orçamento normal. Não reative fora da T0.29.
- Voz de referência: hoje é uma voz pt-BR sintética **fixa** para todo segmento e todo locutor (`POD_VOICE_REFERENCE`, default `voices/default_pt_br.wav`). Sem referência, o modelo sorteia um timbre por chamada. A referência também carrega o tom da fala: referência neutra puxa a entrega para o neutro (ver T0.29).
- Piso de duração: `MIN_SYNTHESIS_SECONDS = 3,2 s`, e segmentos menores são fundidos com o vizinho antes da tradução. A explicação registrada (delay pattern, `n_vq=32`) ainda não foi separada do efeito de "texto curto sem contexto" (T0.26).
- Teto de geração: `max_new_tokens = max(1024, 6 × tokens)`. Com segmentos de até 12 s, o piso de 1024 sempre vence (T0.24).
- Adapter LoRA experimental (`scripts/pod_lora_train.py`, `POD_LORA_ADAPTER_PATH`): melhorou sotaque, não estabilidade. Pausado.

**Pós-processamento de áudio (achados medidos):**
- Transferência de contorno de pitch (phase vocoder) é **prejudicial**: quase dobra o CER round-trip, até 5,5x no pior segmento. Desligada por padrão; `--pitch-transfer` só para experimento.
- Transferência de envelope de energia é neutra em CER e fica ligada (`--prosody-transfer`).
- `time_stretch_to_duration` (phase vocoder) tem teto de 1,5x.

## Ambiente de GPU atual (Fase 0, temporário)

**Decisão:** enquanto não houver produção, todo trabalho de GPU roda num Pod via SSH. Serverless (invariante 7) é o alvo de produção e entra na T1.2a. Hoje:

- `runpod/*/handler.py` está **congelado**: escrito em 2026-09-12, nunca validado e desatualizado. Não edite nem faça deploy. A fonte de verdade da lógica de modelo é `scripts/pod_worker.py` até a T1.2a.

- `scripts/dub.py` roda na máquina local e orquestra via SSH. `scripts/pod_worker.py` roda num RunPod Pod e faz as chamadas de modelo, com um subcomando por estágio: `separate_stems`, `transcribe`, `translate`, `synthesize`, `evaluate`, `match_emotion`.
- Receita estável: GPU **A40** (Ampere, sem MIG). Duas venvs: ASR com `torch 2.2.2+cu121`; TTS com `torch 2.9.1+cu128`.
- Não use pods com partição MIG: houve um bug de PyTorch/NVML específico de MIG (num pod Blackwell) sem correção possível.
- Isso viola a invariante 7 de propósito, como harness de validação. Desligue o Pod ao fim de cada sessão.
- Se o Pod estiver inacessível (timeout de SSH), pare e me avise. Não tente contornar.

## Layout do repositório

```
/apps
  /web              Next.js — landing, dashboard, editor
  /api              FastAPI — REST, auth, orquestração de jobs
/packages
  /pipeline         Lógica de dublagem pura (sem I/O de rede)
    segmentation.py   Segmentação prosódica
    budget.py         Orçamento de duração, contagem de sílabas pt-BR
    translation.py    Tradução com restrição de comprimento
    synthesis.py      Wrapper do TTS, mapeamento duração→tokens
    assembly.py       Montagem da timeline, mix, remux
    quality.py        Gates de qualidade e flags
    prosody.py        Envelope de energia e contorno de pitch: extração e transferência
    scorecard.py      Agregação por vídeo: taxas de flag, métricas médias, piores segmentos
    diagnostics.py    Base flag → causa raiz → correção (congelada até a T0.28)
    report.py         Dashboard HTML local
    subtitles.py      Legenda pt-BR (SRT) a partir dos segmentos
    calibration.json  Coeficientes da regressão de duração (T0.6; revisão na T0.15)
  /ptbr             Camada de texto pt-BR
    normalize.py      Números, moeda, datas, siglas, ordinais
    syllables.py      Contador de sílabas
    g2p.py            Fonemização e dicionário IPA
/runpod             Entrypoints Serverless, um subdiretório por endpoint.
                    Escritos, nunca validados, desatualizados. Congelados até a T1.2a.
  /separate_stems   Dockerfile + handler.py (Demucs)
  /transcribe       Dockerfile + handler.py (WhisperX + pyannote)
  /translate        Dockerfile + handler.py (Qwen2.5-7B-Instruct)
  /synthesize       Dockerfile + handler.py (MOSS-TTS)
  /evaluate         Dockerfile + handler.py (Whisper large-v3, CER; emotion2vec a definir)
/migrations         SQL versionado
/evals              Harness de avaliação pt-BR
/scripts            CLI de desenvolvimento
  dub.py            CLI end-to-end; na Fase 0 orquestra o Pod via SSH
  pod_worker.py     Roda no Pod; um subcomando por estágio
  pod_runner.py     SSH/scp para o Pod; configuração via POD_* no .env
  pod_lora_train.py Treino do adapter LoRA (experimental, pausado)
  calibration_*.py  Coleta e ajuste da calibração de duração (T0.6/T0.15)
  r2.py             Download do bucket R2
/phase1test         Transcrições, traduções e scorecards de execuções reais
/tests              Testes do CLI (dub.py)
/docs               Investigações e notas (ptbr.md, moss_tts_investigation.md)
/reports            Saída local: dashboard.html + history.json (append-only)
```

**Regra de dependência:** `packages/pipeline` e `packages/ptbr` são puros — sem chamadas de rede, sem acesso a banco, sem dependência de framework. Recebem dados, devolvem dados. Isso os torna testáveis sem GPU e sem infraestrutura. A orquestração vive em `apps/api`.

## Comandos

```bash
make dev          # sobe api + web local
make test         # pytest + vitest
make lint         # ruff + eslint + tsc --noEmit
make migrate      # aplica migrations
make eval         # roda o harness pt-BR
make pipeline-cli VIDEO=path/to.mp4   # pipeline completo, local
```

Antes de dizer que uma tarefa está pronta, rode `make lint && make test`.

## Avaliação de qualidade

- Flags por segmento em `quality.py`; agregação em `scorecard.py`. `traducao_infiel` e `locutor_suspeito` ainda são stubs.
- **A métrica do portão da Fase 0 é a taxa de correção medida por rótulo humano (T0.28), não `any_flag_rate`.** As flags são proxies até serem calibradas contra rótulos.
- Todos os limiares são pontos de partida não calibrados: `CER_THRESHOLD=0.15`, `EMOTION_SIMILARITY_THRESHOLD=0.5`, `PROSODY_CORRELATION_THRESHOLD=0.3`, bandas do veredito 20%/50%. **Nunca ajuste um limiar contra um vídeo só** — o produto recebe uploads arbitrários.
- Problemas conhecidos de medição:
  - O "CER final" transcreve o segmento **antes** da mixagem, não depois. Por isso saiu idêntico ao CER do candidato (T0.19).
  - O `evaluate` decodifica com `repetition_penalty=1.3` e `no_repeat_ngram_size=3` para conter alucinação do Whisper; efeito colateral: o ASR não consegue transcrever repetição real, e `has_repetition` fica praticamente cego para loop do TTS (T0.19).
  - A correlação de prosódia é medida depois da transferência de envelope, então mede em parte a própria correção (T0.20).
  - `packages/ptbr/normalize.py` só é usado no CER, não no texto que vai para o TTS (T0.22).
- `diagnostics.py` está congelado: não adicione entradas antes da T0.28.

## Convenções

- Python: ruff (linha 100), type hints obrigatórios em funções públicas, pydantic para fronteiras de dados.
- TypeScript: strict. Sem `any`. Sem `!` non-null assertion.
- SQL escrito à mão em migrations numeradas. Sem autogeração de schema.
- Toda função de `packages/pipeline` tem teste unitário com dado sintético (sem GPU).
- Mensagens de commit: imperativo, em inglês, escopo prefixado — `pipeline: add syllable-aware duration budget`.
- Commit ao fim de cada tarefa. Nada fica fora do git entre sessões.
- Nomes de domínio em inglês no código; conteúdo pt-BR só em dados e UI.

## Glossário de domínio

- **Segmento** — bloco de fala delimitado por pausa real no áudio original. Unidade de tradução, síntese e edição. Segmentos abaixo do piso de síntese são fundidos com o vizinho, mas a fusão nunca atravessa troca de locutor (T0.21).
- **Isocronia** — restrição de que o áudio dublado caiba na duração do original.
- **Orçamento de duração** — quantas sílabas cabem num segmento, dado o alvo temporal.
- **On-screen** — locutor visível em quadro; exige tolerância de duração mais apertada (±8% vs ±20%).
- **Flag** — marcação automática de suspeita de erro num segmento; alimenta o editor.
- **Stem** — faixa separada (voz ou fundo) produzida pelo Demucs.
- **Perfil de voz** — voz clonada persistente associada a um canal, reutilizada entre vídeos.
- **Glossário do canal** — termos recorrentes com tradução e pronúncia IPA fixas, por canal.
- **CER round-trip** — sintetizar, transcrever com ASR, comparar com o texto de entrada.
- **CER final** — CER do áudio retranscrito depois do pós-processamento, separado do CER usado para escolher o candidato. Hoje mede o segmento antes da mixagem; o alvo é medir o áudio mixado entregue (T0.19).
- **Taxa de correção** — fração dos segmentos que um humano corrigiria, medida por rótulo. É a métrica do portão da Fase 0. Não confundir com `any_flag_rate`, que é a fração com alguma flag automática.
- **Instrução (MOSS)** — campo de texto livre do MOSS-TTS para estilo e emoção. Desligado; ver detalhes do MOSS-TTS.
- **Harness de GPU** — o par `dub.py` + `pod_worker.py` via SSH da Fase 0. Temporário.

## O que NÃO construir

Estas coisas vão parecer urgentes. Não são. Não implemente sem eu pedir explicitamente:

- Lip-sync facial / manipulação de vídeo
- Dublagem ao vivo/streaming — validar o produto assíncrono (upload → dublagem em lote → editor) primeiro. Live dub é uma v2 possível, arquitetura bem diferente (ASR incremental, tradução e TTS em streaming), não uma extensão incremental deste pipeline.
- Outros pares de idiomas além de EN→pt-BR
- App mobile, API pública, webhooks, integrações
- Gestão de times, papéis, permissões granulares
- Editor colaborativo em tempo real
- Fine-tune de modelo (fase posterior, feita à mão fora do repo). Exceção já existente: o experimento LoRA em `scripts/pod_lora_train.py`, pausado — não expandir.
- Qualquer dashboard de analytics além do básico

## Conformidade (obrigatório, não é backlog)

- Todo upload exige declaração de que o usuário tem direito sobre as vozes do vídeo. Persista o registro com timestamp.
- Voz pode ser dado pessoal sensível (LGPD art. 5º, II). Defina retenção e permita exclusão.
- Watermark de conteúdo sintético em todo áudio gerado (EU AI Act art. 50, vigente desde 2/8/2026). Tem que existir antes de qualquer áudio chegar a uma pessoa de fora: implementação na T1.5a, antes do dashboard da T1.6.
- Nunca logue conteúdo de vídeo do cliente em serviços de terceiros.

## Como trabalhar comigo

- Tarefa por vez, seguindo `TASKS.md` na ordem.
- Antes de começar uma tarefa, releia os critérios de aceite dela.
- Se uma tarefa parecer exigir violar um invariante, pare e me pergunte em vez de contornar.
- Se precisar de uma decisão de produto que não está aqui, pergunte — não invente.
- Não refatore código fora do escopo da tarefa atual.
