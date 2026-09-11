# CLAUDE.md

Contexto permanente do projeto. Leia antes de qualquer tarefa.

## O que é este projeto

SaaS de dublagem automática de vídeo, inglês → português brasileiro, para YouTubers de médio e grande porte que querem alcançar o público brasileiro.

O usuário faz upload de um vídeo, o sistema dubla automaticamente, e um **editor web** permite corrigir segmento a segmento antes de exportar. O produto não promete perfeição automática — promete erro visível e corrigível em dois cliques.

Fundador solo. Prioridade absoluta: **simplicidade operacional e custo baixo em ociosidade**.

## Invariantes de arquitetura (não viole sem me perguntar)

1. **O segmento é a unidade atômica.** Toda operação de dublagem acontece no nível do segmento. Nunca reprocesse um vídeo inteiro para corrigir uma parte.

2. **Toda etapa do pipeline é um job idempotente e retomável.** Separação de fontes, ASR e diarização são caros — rode uma vez, persista o resultado, nunca refaça sem flag explícita `force=True`.

3. **Ressíntese de um segmento isolado deve levar < 5 segundos.** Se uma mudança tornar isso mais lento, pare e me avise.

4. **Nunca destrua a versão anterior de um segmento.** Edições criam uma nova revisão; o histórico é preservado.

5. **Edição de um segmento não altera os vizinhos automaticamente.** Mostre o impacto na timeline, mas não "conserte" nada sozinho. O usuário perde confiança quando coisas mudam sem ele pedir.

6. **`org_id` em toda tabela de domínio**, desde o primeiro dia. Toda query filtra por ele. Não construa gestão de times ainda, mas não pinte o multi-tenant num canto.

7. **Nada de GPU ociosa.** Todo trabalho de GPU roda em Modal (serverless, cobrança por segundo). Não introduza serviços que exijam GPU ligada esperando requisição.

## Stack (travada — não substitua sem me perguntar)

| Camada | Tecnologia |
|---|---|
| Frontend | Next.js (App Router) + TypeScript + Tailwind, na Vercel |
| API | FastAPI (Python 3.12), em Railway |
| Banco / Auth | Supabase (Postgres + Auth) |
| Storage de mídia | Cloudflare R2 (S3-compatible, egresso grátis) |
| GPU / workers de ML | Modal (serverless) |
| Fila | Postgres, `SELECT ... FOR UPDATE SKIP LOCKED` |
| Pagamento | Asaas (PIX e boleto) |
| Erros | Sentry |

**Não introduza:** Kubernetes, Airflow, Temporal, Celery, RabbitMQ, Kafka, microserviços, GraphQL, ORM pesado além do SQLAlchemy Core, ou qualquer AWS service com taxa de egresso.

## Modelos de ML usados

| Etapa | Modelo | Onde roda |
|---|---|---|
| Separação voz/fundo | Demucs (htdemucs) | Modal |
| ASR + timestamps por palavra | WhisperX (large-v3) | Modal |
| Diarização | pyannote (via WhisperX) | Modal |
| Tradução | LLM via API (Claude) | API, não GPU |
| TTS | MOSS-TTS-Local-Transformer-v1.5 (Apache-2.0) | Modal |
| Avaliação (CER) | Whisper large-v3 | Modal |
| Similaridade de locutor | WavLM-TDNN / ECAPA | Modal |

**Detalhes do MOSS-TTS que importam para o código:**
- Controle de duração: parâmetro `tokens` em `processor.build_user_message(text=..., tokens=N)`.
- Frame rate do tokenizer: 12,5 Hz → `tokens = round(segundos * 12.5)`. Granularidade de 80 ms.
- Pausas explícitas: marcador inline `[pause 3.2s]` no texto.
- Pronúncia forçada: aceita IPA entre barras, ex. `/oʊˈpɛn aɪ/`.
- Tag de idioma: `language="Portuguese"` — atenção, o modelo tende ao pt-PT; ver `docs/ptbr.md`.

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
  /ptbr             Camada de texto pt-BR
    normalize.py      Números, moeda, datas, siglas, ordinais
    syllables.py      Contador de sílabas
    g2p.py            Fonemização e dicionário IPA
/modal
  functions.py      Entrypoints de GPU (demucs, whisperx, tts, eval)
/migrations         SQL versionado
/evals              Harness de avaliação pt-BR
/scripts            CLI de desenvolvimento
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

## Convenções

- Python: ruff (linha 100), type hints obrigatórios em funções públicas, pydantic para fronteiras de dados.
- TypeScript: strict. Sem `any`. Sem `!` non-null assertion.
- SQL escrito à mão em migrations numeradas. Sem autogeração de schema.
- Toda função de `packages/pipeline` tem teste unitário com dado sintético (sem GPU).
- Mensagens de commit: imperativo, em inglês, escopo prefixado — `pipeline: add syllable-aware duration budget`.
- Nomes de domínio em inglês no código; conteúdo pt-BR só em dados e UI.

## Glossário de domínio

- **Segmento** — bloco de fala delimitado por pausa real no áudio original. Unidade de tradução, síntese e edição.
- **Isocronia** — restrição de que o áudio dublado caiba na duração do original.
- **Orçamento de duração** — quantas sílabas cabem num segmento, dado o alvo temporal.
- **On-screen** — locutor visível em quadro; exige tolerância de duração mais apertada (±8% vs ±20%).
- **Flag** — marcação automática de suspeita de erro num segmento; alimenta o editor.
- **Stem** — faixa separada (voz ou fundo) produzida pelo Demucs.
- **Perfil de voz** — voz clonada persistente associada a um canal, reutilizada entre vídeos.
- **Glossário do canal** — termos recorrentes com tradução e pronúncia IPA fixas, por canal.
- **CER round-trip** — sintetizar, transcrever com ASR, comparar com o texto de entrada.

## O que NÃO construir

Estas coisas vão parecer urgentes. Não são. Não implemente sem eu pedir explicitamente:

- Lip-sync facial / manipulação de vídeo
- Outros pares de idiomas além de EN→pt-BR
- App mobile, API pública, webhooks, integrações
- Gestão de times, papéis, permissões granulares
- Editor colaborativo em tempo real
- Fine-tune de modelo (fase posterior, feita à mão fora do repo)
- Qualquer dashboard de analytics além do básico

## Conformidade (obrigatório, não é backlog)

- Todo upload exige declaração de que o usuário tem direito sobre as vozes do vídeo. Persista o registro com timestamp.
- Voz pode ser dado pessoal sensível (LGPD art. 5º, II). Defina retenção e permita exclusão.
- Watermark de conteúdo sintético em todo áudio gerado (EU AI Act art. 50, vigente desde 2/8/2026).
- Nunca logue conteúdo de vídeo do cliente em serviços de terceiros.

## Como trabalhar comigo

- Tarefa por vez, seguindo `TASKS.md` na ordem.
- Antes de começar uma tarefa, releia os critérios de aceite dela.
- Se uma tarefa parecer exigir violar um invariante, pare e me pergunte em vez de contornar.
- Se precisar de uma decisão de produto que não está aqui, pergunte — não invente.
- Não refatore código fora do escopo da tarefa atual.
