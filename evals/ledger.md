# Registro do ciclo de qualidade

Append-only. Uma entrada por ciclo, escrita pelo orquestrador (`/ciclo`) depois da sua decisão. O `dub-analyst` lê este arquivo antes de propor: abordagem rejeitada não volta sem evidência nova.

Formato:

## <ciclo> — <idioma> — <título>
- **Classe-alvo:** ...
- **Hipótese:** ...
- **Mudança:** branch / commit, tarefa do TASKS.md
- **Resultado (juiz, proxy):** dev ... → ...; holdout ... → ...; regressões ...
- **Escuta humana:** ...
- **Decisão:** aceita / rejeitada / iterar, por quê
- **Aprendizado:** o que isso diz sobre o pipeline, para os próximos ciclos

---
