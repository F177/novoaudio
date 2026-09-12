.PHONY: dev test lint migrate eval pipeline-cli

dev:
	@echo "make dev: apps/api e apps/web ainda não existem (Fase 1)."

test:
	pytest
	@if [ -f apps/web/package.json ]; then cd apps/web && npm test; fi

lint:
	ruff check .
	@if [ -f apps/web/package.json ]; then cd apps/web && npm run lint && npx tsc --noEmit; fi

migrate:
	@echo "make migrate: sem migrations ainda (Fase 1, T1.1)."

eval:
	@echo "make eval: harness de avaliação ainda não existe (Fase 3, T3.1)."

pipeline-cli:
	python scripts/dub.py $(VIDEO)
