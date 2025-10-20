# Lens Service Makefile

.PHONY: help install dev test lint format clean build up down logs shell init-db deepsource-local deepsource-auth

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  dev         - Run development server"
	@echo "  dev-debug   - Run development server in debug mode"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  clean       - Clean up generated files"
	@echo "  build       - Build Docker images"
	@echo "  up          - Start services with Docker Compose"
	@echo "  down        - Stop services"
	@echo "  logs        - View service logs"
	@echo "  shell       - Open shell in backend container"
	@echo "  migrate     - Run database migrations"
	@echo "  init-db     - Initialize database with seed data"
	@echo "  check       - Run all quality checks"
	@echo "  deepsource-local - Run Deepsource analysis locally"
	@echo "  deepsource-auth  - Authenticate with Deepsource"

install:
	uv sync --frozen --no-cache

dev:
	uv run python main.py

dev-debug:
	uv run python -m debugpy --listen 5678 --wait-for-client main.py

test:
	uv run pytest -v

test-cov:
	uv run pytest --cov=core --cov=api --cov-report=html --cov-report=term

lint:
	-uv run ruff check core
	-uv run mypy core

format:
	-uv run ruff format core
	-uv run ruff check --fix core

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf .ruff_cache

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f lens-service

shell:
	docker compose exec lens-service /bin/bash

migrate:
	uv run alembic upgrade head

init-db:
	@echo "Initializing database with seed data..."
	uv run python scripts/init_data.py

check: lint test
	@echo "All quality checks passed!"


# Database operations
db-up:
	docker compose up -d postgres

db-down:
	docker compose down postgres

# Monitoring
monitoring-up:
	docker compose up -d prometheus grafana

monitoring-down:
	docker compose down prometheus grafana

# Code quality
deepsource-local:
	@echo "Running local Deepsource analysis..."
	@./scripts/run_deepsource_local.sh

deepsource-auth:
	@echo "Setting up Deepsource authentication..."
	@./scripts/deepsource_auth.sh
