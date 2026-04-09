.PHONY: up build daemon down clean logs upd \
	help shell do-fn-validate do-fn-connect do-fn-status do-fn-deploy do-fn-deploy-remote \
	do-fn-list do-fn-get do-fn-invoke do-fn-activations do-fn-logs do-droplet-log

WORKER_CONTAINER=credit_liquidity_monitor

# Default Compose files
MAIN_COMPOSE_FILE := docker-compose.yml
DROPLET_USER ?= root
DROPLET_LOG_FILE ?= /var/log/job.log

# Quickly view available targets
help:
	@echo "Available make targets:"
	@echo "  build        Build main Airflow image"
	@echo "  up           Start main stack (foreground)"
	@echo "  upd          Start main stack (daemon)"
	@echo "  down         Stop main stack"
	@echo "  clean        Stop and remove volumes for main stack"
	@echo "  logs         Follow logs for main stack"

# ---------------------------------------------------------------------
# MAINCOMMANDS
# ---------------------------------------------------------------------

build:
	docker-compose -f $(MAIN_COMPOSE_FILE) build

up:
	docker-compose -f $(MAIN_COMPOSE_FILE) up

upd:
	docker-compose -f $(MAIN_COMPOSE_FILE) up -d

down:
	docker-compose -f $(MAIN_COMPOSE_FILE) down --remove-orphans

clean:
	docker-compose -f $(MAIN_COMPOSE_FILE) down --volumes --remove-orphans
	#rm -rf airflow/logs airflow/db

logs:
	docker-compose -f $(MAIN_COMPOSE_FILE) logs -f

shell:
	docker exec -it $(WORKER_CONTAINER) /bin/bash
