.PHONY: up build daemon down clean logs upd \
	help shell do-fn-validate do-fn-connect do-fn-status do-fn-deploy do-fn-deploy-remote \
	do-fn-list do-fn-get do-fn-invoke do-fn-activations do-fn-logs do-droplet-log

WORKER_CONTAINER=credit_liquidity_monitor

# Default Compose files
MAIN_COMPOSE_FILE := docker-compose.yml
DO_FN_DIR ?= infra/do-functions
DO_FN_ENV ?= infra/do-functions/.env
DO_FN_NAME ?= launcher/credit-liquidity-monitor
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
	@echo "  do-fn-validate      Validate DO Functions project metadata"
	@echo "  do-fn-connect       Connect doctl to a DO Functions namespace"
	@echo "  do-fn-status        Show DO Functions connection status"
	@echo "  do-fn-deploy        Deploy infra/do-functions with runtime env"
	@echo "  do-fn-deploy-remote Deploy infra/do-functions using remote build"
	@echo "  do-fn-list          List deployed DO functions"
	@echo "  do-fn-get           Show deployed function metadata"
	@echo "  do-fn-invoke        Invoke $(DO_FN_NAME)"
	@echo "  do-fn-activations   List recent activations"
	@echo "  do-fn-logs          Show logs for ACTIVATION=<id>"
	@echo "  do-droplet-log      Tail droplet log over SSH with DROPLET_IP=<ip>"


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


do-fn-validate:
	doctl serverless get-metadata $(DO_FN_DIR)

do-fn-connect:
	doctl serverless connect

do-fn-status:
	doctl serverless status

do-fn-deploy:
	doctl serverless deploy $(DO_FN_DIR) --env $(DO_FN_ENV)

do-fn-deploy-remote:
	doctl serverless deploy $(DO_FN_DIR) --env $(DO_FN_ENV) --remote-build

do-fn-list:
	doctl serverless functions list

do-fn-get:
	doctl serverless functions get $(DO_FN_NAME)

do-fn-invoke:
	doctl serverless functions invoke $(DO_FN_NAME)

do-fn-activations:
	doctl serverless activations list

do-fn-logs:
	test -n "$(ACTIVATION)" || (echo "Set ACTIVATION=<id>" && exit 1)
	doctl serverless activations logs $(ACTIVATION)

do-droplet-log:
	test -n "$(DROPLET_IP)" || (echo "Set DROPLET_IP=<ip>" && exit 1)
	ssh $(DROPLET_USER)@$(DROPLET_IP) "sudo tail -f $(DROPLET_LOG_FILE)"
