# Get the current branch name (e.g., dev, main)
BRANCH       := $(shell git rev-parse --abbrev-ref HEAD)
ifeq ($(BRANCH),main)
NS           ?= fraud-det-v31
PORT         ?= 30880
else
NS           ?= fraud-det-$(BRANCH)
# Use a different NodePort for non-main branches to avoid conflicts
PORT         ?= 30881
endif
TAG          ?= $(BRANCH)

REGISTRY     = 10.23.181.247:5000/$(NS)
BUILD_VM     = tduser@10.23.181.247
BUILD_DIR    = /home/tduser/fraud-det-v31
BACKEND_URL  = http://10.23.181.44:$(PORT)

.PHONY: help build push deploy up start stop reset stress stress-stop logs status clean seed

help:
	@echo "fraud-det-v3.1  (Dynamic environment: $(NS), Tag: $(TAG), Port: $(PORT))"
	@echo ""
	@echo "  make build      - git push → pull on .247 → docker build all images"
	@echo "  make push       - push built images to private registry"
	@echo "  make deploy     - kubectl apply all k8s/ manifests (parameterized by $(NS))"
	@echo "  make up         - deploy + wait for backend to be Ready"
	@echo "  make start      - POST /api/control/start  (triggers pipeline via dashboard)"
	@echo "  make stop       - POST /api/control/stop"
	@echo "  make reset      - POST /api/control/reset"
	@echo "  make stress     - POST /api/control/stress"
	@echo "  make stress-stop- POST /api/control/stress-stop"
	@echo "  make logs       - Stream backend pod logs"
	@echo "  make status     - kubectl get pods -n $(NS)"
	@echo "  make seed       - scp seed-data zip to build VM (run once)"
	@echo "  make clean      - Delete all k8s resources in namespace"
	@echo ""
	@echo "  Dashboard:  $(BACKEND_URL)"
	@echo "  Prometheus: http://10.23.181.44:30090"

# ── Build & push ─────────────────────────────────────────────────────────────
build:
	git push origin $(BRANCH)
	ssh -i ~/.ssh/id_rsa $(BUILD_VM) "\
	  cd $(BUILD_DIR) && git fetch v4 $(BRANCH) && git checkout $(BRANCH) && git pull v4 $(BRANCH) && \
	  docker build --no-cache -t $(REGISTRY)/data-gather:$(TAG) -f pods/data-gather/Dockerfile . && \
	  docker build --no-cache -t $(REGISTRY)/data-prep:$(TAG)   -f pods/data-prep/Dockerfile   . && \
	  docker build --no-cache -t $(REGISTRY)/model-build:$(TAG) -f pods/model-build/Dockerfile . && \
	  docker build --no-cache -t $(REGISTRY)/model-train:$(TAG) -f pods/model-train/Dockerfile . && \
	  docker build --no-cache -t $(REGISTRY)/scoring:$(TAG)     -f pods/scoring/Dockerfile     . && \
	  docker build --no-cache -t $(REGISTRY)/triton:$(TAG)      -f pods/triton/Dockerfile      . && \
	  docker build --no-cache -t $(REGISTRY)/backend:$(TAG)     -f pods/backend/Dockerfile     ."

push:
	ssh -i ~/.ssh/id_rsa $(BUILD_VM) "\
	  docker push $(REGISTRY)/data-gather:$(TAG) && \
	  docker push $(REGISTRY)/data-prep:$(TAG)   && \
	  docker push $(REGISTRY)/model-build:$(TAG) && \
	  docker push $(REGISTRY)/model-train:$(TAG) && \
	  docker push $(REGISTRY)/scoring:$(TAG)     && \
	  docker push $(REGISTRY)/triton:$(TAG)      && \
	  docker push $(REGISTRY)/backend:$(TAG)"

# ── K8s deploy ───────────────────────────────────────────────────────────────
deploy:
	@echo "Deploying to namespace: $(NS) with tag: $(TAG) on port: $(PORT)"
	# 1. Apply namespace first
	sed "s/fraud-det-v31/$(NS)/g" k8s/namespace.yaml | kubectl apply -f -

	# 2. Apply other manifests with isolation logic:
	@for f in k8s/rbac.yaml k8s/storage.yaml k8s/configmap.yaml k8s/deployments.yaml k8s/services.yaml; do \
		sed "s/fraud-det-v31/$(NS)/g; s/fraud-det-v4/$(NS)/g; s/:latest/:$(TAG)/g; s/\/v31\//\/$(TAG)\//g; s/30880/$(PORT)/g" $$f | kubectl apply -n $(NS) -f - ; \
	done

	@echo "Waiting for backend to be Ready..."
	kubectl rollout status deployment/backend -n $(NS) --timeout=120s
	@echo ""
	@echo "  Dashboard:  $(BACKEND_URL)"
up: build push deploy

# ── Pipeline control ──────────────────────────────────────────────────────────
start:
	curl -s -X POST $(BACKEND_URL)/api/control/start | python3 -m json.tool

stop:
	curl -s -X POST $(BACKEND_URL)/api/control/stop | python3 -m json.tool

reset:
	curl -s -X POST $(BACKEND_URL)/api/control/reset | python3 -m json.tool

stress:
	curl -s -X POST $(BACKEND_URL)/api/control/stress | python3 -m json.tool

stress-stop:
	curl -s -X POST $(BACKEND_URL)/api/control/stress-stop | python3 -m json.tool

# ── Observability ─────────────────────────────────────────────────────────────
logs:
	kubectl logs -n $(NS) -l app=backend -f

status:
	kubectl get pods -n $(NS)

# ── One-time seed data copy ───────────────────────────────────────────────────
seed:
	@echo "Copying seed data to build VM (then rebuild data-gather image)..."
	scp -i ~/.ssh/id_rsa seed-data/credit_card_transactions.csv.zip \
	    $(BUILD_VM):$(BUILD_DIR)/seed-data/credit_card_transactions.csv.zip
	@echo "Done. Run: make build to bake seed data into the data-gather image."

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	kubectl delete namespace $(NS) --ignore-not-found
