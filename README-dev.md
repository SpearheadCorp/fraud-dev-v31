# Fraud Detection Demo - Development Environment (`dev` branch)

This document describes the development and deployment workflow for the `dev` branch of the Fraud Detection Demo v4.

## 1. Environment Overview
- **Branch:** `dev`
- **Kubernetes Namespace:** `fraud-det-dev`
- **Image Tags:** `:dev`
- **Image Registry:** `10.23.181.247:5000/fraud-det-dev`
- **Storage Path (NFS):** `/financial-fraud-detection-demo/dev/`
- **NodePort:** `30881` (Isolated from prod on `30880`)

## 2. Developer Workflow

### Step 1: Push changes to GitHub
The Build VM pulls from the `dev` branch on GitHub. Always push your local changes first:
```bash
git add .
git commit -m "Your changes"
git push origin dev
```

### Step 2: Build & Push Images
Since the images must be built on the NVIDIA Build VM (`.247`), you can trigger it remotely.

**If you have `make` locally:**
```bash
make build
make push
```

**If you are on Windows (Manual SSH):**
```powershell
ssh -i ~/.ssh/id_rsa tduser@10.23.181.247 "cd /home/tduser/fraud-det-v31 && git fetch v4 dev && git checkout dev && git pull v4 dev && docker build -t 10.23.181.247:5000/fraud-det-dev/backend:dev -f pods/backend/Dockerfile . && docker build -t 10.23.181.247:5000/fraud-det-dev/data-gather:dev -f pods/data-gather/Dockerfile . && docker build -t 10.23.181.247:5000/fraud-det-dev/data-prep:dev -f pods/data-prep/Dockerfile . && docker build -t 10.23.181.247:5000/fraud-det-dev/model-build:dev -f pods/model-build/Dockerfile . && docker build -t 10.23.181.247:5000/fraud-det-dev/model-train:dev -f pods/model-train/Dockerfile . && docker build -t 10.23.181.247:5000/fraud-det-dev/scoring:dev -f pods/scoring/Dockerfile . && docker build -t 10.23.181.247:5000/fraud-det-dev/triton:dev -f pods/triton/Dockerfile . && docker push 10.23.181.247:5000/fraud-det-dev/backend:dev && docker push 10.23.181.247:5000/fraud-det-dev/data-gather:dev && docker push 10.23.181.247:5000/fraud-det-dev/data-prep:dev && docker push 10.23.181.247:5000/fraud-det-dev/model-build:dev && docker push 10.23.181.247:5000/fraud-det-dev/model-train:dev && docker push 10.23.181.247:5000/fraud-det-dev/scoring:dev && docker push 10.23.181.247:5000/fraud-det-dev/triton:dev"
```

### Step 3: Deploy to Kubernetes
Use the following commands to apply the isolated manifests. The `sed` (or PowerShell replace) ensures that no production resources are overwritten.

**If you have `bash`/`sed`:**
```bash
# Create the dev namespace
sed "s/fraud-det-v31/fraud-det-dev/g" k8s/namespace.yaml | kubectl apply -f -

# Deploy isolated resources
for f in k8s/rbac.yaml k8s/storage.yaml k8s/deployments.yaml k8s/services.yaml; do
    sed "s/fraud-det-v31/fraud-det-dev/g; s/fraud-det-v4/fraud-det-dev/g; s/:latest/:dev/g; s/\/v31\//\/dev\//g; s/30880/30881/g" $f | kubectl apply -n fraud-det-dev -f -
done
```

**If you are on PowerShell:**
```powershell
# Create the namespace
(Get-Content k8s/namespace.yaml) -replace 'fraud-det-v31', 'fraud-det-dev' | kubectl apply -f -

# Deploy all isolated resources
foreach ($f in "k8s/rbac.yaml", "k8s/storage.yaml", "k8s/deployments.yaml", "k8s/services.yaml") {
    (Get-Content $f) -replace 'fraud-det-v31', 'fraud-det-dev' `
                     -replace 'fraud-det-v4', 'fraud-det-dev' `
                     -replace ':latest', ':dev' `
                     -replace '/v31/', '/dev/' `
                     -replace '30880', '30881' | kubectl apply -n fraud-det-dev -f -
}
```

## 3. Accessing the Environment
- **Dashboard:** `http://10.23.181.44:30881`
- **Prometheus:** `http://10.23.181.153:9090` (Cluster-wide)
- **Grafana:** `http://10.23.181.153:3000` (Cluster-wide)

## 4. Resetting the Dev Environment
To wipe the `dev` data and re-queue raw files for a fresh test:
```bash
curl -X POST http://10.23.181.44:30881/api/control/reset
```
