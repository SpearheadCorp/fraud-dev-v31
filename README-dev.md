# Fraud Detection Demo - Development Environment (`dev` branch)

This document describes the development and deployment workflow for the `dev` branch of the Fraud Detection Demo v4.

## 1. Environment Overview
- **Branch:** `dev`
- **Kubernetes Namespace:** `fraud-det-dev`
- **Image Tags:** `:dev`
- **Registry:** `10.23.181.247:5000/fraud-det-dev`

## 2. Deployment Steps

### Build & Push
To build the images on the remote Build VM and push them to the registry:
```bash
make build
make push
```

### Deploy to Kubernetes
To deploy the `dev` environment to the cluster:
```bash
make deploy
```
The `make deploy` command automatically:
1. Creates the `fraud-det-dev` namespace.
2. Replaces `fraud-det-v31` with `fraud-det-dev` in all manifests.
3. Sets image tags to `:dev`.

### Accessing the Dashboard
By default, the backend is exposed on **NodePort 30880**.
URL: `http://10.23.181.44:30880`

> [!WARNING]
> Since NodePorts must be unique, you cannot run both the `v31` (main) and `dev` environments simultaneously if they both use `30880`. If `v31` is running, use a different `nodePort` (e.g., `30881`) for dev.

## 3. Operational Commands
- Check status: `make status`
- View logs: `make logs`
- Start pipeline: `make start`
- Stop pipeline: `make stop`
- Reset environment: `make reset`
