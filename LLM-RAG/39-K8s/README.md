# LLM Agent GPU Auto-Scaling System

This directory contains Kubernetes manifests and helper scripts for deploying a GPU-backed LLM agent with monitoring and KEDA-based autoscaling.

## Files

### Core deployment
- `deployment.sh`: applies all manifests in order and waits for the stack to become ready
- `install-gpu-monitoring.sh`: installs or upgrades NVIDIA GPU Operator, DCGM exporter, and KEDA via Helm

### Manifests
- `agent_configmap.yaml`: `Deployment` (with model-warmup initContainer), `Service` (ClientIP session affinity), `Ingress` (cookie-based stickiness, `/warmup` endpoint), `PersistentVolumeClaim`, `PodDisruptionBudget`, and `ServiceMonitor` for the LLM agent
- `metrics_configmap.yaml`: Prometheus scrape config and GPU alert rules (`ConfigMap`, `Service`, `ServiceMonitor`)
- `hpa_scale_configmap.yaml`: KEDA `ScaledObject` — scales on GPU utilization, GPU memory, queue length, p95 latency, and CPU
- `hpa_backup.yaml`: fallback `HorizontalPodAutoscaler` (CPU/memory only) — apply only if KEDA is unavailable; conflicts with `hpa_scale_configmap.yaml`

### VPA (Vertical Pod Autoscaler)
- `vpa_install.sh`: installs VPA via the official upstream script into a temporary directory
- `vpa_backup.yaml`: reference VPA configs — dev (`Auto` mode), prod (`Initial` mode), and recommendation-only (`Off` mode for use alongside HPA); includes a `PodDisruptionBudget`
- `vpa_monitoring.yaml`: `ServiceMonitor` for all three VPA components (recommender, updater, admission-controller) and `PrometheusRule` alerts for recommendation deviation and component downtime
- `vpa_scale_configmap.yaml`: Karpenter `NodePool` for GPU node provisioning with spot-first cost optimization

## Quick Start

Prerequisites:

- `kubectl`
- `helm`
- a Kubernetes cluster with NVIDIA GPU nodes

Deploy everything:

```bash
cd LLM-RAG/39-K8s
./deployment.sh
```

Optional overrides:

```bash
# Custom namespaces and GPU quota limit (default: 10 GPUs)
NAMESPACE=llm-production GPU_QUOTA_MAX=20 ./deployment.sh

# Full override set
NAMESPACE=default GPU_OPERATOR_NAMESPACE=gpu-operator KEDA_NAMESPACE=keda-system GPU_QUOTA_MAX=10 ./deployment.sh

# Use the backup HPA instead of KEDA ScaledObject
AUTOSCALER_MODE=hpa ./deployment.sh

# Include a post-deploy service health check
LLM_SERVICE_URL=https://llm-api.yourdomain.com ./deployment.sh
```

If the cluster already has GPU Operator, DCGM exporter, and KEDA installed:

```bash
SKIP_INSTALL=true ./deployment.sh
```

## Tuning Points

Update these values in `agent_configmap.yaml` for your workload:

- `image`: container image for the LLM service
- `MODEL_PATH`: model mount path
- `MAX_BATCH_SIZE`
- `MAX_CONCURRENT_REQUESTS`
- `MODEL_CACHE_SIZE`
- CPU, memory, GPU, and ephemeral storage requests and limits

Update these values in `hpa_scale_configmap.yaml` (`ScaledObject`) for scaling behavior:

- `minReplicaCount` and `maxReplicaCount`
- `pollingInterval` and `cooldownPeriod`
- GPU utilization threshold: `65`
- GPU memory threshold: `75`
- queue length threshold: `5`
- p95 latency threshold in milliseconds: `3000`

## VPA Usage

Install VPA into the cluster (runs once):

```bash
# Default: clones kubernetes/autoscaler and runs the official vpa-up.sh
./vpa_install.sh

# Helm (recommended for production — idempotent, upgradeable)
INSTALL_METHOD=helm ./vpa_install.sh

# Override namespace or Helm release name
VPA_NAMESPACE=kube-system VPA_HELM_RELEASE=vpa INSTALL_METHOD=helm ./vpa_install.sh
```

`vpa_backup.yaml` contains three reference configs — pick the one matching your environment:

| Config | Namespace | `updateMode` | Use when |
|---|---|---|---|
| `llm-agent-vpa-dev` | `development` | `Auto` | dev cluster, restarts OK |
| `llm-agent-vpa-prod` | `production` | `Initial` | prod, apply only at pod start |
| `llm-agent-vpa-with-hpa` | `default` | `Off` | running alongside KEDA HPA (recommendations only) |

`vpa_scale_configmap.yaml` defines two Karpenter `NodePool`s for GPU node provisioning:
- `gpu-nodepool-spot` (weight 100) — spot-only, g4dn.xlarge/2xlarge, saves ~70% cost
- `gpu-nodepool-ondemand` (weight 10) — on-demand fallback, includes p3.2xlarge, slower consolidation to reduce churn

Apply separately if Karpenter is installed (requires `CLUSTER_NAME` env var for subnet/SG discovery):

```bash
CLUSTER_NAME=my-cluster envsubst < vpa_scale_configmap.yaml | kubectl apply -f -
```

## Validation

Useful checks after deployment:

```bash
kubectl get pods -n gpu-operator
kubectl get pods -n keda-system
kubectl get pods -n default -l app=llm-agent
kubectl get scaledobjects -n default
kubectl get hpa -n default
kubectl get vpa -A
```

Inspect GPU metrics:

```bash
kubectl port-forward -n gpu-operator svc/dcgm-exporter 9400:9400
curl http://localhost:9400/metrics
```

## Notes

- The manifests assume Prometheus is reachable at `http://prometheus.monitoring.svc.cluster.local:9090`.
- The autoscaling queries rely on application metrics such as `llm_request_queue_length` and `llm_request_duration_seconds_bucket`; the LLM service must expose them on `/metrics`.
- If your cluster uses different GPU node labels or storage classes, adjust `agent_configmap.yaml` before deployment.
- Namespace overrides in `deployment.sh` are applied at deploy time, including the Prometheus query namespace filter used by the CPU-based scaler.
- `deployment.sh` now supports `AUTOSCALER_MODE=keda` and `AUTOSCALER_MODE=hpa` so the deployment flow matches the manifest set in this directory.
