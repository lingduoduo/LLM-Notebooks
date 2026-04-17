#!/bin/bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_NAMESPACE="${NAMESPACE:-default}"
GPU_OPERATOR_NAMESPACE="${GPU_OPERATOR_NAMESPACE:-gpu-operator}"
KEDA_NAMESPACE="${KEDA_NAMESPACE:-keda-system}"
GPU_QUOTA_MAX="${GPU_QUOTA_MAX:-10}"
LLM_SERVICE_URL="${LLM_SERVICE_URL:-}"
AUTOSCALER_MODE="${AUTOSCALER_MODE:-keda}"
HPA_SCALE_MANIFEST="${HPA_SCALE_MANIFEST:-$SCRIPT_DIR/hpa_scale_configmap.yaml}"
HPA_BACKUP_MANIFEST="${HPA_BACKUP_MANIFEST:-$SCRIPT_DIR/hpa_backup.yaml}"

on_error() {
    local line_no=${1:-unknown}
    local cmd=${2:-unknown}

    echo "ERROR: Deployment failed at line ${line_no}: ${cmd}"
}

trap 'on_error $LINENO "$BASH_COMMAND"' ERR

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    if ! command_exists "$1"; then
        echo "ERROR: Required command '$1' is not installed or not in PATH"
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: Required file '$1' was not found"
        exit 1
    fi
}

verify_cluster_access() {
    echo "Checking Kubernetes cluster access..."
    kubectl cluster-info >/dev/null
}

render_manifest() {
    local manifest=$1

    sed -E \
        -e "s|(  *namespace: )default$|\1${DEFAULT_NAMESPACE}|g" \
        -e "s|(  *namespace: )gpu-operator$|\1${GPU_OPERATOR_NAMESPACE}|g" \
        -e "s|(  *namespace: )keda-system$|\1${KEDA_NAMESPACE}|g" \
        -e "s|namespace=\"default\"|namespace=\"${DEFAULT_NAMESPACE}\"|g" \
        -e "s|([[:space:]]*- )gpu-operator$|\1${GPU_OPERATOR_NAMESPACE}|g" \
        -e "s|\\\$\{GPU_QUOTA_MAX\}|${GPU_QUOTA_MAX}|g" \
        "$manifest"
}

apply_manifest() {
    local manifest=$1

    echo "Applying $(basename "$manifest")..."
    render_manifest "$manifest" | kubectl apply -f -
}

print_autoscaler_status() {
    echo "ScaledObjects:"
    kubectl get scaledobjects -n "$DEFAULT_NAMESPACE" 2>/dev/null || echo "No ScaledObjects found"
    echo ""
    echo "HPAs:"
    kubectl get hpa -n "$DEFAULT_NAMESPACE" 2>/dev/null || echo "No HPAs found"
}

wait_for_pods() {
    local namespace=$1
    local label=$2
    local timeout=${3:-300}

    echo "Waiting for pods with label '$label' in namespace '$namespace'..."
    if ! kubectl wait --for=condition=ready pod -l "$label" -n "$namespace" --timeout="${timeout}s"; then
        echo "ERROR: Pods with label '$label' in namespace '$namespace' were not ready within ${timeout}s"
        kubectl get pods -l "$label" -n "$namespace" -o wide || true
        exit 1
    fi
}

wait_for_rollout() {
    local namespace=$1
    local deployment=$2
    local timeout=${3:-600}

    echo "Waiting for rollout of deployment/$deployment in namespace '$namespace'..."
    kubectl rollout status "deployment/$deployment" -n "$namespace" --timeout="${timeout}s"
}

echo "Starting deployment of the GPU monitoring and LLM auto-scaling stack..."

require_command kubectl
require_command helm
require_file "$SCRIPT_DIR/install-gpu-monitoring.sh"
require_file "$SCRIPT_DIR/metrics_configmap.yaml"
require_file "$SCRIPT_DIR/agent_configmap.yaml"
verify_cluster_access

echo "Step 0: Setting up namespace and GPU resource quota"
kubectl create namespace "$DEFAULT_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
echo "Namespace '${DEFAULT_NAMESPACE}' ready (GPU quota: ${GPU_QUOTA_MAX} GPUs)"

if [[ "${SKIP_INSTALL:-false}" == "true" ]]; then
    echo "Step 1: Skipping GPU monitoring installation because SKIP_INSTALL=true"
else
    echo "Step 1: Installing GPU monitoring components"
    bash "$SCRIPT_DIR/install-gpu-monitoring.sh"
fi

wait_for_pods "$GPU_OPERATOR_NAMESPACE" "app=nvidia-operator-validator" 600
wait_for_pods "$KEDA_NAMESPACE" "app.kubernetes.io/name=keda-operator" 300

echo "Step 2: Configuring GPU metrics collection"
apply_manifest "$SCRIPT_DIR/metrics_configmap.yaml"

echo "Step 3: Deploying the LLM agent"
apply_manifest "$SCRIPT_DIR/agent_configmap.yaml"
wait_for_rollout "$DEFAULT_NAMESPACE" "llm-agent" 900

echo "Step 4: Configuring GPU-based autoscaling"
case "$AUTOSCALER_MODE" in
    keda)
        apply_manifest "$HPA_SCALE_MANIFEST"
        ;;
    hpa)
        echo "Applying backup HPA only because AUTOSCALER_MODE=hpa"
        apply_manifest "$HPA_BACKUP_MANIFEST"
        ;;
    *)
        echo "ERROR: Unknown AUTOSCALER_MODE '${AUTOSCALER_MODE}'. Use 'keda' or 'hpa'."
        exit 1
        ;;
esac

echo "Verifying deployment status..."
echo "GPU Operator pods:"
kubectl get pods -n "$GPU_OPERATOR_NAMESPACE"
echo ""
echo "KEDA pods:"
kubectl get pods -n "$KEDA_NAMESPACE"
echo ""
echo "LLM Agent pods:"
kubectl get pods -n "$DEFAULT_NAMESPACE" -l app=llm-agent
echo ""
print_autoscaler_status
echo ""
echo "ResourceQuota:"
kubectl get resourcequota -n "$DEFAULT_NAMESPACE"
echo ""
echo "GPU node resources:"
kubectl describe nodes -l node-type=gpu 2>/dev/null \
    | grep -E "(Name:|nvidia.com/gpu|Capacity|Allocatable)" \
    || echo "No GPU nodes with label node-type=gpu found"

if [[ -n "${LLM_SERVICE_URL:-}" ]]; then
    echo ""
    echo "Testing service availability at ${LLM_SERVICE_URL}..."
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 10 "${LLM_SERVICE_URL}/health/ready" || echo "000")
    if [[ "$HTTP_CODE" == "200" ]]; then
        echo "Service health check passed (HTTP ${HTTP_CODE})"
    else
        echo "WARNING: Service health check returned HTTP ${HTTP_CODE} — the pod may still be warming up"
    fi
fi

echo "Deployment complete."
echo "Autoscaler mode: ${AUTOSCALER_MODE}"
echo "KEDA manifest: $HPA_SCALE_MANIFEST"
echo "Backup HPA manifest: $HPA_BACKUP_MANIFEST"
echo "Do not run KEDA ScaledObject and backup HPA together for the same Deployment."
echo ""
echo "To inspect raw GPU metrics:"
echo "kubectl port-forward -n $GPU_OPERATOR_NAMESPACE svc/dcgm-exporter 9400:9400"
echo "Then open http://localhost:9400/metrics"
