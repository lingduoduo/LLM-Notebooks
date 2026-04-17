#!/bin/bash

set -Eeuo pipefail

GPU_OPERATOR_NAMESPACE="${GPU_OPERATOR_NAMESPACE:-gpu-operator}"
KEDA_NAMESPACE="${KEDA_NAMESPACE:-keda-system}"

on_error() {
    local line_no=${1:-unknown}

    echo "ERROR: Installation failed near line ${line_no}"
}

trap 'on_error $LINENO' ERR

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

require_command() {
    if ! command_exists "$1"; then
        echo "ERROR: Required command '$1' is not installed or not in PATH"
        exit 1
    fi
}

add_repo() {
    local name=$1
    local url=$2

    if helm repo list | awk 'NR > 1 {print $1}' | grep -qx "$name"; then
        echo "Helm repo '$name' already configured"
        return
    fi

    helm repo add "$name" "$url"
}

echo "Installing GPU monitoring dependencies..."

require_command helm
require_command kubectl
kubectl cluster-info >/dev/null

add_repo nvidia https://helm.ngc.nvidia.com/nvidia
add_repo gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
add_repo kedacore https://kedacore.github.io/charts

helm repo update

echo "Installing or upgrading NVIDIA GPU Operator..."
helm upgrade --install gpu-operator nvidia/gpu-operator \
  --namespace "$GPU_OPERATOR_NAMESPACE" \
  --create-namespace \
  --wait \
  --timeout=600s \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true \
  --set gfd.enabled=true

echo "Installing or upgrading DCGM Exporter..."
helm upgrade --install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace "$GPU_OPERATOR_NAMESPACE" \
  --wait \
  --timeout=300s \
  --set serviceMonitor.enabled=false

echo "Installing or upgrading KEDA..."
helm upgrade --install keda kedacore/keda \
  --namespace "$KEDA_NAMESPACE" \
  --create-namespace \
  --wait \
  --timeout=300s

echo "Verifying installation status..."
echo "GPU Operator pods:"
kubectl get pods -n "$GPU_OPERATOR_NAMESPACE" || echo "Warning: Unable to list GPU Operator pods"
echo ""
echo "KEDA pods:"
kubectl get pods -n "$KEDA_NAMESPACE" || echo "Warning: Unable to list KEDA pods"

echo "GPU monitoring installation completed successfully."
