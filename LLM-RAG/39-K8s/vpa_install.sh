#!/bin/bash

set -Eeuo pipefail

# INSTALL_METHOD: "upstream" (default) clones the official repo and runs vpa-up.sh
#                 "helm" installs via the cowboysysop Helm chart (recommended for production)
INSTALL_METHOD="${INSTALL_METHOD:-upstream}"
VPA_NAMESPACE="${VPA_NAMESPACE:-kube-system}"
VPA_HELM_RELEASE="${VPA_HELM_RELEASE:-vpa}"

on_error() {
    local line_no=${1:-unknown}
    local cmd=${2:-unknown}

    echo "ERROR: VPA installation failed at line ${line_no}: ${cmd}"
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

install_upstream() {
    require_command git

    TMPDIR=$(mktemp -d)
    trap 'rm -rf "$TMPDIR"' EXIT

    echo "Cloning autoscaler repository..."
    git clone --depth=1 https://github.com/kubernetes/autoscaler.git "$TMPDIR/autoscaler"

    echo "Running VPA installation script..."
    bash "$TMPDIR/autoscaler/vertical-pod-autoscaler/hack/vpa-up.sh"
}

install_helm() {
    require_command helm

    echo "Adding cowboysysop Helm repository..."
    if ! helm repo list 2>/dev/null | awk '{print $1}' | grep -qx "cowboysysop"; then
        helm repo add cowboysysop https://cowboysysop.github.io/charts/
    else
        echo "Helm repo 'cowboysysop' already configured"
    fi
    helm repo update cowboysysop

    echo "Installing or upgrading VPA via Helm..."
    helm upgrade --install "$VPA_HELM_RELEASE" cowboysysop/vertical-pod-autoscaler \
        --namespace "$VPA_NAMESPACE" \
        --create-namespace \
        --wait \
        --timeout=300s \
        --set recommender.enabled=true \
        --set updater.enabled=true \
        --set admissionController.enabled=true
}

echo "Installing Vertical Pod Autoscaler (VPA) — method: ${INSTALL_METHOD}..."

require_command kubectl

KUBE_VERSION=$(kubectl version -o json 2>/dev/null \
    | grep '"gitVersion"' | tail -1 | grep -o 'v[0-9.]*' \
    || echo "unknown")
echo "Detected Kubernetes version: ${KUBE_VERSION}"

case "$INSTALL_METHOD" in
    upstream) install_upstream ;;
    helm)     install_helm ;;
    *)
        echo "ERROR: Unknown INSTALL_METHOD '${INSTALL_METHOD}'. Use 'upstream' or 'helm'."
        exit 1
        ;;
esac

echo "Verifying VPA component installation..."
kubectl get pods -n "$VPA_NAMESPACE" | grep vpa \
    || echo "Warning: No VPA pods found in namespace ${VPA_NAMESPACE}"
kubectl get crd | grep verticalpodautoscaler \
    || echo "Warning: VPA CRD not found — installation may have failed"

echo "VPA installation completed!"
