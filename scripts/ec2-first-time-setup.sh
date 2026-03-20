#!/usr/bin/env bash

set -eu

APP_DIR="/opt/ai-content-detection"

sudo mkdir -p "${APP_DIR}"
sudo chown "$(whoami)":"$(whoami)" "${APP_DIR}"

if [ ! -f "${APP_DIR}/.env" ]; then
  cat <<'EOF'
Create your runtime env file at:
  /opt/ai-content-detection/.env

Example:
  PROJECT_NAME=AI Content Detection ML Server
  USE_GPU=False
EOF
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker on this EC2 host first."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is not installed. Install curl on this EC2 host first."
  exit 1
fi

echo "EC2 host is ready for GitHub Actions deployments."
