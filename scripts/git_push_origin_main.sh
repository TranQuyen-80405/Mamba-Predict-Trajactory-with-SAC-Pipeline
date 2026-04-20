#!/usr/bin/env bash
# Push main to origin using HTTPS + Personal Access Token (method A).
#
# 1) Create a PAT: GitHub → Settings → Developer settings → Personal access tokens
#    (classic: enable "repo" scope; fine-grained: Contents read/write for this repo)
# 2) Run ONE of:
#      export GITHUB_TOKEN='ghp_xxxxxxxx'
#      ./scripts/git_push_origin_main.sh
#    Or pass token without storing in shell history:
#      GITHUB_TOKEN='ghp_xxx' ./scripts/git_push_origin_main.sh
#
# Do not commit tokens. Do not paste tokens into chat.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "GITHUB_TOKEN is not set." >&2
  echo "Create a PAT on GitHub, then run:" >&2
  echo "  export GITHUB_TOKEN='ghp_...'" >&2
  echo "  $0" >&2
  exit 1
fi

REMOTE_URL="$(git remote get-url origin)"
# Expect https://github.com/OWNER/REPO.git
if [[ "$REMOTE_URL" != https://github.com/* ]]; then
  echo "origin is not an https://github.com/ URL: $REMOTE_URL" >&2
  echo "Fix: git remote set-url origin https://github.com/TranQuyen-80405/Mamba-Predict-Trajactory-with-SAC-Pipeline.git" >&2
  exit 1
fi

# Strip https:// for user@host:path form
REST="${REMOTE_URL#https://}"
USER="TranQuyen-80405"
PUSH_URL="https://${USER}:${GITHUB_TOKEN}@${REST}"

echo "Pushing main to origin (HTTPS + token from env)..."
git push "$PUSH_URL" main

echo "Done."
