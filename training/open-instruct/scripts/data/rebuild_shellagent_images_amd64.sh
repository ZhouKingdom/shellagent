#!/bin/bash
set -euo pipefail

# Rebuild and republish shellagent task images for x86_64 nodes.
# Usage:
#   bash scripts/data/rebuild_shellagent_images_amd64.sh [extra build_shellagent_images.py args...]

uv run python scripts/data/build_shellagent_images.py \
  --input hamishivi/swerl-shellagent-15k-verified \
  --output-dataset hamishivi/swerl-shellagent-15k-verified \
  --registry hamishi740 \
  --repo-prefix swerl-shellagent \
  --platform linux/amd64 \
  "$@"
