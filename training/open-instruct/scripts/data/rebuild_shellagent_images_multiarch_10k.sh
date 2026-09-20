#!/bin/bash
set -euo pipefail

# Rebuild and republish non-verified shellagent task images as multi-platform tags.
# Usage:
#   bash scripts/data/rebuild_shellagent_images_multiarch_10k.sh [extra build_shellagent_images.py args...]

uv run python scripts/data/build_shellagent_images.py \
  --input osieosie/shellagent-tasks-skill-taxonomy-20260401-10k \
  --output-dataset hamishivi/swerl-shellagent-15k \
  --registry hamishi740 \
  --repo-prefix swerl-shellagent \
  --platform linux/amd64,linux/arm64 \
  --use-buildx \
  "$@"
