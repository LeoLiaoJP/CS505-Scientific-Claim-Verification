#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"
cd "$PROJECT_ROOT"

python -m pip install -r requirements_gpu.txt
python scripts/run_midway_experiments.py \
  --preset gpu_midway \
  --output-dir results/midway_local_gpu_10ep
