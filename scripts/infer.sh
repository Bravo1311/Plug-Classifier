#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m detector.cli infer \
  --ckpt runs/exp_001/checkpoints/best.pt \
  --input data/inference \
  --topk 5
