#!/usr/bin/env bash
set -e
source .venv/bin/activate
python -m detector.cli train \
  --dataset configs/dataset.yaml \
  --augment configs/augment.yaml \
  --model configs/model.yaml \
  --train configs/train.yaml
