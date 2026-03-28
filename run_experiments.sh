#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=(61 80 33 314)

MLP_CONFIGS=(
    src/config/model_setup/mlp/mlp.yaml
)

GNN_CONFIGS=(
    src/config/model_setup/graphSAGE/graphSAGE_all_features.yaml
    src/config/model_setup/graphSAGE/graphSAGE_no_features.yaml
    src/config/model_setup/gat/gat_all_features.yaml
    src/config/model_setup/gat/gat_no_features.yaml
    src/config/model_setup/gin/gin_all_features.yaml
    src/config/model_setup/rgcn/rgcn_all_features.yaml
    src/config/model_setup/rgcn/rgcn_no_features.yaml
    src/config/model_setup/han/han_all_features.yaml
    src/config/model_setup/han/han_no_features.yaml
    src/config/model_setup/rgat/rgat_all_features.yaml
    src/config/model_setup/rgat/rgat_no_features.yaml
)

BERT_CONFIGS=(
    src/config/model_setup/bert/bert.yaml
)

for SEED in "${SEEDS[@]}"; do

    echo "========================================"
    echo "SEED: ${SEED}"
    echo "========================================"

    # -------------------------
    # MLP
    # -------------------------
    for CONFIG in "${MLP_CONFIGS[@]}"; do
        python main.py \
            --seed "${SEED}" \
            --config "${CONFIG}" \
            --model-type mlp
    done

    # -------------------------
    # GNN
    # -------------------------
    for CONFIG in "${GNN_CONFIGS[@]}"; do
        python main.py \
            --seed "${SEED}" \
            --config "${CONFIG}" \
            --model-type gnn
    done

    # -------------------------
    # BERT
    # -------------------------
    for CONFIG in "${BERT_CONFIGS[@]}"; do
        python main.py \
            --seed "${SEED}" \
            --config "${CONFIG}" \
            --model-type bert
    done

done

echo "All experiments completed."