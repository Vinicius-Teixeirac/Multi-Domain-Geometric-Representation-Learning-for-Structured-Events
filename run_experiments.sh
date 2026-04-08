#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=(42 61 80 33 314)

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

MULTI_DOMAIN_CONFIGS=(
    # ---- existing / hand-crafted configs ----
    src/config/model_setup/multi_domain/multi_domain.yaml
    src/config/model_setup/multi_domain/multi_domain_gat_attention.yaml
    src/config/model_setup/multi_domain/multi_domain_region_fourier_gated.yaml
    src/config/model_setup/multi_domain/multi_domain_riemannian.yaml
    src/config/model_setup/multi_domain/multi_domain_riemannian_aware.yaml

    # ---- actor encoder ablations (geo=hyper, temporal=riem, fusion=ga_gated) ----
    src/config/model_setup/multi_domain/actor_gat.yaml
    src/config/model_setup/multi_domain/actor_weighted.yaml
    src/config/model_setup/multi_domain/actor_attribute_only.yaml

    # ---- geo encoder ablations (actor=sage, temporal=riem, fusion=ga_gated) ----
    src/config/model_setup/multi_domain/geo_projected.yaml
    src/config/model_setup/multi_domain/geo_euclidean.yaml
    src/config/model_setup/multi_domain/geo_region_aware.yaml

    # ---- temporal encoder ablations (actor=sage, geo=hyper, fusion=ga_gated) ----
    src/config/model_setup/multi_domain/temporal_product_manifold.yaml
    src/config/model_setup/multi_domain/temporal_learnable_period.yaml
    src/config/model_setup/multi_domain/temporal_fourier.yaml

    # ---- fusion ablations (actor=sage, geo=hyper, temporal=riem) ----
    src/config/model_setup/multi_domain/fusion_concat.yaml
    src/config/model_setup/multi_domain/fusion_attention.yaml
    src/config/model_setup/multi_domain/fusion_ga_concat.yaml
    src/config/model_setup/multi_domain/fusion_ga_attention.yaml

    # ---- geometry inductive bias ablations ----
    src/config/model_setup/multi_domain/ablation_no_geometry.yaml
    src/config/model_setup/multi_domain/ablation_geo_sphere_only.yaml
    src/config/model_setup/multi_domain/ablation_temporal_sphere_only.yaml
    src/config/model_setup/multi_domain/ablation_full_riemannian_concat.yaml
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
            --mlp-configs "${CONFIG}"
    done

    # -------------------------
    # GNN
    # -------------------------
    for CONFIG in "${GNN_CONFIGS[@]}"; do
        python main.py \
            --seed "${SEED}" \
            --gnn-configs "${CONFIG}"
    done

    # -------------------------
    # BERT
    # -------------------------
    for CONFIG in "${BERT_CONFIGS[@]}"; do
        python main.py \
            --seed "${SEED}" \
            --bert-configs "${CONFIG}"
    done

    # -------------------------
    # Multi-Domain
    # -------------------------
    python main.py \
        --seed "${SEED}" \
        --multi-domain-configs "${MULTI_DOMAIN_CONFIGS[@]}"

done

echo "All experiments completed."
