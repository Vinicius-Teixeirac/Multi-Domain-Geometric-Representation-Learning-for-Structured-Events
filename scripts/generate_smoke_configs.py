"""Dev utility: regenerate the low-config smoke-test YAMLs from the real configs.

Reads a representative config per model family, overrides training
hyperparameters (epochs, batch_size, patience, num_neighbors) down to
values suitable for the 500-row smoke-test dataset, and writes the result
under src/config/model_setup/smoke/. Not part of the runtime pipeline;
run manually (`python scripts/generate_smoke_configs.py`) if the smoke
configs need to be refreshed after a real config changes shape.
"""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "config" / "model_setup"
OUT = SRC / "smoke"
OUT.mkdir(parents=True, exist_ok=True)

SMOKE_TRAINING_COMMON = dict(batch_size=32, epochs=5, patience=5)

# (source relative path, output filename, extra overrides)
TARGETS = [
    ("mlp/mlp.yaml", "mlp.yaml", {}),
    ("graphSAGE/graphSAGE_all_features.yaml", "graphSAGE.yaml", {"graph.num_neighbors": [5, 5]}),
    ("gat/gat_all_features.yaml", "gat.yaml", {"graph.num_neighbors": [5, 5]}),
    ("gin/gin_all_features.yaml", "gin.yaml", {"graph.num_neighbors": [5, 5]}),
    ("han/han_all_features.yaml", "han.yaml", {"graph.num_neighbors": [5, 5]}),
    ("rgat/rgat_all_features.yaml", "rgat.yaml", {"graph.num_neighbors": [5, 5]}),
    ("rgcn/rgcn_all_features.yaml", "rgcn.yaml", {"graph.num_neighbors": [5, 5]}),
    ("bert/bert.yaml", "bert.yaml", {"text.max_length": 64, "training.epochs": 2}),
    ("multi_domain/multi_domain.yaml", "multi_domain.yaml", {}),
    ("multi_domain/multi_domain_riemannian.yaml", "multi_domain_riemannian.yaml", {}),
    ("multi_domain/multi_domain_riemannian_aware.yaml", "multi_domain_riemannian_aware.yaml", {}),
    ("multi_domain/multi_domain_gat_attention.yaml", "multi_domain_gat_attention.yaml", {}),
    ("multi_domain/multi_domain_region_fourier_gated.yaml", "multi_domain_region_fourier_gated.yaml", {}),
]


def set_path(d: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def main():
    for src_rel, out_name, overrides in TARGETS:
        src_path = SRC / src_rel
        with open(src_path) as f:
            cfg = yaml.safe_load(f)

        cfg.setdefault("training", {}).update(SMOKE_TRAINING_COMMON)

        for dotted_key, value in overrides.items():
            set_path(cfg, dotted_key, value)

        out_path = OUT / out_name
        with open(out_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
