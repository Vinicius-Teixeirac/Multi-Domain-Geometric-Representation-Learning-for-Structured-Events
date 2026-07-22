# Multi-Domain Geometric Representation Learning for Structured Events

**This repository is the official experimental code release for an accepted academic paper.** It is not a general-purpose library: it is the exact codebase used to produce every baseline and ablation result reported in **"Multi-Domain Geometric Representation Learning for Structured Events,"** accepted at **BRACIS 2026** (Brazilian Conference on Intelligent Systems).

> **Status:** Accepted, awaiting camera-ready / proceedings publication. The paper is not yet published; this repo will be updated with the camera-ready link/DOI once available.

**Authors:** Vinícius T. C. Freitas, Kenzo M. Sakiyama, Ricardo M. Marcacini
Institute of Mathematics and Computer Sciences (ICMC), University of São Paulo (USP), São Carlos, SP, Brazil

### Citation

A formal citation (with DOI and proceedings details) will be added once the paper is published. In the meantime, please cite it as:

```bibtex
@inproceedings{freitas2026multidomain,
  title     = {Multi-Domain Geometric Representation Learning for Structured Events},
  author    = {Freitas, Vin{\'i}cius T. C. and Sakiyama, Kenzo M. and Marcacini, Ricardo M.},
  booktitle = {Proceedings of the Brazilian Conference on Intelligent Systems (BRACIS)},
  year      = {2026},
  note      = {To appear}
}
```

---

## Objectives

This repository backs a comparative study on **representation learning and inductive biases**, using the **GDELT Event Database**, in two parts:

1. **Baseline study.** Hold the informational content of the data constant and vary only its structural representation, to isolate the effect of inductive bias on predictive event classification (`QuadClass`):
   - **Tabular**: events as independent, orderless feature vectors (Actor + Geo + Time concatenated). Assumption: the label depends only on the event's own attributes.
   - **Graph**: events (homogeneous) or events+actors+locations+dates (heterogeneous) as nodes in a relational structure. Assumption: the label depends on context, who else is involved, where else, what neighbors are doing.
   - **Text**: events as prompt-engineered natural-language sentences. Assumption: the label is recoverable from linguistic patterns learned by a pretrained language model.

   Models tested per representation: **MLP** (tabular baseline), **homogeneous GNNs** — GraphSAGE / GAT / GIN (locality, selective attention, and structural-discrimination biases respectively), **heterogeneous GNNs** — RGCN / HAN / RGAT (typed-relation bias), and **BERT** (linguistic pretraining as inductive bias).

2. **Proposed model.** The baseline results motivate the **Multi-Domain Geometric Model**: rather than forcing a GDELT event into one representational lens, it decomposes the event into three geometrically distinct domains and encodes each with the inductive bias native to its geometry, then fuses them:

   | Dimension | GDELT columns | Natural geometry |
   |-----------|--------------|-----------------|
   | **WHO / WHOM** | `Actor1*`, `Actor2*` (8 categorical attrs each) | Graph: actors related by co-occurrence |
   | **WHERE** | `ActionGeo_Lat`, `ActionGeo_Long`, `ActionGeo_CountryCode` | Sphere $S^2$: coordinates live on the Earth's surface |
   | **WHEN** | `Day` (YYYYMMDD) | Product manifold $\mathbb{R} \times S^1 \times S^1$: linear time + annual + weekly cycles |
   | **WHAT** | `QuadClass` (4-class CAMEO label) | Target $y \in \{0, 1, 2, 3\}$ |

   $$\varphi(e) = \Psi\!\bigl(\varphi_{\text{who}}(e),\; \varphi_{\text{where}}(e),\; \varphi_{\text{when}}(e)\bigr)$$

   Each domain encoder (actor graph GNN, spherical Riemannian encoder, circular-temporal encoder) and the fusion mechanism $\Psi$ are swappable via config, which is what the [Configuration & Ablations](#configuration--ablations) section below drives.

   **The full mathematical derivations (Riemannian log/exp maps, `SphericalLinear`, product-manifold encoding) and their exact PyTorch implementations are documented, derived, and executed step-by-step in [`notebooks/multi_domain_architecture.ipynb`](notebooks/multi_domain_architecture.ipynb).** That notebook is the canonical architecture reference, this README only summarizes it.

---

## Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | ≥3.12 | Runtime |
| PyTorch | 2.10.0 (cu126) | Core tensor / autograd engine |
| PyTorch Geometric | 2.8.0.post1 (+ `pyg-lib`) | GNN layers (SAGE, GAT, GIN, RGCN, HAN, RGAT) |
| Transformers | 5.5.0 | BERT baseline |
| pandas / pyarrow | 2.3.3 / 23.0.1 | Tabular data + parquet I/O |
| scikit-learn / scipy | 1.8.0 / 1.16.3 | Preprocessing, metrics |
| matplotlib / seaborn | 3.10.8 / 0.13.2 | EDA and figures (notebooks) |
| networkx | 3.6.1 | Actor co-occurrence graph tooling |
| pytest | ≥9.0.3 | Unit + integration tests |

Dependencies are pinned in `pyproject.toml` / `uv.lock`.

### Installation

**Recommended, using [uv](https://docs.astral.sh/uv/) (Linux and Windows):**
```bash
uv sync
```
The `pyproject.toml` and `uv.lock` already point to the correct PyTorch CUDA 12.6 and PyG wheel indices, no extra flags needed. Verified end-to-end on both platforms (full test suite + real CUDA training): PyG ships `pyg-lib` wheels for Windows too, so no platform-specific workaround is needed. Note the CUDA index is 12.6, not 12.1: PyG stopped publishing Windows wheels for `pyg-lib` against cu121 past PyTorch 2.4, so cu126 is required from PyTorch 2.6 onward.

**Plain pip (Linux and Windows):**
```bash
pip install torch==2.10.0+cu126 torchvision==0.25.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126

pip install pyg_lib \
  -f https://data.pyg.org/whl/torch-2.10.0+cu126.html

pip install torch-geometric transformers pandas scikit-learn scipy \
  matplotlib seaborn pyarrow pyyaml tqdm networkx
```

> If `pyg-lib` ever has no wheel for your exact Python version, OS, or CUDA combo (e.g. a very new Python release PyG hasn't published for yet), fall back to an older, longer-supported combo such as `torch==2.4.1+cu121` (`torchvision==0.19.1+cu121`) with the matching `-f https://data.pyg.org/whl/torch-2.4.1+cu121.html` index, and drop `pyg_lib` from the install if it's unavailable there: torch-geometric falls back to pure-Python implementations automatically, just slower on sparse ops. Beware that PyG stopped publishing Windows `pyg-lib` wheels for some cu121 builds past PyTorch 2.4 (e.g. `torch==2.6.0+cu121` has none), check https://data.pyg.org/whl/ for your target combo before committing to it.

**Conda (any platform):**
```bash
conda install pyg -c pyg
```

---

## Repository Structure

```
.
├── data/                   # Raw parquet files and processed artifacts
├── notebooks/              # EDA, CAMEO code dictionaries, and architecture docs
│   └── multi_domain_architecture.ipynb  # Full architecture reference
├── src/
│   ├── config/
│   │   ├── data/           # Column definitions and dtype schema
│   │   ├── schema/         # Encoding and column selection logic
│   │   └── model_setup/    # Hyperparameters for all models
│   │       ├── mlp/
│   │       ├── gat/ gin/ graphSAGE/ han/ rgat/ rgcn/
│   │       ├── bert/
│   │       ├── multi_domain/   # Multi-domain geometric model configs (headline + ablations)
│   │       └── smoke/          # Tiny fast-running configs for pipeline smoke tests
│   ├── models/             # PyTorch modules
│   │   ├── mlp/            # MLP baseline
│   │   ├── gnn/            # Homogeneous & Heterogeneous GNNs
│   │   ├── bert/           # BERT fine-tuning module
│   │   └── multi_domain/   # Proposed model
│   │       ├── model.py            # MultiDomainGeometricModel (top-level)
│   │       ├── actor_encoders.py   # WHO/WHOM: SAGE, GAT, Weighted, Attribute-only
│   │       ├── geo_encoders.py     # WHERE: Hyperspherical, Projected, Euclidean, Region-aware
│   │       ├── temporal_encoders.py # WHEN: Product, Learnable, Fourier, Riemannian
│   │       ├── fusion.py           # Concat-MLP, Attention, Gated, Geometry-aware
│   │       ├── riemannian.py       # log/exp maps, SphericalLinear, SphereReLU
│   │       ├── dataset.py
│   │       └── datamodule.py
│   ├── preprocessing/      # Cleaning, Splitting, Entity Construction
│   ├── representation/     # Feature engineering pipelines
│   │   ├── tabular/        # Encoders and Transformers
│   │   ├── graph/          # Homogeneous & Heterogeneous builders
│   │   ├── text/           # Natural Language Template builders
│   │   └── multi_domain/   # Actor co-occurrence graph builder
│   ├── runners/            # Single-responsibility execution scripts (one per model family/stage)
│   ├── training/           # Shared training loop (train.py)
│   ├── testing/            # Shared evaluation loop (evaluate.py)
│   └── utils/              # Logging, Metrics, Seeding, idempotency helpers
├── tests/                  # Pytest unit suite (+ opt-in integration smoke test)
├── scripts/                # One-off maintenance scripts (e.g. generate_smoke_configs.py)
├── logs/                   # Run logs written by src/utils/experiments_logging.py
├── results/                # Per-dataset, per-model JSON results
├── main.py                 # Primary entry point for all experiments
├── run_experiments.sh      # Full paper reproduction: every config below, across 5 seeds
├── pyproject.toml          # Project dependencies
└── uv.lock                 # Pinned lockfile
```

---

## Pipeline Flow

`main.py` discovers every dataset (parquet stem) under `data/raw/` and, for each one, runs the following stages in order. Every stage is **idempotent**, it checks for existing cached artifacts and skips recomputation unless `--force` is passed.

1. **Cleaning**: validate/cast columns against the GDELT codebook schema, drop rows with missing `QuadClass`, remap `QuadClass` to 0–3, apply per-column missing-value policies.
2. **Splitting**: stratified train/val/test split, tagged by `{base_tag}_s{seed}` so different seeds and different config groups never collide. Deduplicated across configs that share a split tag.
3. **Entity construction, tabular features, text building**: only for MLP/GNN/BERT configs; the multi-domain model reads raw splits directly and builds its own actor co-occurrence graph and per-domain views inside its runner.
4. **Train + evaluate**, once per supplied YAML config, for each of the four model families:
   - `--mlp-configs` -> `src/runners/mlp_runner.py`
   - `--gnn-configs` -> `src/runners/gnn_runner.py` (homogeneous or heterogeneous, inferred from config)
   - `--bert-configs` -> `src/runners/bert_runner.py`
   - `--multi-domain-configs` -> `src/runners/multi_domain_runner.py`

   Each run gets a stable, MD5-derived `exp_id` used to skip already-completed runs (a results JSON must exist, not just a checkpoint), GPU memory is released between runs, and results are written as JSON under `results/{dataset}/{ModelClass}/`. Each results JSON also carries exact per-sample GFLOPs and forward-pass latency/throughput, for compute-cost comparisons across model families (`src/utils/profiling.py`).

---

## Usage

You can trigger multiple model runs across model families and config variants in a single `main.py` invocation.

**Baseline models:**
- Tabular baseline (MLP): `main.py --mlp-configs src/config/model_setup/mlp/mlp.yaml`
- Homogeneous GNNs (SAGE, GAT, GIN): `main.py --gnn-configs src/config/model_setup/graphSAGE/graphSAGE_all_features.yaml src/config/model_setup/gat/gat_all_features.yaml`
- Heterogeneous GNNs (RGCN, HAN, RGAT): `main.py --gnn-configs src/config/model_setup/rgcn/rgcn_all_features.yaml`
- NLP baseline (BERT): `main.py --bert-configs src/config/model_setup/bert/bert.yaml`

**Proposed multi-domain model (headline configs):**
```bash
# Default: SAGE actor + hyperspherical geo + product-manifold temporal + concat-MLP fusion
python main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain.yaml

# Fully Riemannian variant (reference full model)
python main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain_riemannian_aware.yaml
```

**Run everything at once:**
```bash
python main.py \
  --mlp-configs src/config/model_setup/mlp/mlp.yaml \
  --gnn-configs src/config/model_setup/gat/gat_all_features.yaml \
               src/config/model_setup/gin/gin_all_features.yaml \
               src/config/model_setup/graphSAGE/graphSAGE_all_features.yaml \
               src/config/model_setup/han/han_all_features.yaml \
               src/config/model_setup/rgat/rgat_all_features.yaml \
               src/config/model_setup/rgcn/rgcn_all_features.yaml \
  --bert-configs src/config/model_setup/bert/bert.yaml \
  --multi-domain-configs src/config/model_setup/multi_domain/multi_domain_riemannian_aware.yaml
```

**Options:**
- `--seed`: Global random seed (default: 42). PyTorch's randomized linear algebra is intentionally left enabled for speed, so results may vary slightly even for the same seed.
- `--force`: Recompute all stages (ignore cached artifacts).
- `--datasets`: Restrict the run to specific dataset stems (`.parquet` filenames without extension) under `data/raw/`. Defaults to all discovered datasets.

> The provided config files in `src/config/model_setup` were the ones used across the experiments reported in the paper. An interested reader can edit them to expand the experiments.

---

## Configuration & Ablations

Every model is fully specified by a YAML file, no code changes needed to change hyperparameters or swap an inductive bias. For the multi-domain model specifically, swapping a single field (e.g. `model.geo.type: hyperspherical` -> `euclidean`) changes the geometry of one domain in isolation.

Five headline configs (`src/config/model_setup/multi_domain/`) cover the main variants:

| Config file | Actor | Geo | Temporal | Fusion |
|-------------|-------|-----|----------|--------|
| `multi_domain.yaml` | `sage_gnn` | `hyperspherical` | `product_manifold` | `concat_mlp` |
| `multi_domain_riemannian.yaml` | `sage_gnn` | `hyperspherical` | `riemannian_product` | `gated` |
| `multi_domain_riemannian_aware.yaml` | `sage_gnn` | `hyperspherical` | `riemannian_product` | `geometry_aware(gated)` |
| `multi_domain_gat_attention.yaml` | `gat_gnn` | `hyperspherical` | `product_manifold` | `attention` |
| `multi_domain_region_fourier_gated.yaml` | `weighted_gnn` | `region_aware` | `fourier` | `gated` |

`multi_domain_riemannian_aware.yaml` is the reference full model; the other four vary actor, temporal, or fusion choices around it.

Beyond the headline five, the same directory ships ~25 single-factor ablation configs that hold the reference model fixed and swap exactly one component:

- **Actor** (`actor_gat.yaml`, `actor_weighted.yaml`, `actor_attribute_only.yaml`)
- **Geo** (`geo_euclidean.yaml`, `geo_projected.yaml`, `geo_region_aware.yaml`)
- **Temporal** (`temporal_fourier.yaml`, `temporal_learnable_period.yaml`, `temporal_product_manifold.yaml`)
- **Fusion** (`fusion_concat.yaml`, `fusion_attention.yaml`, `fusion_ga_concat.yaml`, `fusion_ga_attention.yaml`)
- **Geometry ladder** (`ablation_no_geometry.yaml` -> `ablation_geo_sphere_only.yaml` / `ablation_temporal_sphere_only.yaml` -> `ablation_full_riemannian_concat.yaml`): progressively reintroduces Riemannian structure, from a fully Euclidean model up to the fully Riemannian one.

A parallel `src/config/model_setup/smoke/` directory mirrors every model family with tiny, fast-running configs for smoke-testing the pipeline (generated by `scripts/generate_smoke_configs.py`).

Every encoder/fusion type referenced above, its exact formula, and a runnable code demo are covered in [`notebooks/multi_domain_architecture.ipynb`](notebooks/multi_domain_architecture.ipynb).

Results are written to `results/{dataset_name}/MultiDomainGeometricModel/` as JSON, following the same structure as the unidomain model outputs.

---

## Data

The GDELT Event Database stores events in an expanded dyadic CAMEO format, capturing two actors and the action performed by Actor1 upon Actor2. See the [official codebook (PDF)](http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf), summarized in [`notebooks/GDELT_metadata.ipynb`](notebooks/GDELT_metadata.ipynb); column-level distributions, encoding choices, and cleaning steps are documented in [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb).

The data used in these experiments is provided via [Git LFS](https://git-lfs.com/) in `data/raw/` (it was acquired using this companion [pipeline repository](https://github.com/Vinicius-Teixeirac/GDELT-2.0-EVENT-DATABASE-Pipeline)). Make sure you have Git LFS installed (`git lfs install`) before cloning, otherwise the parquet files will be downloaded as pointer stubs.

---

## Reproducibility

- **Seeds:** `--seed` (default 42) seeds Python/NumPy/PyTorch RNGs. `run_experiments.sh` reproduces the paper's full experimental protocol: every baseline and multi-domain config above, run across 5 seeds (`42, 61, 80, 33, 314`).
- **Caching / idempotency:** every pipeline stage (cleaning, splitting, feature building, training) checks for existing artifacts keyed by dataset, split tag, and config hash, and skips recomputation unless `--force` is passed.
- **Non-determinism caveat:** PyTorch's randomized linear algebra is intentionally left enabled for speed, so exact metrics may vary slightly run-to-run even at a fixed seed.
- **Tests:** `tests/` contains the unit suite covering preprocessing, encoders, and idempotency logic (`pytest`); an opt-in integration smoke test exercises the full pipeline end-to-end (`pytest -m integration`).

---

## License

This project is licensed under the [MIT License](LICENSE).
