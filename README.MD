# GDELT Representation Learning: A Comparative Study on Inductive Biases

This repository contains the experimental framework for a comparative study on **representation learning and inductive biases** using the **GDELT Event Database**.

The study investigates how different deep learning architectures—operating under different data modeling assumptions—perform on the task of predictive event classification (`QuadClass`). By holding the informational content of the data constant while varying its structural representation (Tabular, Graph, Text), I isolate the effect of inductive bias on performance.

The baseline experiments motivate a **Multi-Domain Geometric Model** (the proposal), which decomposes events into orthogonal geometric domains—relational, spherical, and circular-temporal—and fuses them in a geometry-aware manner.

## 📂 Repository Structure

```
preliminary_experiments/
├── data/                   # Raw parquet files and processed artifacts
├── notebooks/              # EDA, CAMEO code dictionaries, and architecture docs
│   └── multi_domain_architecture.ipynb  # Full architecture reference
├── src/
│   ├── config/             # YAML configs and Schema definitions
│   │   ├── schema/         # Encoding and Column selection logic
│   │   └── model_setup/    # Hyperparameters for all models
│   │       ├── mlp/
│   │       ├── gat/ gin/ graphSAGE/ han/ rgat/ rgcn/
│   │       ├── bert/
│   │       └── multi_domain/   # Multi-domain geometric model configs
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
│   ├── runners/            # Single-responsibility execution scripts
│   └── utils/              # Logging, Metrics, Seeding
├── main.py                 # Primary entry point for all experiments
├── pyproject.toml          # Project dependencies
└── uv.lock                 # Pinned lockfile
```

## Usage

The framework is orchestrated via `main.py`. You can run specific experimental tracks by passing the corresponding configuration files.

### Installation

**Linux (recommended — using [uv](https://docs.astral.sh/uv/)):**
```bash
uv sync
```
The `pyproject.toml` and `uv.lock` already point to the correct PyTorch CUDA 12.1 and PyG wheel indices — no extra flags needed.

**Linux (plain pip):**
```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

pip install torch-geometric transformers pandas scikit-learn scipy \
  matplotlib seaborn pyarrow pyyaml tqdm networkx
```

**Windows:**

pyg-lib has no pre-built Windows wheels, so the install must be done in two steps with the matching torch+CUDA wheel index. The Windows-compatible stack uses torch 2.2.x:

```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

pip install torch-geometric transformers pandas scikit-learn scipy \
  matplotlib seaborn pyarrow pyyaml tqdm networkx
```

> `pyg-lib` (the low-level C++ kernel library) is skipped on Windows — torch-geometric falls back to pure-Python implementations automatically. All models run correctly without it; some sparse operations will be slower.

**Conda (any platform):**
```bash
conda install pyg -c pyg
```

2. Running Experiments

You can trigger multiple model runs in a single execution. The system handles caching, so re-running a command will skip already processed data stages unless `--force` is used.

**Baseline models:**
- Run Tabular Baselines (MLP): `main.py --mlp-configs src/config/model_setup/mlp/mlp.yaml`
- Run Homogeneous GNNs (SAGE, GAT, GIN): `main.py --gnn-configs src/config/model_setup/graphSAGE/graphSAGE_all_features.yaml src/config/model_setup/gat/gat_all_features.yaml`
- Run Heterogeneous GNNs (RGCN, HAN, RGAT): `main.py --gnn-configs src/config/model_setup/rgcn/rgcn_all_features.yaml`
- Run NLP Baselines (BERT): `main.py --bert-configs src/config/model_setup/bert/bert.yaml`

**Proposed multi-domain model:**
- Default variant (SAGE + Euclidean geo + Product-manifold temporal): `main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain.yaml`
- Fully Riemannian (Hyperspherical geo + Riemannian temporal + Gated fusion): `main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain_riemannian.yaml`
- Riemannian + geometry-aware fusion: `main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain_riemannian_aware.yaml`
- GAT actor + attention fusion: `main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain_gat_attention.yaml`
- Region-aware geo + Fourier temporal + gated fusion: `main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain_region_fourier_gated.yaml`

**Run everything at once:**
```
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

Options:
- `--seed`: Set global random seed (default: 42). I acknowledge that pytorch uses randomized linear algebra, which speeds up computations and whose behavior was intentionally kept. Results may vary slightly even for the same seed.
- `--force`: Force re-computation of preprocessing, encoding, and graph building stages.

> The data used in these experiments are far large than what GitHub allows to upload. It was acquired using this other [repository](https://github.com/Vinicius-Teixeirac/GDELT-2.0-EVENT-DATABASE-Pipeline) and you can contact me if you want to see it. I will create a Huggin Face datasets repo soon and update here.

> The provided config files in `src/config/model_setup` were the ones used across the experiments. Of course, an interested reader can edit them to expand the experiments.

---

# The real deal 🔥

What one may need to know about the GDELT Event Database metadata is [on this file](http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf), which is gracefully summarized by me on `notebooks/GDELT_metadata.ipynb`. There it says:

```
GDELT Event records are stored in an expanded version of the dyadic CAMEO format,
capturing two actors and the action performed by Actor1 upon Actor2.
```

These events are extracted from newspaper articles and concern geopolitical happenings. Formally, an "Event" is a first-class semantic unit for representing, interpreting, and reasoning about changes in world state as expressed in data. They are commonly decomposed into "W's" and "H's": *What* happened, *When* it happened, *Where* it happened, *Why* it happened, *Who* did it to *Whom*, ... *How* it happened, *How many/much* ...

GDELT seamlessly supports *What* (action perfomed), *Who* (Actor1), *Whom* (Actor2), *When* (Time Attributes) and *Where* (Geographic Attributes). *What* happens to be the most interesting component to predict in real world applications.

I build three distinct representations of the same underlying data.

> Note on Data Engineering: Detailed analysis of column's roles, features distributions, heavy encoding decisions and cleaning steps are documented in `notebooks/EDA.ipynb`.

## The Builds: Three Views of the Same World (Baseline Experiments)
I hold the informational content constant across all experiments. No model sees "more" data than another; they just see it structured differently.

- **Tabular Build (The Euclidean View):** It treats events as independent, orderless observations. Structure: A flat vector of concatenated features (Actor properties + Geo coordinates + Time).

  Assumption: The class label depends solely on the intrinsic attributes of the event itself, not on its relationship to other events. Format: A standard feature matrix $X \in \mathbb{R}^{N \times F}$.

- **Graph Build (The Relational View):** It treats events as nodes in a connected manifold.

  * Homogeneous Graph: Nodes are Events. Edges exist if events share attributes (e.g., same Actor, same Location, same Day). This explicitly models co-occurrence.
  
  * Heterogeneous Graph: A bipartite structure. Events are nodes; Actors, Locations, and Dates are also nodes. Edges represent semantic typed relations (e.g., Event A $\xrightarrow{\text{has\_actor}}$ Actor B).
  
  Assumption: The class label is heavily influenced by the context—who else is involved, where else this is happening, and what neighbors are doing (homophily).

- **Text Build (The Linguistic View):** It treat events as semantic narratives. Structure: "Prompt-engineered" natural language sentences generated from raw column values. "[WHO] Senator ... involved with [WHOM] New York ... [WHERE] Russia [WHEN] January 1979"

  Assumption: The class label can be inferred from linguistic patterns and semantic compositionality found in pre-trained language models.

## The Models: A Study in Inductive Bias
I employ a suite of architectures to test how different inductive biases—the set of assumptions a model makes to predict outputs for unseen inputs—affect representation learning.

1. **MLP (Multi-Layer Perceptron)** Bias: Weak and general (Universal Function Approximator).
  - Mechanism: MLPs assume functions are smooth or hierarchical but lack domain-specific knowledge like locality (CNNs) or sequence (RNNs). Implication: They are powerful when scaled but data-hungry. They must learn structure purely from the dataset statistics rather than relying on built-in architectural assumptions.

  > In our study, the MLP serves as the Euclidean baseline: how well can we classify events without explicitly modeling their relationships?

2. **Homogeneous GNNs (SAGE, GAT, GIN):** A GNN contrasts with the MLP by requiring a graph structure. It inherently assumes that relationships matter. If the data is truly relational, GNNs exploit this "structural knowledge" to outperform baselines. We test three common "Simple Graph" architectures (Undirected, Unweighted):

  * GraphSAGE (Spatial): Aggregates information from local neighborhoods. It tests the bias of Locality: "Tell me about an event by summarizing its neighbors."
  * GAT (Attention): Uses attention mechanisms to weigh edges dynamically. It tests the bias of Selectivity: "Not all connections are equal; pay attention to the most relevant neighbors."
  * GIN (Isomorphism): Designed to be as powerful as the Weisfeiler-Lehman test. It tests the bias of Discriminative Power: "Can we distinguish between different structural roles in the graph?"

3. **Heterogeneous GNNs (HAN, RGCN, RGAT):** These extend the graph bias to Typed Relations. They assume that a "Shared Location" connection is fundamentally different from a "Shared Actor" connection.

  * RGCN: Relational Graph Convolutional Networks (distinct weights per edge type).
  * HAN: Heterogeneous Graph Attention Network (hierarchical attention over meta-paths).
  * RGAT: Relational Graph Attention Networks.

4. **BERT (Bidirectional Encoder Representations from Transformers)** Setup: BERT-base uncased → Linear Classification Head.

  * Training: I freeze the backbone and fine-tune only the last 1–2 layers on the training split.

  > But BERT was trained over wikipedia, which might contain tons of geopolitical information. Isn't this cheating? No. BERT's pretraining is an inductive bias, not task supervision. Just as Graph models encode priors about connectivity, BERT encodes priors about linguistic structure. But it's true that BERT is by far the largest model.

  * The "Latent Graph" Hypothesis: We use BERT to test a specific theoretical question: Can event semantics be recovered purely from linguistic abstraction? Do co-occurrence patterns and syntax in language approximate relational reasoning? In short: "Can language act as a latent graph?"

---

# The Proposal: Multi-Domain Geometric Model 🌐

The baseline experiments above are all *unidomain*: each model receives events through a single representational lens (tabular, relational graph, or text). However, a GDELT event bundles at least three geometrically distinct information types simultaneously:

| Dimension | GDELT columns | Natural geometry |
|-----------|--------------|-----------------|
| **WHO / WHOM** | `Actor1*`, `Actor2*` (8 categorical attrs each) | Graph — actors are related by co-occurrence |
| **WHERE** | `ActionGeo_Lat`, `ActionGeo_Long`, `ActionGeo_CountryCode` | Sphere $S^2$ — coordinates live on the Earth's surface |
| **WHEN** | `Day` (YYYYMMDD) | Product manifold $\mathbb{R} \times S^1 \times S^1$ — linear time + annual + weekly cycles |
| **WHAT** | `QuadClass` (4-class CAMEO label) | Target $y \in \{0, 1, 2, 3\}$ |

Forcing all three into a single representation space (e.g., a flat vector for MLP, a homogeneous graph for SAGE) necessarily loses structure. The proposed model instead builds one encoder per domain, each with the *right* inductive bias for that domain's geometry, and fuses them:

$$\varphi(e) = \Psi\!\bigl(\varphi_{\text{who}}(e),\; \varphi_{\text{where}}(e),\; \varphi_{\text{when}}(e)\bigr)$$

This decomposition gives the model **orthogonal inductive biases**: relational topology (actor graph), spherical geometry (spatial), and circular periodicity (temporal). Each encoder is responsible for exactly one type of structure, with no overlap in the raw features it consumes.

## Architecture Overview

The full model is detailed in `notebooks/multi_domain_architecture.ipynb`, which covers the mathematical derivations (Riemannian log/exp maps, SphericalLinear, product manifold encoding) and the corresponding PyTorch implementations. A summary follows.

### Actor Domain — WHO/WHOM

An **actor co-occurrence graph** $G = (\mathcal{V}, \mathcal{E})$ is built from the training split: nodes are unique actor profiles (compound key over 8 GDELT attributes), and edges connect actors that co-appeared in at least one training event. Node features are per-attribute learnable embeddings. Message passing runs once per batch on the full graph; per-event actor representations are retrieved by indexing the resulting node embedding matrix for the Actor1/Actor2 pair.

Four encoder variants are available:

| Type | Aggregation | Inductive bias |
|------|------------|----------------|
| `sage_gnn` (default) | Mean pooling (SAGEConv) | Uniform neighbourhood influence |
| `gat_gnn` | Multi-head attention (GATConv) | Selective focus on relevant co-actors |
| `weighted_gnn` | Degree-normalised weighted sum (GCN) | Frequent co-actors more influential |
| `attribute_only` | None — pure MLP on attributes | Fully inductive; no graph required |

After message passing, the Actor1 and Actor2 embeddings are concatenated and projected via a shared MLP: $\varphi_{\text{who}}(e) \in \mathbb{R}^{d_a}$.

### Geo Domain — WHERE

Latitude and longitude are **angles on a sphere**. The natural representation is a point on $S^2 \subset \mathbb{R}^3$ via:

$$\phi_{S^2}(\text{lat}, \text{lon}) = \bigl(\cos(\text{lat})\cos(\text{lon}),\; \cos(\text{lat})\sin(\text{lon}),\; \sin(\text{lat})\bigr)$$

This eliminates the date-line artefact (Samoa and Tonga are correctly adjacent) and the latitude-distortion of raw degrees. The `SphericalLinear` layer then maps $S^2 \to S^{d-1}$ *without ever leaving the manifold*, operating via log/exp maps at the north pole:

$$v = \log_p(x)_{1:d-1} \in \mathbb{R}^{d-1} \xrightarrow{W,b} u \in \mathbb{R}^{d'-1} \xrightarrow{\exp_p} y \in S^{d'-1}$$

Four encoder variants are available:

| Type | Output space | Description |
|------|------------|-------------|
| `hyperspherical` (default) | $S^{d-1}$ | Fully Riemannian: $S^2 \to$ SphericalLinear $\to$ SphereReLU $\to S^{d-1}$ |
| `region_aware` | $S^{d-1}$ | Riemannian path blended with country embedding in tangent space |
| `projected` | $S^{d-1}$ | $S^2 \to$ Euclidean MLP $\to$ L2-normalize (ablation baseline) |
| `euclidean` | $\mathbb{R}^d$ | Plain MLP on (lat, lon) — no spherical bias |

### Temporal Domain — WHEN

Time has a natural product-manifold structure: linear progression $\mathbb{R}$, annual cycle $S^1_{365}$, and weekly cycle $S^1_7$. Fixed sin/cos features capture the fundamental frequency of each cycle; the Riemannian encoder generalises this to a learned multi-frequency bank.

Four encoder variants are available:

| Type | Key idea |
|------|---------|
| `product_manifold` (default) | Fixed sin/cos periods + MLP |
| `learnable_period` | Log-space learnable period parameters |
| `fourier` | Learnable frequency matrix $W \in \mathbb{R}^{K \times 3}$; discovers arbitrary periodicities |
| `riemannian_product` | SphericalLinear on each $S^1$ component; learns a bank of frequencies while maintaining circular topology |

### Fusion — Ψ

The fusion module receives $[\varphi_{\text{who}},\, \varphi_{\text{where}},\, \varphi_{\text{when}}]$ and outputs class logits. Four strategies are available:

| Type | Mechanism |
|------|-----------|
| `concat_mlp` | Late concatenation followed by MLP |
| `attention` | Softmax attention scores per domain; dynamically weights views per event |
| `gated` | Sigmoid gates from global context; soft-suppresses uninformative views (e.g., missing geo) |
| `geometry_aware` | Maps sphere views to tangent space via $\log_p$ before any inner fusion strategy; resolves the scale mismatch between unit-norm sphere outputs and unconstrained GNN outputs |

The `geometry_aware` wrapper is the principled solution to mixed-manifold fusion: a sphere view $z \in S^{d-1}$ is mapped to $v = \log_p(z)_{1:d-1} \in \mathbb{R}^{d-1}$, recovering exactly the $d-1$ degrees of freedom of the manifold as a flat Euclidean vector whose norm equals the geodesic distance from the reference point. Any inner fusion strategy (concat_mlp, attention, or gated) then operates on geometrically homogeneous inputs.

## Config-Driven Ablation

All design choices are controlled by a single YAML file. Swapping a single field (e.g., `model.geo.type: hyperspherical` → `euclidean`) changes the inductive bias without touching any Python code, enabling systematic one-factor-at-a-time ablation.

The five shipped configs cover the main variants:

| Config file | Actor | Geo | Temporal | Fusion |
|-------------|-------|-----|----------|--------|
| `multi_domain.yaml` | `sage_gnn` | `euclidean` | `product_manifold` | `concat_mlp` |
| `multi_domain_riemannian.yaml` | `sage_gnn` | `hyperspherical` | `riemannian_product` | `gated` |
| `multi_domain_riemannian_aware.yaml` | `sage_gnn` | `hyperspherical` | `riemannian_product` | `geometry_aware(gated)` |
| `multi_domain_gat_attention.yaml` | `gat_gnn` | `hyperspherical` | `product_manifold` | `attention` |
| `multi_domain_region_fourier_gated.yaml` | `weighted_gnn` | `region_aware` | `fourier` | `gated` |

The `multi_domain.yaml` config (SAGE + Euclidean geo + fixed product manifold + concat MLP) serves as the weakest multi-domain baseline, isolating the contribution of multi-domain decomposition alone from geometric encoding. Each subsequent config progressively adds geometric structure, and `multi_domain_riemannian_aware.yaml` is the fully geometry-aware variant.

### Running the Proposed Model

```bash
# Default multi-domain (multi-domain decomposition, Euclidean encoders)
python main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain.yaml

# Fully Riemannian variant
python main.py --multi-domain-configs src/config/model_setup/multi_domain/multi_domain_riemannian_aware.yaml

# All five variants at once
python main.py --multi-domain-configs \
  src/config/model_setup/multi_domain/multi_domain.yaml \
  src/config/model_setup/multi_domain/multi_domain_riemannian.yaml \
  src/config/model_setup/multi_domain/multi_domain_riemannian_aware.yaml \
  src/config/model_setup/multi_domain/multi_domain_gat_attention.yaml \
  src/config/model_setup/multi_domain/multi_domain_region_fourier_gated.yaml
```

Results are written to `results/{dataset_name}/MultiDomainGeometricModel/` as JSON, following the same structure as the unidomain model outputs.

For a complete walkthrough of the architecture, mathematical derivations, and code, see `notebooks/multi_domain_architecture.ipynb`.
