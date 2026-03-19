# GDELT Representation Learning: A Comparative Study on Inductive Biases

This repository contains the experimental framework for a comparative study on **representation learning and inductive biases** using the **GDELT Event Database**.

The study investigates how different deep learning architectures—operating under different data modeling assumptions—perform on the task of predictive event classification (`QuadClass`). By holding the informational content of the data constant while varying its structural representation (Tabular, Graph, Text), I isolate the effect of inductive bias on performance.

## 📂 Repository Structure

```
preliminary_experiments/
├── data/                   # Raw parquet files and processed artifacts
├── notebooks/              # EDA and CAMEO code dictionaries
├── src/
│   ├── config/             # YAML configs and Schema definitions
│   │   ├── schema/         # Encoding and Column selection logic
│   │   └── model_setup/    # Hyperparameters for MLP, GNNs, BERT
│   ├── models/             # PyTorch Lightning modules (MLP, GNN, BERT)
│   ├── preprocessing/      # Cleaning, Splitting, Entity Construction
│   ├── representation/     # Feature engineering pipelines
│   │   ├── tabular/        # Encoders and Transformers
│   │   ├── graph/          # Homogeneous & Heterogeneous builders
│   │   └── text/           # Natural Language Template builders
│   ├── runners/            # Single-responsibility execution scripts
│   └── utils/              # Logging, Metrics, Seeding
├── main.py                 # Primary entry point for experiments
└── linux_requirements.txt  # Dependencies
```

## Usage

The framework is orchestrated via main.py. You can run specific experimental tracks by passing the corresponding configuration files.

1. `pip install -r linux_requirements.txt` or `pip install -r windows_requirements.txt`

> The pyg-lib low-level library currently does not offer pre-built Windows wheels, which makes windows_requirements trickier. Also, wheels for pyg-lib aren’t on PyPI — they’re hosted on a separate wheel index (data.pyg.org) with tags that include the PyTorch version and CUDA version. Install PyG dependencies from the correct wheel source:

```
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```

or `conda install pyg -c pyg`


2. Running Experiments

You can trigger multiple model runs in a single execution. The system handles caching, so re-running a command will skip already processed data stages unless --force is used.

- Run Tabular Baselines (MLP): `main.py --mlp-configs src/config/model_setup/mlp/mlp.yaml`
- Run Homogeneous GNNs (SAGE, GAT, GIN): ``` main.py --gnn-configs 
    src/config/model_setup/graphSAGE/graphSAGE_all_features.yaml 
    src/config/model_setup/gat/gat_all_features.yaml ```
- Run Heterogeneous GNNs (RGCN, HAN, RGAT): `main.py --gnn-configs src/config/model_setup/rgcn/rgcn_all_features.yaml`
- Run NLP Baselines (BERT): `main.py --bert-configs src/config/model_setup/bert/bert.yaml`
- Every model: ```python main.py --mlp-configs src/config/model_setup/mlp/mlp.yaml --gnn-configs src/config/model_setup/gat/gat_all_features.yaml src/config/model_setup/gat/gat_no_features.yaml src/config/model_setup/gin/gin_all_features.yaml src/config/model_setup/graphSAGE/graphSAGE_all_features.yaml src/config/model_setup/graphSAGE/graphSAGE_no_features.yaml src/config/model_setup/han/han_all_features.yaml src/config/model_setup/han/han_no_features.yaml src/config/model_setup/rgat/rgat_all_features.yaml src/config/model_setup/rgat/rgat_no_features.yaml src/config/model_setup/rgcn/rgcn_all_features.yaml src/config/model_setup/rgcn/rgcn_no_features.yaml --bert-configs src/config/model_setup/bert/bert.yaml```
Options: 
- `--seed`: Set global random seed (default: 42). I acknowledge that pytorch uses randomized linear algebra, which speeds up computations and whose behavior was intentionally kept. Results may vary slightly even for the same seed.

- `--force`: Force re-computation of preprocessing, encoding, and graph building stages.

> The data used in these experiments are far large than what GitHub allows to upload. It was acquired using this other [repository](https://github.com/Vinicius-Teixeirac/GDELT-2.0-EVENT-DATABASE-Pipeline) and you can contact me if you want to see it. I will create a Huggin Face datasets repo soon and update here.

> The provided config files in src/config/model_setup were the ones used across the experiments. Of course, an interested reader can edit them to expand the experiments.

# The real deal 🔥

What one may need to know about the GDELT Event Database metadata is [on this file](http://data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf), which is gracefully summarized by me on notebooks/GDELT_metadata.ipynb. There it says:

```GDELT Event records are stored in an expanded version of the dyadic CAMEO format, **capturing two actors and the action performed by Actor1 upon Actor2**. ```

These events are extracted from newspaper articles and concern geopolitical happenings. Formally, an "Event" is a first-class semantic unit for representing, interpreting, and reasoning about changes in world state as expressed in data. They are commonly decomposed into "W's" and "H's": *What* happened, *When* it happened, *Where* it happened, *Why* it happened, *Who* did it to *Whom*, ... *How* it happened, *How many/much* ...

GDELT seamlessly supports *What* (action perfomed), *Who* (Actor1), *Whom* (Actor2), *When* (Time Attributes) and *Where* (Geographic Attributes). *What* happens to be the most interesting component to predict in real world applications. 

I build three distinct representations of the same underlying data.

> Note on Data Engineering: Detailed analysis of column's roles, features distributions, heavy encoding decisions and cleaning steps are documented in notebooks/EDA.ipynb.

## The Builds: Three Views of the Same World 
I hold the informational content constant across all experiments. No model sees "more" data than another; they just see it structured differently.

- Tabular Build (The Euclidean View): It treats events as independent, orderless observations. Structure: A flat vector of concatenated features (Actor properties + Geo coordinates + Time). 

Assumption: The class label depends solely on the intrinsic attributes of the event itself, not on its relationship to other events. Format: A standard feature matrix $X \in \mathbb{R}^{N \times F}$.

- Graph Build (The Relational View): It treats events as nodes in a connected manifold.

  * Homogeneous Graph: Nodes are Events. Edges exist if events share attributes (e.g., same Actor, same Location, same Day). This explicitly models co-occurrence.
  
  * Heterogeneous Graph: A bipartite structure. Events are nodes; Actors, Locations, and Dates are also nodes. Edges represent semantic typed relations (e.g., Event A $\xrightarrow{\text{has\_actor}}$ Actor B). 
  
Assumption: The class label is heavily influenced by the context—who else is involved, where else this is happening, and what neighbors are doing (homophily).

- Text Build (The Linguistic View): It treat events as semantic narratives. Structure: "Prompt-engineered" natural language sentences generated from raw column values. "[WHO] Senator ... involved with [WHOM] New York ... [WHERE] Russia [WHEN] January 1979" 

Assumption: The class label can be inferred from linguistic patterns and semantic compositionality found in pre-trained language models.

## The Models: A Study in Inductive Bias
I employ a suite of architectures to test how different inductive biases—the set of assumptions a model makes to predict outputs for unseen inputs—affect representation learning.

1. MLP (Multi-Layer Perceptron) Bias: Weak and general (Universal Function Approximator). 
  - Mechanism: MLPs assume functions are smooth or hierarchical but lack domain-specific knowledge like locality (CNNs) or sequence (RNNs).Implication: They are powerful when scaled but data-hungry. They must learn structure purely from the dataset statistics rather than relying on built-in architectural assumptions. 

> In our study, the MLP serves as the Euclidean baseline: how well can we classify events without explicitly modeling their relationships?

2. Homogeneous GNNs (SAGE, GAT, GIN): A GNN contrasts with the MLP by requiring a graph structure. It inherently assumes that relationships matter. If the data is truly relational, GNNs exploit this "structural knowledge" to outperform baselines. We test three common "Simple Graph" architectures (Undirected, Unweighted): 

  * GraphSAGE (Spatial): Aggregates information from local neighborhoods. It tests the bias of Locality: "Tell me about an event by summarizing its neighbors."

  * GAT (Attention): Uses attention mechanisms to weigh edges dynamically. It tests the bias of Selectivity: "Not all connections are equal; pay attention to the most relevant neighbors."

  * GIN (Isomorphism): Designed to be as powerful as the Weisfeiler-Lehman test. It tests the bias of Discriminative Power: "Can we distinguish between different structural roles in the graph?"

3. Heterogeneous GNNs (HAN, RGCN, RGAT): These extend the graph bias to Typed Relations. They assume that a "Shared Location" connection is fundamentally different from a "Shared Actor" connection. 

  * RGCN: Relational Graph Convolutional Networks (distinct weights per edge type). 
  * HAN: Heterogeneous Graph Attention Network (hierarchical attention over meta-paths).
  * RGAT: Relational Graph Attention Networks.

4. BERT (Bidirectional Encoder Representations from Transformers) Setup: BERT-base uncased $\rightarrow$ Linear Classification Head.

  * Training: I freeze the backbone and fine-tune only the last 1–2 layers on the training split.

  > But BERT was trained over wikipedia, which might contain tons of geopolitical information. Isn't this cheating? No. BERT’s pretraining is an inductive bias, not task supervision. Just as Graph models encode priors about connectivity, BERT encodes priors about linguistic structure. But it's true that BERT is by far the largest model.

  * The "Latent Graph" Hypothesis: We use BERT to test a specific theoretical question: Can event semantics be recovered purely from linguistic abstraction?Do co-occurrence patterns and syntax in language approximate relational reasoning? In short: "Can language act as a latent graph?"