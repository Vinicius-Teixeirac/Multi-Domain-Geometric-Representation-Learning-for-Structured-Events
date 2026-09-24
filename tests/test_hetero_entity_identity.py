"""Tests for cross-split entity identity in the featureless heterogeneous graph.

Each split's graph numbers its actor, geo and day nodes locally. The featureless
models learn one embedding per entity on the training graph, so evaluation must
address that table by an identity stable across splits. Addressing it by node
position read another entity's row for every test node, silently: the model
still ran and still produced numbers.
"""

from typing import cast

import pandas as pd
import pytest
import torch
from torch import nn
from torch_geometric.data import HeteroData

import src.representation.graph.heterogeneous.loaders as hetero_loaders
from src.models.gnn.heterogeneous import HeterogeneousGNN
from src.representation.graph.heterogeneous.builder import (
    HeterogeneousEventGraphBuilder,
)

B = HeterogeneousEventGraphBuilder


def _frame(actors, days, start=0):
    n = len(actors)
    return pd.DataFrame({
        "GlobalEventID": list(range(start, start + n)),
        "Actor1ID": actors,
        "Actor2ID": ["X"] * n,
        "Event_GeoID": ["g"] * n,
        "Day": days,
        "QuadClass": [i % 4 for i in range(n)],
    })


# Train sees A, B, C; test sees C and D. C is the last train node (position 2)
# but the first test node (position 0), so position and identity disagree.
TRAIN = _frame(["A", "B", "C", "A"], ["1", "1", "2", "3"])
TEST = _frame(["C", "D", "C"], ["3", "4", "4"], start=100)


def _write(root, split, df, dataset="ident", tag="default"):
    out = root / dataset
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{split}_{tag}_entities.parquet", index=False)


def _entity_at(df, col, position):
    """The entity a split-local node position stands for (sorted order)."""
    return sorted(df[col].unique())[position]


class TestBuilderVocabulary:
    def _build(self, tmp_path):
        _write(tmp_path, "train", TRAIN)
        _write(tmp_path, "test", TEST)
        train_builder = B(tmp_path, "ident", "train")
        train = train_builder.build()
        test = B(tmp_path, "ident", "test", vocab=train_builder.vocab).build()
        return train_builder.vocab, train, test

    def test_training_entities_get_rows_from_one(self, tmp_path):
        vocab, train, _ = self._build(tmp_path)
        assert vocab[B.ACTOR1] == {"A": 1, "B": 2, "C": 3}
        assert train[B.ACTOR1].vocab_id.tolist() == [1, 2, 3]

    def test_a_test_node_carries_its_own_entity_row(self, tmp_path):
        """C is at position 0 in test and 2 in train; it must read row 3."""
        vocab, _, test = self._build(tmp_path)
        for position, row in enumerate(test[B.ACTOR1].vocab_id.tolist()):
            entity = _entity_at(TEST, "Actor1ID", position)
            assert row == vocab[B.ACTOR1].get(entity, 0), entity

    def test_the_row_is_not_the_node_position(self, tmp_path):
        """The defect: position 0 in test would have read A's row."""
        _, _, test = self._build(tmp_path)
        assert test[B.ACTOR1].vocab_id[0].item() == 3

    def test_an_entity_training_never_saw_gets_row_zero(self, tmp_path):
        _, _, test = self._build(tmp_path)
        assert test[B.ACTOR1].vocab_id[1].item() == 0  # D

    def test_an_unseen_day_gets_row_zero(self, tmp_path):
        vocab, _, test = self._build(tmp_path)
        assert test[B.DAY].vocab_id.tolist() == [vocab[B.DAY]["3"], 0]

    def test_events_carry_no_identity(self, tmp_path):
        _, train, test = self._build(tmp_path)
        assert train[B.EVENT].vocab_id.eq(0).all()
        assert test[B.EVENT].vocab_id.eq(0).all()

    def test_every_component_type_is_indexed(self, tmp_path):
        _, _, test = self._build(tmp_path)
        for ntype in (B.ACTOR1, B.ACTOR2, B.GEO, B.DAY):
            assert test[ntype].vocab_id.numel() == test[ntype].num_nodes


class TestLoadersShareTheVocabulary:
    def test_test_graph_is_indexed_with_the_training_vocabulary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hetero_loaders, "ENTITIES_DATA", tmp_path / "entities")
        monkeypatch.setattr(hetero_loaders, "GRAPHS_DATA", tmp_path / "graphs")
        _write(tmp_path / "entities", "train", TRAIN)
        _write(tmp_path / "entities", "test", TEST)

        train_loader, val_loader, test_loader = hetero_loaders.make_hetero_gnn_loaders("ident")

        assert val_loader is None
        assert train_loader.data[B.ACTOR1].vocab_id.tolist() == [1, 2, 3]
        # Fitted on test alone this would be [1, 2]; with the training
        # vocabulary, C keeps row 3 and D falls to 0.
        assert test_loader.data[B.ACTOR1].vocab_id.tolist() == [3, 0]


def _model(num_nodes_per_type):
    return HeterogeneousGNN(
        conv_type="rgcn",
        in_channels=0,
        hidden_channels=8,
        out_channels=4,
        num_relations=2,
        metadata=(
            ["event", "actor1"],
            [("event", "has_actor1", "actor1"), ("actor1", "rev_has_actor1", "event")],
        ),
        num_nodes_per_type=num_nodes_per_type,
        num_layers=2,
    )


def _table(model, ntype) -> nn.Embedding:
    return cast(nn.Embedding, cast(nn.ModuleDict, model.node_embeddings)[ntype])


class TestModelLookup:
    def test_the_unseen_row_is_zero_and_stays_zero(self):
        model = _model({"event": 1, "actor1": 4})
        table = _table(model, "actor1")
        assert table.padding_idx == 0
        assert torch.all(table.weight[0] == 0)

    def test_events_share_one_learned_row(self):
        table = _table(_model({"event": 1, "actor1": 4}), "event")
        assert table.num_embeddings == 1
        assert table.padding_idx is None

    def test_forward_batch_addresses_rows_by_vocabulary_id(self, monkeypatch):
        """Node positions 0, 1 must read rows 3 and 0, not rows 0 and 1."""
        model = _model({"event": 1, "actor1": 4})
        data = HeteroData()
        data["event"].num_nodes = 2
        data["event"].y = torch.tensor([0, 1])
        data["event"].batch_size = 2
        data["event"].vocab_id = torch.zeros(2, dtype=torch.long)
        data["actor1"].num_nodes = 2
        data["actor1"].vocab_id = torch.tensor([3, 0])
        edges = torch.tensor([[0, 1], [0, 1]])
        data["event", "has_actor1", "actor1"].edge_index = edges
        data["actor1", "rev_has_actor1", "event"].edge_index = edges

        seen = {}

        def capture(x_dict, edge_index_dict, n_id_dict):
            seen.update(n_id_dict)
            return {"event": torch.zeros(2, 4)}

        monkeypatch.setattr(model, "forward", capture)
        model.forward_batch(data, "cpu")
        assert seen["actor1"].tolist() == [3, 0]

    def test_an_unseen_entity_embeds_to_zero(self):
        model = _model({"event": 1, "actor1": 4})
        rows = model._featureless_embedding("actor1", {"actor1": torch.tensor([3, 0])}, {})
        assert torch.all(rows[1] == 0)
        assert not torch.all(rows[0] == 0)


def _bipartite(num_events=40, seed=0):
    """Events joined to actor, geo and day nodes, events sharing one row."""
    gen = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["event"].num_nodes = num_events
    data["event"].y = torch.zeros(num_events, dtype=torch.long)
    data["event"].batch_size = num_events
    data["event"].vocab_id = torch.zeros(num_events, dtype=torch.long)
    for ntype, n in (("actor1", 10), ("actor2", 10), ("geo", 8), ("day", 5)):
        data[ntype].num_nodes = n
        data[ntype].vocab_id = torch.arange(1, n + 1)
        dst = torch.randint(0, n, (num_events,), generator=gen)
        src = torch.arange(num_events)
        data["event", f"has_{ntype}", ntype].edge_index = torch.stack([src, dst])
        data[ntype, f"rev_has_{ntype}", "event"].edge_index = torch.stack([dst, src])
    return data


def _featureless(conv_type, data, num_layers=2):
    torch.manual_seed(0)
    meta = data.metadata()
    sizes = {"event": 1, "actor1": 11, "actor2": 11, "geo": 9, "day": 6}
    return HeterogeneousGNN(
        conv_type=conv_type, in_channels=0, hidden_channels=16, out_channels=4,
        metadata=meta, num_relations=len(meta[1]), num_layers=num_layers,
        heads=2, dropout=0.0, num_nodes_per_type=sizes,
    ).eval()


class TestEveryConvSeparatesEvents:
    """With events sharing one row, only the components can tell events apart.

    RGCN and RGAT carry a root term, so their event outputs still differ.
    """

    @pytest.mark.parametrize("conv_type", ["rgcn", "rgat"])
    def test_event_outputs_differ(self, conv_type):
        data = _bipartite()
        with torch.no_grad():
            logits, _ = _featureless(conv_type, data).forward_batch(data, "cpu")
        assert (logits - logits.mean(0)).abs().max().item() > 1e-6
