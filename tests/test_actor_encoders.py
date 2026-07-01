"""Tests for the actor (WHO/WHOM) domain encoders: construction, forward
shapes across GNN and attribute-only variants, weighted-edge handling, and
the build_actor_encoder factory."""

import torch
import pytest

from src.models.multi_domain.actor_encoders import (
    ActorSAGEEncoder,
    ActorGATEncoder,
    ActorWeightedEncoder,
    ActorAttributeEncoder,
    build_actor_encoder,
)

NUM_NODES = 20
NUM_EDGES = 40
BATCH_SIZE = 8
FEAT_EMBED_DIM = 4
HIDDEN_DIM = 16
OUT_DIM = 12

# 8 categorical actor attributes (name, country, group, ethnic, religion x2, type x3
# collapse to 8 in the real pipeline); cardinality includes the unknown slot at index 0.
CARDINALITIES = [10, 8, 6, 5, 4, 4, 3, 3]


@pytest.fixture
def graph_x() -> torch.Tensor:
    """Per-node categorical attribute indices, shape (NUM_NODES, len(CARDINALITIES))."""
    cols = [torch.randint(0, card, (NUM_NODES,)) for card in CARDINALITIES]
    return torch.stack(cols, dim=1)


@pytest.fixture
def graph_edge_index() -> torch.Tensor:
    """Random co-occurrence edges among actor nodes, shape (2, NUM_EDGES)."""
    src = torch.randint(0, NUM_NODES, (NUM_EDGES,))
    dst = torch.randint(0, NUM_NODES, (NUM_EDGES,))
    return torch.stack([src, dst])


@pytest.fixture
def graph_edge_attr() -> torch.Tensor:
    """Random positive co-occurrence edge weights, shape (NUM_EDGES,)."""
    return torch.rand(NUM_EDGES) + 0.1


@pytest.fixture
def actor_pair_idx():
    """actor1_idx/actor2_idx node indices for a batch of event pairs."""
    actor1_idx = torch.randint(0, NUM_NODES, (BATCH_SIZE,))
    actor2_idx = torch.randint(0, NUM_NODES, (BATCH_SIZE,))
    return actor1_idx, actor2_idx


@pytest.mark.parametrize(
    "encoder_cls", [ActorSAGEEncoder, ActorGATEncoder, ActorWeightedEncoder, ActorAttributeEncoder]
)
class TestActorEncoders:
    def test_forward_shape(self, encoder_cls, graph_x, graph_edge_index, actor_pair_idx):
        """Every actor encoder returns one pair embedding per event, shape (B, out_dim)."""
        actor1_idx, actor2_idx = actor_pair_idx
        enc = encoder_cls(
            cardinalities=CARDINALITIES,
            feat_embed_dim=FEAT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            out_dim=OUT_DIM,
        )
        out = enc(graph_x, graph_edge_index, actor1_idx, actor2_idx)
        assert out.shape == (BATCH_SIZE, OUT_DIM)
        assert not torch.isnan(out).any()

    def test_construction_creates_one_embedding_per_attribute(self, encoder_cls):
        """_ActorEncoderBase must create one nn.Embedding per categorical attribute column."""
        enc = encoder_cls(
            cardinalities=CARDINALITIES,
            feat_embed_dim=FEAT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            out_dim=OUT_DIM,
        )
        assert len(enc.feat_embeddings) == len(CARDINALITIES)
        for emb, card in zip(enc.feat_embeddings, CARDINALITIES):
            assert emb.num_embeddings == card
            assert emb.embedding_dim == FEAT_EMBED_DIM


class TestActorWeightedEncoderEdgeWeights:
    def test_with_edge_weights(self, graph_x, graph_edge_index, graph_edge_attr, actor_pair_idx):
        """ActorWeightedEncoder accepts normalised co-occurrence weights without error."""
        actor1_idx, actor2_idx = actor_pair_idx
        enc = ActorWeightedEncoder(
            cardinalities=CARDINALITIES, feat_embed_dim=FEAT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
        )
        out = enc(graph_x, graph_edge_index, actor1_idx, actor2_idx, graph_edge_attr)
        assert out.shape == (BATCH_SIZE, OUT_DIM)

    def test_falls_back_to_unweighted_when_edge_attr_is_none(self, graph_x, graph_edge_index, actor_pair_idx):
        """graph_edge_attr=None must not crash; GCNConv falls back to unweighted aggregation."""
        actor1_idx, actor2_idx = actor_pair_idx
        enc = ActorWeightedEncoder(
            cardinalities=CARDINALITIES, feat_embed_dim=FEAT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
        )
        out = enc(graph_x, graph_edge_index, actor1_idx, actor2_idx, None)
        assert out.shape == (BATCH_SIZE, OUT_DIM)


class TestActorAttributeEncoderIsInductive:
    def test_ignores_graph_structure(self, graph_x, actor_pair_idx):
        """ActorAttributeEncoder performs no message passing, so a garbage/empty edge_index
        must produce the same output as any other edge_index (it's never used).

        eval() is required here: attr_proj/pair_proj contain Dropout layers, so
        without disabling training-mode stochasticity the two forward passes
        would differ from dropout randomness alone, unrelated to edge_index.
        """
        actor1_idx, actor2_idx = actor_pair_idx
        enc = ActorAttributeEncoder(
            cardinalities=CARDINALITIES, feat_embed_dim=FEAT_EMBED_DIM,
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
        )
        enc.eval()
        empty_edges = torch.empty(2, 0, dtype=torch.long)
        dense_edges = torch.randint(0, NUM_NODES, (2, 100))
        out_empty = enc(graph_x, empty_edges, actor1_idx, actor2_idx)
        out_dense = enc(graph_x, dense_edges, actor1_idx, actor2_idx)
        assert torch.allclose(out_empty, out_dense, atol=1e-6)


class TestBuildActorEncoder:
    @pytest.mark.parametrize(
        "type_name,expected_cls",
        [
            ("sage_gnn", ActorSAGEEncoder),
            ("gat_gnn", ActorGATEncoder),
            ("weighted_gnn", ActorWeightedEncoder),
            ("attribute_only", ActorAttributeEncoder),
        ],
    )
    def test_dispatches_to_correct_class(self, type_name, expected_cls):
        """build_actor_encoder must instantiate the class registered for each known type string."""
        cfg = {"type": type_name, "feat_embed_dim": FEAT_EMBED_DIM, "hidden_dim": HIDDEN_DIM, "out_dim": OUT_DIM}
        enc = build_actor_encoder(cfg, CARDINALITIES)
        assert isinstance(enc, expected_cls)

    def test_defaults_to_sage_gnn(self):
        """Omitting 'type' should default to the plain GraphSAGE encoder."""
        enc = build_actor_encoder({}, CARDINALITIES)
        assert isinstance(enc, ActorSAGEEncoder)

    def test_gat_heads_only_applied_to_gat(self):
        """gat_heads should only be forwarded for gat_gnn; other types must ignore it if present."""
        cfg = {"type": "gat_gnn", "gat_heads": 2, "out_dim": OUT_DIM}
        enc = build_actor_encoder(cfg, CARDINALITIES)
        assert isinstance(enc.convs[0], torch.nn.Module)

    def test_unknown_type_raises(self):
        """An unrecognized encoder type must raise ValueError, not silently pick a default."""
        with pytest.raises(ValueError, match="Unknown actor encoder type"):
            build_actor_encoder({"type": "not_a_real_type"}, CARDINALITIES)
