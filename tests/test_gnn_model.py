import torch
import pytest
from torch_geometric.data import Data, HeteroData

from src.models.gnn.homogeneous import HomogeneousGNN
from src.models.gnn.heterogeneous import HeterogeneousGNN
from src.models.tabular_encoder import TabularInputEncoder


NUM_NODES = 20
HIDDEN = 16
OUT_CHANNELS = 4
NUM_EVENTS = 15
NUM_ACTORS = 5

NODE_TYPES = ["event", "actor"]
EDGE_TYPES = [
    ("event", "has_actor", "actor"),
    ("actor", "rev_has_actor", "event"),
]
METADATA = (NODE_TYPES, EDGE_TYPES)


# ── Homogeneous GNN ─────────────────────────────────────────────

class TestHomogeneousGNNForwardBatch:
    def test_forward_batch_with_features(self):
        """``forward_batch`` returns full-graph logits and targets when node features are present."""
        model = HomogeneousGNN(
            conv_type="sage", in_channels=8,
            hidden_channels=HIDDEN, out_channels=OUT_CHANNELS,
            num_layers=2,
        )
        model.eval()

        edge_index = torch.stack([
            torch.randint(0, NUM_NODES, (30,)),
            torch.randint(0, NUM_NODES, (30,)),
        ])
        data = Data(
            x=torch.randn(NUM_NODES, 8),
            edge_index=edge_index,
            y=torch.randint(0, OUT_CHANNELS, (NUM_NODES,)),
        )
        logits, targets = model.forward_batch(data, "cpu")

        assert logits.shape == (NUM_NODES, OUT_CHANNELS)
        assert targets.shape == (NUM_NODES,)
        assert logits.dtype == torch.float32
        assert targets.dtype == torch.long

    def test_forward_batch_featureless(self):
        """In the real code, 'none' policy still provides data.x = ones(N,1)."""
        model = HomogeneousGNN(
            conv_type="sage", in_channels=1,
            hidden_channels=HIDDEN, out_channels=OUT_CHANNELS,
            num_layers=2,
        )
        model.eval()

        edge_index = torch.stack([
            torch.randint(0, NUM_NODES, (30,)),
            torch.randint(0, NUM_NODES, (30,)),
        ])
        data = Data(
            x=torch.ones(NUM_NODES, 1),
            edge_index=edge_index,
            y=torch.randint(0, OUT_CHANNELS, (NUM_NODES,)),
        )
        logits, targets = model.forward_batch(data, "cpu")

        assert logits.shape == (NUM_NODES, OUT_CHANNELS)
        assert targets.shape == (NUM_NODES,)

    def test_forward_batch_seed_node_slicing(self):
        """
        When ``data.batch_size`` is set, ``forward_batch`` slices outputs to
        the seed nodes only (mini-batch neighbor sampling convention).
        """
        model = HomogeneousGNN(
            conv_type="sage", in_channels=8,
            hidden_channels=HIDDEN, out_channels=OUT_CHANNELS,
            num_layers=2,
        )
        model.eval()

        edge_index = torch.stack([
            torch.randint(0, NUM_NODES, (30,)),
            torch.randint(0, NUM_NODES, (30,)),
        ])
        data = Data(
            x=torch.randn(NUM_NODES, 8),
            edge_index=edge_index,
            y=torch.randint(0, OUT_CHANNELS, (NUM_NODES,)),
        )
        data.batch_size = 5
        logits, targets = model.forward_batch(data, "cpu")

        assert logits.shape == (5, OUT_CHANNELS)
        assert targets.shape == (5,)


# ── Heterogeneous GNN ────────────────────────────────────────────

class TestHeterogeneousGNNForwardBatch:
    def _make_hetero_data(self, with_encoder=False):
        """
        Build a minimal ``HeteroData`` object for event–actor graphs.

        Parameters
        ----------
        with_encoder : bool
            When ``True``, attaches ``x_cat`` and ``x_num`` to the event
            node store so that a ``TabularInputEncoder`` can be exercised.

        Returns
        -------
        torch_geometric.data.HeteroData
            Graph with event and actor nodes connected by ``has_actor`` /
            ``rev_has_actor`` edges; ``batch_size`` is set to 10 on event nodes.
        """
        data = HeteroData()
        data["event"].y = torch.randint(0, OUT_CHANNELS, (NUM_EVENTS,))
        data["event"].num_nodes = NUM_EVENTS
        data["actor"].num_nodes = NUM_ACTORS

        data["event"].batch_size = 10

        data[("event", "has_actor", "actor")].edge_index = torch.stack([
            torch.randint(0, NUM_EVENTS, (20,)),
            torch.randint(0, NUM_ACTORS, (20,)),
        ])
        data[("actor", "rev_has_actor", "event")].edge_index = torch.stack([
            torch.randint(0, NUM_ACTORS, (20,)),
            torch.randint(0, NUM_EVENTS, (20,)),
        ])

        if with_encoder:
            data["event"].x_cat = {
                "col_a": torch.randint(0, 10, (NUM_EVENTS,)),
            }
            data["event"].x_num = torch.randn(NUM_EVENTS, 3)

        return data

    def test_forward_batch_featureless(self):
        """
        Featureless heterogeneous GNN uses learned node embeddings and returns
        seed-node-sliced logits and targets matching ``batch_size``.
        """
        model = HeterogeneousGNN(
            conv_type="rgcn", in_channels=0,
            hidden_channels=HIDDEN, out_channels=OUT_CHANNELS,
            num_relations=len(EDGE_TYPES), metadata=METADATA,
            num_layers=2,
            encoder=None, event_type="event",
        )
        model.eval()

        data = self._make_hetero_data(with_encoder=False)
        logits, targets = model.forward_batch(data, "cpu")

        assert logits.shape[0] == data["event"].batch_size
        assert logits.shape[1] == OUT_CHANNELS
        assert targets.shape[0] == data["event"].batch_size

    def test_forward_batch_with_encoder(self):
        """
        When a ``TabularInputEncoder`` is provided, node features flow through
        the encoder before the GNN layers; output shapes still match batch_size.
        """
        encoder = TabularInputEncoder(
            categorical_cardinalities={"col_a": 10},
            numeric_dim=3,
        )
        in_dim = encoder.output_dim

        model = HeterogeneousGNN(
            conv_type="rgcn", in_channels=in_dim,
            hidden_channels=HIDDEN, out_channels=OUT_CHANNELS,
            num_relations=len(EDGE_TYPES), metadata=METADATA,
            num_layers=2,
            encoder=encoder, event_type="event",
        )
        model.eval()

        data = self._make_hetero_data(with_encoder=True)
        logits, targets = model.forward_batch(data, "cpu")

        assert logits.shape[0] == data["event"].batch_size
        assert logits.shape[1] == OUT_CHANNELS
        assert targets.shape[0] == data["event"].batch_size

    def test_featureless_with_prebuilt_embeddings(self):
        """D4 fix: embeddings created at __init__ time via num_nodes_per_type."""
        num_nodes_per_type = {"event": NUM_EVENTS, "actor": NUM_ACTORS}
        model = HeterogeneousGNN(
            conv_type="rgcn", in_channels=0,
            hidden_channels=HIDDEN, out_channels=OUT_CHANNELS,
            num_relations=len(EDGE_TYPES), metadata=METADATA,
            num_layers=2,
            encoder=None, event_type="event",
            num_nodes_per_type=num_nodes_per_type,
        )

        assert "event" in model.node_embeddings
        assert "actor" in model.node_embeddings
        assert model.node_embeddings["event"].num_embeddings == NUM_EVENTS
        assert model.node_embeddings["actor"].num_embeddings == NUM_ACTORS
