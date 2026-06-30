import torch
import pytest

from src.models.gnn.heterogeneous import HeterogeneousGNN


NUM_EVENTS = 15
NUM_ACTORS = 5
IN_CHANNELS = 8
HIDDEN = 16
OUT_CHANNELS = 4

NODE_TYPES = ["event", "actor"]
EDGE_TYPES = [
    ("event", "has_actor", "actor"),
    ("actor", "rev_has_actor", "event"),
]
METADATA = (NODE_TYPES, EDGE_TYPES)
NUM_RELATIONS = len(EDGE_TYPES)


@pytest.fixture
def edge_index_dict():
    """
    Random edge indices for both ``has_actor`` and ``rev_has_actor`` relations.

    Returns
    -------
    dict[tuple, torch.Tensor]
        Mapping from ``(src_type, relation, dst_type)`` to a ``(2, 20)`` edge
        index tensor.
    """
    return {
        ("event", "has_actor", "actor"): torch.stack([
            torch.randint(0, NUM_EVENTS, (20,)),
            torch.randint(0, NUM_ACTORS, (20,)),
        ]),
        ("actor", "rev_has_actor", "event"): torch.stack([
            torch.randint(0, NUM_ACTORS, (20,)),
            torch.randint(0, NUM_EVENTS, (20,)),
        ]),
    }


@pytest.fixture
def x_dict():
    """
    Random node feature tensors for event and actor node types.

    Returns
    -------
    dict[str, torch.Tensor]
        ``"event"`` → ``(NUM_EVENTS, IN_CHANNELS)`` and
        ``"actor"`` → ``(NUM_ACTORS, IN_CHANNELS)``, both float32.
    """
    return {
        "event": torch.randn(NUM_EVENTS, IN_CHANNELS),
        "actor": torch.randn(NUM_ACTORS, IN_CHANNELS),
    }


@pytest.mark.parametrize("conv_type", ["rgcn", "rgat", "han"])
class TestHeterogeneousGNN:
    def test_construction(self, conv_type):
        """Model instantiates with the correct hidden and output dimensions."""
        model = HeterogeneousGNN(
            conv_type=conv_type,
            in_channels=IN_CHANNELS,
            hidden_channels=HIDDEN,
            out_channels=OUT_CHANNELS,
            num_relations=NUM_RELATIONS,
            metadata=METADATA,
            num_layers=2,
        )
        assert model.hidden_dim == HIDDEN
        assert model.out_dim == OUT_CHANNELS

    def test_forward_shape(self, conv_type, x_dict, edge_index_dict):
        """
        Forward pass returns a dict with float32 event embeddings of shape
        ``(NUM_EVENTS, OUT_CHANNELS)``.
        """
        model = HeterogeneousGNN(
            conv_type=conv_type,
            in_channels=IN_CHANNELS,
            hidden_channels=HIDDEN,
            out_channels=OUT_CHANNELS,
            num_relations=NUM_RELATIONS,
            metadata=METADATA,
            num_layers=2,
            heads=2,
        )
        model.eval()
        out = model(x_dict, edge_index_dict)

        assert isinstance(out, dict)
        assert "event" in out
        assert out["event"].shape == (NUM_EVENTS, OUT_CHANNELS)
        assert out["event"].dtype == torch.float32
