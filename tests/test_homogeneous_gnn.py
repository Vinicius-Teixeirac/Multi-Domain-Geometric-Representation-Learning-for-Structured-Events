import torch
import pytest

from src.models.gnn.homogeneous import HomogeneousGNN


NUM_NODES = 20
IN_CHANNELS = 8
HIDDEN = 16
OUT_CHANNELS = 4


@pytest.fixture
def edge_index():
    """
    Random homogeneous edge index with 40 edges over ``NUM_NODES`` nodes.

    Returns
    -------
    torch.Tensor
        Shape ``(2, 40)`` long tensor of source/destination node indices.
    """
    src = torch.randint(0, NUM_NODES, (40,))
    dst = torch.randint(0, NUM_NODES, (40,))
    return torch.stack([src, dst])


@pytest.fixture
def x():
    """Random node feature matrix of shape ``(NUM_NODES, IN_CHANNELS)``."""
    return torch.randn(NUM_NODES, IN_CHANNELS)


@pytest.mark.parametrize("conv_type", ["sage", "gin", "gat"])
class TestHomogeneousGNN:
    def test_construction(self, conv_type):
        """Model instantiates with the correct hidden and output dimensions."""
        model = HomogeneousGNN(
            conv_type=conv_type,
            in_channels=IN_CHANNELS,
            hidden_channels=HIDDEN,
            out_channels=OUT_CHANNELS,
            num_layers=2,
        )
        assert model.hidden_dim == HIDDEN
        assert model.out_dim == OUT_CHANNELS

    def test_forward_shape(self, conv_type, x, edge_index):
        """Forward pass produces float32 logits of shape ``(NUM_NODES, OUT_CHANNELS)``."""
        model = HomogeneousGNN(
            conv_type=conv_type,
            in_channels=IN_CHANNELS,
            hidden_channels=HIDDEN,
            out_channels=OUT_CHANNELS,
            num_layers=2,
            heads=2,
        )
        model.eval()
        out = model(x, edge_index)
        assert out.shape == (NUM_NODES, OUT_CHANNELS)
        assert out.dtype == torch.float32
