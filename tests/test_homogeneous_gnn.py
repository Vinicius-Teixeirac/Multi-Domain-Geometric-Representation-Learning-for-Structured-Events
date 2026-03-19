import torch
import pytest

from src.models.gnn.homogeneous import HomogeneousGNN


NUM_NODES = 20
IN_CHANNELS = 8
HIDDEN = 16
OUT_CHANNELS = 4


@pytest.fixture
def edge_index():
    src = torch.randint(0, NUM_NODES, (40,))
    dst = torch.randint(0, NUM_NODES, (40,))
    return torch.stack([src, dst])


@pytest.fixture
def x():
    return torch.randn(NUM_NODES, IN_CHANNELS)


@pytest.mark.parametrize("conv_type", ["sage", "gin", "gat"])
class TestHomogeneousGNN:
    def test_construction(self, conv_type):
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
