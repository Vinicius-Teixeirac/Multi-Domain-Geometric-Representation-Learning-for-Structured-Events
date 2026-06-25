# src/models/gnn/homogeneous.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv, GATConv


class HomogeneousGNN(nn.Module):
    """Homogeneous GNN supporting SAGE, GIN, and GAT convolutions with configurable depth."""

    def __init__(
        self,
        conv_type: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
        encoder=None,
    ):
        super().__init__()
        self.encoder = encoder

        assert num_layers >= 2, "num_layers must be >= 2"

        self.conv_type = conv_type.lower()
        self.dropout = dropout

        # ARCHITECTURE METADATA
        self.hidden_dim = hidden_channels
        self.out_dim = out_channels

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            self._make_conv(in_channels, hidden_channels, heads=heads)
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                self._make_conv(
                    self._hidden_dim(hidden_channels, heads),
                    hidden_channels,
                    heads=heads,
                )
            )

        # Output layer
        self.convs.append(
            self._make_conv(
                self._hidden_dim(hidden_channels, heads),
                out_channels,
                last=True,
                heads=heads,
            )
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self._activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.convs[-1](x, edge_index)

    def _make_conv(self, in_dim: int, out_dim: int, *, last: bool = False, heads: int = 4) -> nn.Module:
        # last=True collapses GAT to a single head so the output dim equals out_dim exactly
        if self.conv_type == "sage":
            return SAGEConv(in_dim, out_dim)

        if self.conv_type == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(mlp)

        if self.conv_type == "gat":
            if last:
                return GATConv(in_dim, out_dim, heads=1, concat=False, dropout=self.dropout)
            return GATConv(in_dim, out_dim, heads=heads, dropout=self.dropout)

        raise ValueError(f"Unknown conv_type '{self.conv_type}'")

    def _hidden_dim(self, hidden_channels, heads):
        return hidden_channels * heads if self.conv_type == "gat" else hidden_channels

    def _activation(self, x):
        return F.elu(x) if self.conv_type == "gat" else F.relu(x)

    def forward_batch(self, batch, device):
        batch = batch.to(device)

        if hasattr(batch, "x"):
            x = batch.x
        else:
            num_nodes = batch.num_nodes
            x = batch.y.new_zeros((num_nodes, self.hidden_dim)).float()

        logits = self.forward(x, batch.edge_index)
        targets = batch.y

        if hasattr(batch, "batch_size"):
            logits = logits[: batch.batch_size]
            targets = targets[: batch.batch_size]

        return logits, targets