# src/models/gnn/homogeneous.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
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
        """
        Parameters
        ----------
        conv_type : str
            Graph convolution type: 'sage', 'gin', or 'gat'.
        in_channels : int
            Input feature dimension per node.
        hidden_channels : int
            Width of hidden GNN layers.
        out_channels : int
            Output dimension (number of classes for direct classification).
        num_layers : int
            Total number of convolution layers (must be >= 2).
        dropout : float
            Dropout probability applied between layers.
        heads : int
            Number of attention heads (GAT only; ignored by SAGE and GIN).
        encoder : nn.Module or None
            Optional tabular input encoder (TabularInputEncoder) used by
            forward_batch to build node features from raw x_cat/x_num each
            batch, so its embeddings train jointly with the GNN. None means
            batch.x is used as-is (e.g. the constant dummy feature under the
            "none" node-feature policy).
        """
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
        """
        Run message passing and return node-level logits.

        Parameters
        ----------
        x : torch.Tensor of shape (N, in_channels)
        edge_index : torch.Tensor of shape (2, E)

        Returns
        -------
        torch.Tensor of shape (N, out_channels)
        """
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self._activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.convs[-1](x, edge_index)

    def _make_conv(self, in_dim: int, out_dim: int, *, last: bool = False, heads: int = 4) -> nn.Module:
        """Instantiate a single convolution layer of the configured type."""
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
        """Return the effective hidden dimension after a non-final layer (GAT multiplies by heads)."""
        return hidden_channels * heads if self.conv_type == "gat" else hidden_channels

    def _activation(self, x):
        """Apply the appropriate activation for the current conv type (ELU for GAT, ReLU otherwise)."""
        return F.elu(x) if self.conv_type == "gat" else F.relu(x)

    def forward_batch(
        self,
        batch: Data,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Move a PyG batch to device, run forward, and return (logits, targets).

        Slices logits and targets to batch_size when NeighborLoader is used
        (seed nodes only, not sampled neighbourhood nodes).

        Parameters
        ----------
        batch : torch_geometric.data.Data (or Batch, its subclass)
            A (possibly neighbour-sampled) homogeneous graph batch.
        device : str
            Target device string.

        Returns
        -------
        tuple of (logits, targets)
            Both sliced to the seed-node batch size (if applicable) and
            moved to device.
        """
        batch = batch.to(device)

        if self.encoder is not None:
            x = self._encode_batch(batch)
        elif hasattr(batch, "x") and batch.x is not None:
            x = batch.x
            assert isinstance(x, torch.Tensor)
        else:
            num_nodes = batch.num_nodes
            y = batch.y
            assert num_nodes is not None and isinstance(y, torch.Tensor)
            x = y.new_zeros((num_nodes, self.hidden_dim)).float()

        assert batch.edge_index is not None
        logits = self.forward(x, batch.edge_index)
        targets = batch.y
        assert isinstance(targets, torch.Tensor)

        if hasattr(batch, "batch_size"):
            logits = logits[: batch.batch_size]
            targets = targets[: batch.batch_size]

        return logits, targets

    def _encode_batch(self, batch: Data) -> torch.Tensor:
        """
        Build node features from batch.x_cat/x_num via self.encoder.

        x_cat/x_num are attached to the full split graph as full-length
        tensors (see homogeneous/loaders.py); NeighborLoader doesn't slice
        dict-valued attributes automatically, so any tensor whose first
        dimension doesn't already match the sampled batch is indexed by
        batch.n_id (absent in full-batch mode, where sizes already match).
        """
        num_nodes = batch.num_nodes
        x_cat = batch.x_cat
        x_num = getattr(batch, "x_num", None)

        if isinstance(x_cat, dict) and hasattr(batch, "n_id"):
            n_id = batch.n_id
            x_cat = {
                k: (v[n_id] if v.size(0) != num_nodes else v)
                for k, v in x_cat.items()
            }

        if x_num is not None and hasattr(batch, "n_id"):
            n_id = batch.n_id
            if x_num.size(0) != num_nodes:
                x_num = x_num[n_id]

        return self.encoder(x_cat, x_num)