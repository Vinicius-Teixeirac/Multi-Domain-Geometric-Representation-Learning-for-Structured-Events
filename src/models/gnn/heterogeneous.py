# src/models/gnn/heterogeneous.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv, RGATConv, HANConv
from torch_geometric.data import HeteroData


class HeterogeneousGNN(nn.Module):

    def __init__(
        self,
        conv_type: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_relations: int | None = None,
        metadata=None,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
        encoder=None,
        event_type: str = "event",
        num_nodes_per_type: dict[str, int] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.event_type = event_type

        assert num_layers >= 2

        self.conv_type = conv_type.lower()
        self.dropout = dropout

        # ARCHITECTURE METADATA
        self.hidden_dim = hidden_channels
        self.out_dim = out_channels

        if self.conv_type in {"rgcn", "rgat"}:
            assert num_relations is not None

        if self.conv_type == "han":
            assert metadata is not None

        self.convs = nn.ModuleList()
        self.featureless = in_channels == 0

        if self.featureless:
            self.node_embeddings = nn.ModuleDict()
            if num_nodes_per_type is not None:
                for ntype, n in num_nodes_per_type.items():
                    self.node_embeddings[ntype] = nn.Embedding(n, hidden_channels)
            in_channels = hidden_channels

        # First layer
        self.convs.append(
            self._make_conv(
                in_channels,
                hidden_channels,
                heads=heads,
                num_relations=num_relations,
                metadata=metadata,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                self._make_conv(
                    self._hidden_dim(hidden_channels, heads),
                    hidden_channels,
                    heads=heads,
                    num_relations=num_relations,
                    metadata=metadata,
                )
            )

        # Output layer
        self.convs.append(
            self._make_conv(
                self._hidden_dim(hidden_channels, heads),
                out_channels,
                last=True,
                heads=heads,
                num_relations=num_relations,
                metadata=metadata,
            )
        )

    def forward(self, x_dict, edge_index_dict):
        if self.conv_type == "han":
            return self._forward_han(x_dict, edge_index_dict)
        return self._forward_flattened(x_dict, edge_index_dict)

    def _forward_flattened(self, x_dict, edge_index_dict):
        data = HeteroData()

        for node_type, x in x_dict.items():
            if x is None:
                num_nodes = data[node_type].num_nodes
                if node_type not in self.node_embeddings:
                    self.node_embeddings[node_type] = nn.Embedding(
                        num_nodes, self.hidden_dim
                    ).to(next(self.parameters()).device)
                data[node_type].x = self.node_embeddings[node_type].weight
            else:
                data[node_type].x = x

        for (src, rel, dst), edge_index in edge_index_dict.items():
            data[(src, rel, dst)].edge_index = edge_index

        data = data.to_homogeneous()

        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_type)
            x = self._activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_type)
        return self._unflatten(x, data)

    def _forward_han(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            if x is None:
                num_nodes = x_dict[node_type].size(0)
                if node_type not in self.node_embeddings:
                    self.node_embeddings[node_type] = nn.Embedding(
                        num_nodes, self.hidden_dim
                    ).to(next(self.parameters()).device)
                x_dict[node_type] = self.node_embeddings[node_type].weight

        for conv in self.convs[:-1]:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self._activation(v) for k, v in x_dict.items()}
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }

        return self.convs[-1](x_dict, edge_index_dict)

    def _make_conv(
        self,
        in_dim,
        out_dim,
        *,
        last=False,
        heads=4,
        num_relations=None,
        metadata=None,
    ):
        if self.conv_type == "rgcn":
            return RGCNConv(in_dim, out_dim, num_relations)

        if self.conv_type == "rgat":
            if last:
                return RGATConv(in_dim, out_dim, num_relations, heads=1, concat=False, dropout=self.dropout)
            return RGATConv(in_dim, out_dim, num_relations, heads=heads, dropout=self.dropout)

        if self.conv_type == "han":
            return HANConv(
                in_dim,
                out_dim,
                metadata=metadata,
                heads=1 if last else heads,
                dropout=self.dropout,
            )

        raise ValueError(f"Unknown conv_type '{self.conv_type}'")

    def _hidden_dim(self, hidden_channels, heads):
        return hidden_channels * heads if self.conv_type == "rgat" else hidden_channels

    def _activation(self, x):
        return F.elu(x) if self.conv_type == "rgat" else F.relu(x)

    def _unflatten(self, x, data):
        out = {}
        for i, node_type in enumerate(data._node_type_names):
            out[node_type] = x[data.node_type == i]
        return out

    def forward_batch(self, batch, device):
        batch = batch.to(device)

        x_dict = {}
        event_store = batch[self.event_type]
        num_nodes = event_store.y.size(0)

        if self.encoder is None:
            x_event = event_store.y.new_zeros(
                (num_nodes, self.hidden_dim)
            ).float()
        else:
            # When features were attached as full-length tensors on the
            # HeteroData object (one tensor per event in the split), the
            # NeighborLoader / batching will not automatically slice dict
            # attributes. Ensure we index any full-length tensors by the
            # sampled node ids (`n_id`) so the encoder always receives
            # per-batch tensors with matching first-dimension sizes.
            x_cat = event_store.x_cat
            x_num = event_store.x_num

            if isinstance(x_cat, dict) and hasattr(event_store, "n_id"):
                n_id = event_store.n_id
                # build a sliced dict where needed
                x_cat = {
                    k: (v[n_id] if v.size(0) != num_nodes else v)
                    for k, v in x_cat.items()
                }

            if x_num is not None and hasattr(event_store, "n_id"):
                n_id = event_store.n_id
                if x_num.size(0) != num_nodes:
                    x_num = x_num[n_id]

            x_event = self.encoder(x_cat, x_num)

        x_dict[self.event_type] = x_event

        for node_type in batch.node_types:
            if node_type == self.event_type:
                continue
            x_dict[node_type] = x_event.new_zeros(
                (batch[node_type].num_nodes, x_event.size(1))
            )

        out = self.forward(x_dict, batch.edge_index_dict)

        logits = out[self.event_type][: batch[self.event_type].batch_size]
        targets = batch[self.event_type].y[: batch[self.event_type].batch_size]

        return logits, targets