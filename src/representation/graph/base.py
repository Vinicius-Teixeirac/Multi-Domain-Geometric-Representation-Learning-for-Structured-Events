# src/representation/graph/base.py
from abc import ABC, abstractmethod
from typing import Dict


class GraphBuilder(ABC):
    """Abstract base for split-wise graph constructors.

    Concrete subclasses must implement build() and return a PyG Data or HeteroData
    object ready for use with DataLoader/NeighborLoader.
    """

    @abstractmethod
    def build(self) -> Dict:
        """Construct and return the graph for the configured split."""
        pass
