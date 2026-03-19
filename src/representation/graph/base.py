# src/representation/graph/base.py
from abc import ABC, abstractmethod
from typing import Dict


class GraphBuilder(ABC):
    @abstractmethod
    def build(self) -> Dict:
        """
        Returns a graph object (PyG Data or HeteroData)
        """
        pass
