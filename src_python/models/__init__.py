"""疎結合・合成可能な GNN モジュール（応用圏論的 compositionality を意図したインターフェース）。"""

from .category_informed_gnn import CategoryInformedGNN, CategoryInformedGNNLayer

__all__ = ["CategoryInformedGNN", "CategoryInformedGNNLayer"]
