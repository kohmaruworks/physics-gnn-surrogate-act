"""
疎結合なメッセージパッシング層の合成（Compose）を前提とした GNN スケルトン。

圏论的インターフェースで、ばね・ダンパ・流体など異なる物理コンポーネントが
同じ更新枠 ``CategoryInformedGNNLayer`` 上で差し替え可能になるようにする。

ドメイン知識としての 1 ステップ（Kipf & Welling 型の畳み込み）::

    h_i^{(l+1)} = σ( Σ_{j ∈ Ñ(i)} c_ij W^{(l)} h_j^{(l)} )

ここで c_ij は正規化係数（``GCNConv`` が ``edge_index`` から構築するデフォルトに相当）、
σ は非線形。複数の「物理カテゴリ」は **別レート・別 ``edge_index``** の
``CategoryInformedGNNLayer`` を合成するか、上位で ``HeteroConv`` に接続して表現する。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class CategoryInformedGNNLayer(nn.Module):
    """
    単一の関係型に対する 1 層。将来、ここを差し替えて構成則ごとのメッセージに拡張可能。

    ``edge_weight`` を渡せば c_ij をエッジごとに上書きできる（PyG の GCN 仕様に従う）。
    """

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = True) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.conv(x, edge_index, edge_weight)


class CategoryInformedGNN(nn.Module):
    """
    同型の ``CategoryInformedGNNLayer`` を積み重ねたホモジニアス GCN スタック。

    マルチフィジックスでは、このブロックを関係型ごとにインスタンス化し、
    上位で ``torch_geometric.nn.HeteroConv`` 等に束ねる想定。
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = float(dropout)
        dims = [in_channels] + [hidden] * (num_layers - 1) + [out_channels]
        self.layers = nn.ModuleList(
            CategoryInformedGNNLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        return h
