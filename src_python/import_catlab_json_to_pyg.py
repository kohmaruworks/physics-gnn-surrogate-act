"""
Julia（Catlab）からエクスポートされた JSON を torch_geometric.data.Data に変換する。

Single Source of Truth はスキーマ ``format == \"catlab_directed_graph_v1\"`` の JSON。

0-based Contract
-----------------
Julia は頂点番号が 1 始まり、PyG の ``edge_index`` は 0 始まり。**引き算（-1）は Julia の
エクスポート境界（``src_julia/export_catlab_graph_json.jl``）で一度だけ**行う。
本モジュールでは ``edges`` を **そのまま** ``torch.long`` の ``(2, |E|)`` に整形し、
Python 側で ``edge_index -= 1`` 等の操作は行わない。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data


def catlab_json_to_data(path: str | Path, *, dtype_edge_index=torch.int64) -> Data:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    if payload.get("format") != "catlab_directed_graph_v1":
        raise ValueError(
            f"unsupported format: {payload.get('format')!r}; "
            "expected 'catlab_directed_graph_v1'"
        )

    num_nodes = int(payload["num_nodes"])
    edges = payload.get("edges") or []

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=dtype_edge_index)
    else:
        # 0-based Contract: JSON の整数をそのまま PyG 用テンソルにする（引き算しない）
        edge_index = torch.tensor(edges, dtype=dtype_edge_index).t().contiguous()

    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    if "x" in payload:
        data.x = torch.tensor(payload["x"], dtype=torch.float32)

    if "y" in payload:
        data.y = torch.tensor(payload["y"], dtype=torch.float32)

    return data


def _default_sample_json() -> Path:
    """リポジトリ直下の ``data/graph_from_catlab.json``（サンプル用）。"""
    return Path(__file__).resolve().parents[1] / "data" / "graph_from_catlab.json"


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_sample_json()
    d = catlab_json_to_data(p)
    print(d)
    print("edge_index:\n", d.edge_index)
    if d.x is not None:
        print("x:\n", d.x)
