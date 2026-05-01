# Physics-Informed Surrogate Modeling via Applied Category Theory (ACT–GNN) — Application Module

[![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)

[🇺🇸 English](#english) | [🇯🇵 日本語](#japanese)

<a id="english"></a>

## English

### Overview

**ACT–GNN Application Module** extends the **[Foundation series](https://github.com/kohmaruworks/physics-gnn-surrogate-basic)** line of work: it stress-tests **physics-informed graph surrogates** where (i) **graph cardinality** changes across deployment (scale), and (ii) **multiple constitutive laws** share the same discrete substrate (multiphysics). The central design choice is to treat heterogeneous interactions as **schema-level structure**—materialized in PyTorch Geometric as `HeteroData` with `HeteroConv`—rather than as a single mixed edge set with opaque attributes.

This layout is meant to be teachable (e.g. YouTube **Mathematical Science Note**): the intent is that **language boundaries**, the **0-based Contract**, and **compositionality** are visible in the folder layout and in the JSON loader—not buried in ad hoc scripts.

### Architecture design rules

**A. Language boundary and data schema**

- **Julia**: data generation, categorical graph tooling (`Catlab`), and dynamics-oriented tooling (`DifferentialEquations`, `JSON3`).
- **Python**: learning and inference with PyTorch Geometric.
- The interchange artifact is JSON with a fixed tag: **`format == "catlab_directed_graph_v1"`** (single source of truth for topology + optional `x` / `y`).

**B. Index safety — 0-based Contract**

- Julia graph APIs use **1-based** vertex IDs; PyG `edge_index` is **0-based**.
- **Required:** perform **`−1` exactly once**, only on the **Julia export boundary** (`src_julia/export_catlab_graph_json.jl`). The JSON stores **already 0-based** endpoints.
- **Forbidden in Python:** statements like `edge_index -= 1` inside `src_python/import_catlab_json_to_pyg.py`. The loader **casts** the `edges` list to `torch.long` and transposes to shape `(2, |E|)`—nothing else.

**C. Readability and compositionality**

- Physical message passing is organized so multiple domains (springs, dampers, fluids, …) can **compose** as separate blocks later.
- As a reference update, layers follow a Kipf & Welling–style aggregation (implemented here with `GCNConv`):

\[
\mathbf{h}_i^{(\ell+1)} = \sigma\left( \sum_{j\in\tilde{\mathcal{N}}(i)} c_{ij} \mathbf{W}^{(\ell)} \mathbf{h}_j^{(\ell)} \right)
\]

- A composable hook lives in `src_python/models/category_informed_gnn.py` (`CategoryInformedGNNLayer`, `CategoryInformedGNN`); demos may wrap or extend it.

| Layer | Role in this repository |
| --- | --- |
| **Julia / ACT** | `Catlab.jl` and `Catlab.Graphs` for a minimal, auditable **categorical tooling spine** (`test_catlab.jl`). `Project.toml` pins **DifferentialEquations.jl** and **JSON3.jl** for dynamics-heavy extensions and structured interchange toward Python. **Export** of the canonical JSON lives in `src_julia/export_catlab_graph_json.jl`. |
| **Python / PyG** | Executable tracks: **scale** (`demo1_scale_generalization.py`), **multiphysics** (`demo2_category_multiphysics.py`), **Foundation §3–matched** GNN vs MLP benchmark (`compare_loss_visualization.py` + `src_python/import_catlab_json_to_pyg.py`), optional Zenn figure helper (`article_figures_for_zenn.py`), and **device smoke test** (`test_gpu.py`). |

**Implemented physics.** Each node carries a 2D state \([u, v]\) (scalar position and velocity along a 1D chain). **Springs** apply Hooke forces \(\propto (u_j - u_i)\); **dampers** apply forces \(\propto (v_j - v_i)\) where applicable. Supervision is a **single explicit Euler-type step** \(v' = v + (F/m)\,\Delta t\), \(u' = u + v\,\Delta t\) with forces assembled from the active topology (see source for exact update order per script).

| Track | Script | What it fixes / proves |
| --- | --- | --- |
| **Scale** | `demo1_scale_generalization.py` | A flattened **NaiveMLP** is tied to a fixed \(N\); at inference with \(N=50\) after being sized for \(N=10\) it raises **`RuntimeError`** (shape mismatch). A small **GCN** stack (`CategoryInformedGNN`) accepts variable `edge_index` and completes forward on the larger chain (**no training** in this demo; `torch.manual_seed(0)`). |
| **Multiphysics** | `demo2_category_multiphysics.py` | Alternating **spring** and **damper** bonds on an even-length chain; **HomogeneousGNN** mixes both relation types in one `edge_index`; **CategoryHeteroGNN** routes springs and dampers through **separate** `GCNConv` towers inside `HeteroConv`. Writes **`zenn-articles/images/hetero_loss_comparison.png`** when that folder exists (else repo root, gitignored). |
| **JSON + Foundation §3 match** | `compare_loss_visualization.py` | **Same** supervision as [Foundation `train_spring_mass_gcn.py`](https://github.com/kohmaruworks/physics-gnn-surrogate-basic): **`spring_mass_chain_5.json`** (place sibling repo `physics-gnn-surrogate-basic` next to this one), **TwoLayerGCN** (hidden 16) vs **NaiveMLP**, `Adam` **lr=0.02**, **100** epochs, **training** MSE. Writes **`zenn-articles/images/loss_comparison_test.png`** if that folder exists (else this repo root, gitignored). |

**Default hyperparameters (as in source).**

| Setting | `demo1_scale_generalization.py` | `demo2_category_multiphysics.py` | `compare_loss_visualization.py` (when runnable) |
| --- | --- | --- | --- |
| Design \(N\) / nodes | Train layout \(N_{\mathrm{design}}=10\); inference \(N=50\) | `num_nodes = 12` (even; chain topology) | `Data.num_nodes` from **`spring_mass_chain_5.json`** (sibling `physics-gnn-surrogate-basic`) |
| Features / hidden | `feat_dim = 2`, GNN `hidden = 32` | `feat_dim = 2`, `hidden = 64` | `feat_dim = 2`, GNN (TwoLayer) & MLP **`hidden = 16`** |
| Optimizer / epochs | *(none — forward only)* | `Adam`, `lr = 1e-3`, `epochs = 200` | `Adam`, `lr = 0.02`, `epochs = 100`; **training** MSE (Foundation §3 match) |
| Physics constants | — | `k = 1.0`, `c = 0.8`, `m = 1.0`, `dt = 0.05` | — (teacher **`y`** is Julia ODE in JSON) |
| Data randomness | `torch.randn` for \(x_{50}\) | Initial state pool: `pool_size = 64`; `torch.manual_seed(7)`, generator seed `12345`; `pos_scale = 1.0`, `vel_scale = 0.6` | **One** graph, one \((x,y)\) pair — not a multi-split benchmark |
| PyG topology | Directed chain edges \((i,i{+}1)\) | Springs: pairs \((0,1),(2,3),\ldots\); dampers: \((1,2),(3,4),\ldots\); each logical bond expanded **bidirectionally** for `edge_index` | Chain edges from JSON `edge_index` |

**Implementation note (`compare_loss_visualization.py`).** With ODE-supervised single-graph training, a **TwoLayerGCN** and a **NaiveMLP** (same `lr`, `epochs`) can reach different **training** MSE plateaus; this matches the function-class discussion in the [Foundation article §6](https://zenn.dev/kohmaruworks/articles/phase1-06-training-outlook). Figures are versioned in **`zenn-articles/images/loss_comparison_test.png`**, not in this repository.

### Pipeline (stages)

| Stage | Location | Responsibility |
| --- | --- | --- |
| 1. Categorical incidence | Julia (`test_catlab.jl`, future ACSets) | Declare vertices as **objects** and edges as **typed generating morphisms** before any training. |
| 2. Schema → tensors | Python (`demo2_*`, optional JSON loader) | Materialize relation types as **disjoint** `edge_index` blocks (`HeteroData`) or a deliberately mixed baseline (`Data`). |
| 3. Learning & plots | Python (PyG + `matplotlib`) | Fit surrogates against physics-generated one-step targets; export loss curves / PNGs. |

Process detail:

1. **Julia** instantiates the dependency stack (`Catlab`, optional `DifferentialEquations`, `JSON3`) and can emit **structured** graph data via `src_julia/export_catlab_graph_json.jl`. Other JSON blobs remain **gitignored** by default (see `.gitignore`); a **checked-in sample** lives at `data/graph_from_catlab.json` for tutorials.
2. **Python** either **constructs** the hetero schema directly (`demo2_category_multiphysics.py`) or **ingests** JSON via `src_python/import_catlab_json_to_pyg.py` (`compare_loss_visualization.py` loads **`spring_mass_chain_5.json`** from sibling **`physics-gnn-surrogate-basic`**) with the **0-based Contract** above—**no second `−1`** on load.
3. **Baselines** are always co-located: homogeneous vs hetero in multiphysics; GNN vs MLP in optional JSON track; MLP failure mode in scale demo.

```mermaid
flowchart LR
  subgraph Julia["Julia (ACT / dynamics)"]
    C[Catlab.Graphs / ACSets]
    D[DifferentialEquations.jl]
    J[JSON3.jl]
  end
  subgraph Python["Python (PyG)"]
    H[HeteroData + HeteroConv]
    G[Homogeneous GCN baseline]
    T[Training & visualization]
  end
  C --> H
  C -. optional JSON .-> J
  J -. optional .-> T
  D -. future coupling .-> T
  G --> T
  H --> T
```

### Repository layout

```text
physics-gnn-surrogate-act/
├── README.md
├── data/
│   └── graph_from_catlab.json   # sample catlab_directed_graph_v1 (tracked)
├── src_julia/
│   └── export_catlab_graph_json.jl
└── src_python/
    ├── import_catlab_json_to_pyg.py
    └── models/
        └── category_informed_gnn.py
```

| Path | Role |
| --- | --- |
| `Project.toml` / `Manifest.toml` | Julia environment: **Catlab**, **DifferentialEquations**, **JSON3**. |
| `test_catlab.jl` | Loads `Catlab` / `Catlab.Graphs`, builds a tiny graph, prints incidence counts. |
| `data/graph_from_catlab.json` | Example **`catlab_directed_graph_v1`** payload (0-based `edges`); regenerate with `src_julia/export_catlab_graph_json.jl`. |
| `src_julia/export_catlab_graph_json.jl` | **Export boundary:** Catlab **1-based** graph → JSON with **0-based** `edges` (**only place** that applies `vertex_id − 1`). |
| `src_python/import_catlab_json_to_pyg.py` | JSON → `torch_geometric.data.Data` (**no** `edge_index -= 1`). |
| `src_python/models/category_informed_gnn.py` | Composable **GCN-style** stack for multi-physics extension (`CategoryInformedGNNLayer`, `CategoryInformedGNN`). |
| `demo1_scale_generalization.py` | Scale track: `CategoryInformedGNN` vs `NaiveMLP`, variable \(N\), no optimization loop. |
| `demo2_category_multiphysics.py` | Multiphysics track: `HomogeneousGNN` vs `CategoryHeteroGNN`, training loop; figure → **`zenn-articles/images/hetero_loss_comparison.png`**. |
| `compare_loss_visualization.py` | Foundation §3–matched GNN vs MLP training curve; requires **`../physics-gnn-surrogate-basic/spring_mass_chain_5.json`**; figure → **`zenn-articles/images/loss_comparison_test.png`**. |
| `article_figures_for_zenn.py` | Optional: writes Zenn images (e.g. teacher scatter) using `data/graph_from_catlab.json` when present. |
| `test_gpu.py` | CUDA availability and large `matmul` timing. |
| `.gitignore` | Virtualenvs, caches, **`*.json`** (with **exception** `!data/graph_from_catlab.json`), and generated **PNG**s (`loss_comparison_test.png`, `hetero_loss_comparison.png`). |

### Quick Start

**Julia**

```bash
cd /path/to/physics-gnn-surrogate-act
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test_catlab.jl
julia --project=. src_julia/export_catlab_graph_json.jl   # refreshes data/graph_from_catlab.json
```

**Python**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torch-geometric matplotlib

python demo1_scale_generalization.py
python demo2_category_multiphysics.py
python test_gpu.py
```

**Foundation §3 benchmark (`compare_loss_visualization.py`):** clone or place sibling repo **[physics-gnn-surrogate-basic](https://github.com/kohmaruworks/physics-gnn-surrogate-basic)** next to this one so **`../physics-gnn-surrogate-basic/spring_mass_chain_5.json`** exists (generate via Julia export if needed). Training curves are written to **`zenn-articles/images/loss_comparison_test.png`** when **`../zenn-articles/images`** exists.

```bash
python compare_loss_visualization.py
```

**Optional (Zenn article figures):** with `data/graph_from_catlab.json` and **`../zenn-articles/images`**, run `python article_figures_for_zenn.py` (used by the Foundation / Application article pipeline).

**Try the JSON loader alone:**

```bash
python src_python/import_catlab_json_to_pyg.py
python src_python/import_catlab_json_to_pyg.py /path/to/custom.json
```

<a id="japanese"></a>

## 日本語

### 概要

**ACT–GNN 応用モジュール**は **[基礎編シリーズ](https://github.com/kohmaruworks/physics-gnn-surrogate-basic)** の延長であり、(i) デプロイ時に変動する**グラフサイズ（スケール）**と、(ii) 同一離散基盤上に共存する**複数の構成則（マルチフィジックス）**という二つの圧力下で、**物理情報付きグラフ・サロゲート**を検証します。設計上の主眼は、異種相互作用を単一の混在 `edge_index` の不透明な属性ではなく、**スキーマ上の構造**として扱い、PyTorch Geometric では `HeteroData` と `HeteroConv` に具体化することです。

構成は解説（例: YouTube **Mathematical Science Note**）を意識しており、**言語境界**・**0-based Contract**・**合成可能性（compositionality）**がフォルダ配置と JSON ローダから読み取れるようにしてあります。

### アーキテクチャ設計ルール

**A. 言語間の境界とデータスキーマ**

- **Julia**: データ生成、`Catlab` による圏論的グラフ、动力学（`DifferentialEquations`、`JSON3`）。
- **Python**: PyTorch Geometric による学習・推論。
- 中間表現は **`format == "catlab_directed_graph_v1"`** の JSON とし、トポロジ（および任意の `x` / `y`）の **Single Source of Truth** とする。

**B. インデックスの安全保障（0-based Contract）**

- Julia は**1 始まり**、PyG の `edge_index` は**0 始まり**。
- **必須:** **`−1` は Julia のエクスポート境界（`src_julia/export_catlab_graph_json.jl`）でのみ 1 回**。JSON に書き出す `edges` は**すでに 0 始まり**。
- **Python で禁止:** `src_python/import_catlab_json_to_pyg.py` 内の `edge_index -= 1` など。ローダは `edges` を **`torch.long` にキャスト**し `(2, |E|)` に転置する**だけ**。

**C. 可読性と合成可能性（応用圏論: Compositionality）**

- 将来的なマルチフィジックス（ばね、ダンパー、流体など）を、**ブロックの合成**として足し込める前提のインターフェースにする。
- 参照となる 1 ステップの更新は Kipf & Welling 型（本リポでは `GCNConv`）:

\[
\mathbf{h}_i^{(\ell+1)} = \sigma\left( \sum_{j\in\tilde{\mathcal{N}}(i)} c_{ij} \mathbf{W}^{(\ell)} \mathbf{h}_j^{(\ell)} \right)
\]

- 疎結合なフックは `src_python/models/category_informed_gnn.py`（`CategoryInformedGNNLayer`、`CategoryInformedGNN`）に置く。デモはこれをラップ・拡張してよい。

| レイヤ | 本リポジトリでの役割 |
| --- | --- |
| **Julia / ACT** | `Catlab.jl` と `Catlab.Graphs` による、監査しやすい最小の**圏論ツールチェーン**（`test_catlab.jl`）。`Project.toml` では **DifferentialEquations.jl** と **JSON3.jl** を固定。canonical JSON の**書き出し**は `src_julia/export_catlab_graph_json.jl`。 |
| **Python / PyG** | 実行可能なトラックとして、**スケール**（`demo1_scale_generalization.py`）、**マルチフィジックス**（`demo2_category_multiphysics.py`）、**基礎編第3回同条件**の GNN/MLP 比較（`compare_loss_visualization.py` ＋ `src_python/import_catlab_json_to_pyg.py`）、任意の **Zenn 図**（`article_figures_for_zenn.py`）、**デバイス確認**（`test_gpu.py`）。 |

**実装されている物理。** 各ノードは 2 次元状態 \([u, v]\)（1 次元座標上の位置・速度）。**バネ**はフックの法則に従い \(\propto (u_j - u_i)\)、**ダンパ**（該当スクリプト）は \(\propto (v_j - v_i)\) の力を寄与します。教師信号は、合力から \(a=F/m\) を得たうえでの**陽的オイラー型 1 ステップ** \(v' = v + a\,\Delta t\)、\(u' = u + v\,\Delta t\)（スクリプトごとの更新順はソース参照）。

| トラック | スクリプト | 示していること |
| --- | --- | --- |
| **スケール** | `demo1_scale_generalization.py` | 平坦化 **NaiveMLP** は設計時のノード数に固定され、\(N=10\) 用に構築したのち **\(N=50\)** を流すと **`RuntimeError`**（形状不一致）。**`CategoryInformedGNN`** は `edge_index` だけ変えて大きいチェーンでも forward 可能（本デモは**学習なし**、`torch.manual_seed(0)`）。 |
| **マルチフィジックス** | `demo2_category_multiphysics.py` | 偶数長チェーン上で **バネ**・**ダンパ**を交互配置。**HomogeneousGNN** は両者を同一 `edge_index` に混在、**CategoryHeteroGNN** は `HeteroConv` 内の別 **GCNConv** 経路に分離。図は **`zenn-articles/images/hetero_loss_comparison.png`**（隣接 Zenn リポが無い場合は本リポ直下・`.gitignore` 対象）。 |
| **JSON + 基礎編第3回同条件** | `compare_loss_visualization.py` | [基礎編](https://github.com/kohmaruworks/physics-gnn-surrogate-basic) の **`spring_mass_chain_5.json`** と**同一**教師 $y$（Julia ODE）で、**TwoLayerGCN** と**平坦化 MLP**の**訓練**MSE を比較。図は **`zenn-articles/images/loss_comparison_test.png`**。 |

**既定ハイパーパラメータ（ソース値）。**

| 設定 | `demo1_scale_generalization.py` | `demo2_category_multiphysics.py` | `compare_loss_visualization.py`（実行可能な場合） |
| --- | --- | --- | --- |
| 設計 \(N\) / ノード数 | 設計 \(N_{\mathrm{design}}=10\)、推論 \(N=50\) | `num_nodes = 12`（偶数・チェーン位相） | **`spring_mass_chain_5.json`** の `Data.num_nodes` |
| 特徴 / 隠れ次元 | `feat_dim = 2`、GNN `hidden = 32` | `feat_dim = 2`、`hidden = 64` | `feat_dim = 2`、GNN（2層）・MLP とも **`hidden = 16`** |
| 最適化 / エポック | *（forward のみ）* | `Adam`、`lr = 1e-3`、`epochs = 200` | `Adam`、`lr = 0.02`、`epochs = 100`、**訓練**MSE（基礎編第3回同条件） |
| 物理定数 | — | `k = 1.0`、`c = 0.8`、`m = 1.0`、`dt = 0.05` | —（教師 `y` は JSON 内の ODE 参照解） |
| データ乱数 | \(x_{50}\) 用 `torch.randn` | 固定プール `pool_size = 64`；`torch.manual_seed(7)`、ジェネレータ `12345`；`pos_scale = 1.0`、`vel_scale = 0.6` | **1 グラフ**の \((x,y)\) のみ（**200/150/50 分割ではない**） |
| PyG トポロジ | 有向チェーン辺 \((i,i{+}1)\) | バネ \((0,1),(2,3),\ldots\)、ダンパ \((1,2),(3,4),\ldots\)；各結合を **双方向** に展開して `edge_index` 化 | JSON の `edge_index`（有向チェーン） |

**実装上の注記（`compare_loss_visualization.py`）。** 第3回の**単一**教師付き学習と同条件で、**帰納的バイアス**（重み共有 GCN と平坦 MLP）の違いを**訓練**曲線に重ねる。図面は [Zenn 記事（URL は phase1 のまま）](https://zenn.dev/kohmaruworks/articles/phase1-06-training-outlook) 用の **`zenn-articles/images/loss_comparison_test.png`** です。

### パイプライン（ステージ）

| 段階 | 所在 | 役割 |
| --- | --- | --- |
| 1. 圏論的インシデンス | Julia（`test_catlab.jl`、将来の ACSets） | 学習前に、頂点を**対象**、辺を**型付き生成 morphism** として宣言する。 |
| 2. スキーマ → テンソル | Python（`demo2_*`、任意の JSON ローダ） | 関係型を **互いに素な** `edge_index`（`HeteroData`）として具現化するか、意図的に混在させたベースライン（`Data`）とする。 |
| 3. 学習・プロット | Python（PyG + `matplotlib`） | 物理が生成した 1 ステップ先を教師にサロゲートを学習し、損失曲線や PNG を出力する。 |

補足:

1. **Julia** で依存スタックを確立し、`src_julia/export_catlab_graph_json.jl` から**構造化** JSON を出力できる。**チュートリアル用サンプル**は `data/graph_from_catlab.json`（`*.json` の gitignore に対し **`!data/graph_from_catlab.json`** で追跡）。
2. **Python** はスキーマを**直接構築**（`demo2_category_multiphysics.py`）するか、JSON を **`src_python/import_catlab_json_to_pyg` で取り込む**（`compare_loss_visualization.py` は**隣接** `physics-gnn-surrogate-basic` の **`spring_mass_chain_5.json`**）かに分かれ、**読み込み時に再度 `−1` は行わない**。
3. **ベースライン**は常に同所に置きます（マルチフィジックスでは同質対 Hetero、JSON トラックでは GNN 対 MLP、スケールデモでは MLP の失敗モード）。

```mermaid
flowchart LR
  subgraph Julia["Julia (ACT / dynamics)"]
    C[Catlab.Graphs / ACSets]
    D[DifferentialEquations.jl]
    J[JSON3.jl]
  end
  subgraph Python["Python (PyG)"]
    H[HeteroData + HeteroConv]
    G[Homogeneous GCN baseline]
    T[Training & visualization]
  end
  C --> H
  C -. optional JSON .-> J
  J -. optional .-> T
  D -. future coupling .-> T
  G --> T
  H --> T
```

### リポジトリ構成

（英語セクションのツリーと同一）

| パス | 役割 |
| --- | --- |
| `Project.toml` / `Manifest.toml` | Julia 環境: **Catlab**、**DifferentialEquations**、**JSON3**。 |
| `test_catlab.jl` | `Catlab` / `Catlab.Graphs` をロードし、最小グラフを構築してインシデンス数を表示。 |
| `data/graph_from_catlab.json` | **`catlab_directed_graph_v1`** の例（**0 始まり**の `edges`）。`src_julia/export_catlab_graph_json.jl` で再生成可。 |
| `src_julia/export_catlab_graph_json.jl` | **エクスポート境界:** Catlab の **1 始まり** → JSON の **0 始まり**（**ここでのみ** `−1`）。 |
| `src_python/import_catlab_json_to_pyg.py` | JSON → `Data`（**`edge_index -= 1` なし**）。 |
| `src_python/models/category_informed_gnn.py` | マルチフィジックス拡張を想定した合成可能な **GCN 型**スタック。 |
| `demo1_scale_generalization.py` | スケール軸: `CategoryInformedGNN` と `NaiveMLP`、可変 \(N\)、最適化ループなし。 |
| `demo2_category_multiphysics.py` | マルチフィジックス軸: `HomogeneousGNN` と `CategoryHeteroGNN`、学習ループ、図 → `zenn-articles/images/hetero_loss_comparison.png`。 |
| `compare_loss_visualization.py` | 基礎編第3回同条件: TwoLayerGCN vs MLP。要 **`../physics-gnn-surrogate-basic/spring_mass_chain_5.json`**。図 → `zenn-articles/images/loss_comparison_test.png`。 |
| `article_figures_for_zenn.py` | 任意: `data/graph_from_catlab.json` 等から Zenn 用 PNG を `zenn-articles/images` へ。 |
| `test_gpu.py` | CUDA 有無と大規模 `matmul` の計測。 |
| `.gitignore` | 仮想環境・キャッシュ、**`*.json`**（例外 `!data/graph_from_catlab.json`）、生成 **PNG**。 |

### クイックスタート

**Julia**

```bash
cd /path/to/physics-gnn-surrogate-act
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test_catlab.jl
julia --project=. src_julia/export_catlab_graph_json.jl
```

**Python**

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torch-geometric matplotlib

python demo1_scale_generalization.py
python demo2_category_multiphysics.py
python test_gpu.py
```

**基礎編第3回同条件のベンチ（`compare_loss_visualization.py`）:** **[physics-gnn-surrogate-basic](https://github.com/kohmaruworks/physics-gnn-surrogate-basic)** を**同じ親ディレクトリ**に置き、**`../physics-gnn-surrogate-basic/spring_mass_chain_5.json`** があること。図は **`../zenn-articles/images`** があれば **`loss_comparison_test.png`** へ。

```bash
python compare_loss_visualization.py
```

**任意（Zenn 図表）:** `data/graph_from_catlab.json` と **`../zenn-articles/images`** がある場合に `python article_figures_for_zenn.py`。

**JSON ローダ単体:**

```bash
python src_python/import_catlab_json_to_pyg.py
```

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

本プロジェクトは **MIT License** の下で公開されています。詳細はリポジトリ直下の [`LICENSE`](LICENSE) を参照してください。
