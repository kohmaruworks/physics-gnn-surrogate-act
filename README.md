# ACT-GNN: Physics-Informed Surrogate Modeling via Applied Category Theory

[![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)

## Overview

**ACT-GNN** is a proof-of-concept pipeline that unites **applied category theory (ACT)** with **graph neural networks (GNNs)** for physics-informed surrogate modeling. Physical systems are encoded as compositional structure—in this repository, as **directed multigraphs** whose **bidirectional** edges encode pairwise couplings. **High-fidelity** trajectories come from numerical ODE integration; a **PyTorch Geometric** model then learns a map from initial node states to states at a fixed horizon, approximating the exact flow in a **message-passing** parameterization.

The reference scenario is a **one-dimensional spring–mass chain**: **Catlab.jl** supplies the interaction topology, **DifferentialEquations.jl** with **`Tsit5()`** generates ground-truth evolution at time \(t_1\), and a compact **two-layer GCN** performs **node-wise regression** on **CUDA** when available.

## Key Features

- **Categorical physics topology (Julia / Catlab.jl)** — Adjacent masses are linked by **bidirectional directed edges**, yielding a **multigraph** that reflects symmetric couplings while remaining compatible with standard PyG **`edge_index`** conventions.
- **High-fidelity ground truth (DifferentialEquations.jl)** — Coupled second-order dynamics are integrated using **`Tsit5()`** with tight tolerances (`abstol`, `reltol` ≈ `1e-10`). Per-node **position and velocity** at \(t_1\) are exported as supervised targets.
- **Julia ↔ Python JSON bridge** — The versioned schema **`catlab_directed_graph_v1`** carries **`num_nodes`**, **`edges`**, and optional **`x`** / **`y`**. **Edge endpoints are serialized 0-based** for PyTorch Geometric; Julia graph code remains **1-based** internally, eliminating off-by-one drift across the boundary.
- **PyTorch Geometric training (CUDA-ready)** — **`import_catlab_json_to_pyg.py`** builds **`torch_geometric.data.Data`**; **`train_spring_mass_gcn.py`** trains a two-layer **GCN** with **MSE** loss, preferring **`cuda`** and falling back to **`cpu`**.

## Architecture & Data Flow

Execution proceeds from **explicit compositional topology plus ODE semantics** in Julia to **sparse, relation-local updates** in Python:

1. **Graph construction** — **`spring_mass_chain_graph(n)`** builds a Catlab **`Graph`** on **`n`** vertices with bidirectional edges along the chain, fixing the interaction topology.
2. **Feature and label generation** — Initial conditions **`x ∈ ℝ^{n×2}`** (position and velocity at \(t=0\)) are sampled; **`integrate_spring_mass_chain`** integrates the first-order reduction of the spring–mass system over **`(0, t₁)`** and reads off the terminal state as **`y`**.
3. **Serialization** — **`save_catlab_graph_json`** emits JSON: topology, **`x`**, **`y`**, with edges as **`[[src, tgt], …]`** in **0-based** form.
4. **Ingestion** — Python validates **`format`**, constructs **`edge_index`** as a **`2 × |E|`** tensor in the usual PyG layout.
5. **Learning** — The GCN maps **`x`** to predicted node states; **MSE** against **`y`** trains a surrogate of the **ODE flow map** at **`t₁`**.

**End-to-end flow (compact):**

- **Catlab graph** → **ODE solve (Tsit5)** → **JSON (0-based edges; node features and targets)** → **`torch_geometric.data.Data`** → **GCN training (GPU if available)**

**Core scripts:**

| Component | Role |
|-----------|------|
| `export_catlab_graph_json.jl` | Catlab **`Graph`** → JSON; **1-based → 0-based** edge export |
| `spring_mass_chain_export.jl` | Full path: chain graph + Tsit5 integration → **`spring_mass_chain_5.json`** |
| `import_catlab_json_to_pyg.py` | JSON → **`Data`** |
| `train_spring_mass_gcn.py` | GCN training loop |
| `test_catlab.jl` / `test_gpu.py` | Minimal environment checks |

## How to Run

### Prerequisites

- **Julia** with **`Pkg`**. Dependencies are declared in **`Project.toml`**: Catlab, DifferentialEquations, JSON3.
- **Python** with **PyTorch** and **PyTorch Geometric**, built against your **CUDA** toolchain or CPU wheels. For example:

  ```bash
  pip install torch torch-geometric
  ```

  Align versions with the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 1. Julia: instantiate and generate data

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. spring_mass_chain_export.jl
```

This produces **`spring_mass_chain_5.json`** (default **`n = 5`**; adjust via **`main(...)`** in **`spring_mass_chain_export.jl`**).

Optional—export a small toy graph only:

```bash
julia --project=. export_catlab_graph_json.jl
```

### 2. Python: train the surrogate

```bash
python train_spring_mass_gcn.py
```

The script uses **`cuda`** when available, otherwise **`cpu`**. **`spring_mass_chain_5.json`** must exist (see step 1).

### 3. Smoke tests (optional)

```bash
julia --project=. test_catlab.jl
python test_gpu.py
```

**Note:** **`.gitignore`** excludes **`*.json`** and may exclude **`Manifest.toml`**. Keep **`Project.toml`** under version control; regenerate manifests and data locally as needed.

## Philosophical Background: Why Category Theory?

**Compositionality** — Substantial physics is not “just a large matrix” but **typed components glued by structure-preserving maps**. ACT makes that glue explicit: **objects** anchor state or component kinds; **morphisms** anchor admissible interactions or dynamics. Replacing the exact integrator with a learned map does not dissolve that obligation—the **same compositional diagram** should delimit what the surrogate may couple.

**Universality and transfer** — Categorical machinery tends to isolate **stable structure** (limits, colimits, functoriality) from incidental coordinates. For R&D and quant-adjacent stacks, representations that survive **refactoring and subsystem scaling** reduce breakage when simulators, discretizations, or model classes evolve.

**Resonance with message passing** — GNNs implement **local, relation-respecting updates** along a declared graph. That pattern is close to **composing morphisms along a diagram**: information propagates along edges under algebraic constraints. ACT does not supplant learning; it **grounds** the hypothesis class in semantics aligned with how many physical networks are actually assembled.

---

## 日本語版 (Japanese Version)

[![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io/)

以下は英語版と**同一の構成・情報量**を保った全文である。文体は**だ・である調**とする。

**ACT-GNN：応用圏論による物理情報付きサロゲートモデリング**

### 概要

**ACT-GNN** は、**応用圏論（ACT）**と**グラフニューラルネットワーク（GNN）**を接合し、物理情報付きサロゲートモデリングを実証するための概念実証（PoC）パイプラインである。物理系は**合成可能な構造**として符号化される—本リポジトリでは、**双方向の相互作用辺**を備えた**有向マルチグラフ**として表現する。**高忠実度**の軌道は常微分方程式の数値積分により得られ、**PyTorch Geometric** モデルは固定ホライズンにおける初期ノード状態から未来状態への写像を学習し、**メッセージパッシング**のパラメータ化のもとで厳密なフローを近似する。

参照実装の対象は**一次元バネ–質量鎖**である。**Catlab.jl** が相互作用トポロジーを与え、**DifferentialEquations.jl** の **`Tsit5()`** が時刻 \(t_1\) における真値ダイナミクスを生成し、コンパクトな**二層 GCN** が利用可能な場合は **CUDA** 上で**ノード単位の回帰**を行う。

### 主な機能

- **圏論的物理トポロジー（Julia / Catlab.jl）** — 隣接質量間は**双方向の有向辺**で結ばれ、対称な結合を反映した**マルチグラフ**となる。同時に、標準的な PyG の **`edge_index`** 規約との整合性を保つ。
- **高忠実度の真値（DifferentialEquations.jl）** — 結合した二階系は **`Tsit5()`** および厳しい許容誤差（`abstol`、`reltol` ≈ `1e-10`）で積分される。各ノードの \(t_1\) における**位置と速度**は教師信号として書き出される。
- **Julia ↔ Python の JSON ブリッジ** — 版付きスキーマ **`catlab_directed_graph_v1`** が **`num_nodes`**、**`edges`**、任意の **`x`** / **`y`** を運ぶ。**辺の端点は JSON 上で 0 始まり**に直され、PyTorch Geometric 向けに整列する。Julia 側のグラフ操作は内部的に**1 始まり**のままとし、境界での off-by-one を排除する。
- **PyTorch Geometric による学習（CUDA 対応）** — **`import_catlab_json_to_pyg.py`** が **`torch_geometric.data.Data`** を構築し、**`train_spring_mass_gcn.py`** が二層 **GCN** を **MSE** 損失で学習する。デバイスは **`cuda`** を優先し、なければ **`cpu`** にフォールバックする。

### アーキテクチャとデータフロー

Julia における**明示的な合成トポロジと ODE 意味論**から、Python における**疎かつ関係局所的な更新**へと処理が進む。

1. **グラフ構築** — **`spring_mass_chain_graph(n)`** が **`n`** 個の頂点を持つ Catlab の **`Graph`** を生成し、鎖に沿って双方向辺を張る。これにより相互作用トポロジが固定される。
2. **特徴とラベルの生成** — 初期条件 **`x ∈ ℝ^{n×2}`**（\(t=0\) における位置と速度）をサンプリングする。**`integrate_spring_mass_chain`** がバネ–質量系の一階化系を **`(0, t₁)`** 上で積分し、終端状態を **`y`** として読み出す。
3. **シリアライズ** — **`save_catlab_graph_json`** がトポロジ、**`x`**、**`y`** を JSON に書き出す。辺は **`[[src, tgt], …]`** 形式で、**0 始まり**で格納される。
4. **取り込み** — Python が **`format`** を検証し、慣例どおり **`2 × |E|`** 形状のテンソル **`edge_index`** を構築する。
5. **学習** — GCN が **`x`** からノード状態の予測へ写す。**`y`** に対する **MSE** により、時刻 **`t₁`** における **ODE フロー写像**のサロゲートが学習される。

**エンドツーエンドの流れ（要約）：**

- **Catlab グラフ** → **ODE 求解（Tsit5）** → **JSON（0 始まり辺、ノード特徴とターゲット）** → **`torch_geometric.data.Data`** → **GCN 学習（GPU が利用可能なら GPU）**

**中核スクリプト：**

| コンポーネント | 役割 |
|----------------|------|
| `export_catlab_graph_json.jl` | Catlab の **`Graph`** → JSON、**1 始まりから 0 始まり**への辺エクスポート |
| `spring_mass_chain_export.jl` | 鎖グラフ + Tsit5 積分から **`spring_mass_chain_5.json`** までの一連処理 |
| `import_catlab_json_to_pyg.py` | JSON → **`Data`** |
| `train_spring_mass_gcn.py` | GCN 学習ループ |
| `test_catlab.jl` / `test_gpu.py` | 最小限の環境確認 |

### 実行方法

#### 前提条件

- **`Pkg`** を備えた **Julia**。依存関係は **`Project.toml`** に宣言されている：Catlab、DifferentialEquations、JSON3。
- **PyTorch** および **PyTorch Geometric** を導入した **Python**（**CUDA** 用ビルド、または CPU ホイール）。例：

  ```bash
  pip install torch torch-geometric
  ```

  PyTorch と CUDA の組み合わせは [PyG 公式インストール手順](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)に合わせること。

#### 1. Julia：環境のインスタンス化とデータ生成

リポジトリのルートで：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. spring_mass_chain_export.jl
```

これにより **`spring_mass_chain_5.json`** が生成される（既定は **`n = 5`**。**`spring_mass_chain_export.jl`** 内の **`main(...)`** で変更可能）。

任意：小規模な玩具グラフのみをエクスポートする場合：

```bash
julia --project=. export_catlab_graph_json.jl
```

#### 2. Python：サロゲートの学習

```bash
python train_spring_mass_gcn.py
```

スクリプトは利用可能なら **`cuda`**、さもなくば **`cpu`** を用いる。手順 1 で **`spring_mass_chain_5.json`** が存在している必要がある。

#### 3. スモークテスト（任意）

```bash
julia --project=. test_catlab.jl
python test_gpu.py
```

**注記：** **`.gitignore`** は **`*.json`** を除外し、**`Manifest.toml`** を除外する設定になっている場合がある。**`Project.toml`** はバージョン管理下に置き、マニフェストとデータはローカルで再生成すること。

### 思想的背景：なぜ圏論か

**合成性** — 多くの物理は「巨大な行列に過ぎない」のではなく、**型付きされた部品が構造保存写像によって接合された系**として理解できる。ACT はその接合を明示する：**対象**が状態や部品の種を、**射**が許容される相互作用やダイナミクスを支える。厳密積分器を学習写像に置き換えても、その義務は消えない—**同一の合成ダイアグラム**が、サロゲートに何を結合してよいかの境界を規定すべきである。

**普遍性と転移** — 圏論的構成は、場当たり的な座標より**安定した構造**（極限、余極限、関手性）を切り出しやすい。R&D やクオンツに隣接するスタックでは、シミュレータ、離散化、モデルクラスが変化しても**リファクタリングとサブシステムのスケール**に耐える表現が、統合リスクを下げる。

**メッセージパッシングとの共鳴** — GNN は宣言されたグラフに沿った**局所的かつ関係を尊重する更新**を実装する。その様式は**ダイアグラムに沿った射の合成**に構造的に近い：情報は代数的制約のもとで辺に沿って伝播する。ACT は学習を代替しない。多くの物理ネットワークが実際に組み立てられる仕方と整合する**合成意味論**のうえに、仮説クラスを**固定**するのである。
