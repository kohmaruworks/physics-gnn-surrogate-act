# Catlab の有向グラフを `catlab_directed_graph_v1` JSON に書き出す。
#
# 0-based Contract（必須）:
#   Julia の src/tgt は 1 始まり。PyTorch Geometric は 0 始まり。
#   **このファイルがエクスポート境界**であり、ここでのみ頂点番号から 1 を引く。
#   Python の import 側では引き算を二度と行わない。

using Catlab.Graphs
using JSON3

"""有向グラフ `g` の各辺を [src0, tgt0]（0 始まり）のペアのベクトルに変換する。"""
function zero_based_edge_list(g)
    map(1:ne(g)) do i
        [Int(src(g)[i]) - 1, Int(tgt(g)[i]) - 1]
    end
end

"""
    save_catlab_directed_graph_v1(path, g; num_nodes=nothing, x=nothing, y=nothing)

`g` を JSON に保存する。`num_nodes` を省略した場合は `nv(g)` を使う。
`x`, `y` は任意の列挙可能オブジェクト（ノード特徴・教師テンソル用）。
"""
function save_catlab_directed_graph_v1(path::AbstractString, g; num_nodes=nothing, x=nothing, y=nothing)
    n = something(num_nodes, nv(g))
    payload = Dict{String, Any}(
        "format" => "catlab_directed_graph_v1",
        "num_nodes" => n,
        "edges" => zero_based_edge_list(g),
    )
    if x !== nothing
        payload["x"] = collect(x)
    end
    if y !== nothing
        payload["y"] = collect(y)
    end
    open(path, "w") do io
        JSON3.pretty(io, payload)
    end
    return path
end

function main()
    root = dirname(@__DIR__)
    out = joinpath(root, "data", "graph_from_catlab.json")
    mkpath(dirname(out))

    g = Graph()
    add_vertices!(g, 3)
    add_edges!(g, [1, 2], [2, 3])

    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    save_catlab_directed_graph_v1(out, g; x=x)
    println("Wrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
