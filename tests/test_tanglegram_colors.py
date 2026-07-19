"""Tests for cluster colouring across a tanglegram.

The two dendrograms are drawn by separate calls, so without a shared palette
the same cluster can come out a different colour on each side.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from matplotlib.collections import LineCollection
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import pdist, squareform

import tanglegram as tg


N = 12
LABELS = [f"n{i}" for i in range(N)]


def frame(seed):
    X = np.random.default_rng(seed).random((N, 4))
    return pd.DataFrame(squareform(pdist(X)), columns=LABELS, index=LABELS)


def link_colors(ax):
    return {
        tuple(np.round(c.get_colors()[0], 6))
        for c in ax.collections
        if isinstance(c, LineCollection)
    }


def rgba(color):
    return tuple(np.round(mcl.to_rgba(color), 6))


@pytest.fixture
def matrices():
    return frame(1), frame(2)


# ------------------------------------------------------- shared colour map


def test_same_cluster_same_colour_on_both_sides(matrices):
    """The two sides have different cluster sets, which used to shift the
    palette and give cluster "y" a different colour on each side."""
    m1, m2 = matrices
    left = np.array(["x"] * 4 + ["y"] * 4 + ["z"] * 4)
    right = np.array(["y"] * 6 + ["z"] * 6)

    fig, dn1, dn2 = tg.tanglegram(
        m1,
        m2,
        sort=False,
        clusters_left=left,
        clusters_right=right,
        cluster_colors={"x": "#ff0000", "y": "#00ff00", "z": "#0000ff"},
    )
    ax1, ax2 = fig.get_axes()[0], fig.get_axes()[1]

    # "y" and "z" appear on both sides and must match; "x" only on the left
    assert rgba("#00ff00") in link_colors(ax1)
    assert rgba("#00ff00") in link_colors(ax2)
    assert rgba("#0000ff") in link_colors(ax1)
    assert rgba("#0000ff") in link_colors(ax2)
    assert rgba("#ff0000") in link_colors(ax1)
    assert rgba("#ff0000") not in link_colors(ax2)
    plt.close("all")


def test_shared_palette_without_explicit_colours(matrices):
    m1, m2 = matrices
    left = np.array(["x"] * 4 + ["y"] * 4 + ["z"] * 4)
    right = np.array(["y"] * 6 + ["z"] * 6)

    fig, dn1, dn2 = tg.tanglegram(
        m1, m2, sort=False, clusters_left=left, clusters_right=right
    )
    ax1, ax2 = fig.get_axes()[0], fig.get_axes()[1]

    # left: x, y, z + grey; right: y, z + grey -> three colours in common
    assert len(link_colors(ax1) & link_colors(ax2)) == 3
    plt.close("all")


def test_cluster_color_map_spans_several_assignments():
    cmap = tg.cluster_color_map(["a", "b"], ["b", "c"])

    assert set(cmap) == {"a", "b", "c"}
    assert len(set(cmap.values())) == 3


def test_cluster_color_map_ignores_none_assignments():
    assert tg.cluster_color_map(None, ["a"], None) == tg.cluster_color_map(["a"])


def test_cluster_color_map_skips_unassigned_leafs():
    cmap = tg.cluster_color_map(["a", None, np.nan])
    assert set(cmap) == {"a"}


# ---------------------------------------------------------- edge colouring


def edge_colors(fig):
    ax3 = fig.get_axes()[2]
    return [rgba(line.get_color()) for line in ax3.lines]


def test_edges_coloured_by_left_cluster(matrices):
    m1, m2 = matrices
    left = np.array(["x"] * 4 + ["y"] * 4 + ["z"] * 4)

    fig, _, _ = tg.tanglegram(
        m1,
        m2,
        sort=False,
        clusters_left=left,
        cluster_colors={"x": "#ff0000", "y": "#00ff00", "z": "#0000ff"},
        color_edges_by="left",
    )

    assert set(edge_colors(fig)) == {
        rgba("#ff0000"),
        rgba("#00ff00"),
        rgba("#0000ff"),
    }
    plt.close("all")


def test_edges_coloured_by_right_cluster(matrices):
    m1, m2 = matrices
    right = np.array(["p"] * 6 + ["q"] * 6)

    fig, _, _ = tg.tanglegram(
        m1,
        m2,
        sort=False,
        clusters_right=right,
        cluster_colors={"p": "#ff0000", "q": "#0000ff"},
        color_edges_by="right",
    )

    assert set(edge_colors(fig)) == {rgba("#ff0000"), rgba("#0000ff")}
    plt.close("all")


def test_unassigned_leafs_keep_the_default_edge_colour(matrices):
    m1, m2 = matrices
    left = np.array(["x"] * 6 + [None] * 6, dtype=object)

    fig, _, _ = tg.tanglegram(
        m1,
        m2,
        sort=False,
        clusters_left=left,
        cluster_colors={"x": "#ff0000"},
        color_edges_by="left",
        edge_kwargs=dict(color="black"),
    )

    colors = set(edge_colors(fig))
    assert colors == {rgba("#ff0000"), rgba("black")}
    plt.close("all")


def test_explicit_edge_colour_wins_over_cluster(matrices):
    m1, m2 = matrices
    left = np.array(["x"] * 12)
    edges = [(lb, lb) for lb in LABELS]
    edges[0] = (LABELS[0], LABELS[0], "#123456")

    fig, _, _ = tg.tanglegram(
        m1,
        m2,
        sort=False,
        edges=edges,
        clusters_left=left,
        cluster_colors={"x": "#ff0000"},
        color_edges_by="left",
    )

    assert rgba("#123456") in edge_colors(fig)
    plt.close("all")


def test_per_edge_colour_does_not_leak_to_later_edges(matrices):
    """`edge_kwargs["c"]` used to be mutated in the loop, so one coloured edge
    recoloured every edge after it."""
    m1, m2 = matrices
    edges = [(lb, lb) for lb in LABELS]
    edges[0] = (LABELS[0], LABELS[0], "#123456")

    fig, _, _ = tg.tanglegram(
        m1, m2, sort=False, edges=edges, edge_kwargs=dict(color="black")
    )

    colors = edge_colors(fig)
    assert colors.count(rgba("#123456")) == 1
    assert colors.count(rgba("black")) == len(edges) - 1
    plt.close("all")


def test_color_edges_by_without_clusters_warns(matrices, caplog):
    m1, m2 = matrices
    fig, _, _ = tg.tanglegram(m1, m2, sort=False, color_edges_by="left")

    assert "clusters_left" in caplog.text
    plt.close("all")


def test_bad_color_edges_by_raises(matrices):
    m1, m2 = matrices
    with pytest.raises(ValueError, match="left.*right"):
        tg.tanglegram(
            m1, m2, sort=False, clusters_left=np.zeros(N), color_edges_by="middle"
        )
    plt.close("all")


# ------------------------------------------------------------ plotly backend


def test_plotly_backend_returns_a_figure(matrices):
    plotly = pytest.importorskip("plotly")
    m1, m2 = matrices

    fig, dn1, dn2 = tg.tanglegram(m1, m2, sort=False, backend="plotly")

    assert isinstance(fig, plotly.graph_objects.Figure)
    assert len(dn1["leaves"]) == len(dn2["leaves"]) == N


def test_plotly_draws_one_edge_per_pair(matrices):
    pytest.importorskip("plotly")
    m1, m2 = matrices

    fig, dn1, dn2 = tg.tanglegram(m1, m2, sort=False, backend="plotly")

    # The connecting edges live on the middle subplot
    middle = [t for t in fig.data if t.xaxis == "x2"]
    assert len(middle) == N


def test_plotly_edges_line_up_with_the_leafs(matrices):
    """Both panels are the same height, so an edge has to leave the left tree
    at the same fraction of the way up as its leaf sits."""
    pytest.importorskip("plotly")
    m1, m2 = matrices

    fig, dn1, dn2 = tg.tanglegram(m1, m2, sort=False, backend="plotly")
    middle = [t for t in fig.data if t.xaxis == "x2"]

    label = middle[0].text.split(" - ")[0]
    expected = (dn1["ivl"].index(label) + 0.5) / N
    assert middle[0].y[0] == pytest.approx(expected)


def test_plotly_edges_carry_cluster_colours(matrices):
    pytest.importorskip("plotly")
    m1, m2 = matrices
    left = np.array(["x"] * 4 + ["y"] * 4 + ["z"] * 4)

    fig, _, _ = tg.tanglegram(
        m1,
        m2,
        sort=False,
        backend="plotly",
        clusters_left=left,
        cluster_colors={"x": "#ff0000", "y": "#00ff00", "z": "#0000ff"},
        color_edges_by="left",
    )

    middle = [t for t in fig.data if t.xaxis == "x2"]
    assert {t.line.color for t in middle} == {"#ff0000", "#00ff00", "#0000ff"}


def test_plotly_edges_hover_shows_both_labels(matrices):
    pytest.importorskip("plotly")
    m1, m2 = matrices

    fig, _, _ = tg.tanglegram(m1, m2, sort=False, backend="plotly")
    middle = [t for t in fig.data if t.xaxis == "x2"]

    assert all(" - " in t.text for t in middle)


def test_tanglegram_unknown_backend_raises(matrices):
    m1, m2 = matrices
    with pytest.raises(ValueError, match="[Uu]nknown backend"):
        tg.tanglegram(m1, m2, sort=False, backend="ggplot")


# --------------------------------------------------------- tanglegram_many


def many_matrices(k=3, n=15):
    labels = [f"n{i}" for i in range(n)]
    return [
        pd.DataFrame(
            squareform(pdist(np.random.default_rng(s).random((n, 4)))),
            columns=labels,
            index=labels,
        )
        for s in range(1, k + 1)
    ], np.array(["x"] * 5 + ["y"] * 5 + ["z"] * 5)


def test_many_shares_one_palette():
    mats, clusters = many_matrices()
    fig, dns = tg.tanglegram_many(mats, sort=False, clusters=[clusters] * 3)

    panels = [link_colors(ax) for ax in fig.get_axes()]
    assert panels[0] == panels[1] == panels[2]
    plt.close("all")


def test_many_does_not_touch_the_global_style():
    """It used to call plt.style.use("ggplot"), changing every later plot."""
    mats, _ = many_matrices()
    before = dict(plt.rcParams)

    tg.tanglegram_many(mats, sort=False)

    changed = [k for k, v in plt.rcParams.items() if before.get(k) != v]
    assert not changed
    plt.close("all")


def test_many_cluster_count_must_match():
    mats, clusters = many_matrices()
    with pytest.raises(ValueError, match="cluster assignments"):
        tg.tanglegram_many(mats, sort=False, clusters=[clusters] * 2)
    plt.close("all")


def test_many_returns_one_dendrogram_per_input():
    """Mirrors what tanglegram() returns for its two dendrograms."""
    mats, clusters = many_matrices(k=3)
    fig, dns = tg.tanglegram_many(mats, sort=False, clusters=[clusters] * 3)

    assert len(dns) == 3
    for dn in dns:
        assert set(dn) >= {"icoord", "dcoord", "ivl", "leaves", "color_list"}
        assert sorted(dn["ivl"]) == sorted(mats[0].columns)
    plt.close("all")


def test_many_dendrograms_are_in_input_order():
    """dns[i] must describe x[i], not whatever order they were drawn in."""
    mats, _ = many_matrices(k=3)
    fig, dns = tg.tanglegram_many(mats, sort=False)

    for mat, dn in zip(mats, dns):
        # Leaf order has to match the tree that matrix produces on its own
        own = sch.dendrogram(
            sch.linkage(squareform(mat.values, checks=False)),
            labels=list(mat.columns),
            no_plot=True,
        )["ivl"]
        assert dn["ivl"] == own
    plt.close("all")
