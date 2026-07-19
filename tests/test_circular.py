"""Tests for the circular tanglegram layout.

The point of the layout is topological: with the trees pointing outwards from a
shared inner ring, every leaf row lies on one circle, so a connector is a chord
across the middle that crosses no tree. Most of these tests check exactly that.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from matplotlib.collections import LineCollection
from scipy.spatial.distance import pdist, squareform

import tanglegram as tg
from tanglegram.tangle import _LABEL_BAND


N = 12
LABELS = [f"s{i}" for i in range(N)]
CLUSTERS = np.array(["x"] * 4 + ["y"] * 4 + ["z"] * 4)


def view(seed):
    X = np.random.default_rng(seed).random((N, 3))
    return pd.DataFrame(squareform(pdist(X)), index=LABELS, columns=LABELS)


def views(k=3):
    return [view(s) for s in range(1, k + 1)]


def tree_radii(ax):
    """Distance from the centre of every point of every drawn tree."""
    out = []
    for coll in ax.collections:
        if not isinstance(coll, LineCollection):
            continue
        for seg in coll.get_segments():
            seg = np.asarray(seg)
            out.extend(np.hypot(seg[:, 0], seg[:, 1]))
    return np.asarray(out)


def chord_radii(ax):
    """Distance from the centre of every point of every connector."""
    out = []
    for line in ax.lines:
        x, y = line.get_xdata(), line.get_ydata()
        out.extend(np.hypot(x, y))
    return np.asarray(out)


def rgba(color):
    return tuple(np.round(mcl.to_rgba(color), 6))


# ------------------------------------------------------------------- basics


def test_returns_figure_and_dendrograms():
    fig, dns = tg.tanglegram_many(views(), sort=False, layout="circular")

    assert len(dns) == 3
    for dn in dns:
        assert sorted(dn["ivl"]) == sorted(LABELS)
    plt.close("all")


def test_draws_onto_a_single_axis():
    """Unlike the rows layout, everything shares one axis."""
    fig, dns = tg.tanglegram_many(views(), sort=False, layout="circular")

    assert len(fig.get_axes()) == 1
    ax = fig.get_axes()[0]
    assert ax.get_aspect() == 1.0
    assert not ax.axison
    plt.close("all")


@pytest.mark.parametrize("k", [2, 3, 5])
def test_works_for_any_number_of_trees(k):
    fig, dns = tg.tanglegram_many(views(k), sort=False, layout="circular")
    assert len(dns) == k
    plt.close("all")


# ------------------------------------------------------- the whole point


def test_chords_never_reach_the_trees():
    """The defining property: no connector crosses a tree.

    Every tree sits at radius >= the leaf ring, every chord at radius <= it, so
    the two can never intersect.
    """
    inner = 0.45
    fig, dns = tg.tanglegram_many(
        views(), sort=True, layout="circular", inner_radius=inner
    )
    ax = fig.get_axes()[0]

    leaf_ring = inner + _LABEL_BAND
    assert tree_radii(ax).min() >= leaf_ring - 1e-9
    assert chord_radii(ax).max() <= inner + 1e-9
    plt.close("all")


def test_trees_reach_the_outer_rim():
    fig, dns = tg.tanglegram_many(views(), sort=False, layout="circular")
    ax = fig.get_axes()[0]

    assert tree_radii(ax).max() == pytest.approx(1.0, abs=1e-9)
    plt.close("all")


def test_inner_radius_moves_the_ring():
    fig, dns = tg.tanglegram_many(
        views(),
        sort=False,
        layout="circular",
        inner_radius=0.2,
        dend_kwargs=dict(no_labels=True),
    )
    ax = fig.get_axes()[0]
    assert chord_radii(ax).max() <= 0.2 + 1e-9
    plt.close("all")


def test_trees_occupy_separate_sectors():
    """Each tree gets its own wedge of angle, with a gap between them."""
    fig, dns = tg.tanglegram_many(views(3), sort=False, layout="circular")
    ax = fig.get_axes()[0]

    # Angles covered by the trees, bucketed into thirds of the circle
    angles = []
    for coll in ax.collections:
        for seg in coll.get_segments():
            seg = np.asarray(seg)
            angles.extend(np.arctan2(seg[:, 1], seg[:, 0]) % (2 * np.pi))
    angles = np.asarray(angles)

    per_sector = [
        ((angles >= i * 2 * np.pi / 3) & (angles < (i + 1) * 2 * np.pi / 3)).sum()
        for i in range(3)
    ]
    assert all(count > 0 for count in per_sector)
    plt.close("all")


def test_one_chord_per_edge():
    fig, dns = tg.tanglegram_many(views(3), sort=False, layout="circular")
    ax = fig.get_axes()[0]

    # Two consecutive pairs, N edges each
    assert len(ax.lines) == 2 * N
    plt.close("all")


# ----------------------------------------------------------------- styling


def test_edges_coloured_by_source_cluster():
    fig, dns = tg.tanglegram_many(
        views(),
        sort=False,
        layout="circular",
        clusters=[CLUSTERS] * 3,
        cluster_colors={"x": "#ff0000", "y": "#00ff00", "z": "#0000ff"},
        color_edges_by="source",
    )
    ax = fig.get_axes()[0]

    colors = {rgba(line.get_color()) for line in ax.lines}
    assert colors == {rgba("#ff0000"), rgba("#00ff00"), rgba("#0000ff")}
    plt.close("all")


def test_edges_coloured_by_target_cluster():
    fig, dns = tg.tanglegram_many(
        views(),
        sort=False,
        layout="circular",
        clusters=[CLUSTERS] * 3,
        cluster_colors={"x": "#ff0000", "y": "#00ff00", "z": "#0000ff"},
        color_edges_by="target",
    )
    ax = fig.get_axes()[0]

    assert len({rgba(line.get_color()) for line in ax.lines}) == 3
    plt.close("all")


def test_clusters_share_one_palette_across_sectors():
    fig, dns = tg.tanglegram_many(
        views(), sort=False, layout="circular", clusters=[CLUSTERS] * 3
    )
    ax = fig.get_axes()[0]

    colors = {
        tuple(np.round(c.get_colors()[0], 6))
        for c in ax.collections
        if isinstance(c, LineCollection)
    }
    # three clusters plus the grey links joining them
    assert len(colors) == 4
    plt.close("all")


def test_names_are_drawn():
    fig, dns = tg.tanglegram_many(
        views(), sort=False, layout="circular", names=["one", "two", "three"]
    )
    ax = fig.get_axes()[0]

    assert {"one", "two", "three"} <= {t.get_text() for t in ax.texts}
    plt.close("all")


def test_leaf_labels_are_drawn_by_default():
    fig, dns = tg.tanglegram_many(views(), sort=False, layout="circular")
    ax = fig.get_axes()[0]

    assert sum(t.get_text() in LABELS for t in ax.texts) == 3 * N
    plt.close("all")


def test_no_labels_drops_them_and_the_label_band():
    fig, dns = tg.tanglegram_many(
        views(), sort=False, layout="circular", dend_kwargs=dict(no_labels=True)
    )
    ax = fig.get_axes()[0]

    assert not [t for t in ax.texts if t.get_text() in LABELS]
    # Without labels the trees start right at the ring
    assert tree_radii(ax).min() == pytest.approx(0.45, abs=1e-9)
    plt.close("all")


def test_edge_kwargs_style_the_chords():
    fig, dns = tg.tanglegram_many(
        views(), sort=False, layout="circular", edge_kwargs=dict(color="red", lw=3)
    )
    ax = fig.get_axes()[0]

    assert all(rgba(line.get_color()) == rgba("red") for line in ax.lines)
    assert all(line.get_linewidth() == 3 for line in ax.lines)
    plt.close("all")


def test_explicit_edge_colour_wins():
    edges = [[(lb, lb) for lb in LABELS] for _ in range(2)]
    edges[0][0] = (LABELS[0], LABELS[0], "#123456")

    fig, dns = tg.tanglegram_many(views(), sort=False, layout="circular", edges=edges)
    ax = fig.get_axes()[0]

    assert rgba("#123456") in {rgba(line.get_color()) for line in ax.lines}
    plt.close("all")


# --------------------------------------------------------------- validation


def test_unknown_layout_raises():
    with pytest.raises(ValueError, match="[Uu]nknown layout"):
        tg.tanglegram_many(views(), sort=False, layout="spiral")


def test_bad_inner_radius_raises():
    with pytest.raises(ValueError, match="inner_radius"):
        tg.tanglegram_many(views(), sort=False, layout="circular", inner_radius=0)


def test_names_length_must_match():
    with pytest.raises(ValueError, match="names"):
        tg.tanglegram_many(views(3), sort=False, names=["only", "two"])


def test_bad_color_edges_by_raises():
    with pytest.raises(ValueError, match="source.*target"):
        tg.tanglegram_many(
            views(), sort=False, clusters=[CLUSTERS] * 3, color_edges_by="left"
        )


def test_color_edges_by_without_clusters_warns(caplog):
    tg.tanglegram_many(views(), sort=False, layout="circular", color_edges_by="source")

    assert "clusters" in caplog.text
    plt.close("all")
