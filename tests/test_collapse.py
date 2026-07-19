"""Tests for drawing clades as collapsed wedges.

scipy cannot do this, so the layout is computed here from scratch. The leaf
*order* still comes from scipy, and the tests below lean on that: every subtree
occupies a contiguous block of slots, so a collapsed clade is one wedge.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.cluster.hierarchy as sch

from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from scipy.spatial.distance import pdist

import tanglegram as tg


N = 60


def tree(n=N, seed=0):
    return sch.linkage(pdist(np.random.default_rng(seed).random((n, 4))), "ward")


def wedges(ax):
    return [p for p in ax.patches if isinstance(p, Polygon)]


def rgba(color):
    return tuple(np.round(mcl.to_rgba(color), 6))


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


# ------------------------------------------------------------- what collapses


def test_threshold_collapses_short_clades(ax):
    Z = tree()
    threshold = 0.4 * Z[:, 2].max()
    dn, _ = tg.dendrogram(Z, collapse=threshold, ax=ax)

    assert dn["collapsed"]
    # Every wedge is a clade no taller than the threshold ...
    for node in dn["collapsed"]:
        assert Z[node - N, 2] <= threshold
    # ... and every remaining link is taller than it
    assert all(d[1] > threshold for d in dn["dcoord"])


def test_callable_collapses_small_clades(ax):
    dn, _ = tg.dendrogram(tree(), collapse=lambda leaves: len(leaves) <= 10, ax=ax)

    assert dn["collapsed"]
    assert all(len(v) <= 10 for v in dn["collapsed"].values())


def test_clusters_mode_collapses_each_cluster(ax):
    Z = tree()
    clusters = sch.fcluster(Z, 5, criterion="maxclust")

    dn, _ = tg.dendrogram(Z, clusters=clusters, collapse="clusters", ax=ax)

    # Each cluster forms a clade of its own here, so we get one wedge each
    assert len(dn["collapsed"]) == 5
    sizes = sorted(len(v) for v in dn["collapsed"].values())
    assert sizes == sorted(np.bincount(clusters)[1:])


def test_only_the_topmost_match_collapses(ax):
    """Collapsing a clade inside an already collapsed one would do nothing."""
    Z = tree()
    dn, _ = tg.dendrogram(Z, collapse=lambda leaves: True, ax=ax)

    assert len(dn["collapsed"]) == 1
    assert len(next(iter(dn["collapsed"].values()))) == N


def test_every_observation_is_accounted_for_exactly_once(ax):
    dn, _ = tg.dendrogram(tree(), collapse=lambda leaves: len(leaves) <= 8, ax=ax)

    covered = [obs for members in dn["collapsed"].values() for obs in members]
    covered += [u for u in dn["leaves"] if u < N]

    assert sorted(covered) == list(range(N))


def test_collapsing_reduces_the_number_of_units(ax):
    Z = tree()
    plain, _ = tg.dendrogram(Z, ax=ax)
    ax.clear()
    collapsed, _ = tg.dendrogram(Z, collapse=lambda lv: len(lv) <= 8, ax=ax)

    assert len(plain["leaves"]) == N
    assert len(collapsed["leaves"]) < N


# ------------------------------------------------------------------- drawing


def test_one_wedge_is_drawn_per_collapsed_clade(ax):
    dn, _ = tg.dendrogram(tree(), collapse=lambda lv: len(lv) <= 8, ax=ax)

    assert len(wedges(ax)) == len(dn["collapsed"])


def test_wedge_spans_from_its_height_down_to_the_baseline(ax):
    Z = tree()
    dn, _ = tg.dendrogram(Z, collapse=lambda lv: len(lv) <= 8, ax=ax)

    node = next(iter(dn["collapsed"]))
    height = Z[node - N, 2]

    tips = [w.get_xy()[:, 1].max() for w in wedges(ax)]
    assert pytest.approx(height, rel=1e-9) in tips
    assert all(w.get_xy()[:, 1].min() == pytest.approx(0) for w in wedges(ax))


def test_wedge_width_follows_collapse_width(ax):
    Z = tree()
    dn, _ = tg.dendrogram(Z, collapse=lambda lv: len(lv) <= 8, collapse_width=3, ax=ax)

    widths = {round(float(np.ptp(w.get_xy()[:, 0])), 6) for w in wedges(ax)}
    assert widths == {30.0}


def test_wedges_take_the_cluster_colour(ax):
    Z = tree()
    clusters = sch.fcluster(Z, 4, criterion="maxclust")
    colors = {c: f"#{c:02x}0000" for c in np.unique(clusters)}

    tg.dendrogram(
        Z, clusters=clusters, cluster_colors=colors, collapse="clusters", ax=ax
    )

    drawn = {rgba(w.get_edgecolor()[:3]) for w in wedges(ax)}
    assert drawn == {rgba(c) for c in colors.values()}


def test_wedges_inherit_scipy_colours_without_clusters(ax):
    """Uncollapsed and collapsed renderings should agree on colour."""
    Z = tree()
    tg.dendrogram(Z, collapse=lambda lv: len(lv) <= 8, ax=ax)

    drawn = {rgba(w.get_edgecolor()[:3]) for w in wedges(ax)}
    assert len(drawn) > 1  # not all fall back to above_threshold_color


def test_labels_report_the_leaf_count(ax):
    dn, _ = tg.dendrogram(tree(), collapse=lambda lv: len(lv) <= 8, ax=ax)

    for unit, label in zip(dn["leaves"], dn["ivl"]):
        if unit in dn["collapsed"]:
            assert label == f"({len(dn['collapsed'][unit])})"


def test_cluster_labels_name_the_cluster(ax):
    Z = tree()
    clusters = sch.fcluster(Z, 4, criterion="maxclust")
    dn, _ = tg.dendrogram(Z, clusters=clusters, collapse="clusters", ax=ax)

    assert all(" (" in label for label in dn["ivl"])
    assert any(label.startswith("1 (") for label in dn["ivl"])


def test_custom_collapse_label(ax):
    dn, _ = tg.dendrogram(
        tree(),
        collapse=lambda lv: len(lv) <= 8,
        collapse_label=lambda node, members: f"n={len(members)}",
        ax=ax,
    )

    assert all(label.startswith("n=") for label in dn["ivl"])


def test_axis_limits_cover_the_wedges(ax):
    """With everything collapsed there are no links left to scale the axis."""
    Z = tree()
    tg.dendrogram(Z, collapse=lambda lv: True, ax=ax)

    lo, hi = sorted(ax.get_ylim())
    assert hi >= Z[-1, 2]
    assert lo < hi


@pytest.mark.parametrize("orientation", ["top", "bottom", "left", "right"])
def test_all_orientations(orientation, ax):
    dn, _ = tg.dendrogram(
        tree(), collapse=lambda lv: len(lv) <= 8, orientation=orientation, ax=ax
    )
    assert len(wedges(ax)) == len(dn["collapsed"])


def test_polar():
    Z = tree()
    dn, ax = tg.dendrogram(Z, collapse=lambda lv: len(lv) <= 8, polar=True)

    assert len(wedges(ax)) == len(dn["collapsed"])
    # Wedge bases are arcs, so they carry more than three points
    assert all(len(w.get_xy()) > 4 for w in wedges(ax))
    plt.close("all")


def test_plotly():
    pytest.importorskip("plotly")
    dn, fig = tg.dendrogram(tree(), collapse=lambda lv: len(lv) <= 8, backend="plotly")

    filled = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
    assert len(filled) == len(dn["collapsed"])


def test_links_above_the_wedges_are_still_drawn(ax):
    dn, _ = tg.dendrogram(tree(), collapse=lambda lv: len(lv) <= 8, ax=ax)

    lines = [c for c in ax.collections if isinstance(c, LineCollection)]
    drawn = sum(len(c.get_segments()) for c in lines)
    # one bar plus two legs per remaining link
    assert drawn == 3 * len(dn["icoord"])


# --------------------------------------------------------------- validation


def test_clusters_mode_without_clusters_raises(ax):
    with pytest.raises(ValueError, match="requires `clusters`"):
        tg.dendrogram(tree(), collapse="clusters", ax=ax)


def test_unknown_string_raises(ax):
    with pytest.raises(ValueError, match="clusters"):
        tg.dendrogram(tree(), collapse="everything", ax=ax)


def test_bad_width_raises(ax):
    with pytest.raises(ValueError, match="collapse_width"):
        tg.dendrogram(tree(), collapse=0.5, collapse_width=0, ax=ax)


def test_truncate_mode_raises(ax):
    with pytest.raises(ValueError, match="truncate_mode"):
        tg.dendrogram(tree(), collapse=0.5, truncate_mode="lastp", p=5, ax=ax)


def test_leaf_annot_raises(ax):
    with pytest.raises(ValueError, match="collapse"):
        tg.dendrogram(tree(), collapse=0.5, leaf_annot=["a"] * N, ax=ax)


def test_per_observation_leaf_color_raises(ax):
    with pytest.raises(ValueError, match="collapse"):
        tg.dendrogram(
            tree(), collapse=0.5, leaf_marker="o", leaf_color=["red"] * N, ax=ax
        )


def test_single_leaf_color_still_works(ax):
    """A single color needs no per-observation mapping, so it is fine."""
    dn, _ = tg.dendrogram(
        tree(),
        collapse=lambda lv: len(lv) <= 8,
        leaf_marker="o",
        leaf_color="red",
        ax=ax,
    )
    assert dn["collapsed"]
