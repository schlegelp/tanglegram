"""Tests for the enhanced dendrogram function.

The renderer no longer delegates drawing to scipy - it takes scipy's
coordinates and draws the segments itself so that each half of a link can be
styled independently. The tests below therefore lean heavily on comparing the
drawn geometry against what scipy itself would have produced.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcl
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import pdist

import tanglegram as tg
from tanglegram import dend


ORIENTATIONS = ("top", "bottom", "left", "right")


# ---------------------------------------------------------------- helpers


def random_tree(n=25, seed=0, dims=4, method="average"):
    return sch.linkage(pdist(np.random.default_rng(seed).random((n, dims))), method)


def _undirected(seg):
    """A segment as an orientation-independent key."""
    a, b = tuple(seg[:2]), tuple(seg[2:])
    return tuple(sorted([a, b]))


def drawn_segments(ax):
    """Every line segment on an axis, as a sorted list of comparable keys."""
    out = []
    for coll in line_collections(ax):
        for seg in coll.get_segments():
            seg = np.asarray(seg)
            for a, b in zip(seg[:-1], seg[1:]):
                if not np.allclose(a, b):
                    out.append(_undirected(np.round(np.concatenate([a, b]), 6)))
    return sorted(out)


def line_collections(ax):
    """Only the link collections - leaf markers are a PathCollection."""
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def rgba(color):
    """A color rounded the same way `segment_colors` rounds, so they compare."""
    return tuple(np.round(mcl.to_rgba(color), 6))


def segment_colors(ax):
    """Map each drawn segment to the RGBA it was drawn in."""
    out = {}
    for coll in line_collections(ax):
        color = tuple(np.round(coll.get_colors()[0], 6))
        for seg in coll.get_segments():
            seg = np.asarray(seg)
            for a, b in zip(seg[:-1], seg[1:]):
                if not np.allclose(a, b):
                    out[_undirected(np.round(np.concatenate([a, b]), 6))] = color
    return out


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


# ---------------------------------------------------------------- parity with scipy


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_geometry_matches_scipy(orientation, ax):
    """We must draw exactly the lines scipy would have drawn."""
    Z = random_tree()

    fig_ref, ax_ref = plt.subplots()
    sch.dendrogram(Z, orientation=orientation, ax=ax_ref)
    expected = drawn_segments(ax_ref)
    plt.close(fig_ref)

    tg.dendrogram(Z, orientation=orientation, ax=ax)

    assert drawn_segments(ax) == expected


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_axis_limits_match_scipy(orientation, ax):
    """`tanglegram()` positions its subplots off these limits."""
    Z = random_tree()

    fig_ref, ax_ref = plt.subplots()
    sch.dendrogram(Z, orientation=orientation, ax=ax_ref)
    xlim, ylim = ax_ref.get_xlim(), ax_ref.get_ylim()
    plt.close(fig_ref)

    tg.dendrogram(Z, orientation=orientation, ax=ax)

    assert np.allclose(ax.get_xlim(), xlim)
    assert np.allclose(ax.get_ylim(), ylim)


def test_returns_scipy_dendrogram_dict(ax):
    Z = random_tree()
    dn, returned = tg.dendrogram(Z, ax=ax)

    assert returned is ax
    assert set(dn) >= {"icoord", "dcoord", "ivl", "leaves", "color_list"}
    assert dn["leaves"] == sch.dendrogram(Z, no_plot=True)["leaves"]


def test_defaults_to_scipy_orientation(ax):
    """The old implementation assumed "bottom" while scipy defaults to "top"."""
    Z = random_tree()

    fig_ref, ax_ref = plt.subplots()
    sch.dendrogram(Z, ax=ax_ref)
    ylim = ax_ref.get_ylim()
    plt.close(fig_ref)

    tg.dendrogram(Z, ax=ax)
    assert np.allclose(ax.get_ylim(), ylim)


def test_color_threshold_still_honoured(ax):
    """Passing no clusters must leave scipy's own colouring intact."""
    Z = random_tree()
    dn, _ = tg.dendrogram(Z, color_threshold=0.7 * Z[:, 2].max(), ax=ax)

    assert len(set(dn["color_list"])) > 1


# ---------------------------------------------------------------- cluster colouring


def test_clusters_colour_links(ax):
    Z = random_tree()
    clusters = sch.fcluster(Z, 4, criterion="maxclust")

    tg.dendrogram(Z, clusters=clusters, ax=ax)

    colors = set(segment_colors(ax).values())
    # four clusters plus the grey links joining them
    assert len(colors) == 5
    assert rgba("lightgrey") in colors


def test_cluster_colours_are_deterministic(ax):
    """Colours used to depend on set iteration order, so could shuffle."""
    Z = random_tree()
    clusters = np.array(["c", "a", "b"] * 8 + ["a"])

    runs = []
    for _ in range(3):
        fig, this_ax = plt.subplots()
        tg.dendrogram(Z, clusters=clusters, ax=this_ax)
        runs.append(segment_colors(this_ax))
        plt.close(fig)

    assert runs[0] == runs[1] == runs[2]


def test_cluster_colour_follows_first_appearance(ax):
    """Cluster order is defined by first appearance, not by sort order."""
    Z = random_tree(n=6)
    clusters = np.array(["z", "z", "z", "a", "a", "a"])

    tg.dendrogram(Z, clusters=clusters, cluster_colors=["#ff0000", "#00ff00"], ax=ax)

    colors = set(segment_colors(ax).values())
    assert rgba("#ff0000") in colors  # "z" appears first
    assert rgba("#00ff00") in colors


def test_singleton_cluster_gets_its_own_colour(ax):
    """A cluster of one leaf has no link of its own - only a leg of its parent."""
    Z = random_tree(n=12)
    clusters = np.zeros(12, dtype=int)
    clusters[3] = 1  # a lone leaf

    tg.dendrogram(
        Z, clusters=clusters, cluster_colors={0: "#0000ff", 1: "#ff0000"}, ax=ax
    )

    colors = list(segment_colors(ax).values())
    assert colors.count(rgba("#ff0000")) == 1


def test_unassigned_leafs_are_above_threshold(ax):
    Z = random_tree(n=10)
    clusters = np.array([0.0] * 5 + [np.nan] * 5)

    tg.dendrogram(Z, clusters=clusters, above_threshold_color="black", ax=ax)

    colors = set(segment_colors(ax).values())
    assert rgba("black") in colors


def test_none_is_treated_as_unassigned(ax):
    Z = random_tree(n=10)
    clusters = np.array([0] * 5 + [None] * 5, dtype=object)

    tg.dendrogram(Z, clusters=clusters, above_threshold_color="black", ax=ax)

    assert rgba("black") in set(segment_colors(ax).values())


def test_wrong_number_of_clusters_raises(ax):
    Z = random_tree(n=10)
    with pytest.raises(ValueError, match="clusters for"):
        tg.dendrogram(Z, clusters=np.zeros(9), ax=ax)


def test_missing_cluster_colour_raises(ax):
    Z = random_tree(n=10)
    clusters = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="missing entries"):
        tg.dendrogram(Z, clusters=clusters, cluster_colors={0: "red"}, ax=ax)


def test_clusters_with_truncate_mode_raises(ax):
    Z = random_tree(n=20)
    with pytest.raises(ValueError, match="truncate_mode"):
        tg.dendrogram(Z, clusters=np.zeros(20), truncate_mode="lastp", p=5, ax=ax)


def test_above_threshold_linestyle(ax):
    Z = random_tree(n=12)
    clusters = np.array([0] * 6 + [1] * 6)

    tg.dendrogram(
        Z,
        clusters=clusters,
        above_threshold_color="black",
        above_threshold_ls="--",
        ax=ax,
    )

    dashed, solid = set(), set()
    for coll in line_collections(ax):
        color = tuple(np.round(coll.get_colors()[0], 6))
        # matplotlib reports a dash pattern as (offset, onoffseq); solid is None
        bucket = dashed if coll.get_linestyle()[0][1] is not None else solid
        bucket.add(color)

    # Only the links spanning clusters are dashed; the clusters stay solid
    assert dashed == {rgba("black")}
    assert solid and rgba("black") not in solid


# ---------------------------------------------------------------- hanging


def test_hanging_lifts_leafs_off_the_baseline(ax):
    Z = random_tree(n=15)

    tg.dendrogram(Z, hanging=True, ax=ax)
    ys = [p[1] for seg in drawn_segments(ax) for p in seg]

    # Without hanging the leafs sit exactly at 0
    assert min(ys) > 0


def test_hanging_drop_is_fraction_of_height(ax):
    Z = random_tree(n=15)
    height = Z[:, 2].max()

    tg.dendrogram(Z, hanging=0.5, ax=ax)
    ys = [p[1] for seg in drawn_segments(ax) for p in seg]

    # Every leaf hangs 0.5 * height below its parent, clipped at 0
    assert min(ys) >= 0
    assert max(ys) == pytest.approx(height)


def test_hanging_off_by_default_touches_baseline(ax):
    Z = random_tree(n=15)
    tg.dendrogram(Z, ax=ax)
    ys = [p[1] for seg in drawn_segments(ax) for p in seg]
    assert min(ys) == 0


def test_hanging_with_zero_distance_merges(ax):
    """Duplicate observations merge at distance 0, so inner nodes also sit on
    the baseline - they must not be mistaken for leafs."""
    X = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [5.0, 5.0]]
    )
    Z = sch.linkage(pdist(X), "average")
    assert (Z[:, 2] == 0).sum() > 0  # guard the premise

    tg.dendrogram(Z, hanging=True, leaf_marker="o", ax=ax)

    offsets = np.asarray(ax.collections[-1].get_offsets())
    assert len(offsets) == 6
    assert sorted(offsets[:, 0]) == pytest.approx(np.arange(5, 65, 10))


def test_negative_hanging_raises(ax):
    with pytest.raises(ValueError, match="hanging"):
        tg.dendrogram(random_tree(n=8), hanging=-0.1, ax=ax)


# ---------------------------------------------------------------- leafs & labels


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_leaf_markers_sit_at_leaf_positions(orientation, ax):
    Z = random_tree(n=10)
    dn, _ = tg.dendrogram(Z, leaf_marker="o", orientation=orientation, ax=ax)

    offsets = np.asarray(ax.collections[-1].get_offsets())
    assert len(offsets) == 10

    independent = offsets[:, 0] if orientation in ("top", "bottom") else offsets[:, 1]
    assert sorted(independent) == pytest.approx(np.arange(5, 105, 10))


def test_leaf_colour_defaults_to_link_colour(ax):
    Z = random_tree(n=12)
    clusters = np.array([0] * 6 + [1] * 6)

    tg.dendrogram(
        Z,
        clusters=clusters,
        leaf_marker="o",
        cluster_colors={0: "#ff0000", 1: "#0000ff"},
        ax=ax,
    )

    facecolors = {tuple(np.round(c, 6)) for c in ax.collections[-1].get_facecolors()}
    assert facecolors == {rgba("#ff0000"), rgba("#0000ff")}


def test_explicit_leaf_colour_is_reordered_to_leaf_order(ax):
    """`leaf_color` is given per observation, like scipy's `labels`."""
    Z = random_tree(n=6)
    dn = sch.dendrogram(Z, no_plot=True)
    per_obs = ["#ff0000"] * 6
    per_obs[dn["leaves"][0]] = "#00ff00"  # colour whichever obs is drawn first

    tg.dendrogram(Z, leaf_marker="o", leaf_color=per_obs, ax=ax)

    facecolors = ax.collections[-1].get_facecolors()
    assert tuple(np.round(facecolors[0], 6)) == rgba("#00ff00")


def test_single_leaf_colour_applies_to_all(ax):
    Z = random_tree(n=8)
    tg.dendrogram(Z, leaf_marker="o", leaf_color="red", ax=ax)

    facecolors = {tuple(np.round(c, 6)) for c in ax.collections[-1].get_facecolors()}
    assert facecolors == {rgba("red")}


def test_wrong_number_of_leaf_colours_raises(ax):
    with pytest.raises(ValueError, match="leaf_color"):
        tg.dendrogram(random_tree(n=8), leaf_marker="o", leaf_color=["red"] * 7, ax=ax)


def test_label_dist_adds_one_label_per_link(ax):
    Z = random_tree(n=10)
    tg.dendrogram(Z, label_dist=True, ax=ax)

    assert len(ax.texts) == len(Z)


def test_label_dist_respects_format(ax):
    Z = random_tree(n=6)
    tg.dendrogram(Z, label_dist=True, dist_fmt="{:.4f}", ax=ax)

    assert all(len(t.get_text().split(".")[1]) == 4 for t in ax.texts)


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_label_dist_sits_on_the_link(orientation, ax):
    """The old code negated coordinates for some orientations, which put the
    labels outside the axes."""
    Z = random_tree(n=8)
    tg.dendrogram(Z, label_dist=True, orientation=orientation, ax=ax)

    xlim = sorted(ax.get_xlim())
    ylim = sorted(ax.get_ylim())
    for text in ax.texts:
        x, y = text.get_position()
        assert xlim[0] <= x <= xlim[1]
        assert ylim[0] <= y <= ylim[1]


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_leaf_labels(orientation, ax):
    Z = random_tree(n=8)
    labels = [f"leaf{i}" for i in range(8)]
    dn, _ = tg.dendrogram(Z, labels=labels, orientation=orientation, ax=ax)

    ticklabels = (
        ax.get_xticklabels()
        if orientation in ("top", "bottom")
        else ax.get_yticklabels()
    )
    assert [t.get_text() for t in ticklabels] == dn["ivl"]


def test_no_labels(ax):
    Z = random_tree(n=8)
    tg.dendrogram(Z, no_labels=True, ax=ax)
    assert len(ax.get_xticks()) == 0


# ---------------------------------------------------------------- polar


def test_polar_creates_polar_axes():
    Z = random_tree(n=12)
    dn, ax = tg.dendrogram(Z, polar=True)

    assert ax.name == "polar"
    # Everything must sit within one full turn
    thetas = [p[0] for seg in drawn_segments(ax) for p in seg]
    assert 0 <= min(thetas) and max(thetas) <= 2 * np.pi
    plt.close("all")


def test_polar_puts_leafs_on_the_rim():
    """Leafs belong on the outside, next to their labels; the root at the centre."""
    Z = random_tree(n=12)
    dn, ax = tg.dendrogram(Z, polar=True, leaf_marker="o")

    radii = [p[1] for seg in drawn_segments(ax) for p in seg]
    r_max = ax.get_ylim()[1]

    # The root sits at the centre, the leafs out at the rim. `drawn_segments`
    # rounds to 6 decimals, hence the tolerance.
    assert min(radii) == pytest.approx(r_max - Z[:, 2].max(), abs=1e-6)
    assert max(radii) == pytest.approx(r_max, abs=1e-6)

    leaf_radii = np.asarray(ax.collections[-1].get_offsets())[:, 1]
    assert np.allclose(leaf_radii, r_max)
    plt.close("all")


def test_polar_draws_bars_as_arcs():
    """A straight line between two angles would cut a chord across the circle."""
    Z = random_tree(n=12)
    dn, ax = tg.dendrogram(Z, polar=True)

    # Arcs are subdivided, so there are more segments than in the cartesian case
    fig, cart = plt.subplots()
    tg.dendrogram(Z, ax=cart)
    assert len(drawn_segments(ax)) > len(drawn_segments(cart))
    plt.close("all")


def test_polar_radial_labels_sit_in_the_gap():
    """Leafs occupy angles 5, 15, ...; the gap between the last and the first
    straddles 0, which is where the radial labels belong."""
    dn, ax = tg.dendrogram(random_tree(n=12), polar=True)

    assert ax.get_rlabel_position() == pytest.approx(0)
    plt.close("all")


def test_polar_has_no_grid_spokes():
    """One angular tick per leaf means one spoke per leaf - a solid disc on any
    sizeable tree."""
    dn, ax = tg.dendrogram(random_tree(n=40), polar=True)

    assert not any(line.get_visible() for line in ax.xaxis.get_gridlines())
    # the rings carrying the distance scale stay
    assert any(line.get_visible() for line in ax.yaxis.get_gridlines())
    plt.close("all")


def test_polar_rejects_cartesian_axes(ax):
    with pytest.raises(ValueError, match="polar projection"):
        tg.dendrogram(random_tree(n=8), polar=True, ax=ax)


def test_polar_rejects_plotly():
    pytest.importorskip("plotly")
    with pytest.raises(NotImplementedError, match="[Pp]olar"):
        tg.dendrogram(random_tree(n=8), polar=True, backend="plotly")


# ---------------------------------------------------------------- plotly


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_plotly_geometry_matches_matplotlib(orientation, ax):
    """Both backends must render the same tree."""
    pytest.importorskip("plotly")
    Z = random_tree(n=15)

    tg.dendrogram(Z, orientation=orientation, ax=ax)
    expected = drawn_segments(ax)

    dn, fig = tg.dendrogram(Z, orientation=orientation, backend="plotly")

    got = []
    for trace in fig.data:
        if trace.mode != "lines":
            continue
        xs, ys = trace.x, trace.y
        for i in range(0, len(xs), 3):
            got.append(_undirected(np.round([xs[i], ys[i], xs[i + 1], ys[i + 1]], 6)))
    assert sorted(got) == expected


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_plotly_axis_direction_matches_matplotlib(orientation, ax):
    """The old plotly backend had the orientation semantics inverted."""
    pytest.importorskip("plotly")
    Z = random_tree(n=10)

    tg.dendrogram(Z, orientation=orientation, ax=ax)
    dn, fig = tg.dendrogram(Z, orientation=orientation, backend="plotly")

    mpl_reversed = ax.get_ylim()[0] > ax.get_ylim()[1]
    plotly_reversed = fig.layout.yaxis.autorange == "reversed"
    assert mpl_reversed == plotly_reversed


def test_plotly_labels():
    pytest.importorskip("plotly")
    Z = random_tree(n=8)
    labels = [f"leaf{i}" for i in range(8)]
    dn, fig = tg.dendrogram(Z, labels=labels, backend="plotly")

    assert list(fig.layout.xaxis.ticktext) == dn["ivl"]


def test_plotly_hover_info():
    pytest.importorskip("plotly")
    Z = random_tree(n=8)
    info = [f"info{i}" for i in range(8)]
    dn, fig = tg.dendrogram(Z, hover_info=info, backend="plotly")

    marker = [t for t in fig.data if t.mode == "markers"][0]
    assert "info" in marker.text[0]


def test_plotly_accepts_subplot_target():
    pytest.importorskip("plotly")
    from plotly.subplots import make_subplots

    Z = random_tree(n=8)
    fig = make_subplots(rows=1, cols=2)
    dn, out = tg.dendrogram(Z, backend="plotly", fig={"fig": fig, "row": 1, "col": 2})

    assert out is fig
    assert len(fig.data) > 0


def test_plotly_rejects_unknown_fig_keys():
    pytest.importorskip("plotly")
    with pytest.raises(ValueError, match="Unexpected keys"):
        tg.dendrogram(random_tree(n=8), backend="plotly", fig={"nope": 1})


# ---------------------------------------------------------------- validation


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="[Uu]nknown backend"):
        tg.dendrogram(random_tree(n=8), backend="ggplot")


def test_unknown_orientation_raises():
    with pytest.raises(ValueError, match="[Uu]nknown orientation"):
        tg.dendrogram(random_tree(n=8), orientation="sideways")


@pytest.mark.parametrize("reserved", ["no_plot", "link_color_func"])
def test_reserved_kwargs_raise(reserved):
    with pytest.raises(ValueError, match=reserved):
        tg.dendrogram(random_tree(n=8), **{reserved: True})


# ---------------------------------------------------------------- integration


def test_tanglegram_still_works():
    """`tanglegram()` used to rely on `plt.gca()` picking the right axis."""
    import pandas as pd

    labels = ["A", "B", "C", "D"]
    data = [
        [0, 0.1, 0.4, 0.3],
        [0.1, 0, 0.5, 0.6],
        [0.4, 0.5, 0, 0.2],
        [0.3, 0.6, 0.2, 0],
    ]
    mat = pd.DataFrame(data, columns=labels, index=labels)

    fig, dn1, dn2 = tg.tanglegram(mat, mat, sort=False)

    ax1, ax2 = fig.get_axes()[0], fig.get_axes()[1]
    assert len(ax1.collections) > 0
    assert len(ax2.collections) > 0
    assert sorted(dn1["ivl"]) == labels
    plt.close("all")


def test_tanglegram_with_clusters():
    import pandas as pd

    labels = ["A", "B", "C", "D"]
    data = [
        [0, 0.1, 0.4, 0.3],
        [0.1, 0, 0.5, 0.6],
        [0.4, 0.5, 0, 0.2],
        [0.3, 0.6, 0.2, 0],
    ]
    mat = pd.DataFrame(data, columns=labels, index=labels)

    fig, dn1, dn2 = tg.tanglegram(
        mat, mat, sort=False, clusters_left=[0, 0, 1, 1], clusters_right=[0, 0, 1, 1]
    )
    assert len(fig.get_axes()) >= 3
    plt.close("all")


# ---------------------------------------------------------------- internals


def test_assign_node_clusters_marks_mixed_nodes():
    Z = random_tree(n=6)
    n = len(Z) + 1
    node_cluster = dend._assign_node_clusters(Z, [0, 0, 0, 1, 1, 1])

    assert list(node_cluster[:n]) == [0, 0, 0, 1, 1, 1]
    # the root must span both clusters
    assert node_cluster[-1] is dend._MIXED


def test_assign_node_clusters_propagates_uniform_cluster():
    Z = random_tree(n=6)
    node_cluster = dend._assign_node_clusters(Z, [7] * 6)
    assert all(c == 7 for c in node_cluster)
