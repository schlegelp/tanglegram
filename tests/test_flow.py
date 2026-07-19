"""Tests for the alluvial cluster-flow view.

The ribbons carry counts, so most of these check that what is drawn adds up:
every observation that has a cluster on both sides ends up in exactly one
ribbon, and block heights track cluster sizes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch

from matplotlib.patches import Polygon, Rectangle
from scipy.spatial.distance import pdist

import tanglegram as tg


N = 120


def cuts(ks=(2, 4, 8), n=N, seed=0):
    Z = sch.linkage(pdist(np.random.default_rng(seed).random((n, 4))), "ward")
    return [sch.fcluster(Z, k, criterion="maxclust") for k in ks]


def blocks(ax):
    return [p for p in ax.patches if isinstance(p, Rectangle)]


def ribbons(ax):
    return [p for p in ax.patches if isinstance(p, Polygon)]


def rgba(color):
    return tuple(np.round(mcl.to_rgba(color), 6))


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------------ counts


def test_returns_flows_and_axis():
    flows, ax = tg.cluster_flow(cuts())

    assert isinstance(flows, list)
    assert {"step", "source", "target", "count"} == set(flows[0])
    assert ax.figure is not None


def test_every_observation_is_carried_exactly_once():
    """No nulls here, so each step's ribbons must account for all of them."""
    stages = cuts()
    flows, ax = tg.cluster_flow(stages)

    for step in range(len(stages) - 1):
        assert sum(f["count"] for f in flows if f["step"] == step) == N


def test_counts_match_the_actual_membership():
    stages = cuts(ks=(2, 4))
    flows, ax = tg.cluster_flow(stages)

    for f in flows:
        expected = np.sum((stages[0] == f["source"]) & (stages[1] == f["target"]))
        assert f["count"] == expected


def test_one_ribbon_per_flow():
    flows, ax = tg.cluster_flow(cuts())
    assert len(ribbons(ax)) == len(flows)


def test_one_block_per_cluster_per_column():
    stages = cuts(ks=(2, 4, 8))
    flows, ax = tg.cluster_flow(stages)

    expected = sum(len(np.unique(s)) for s in stages)
    assert len(blocks(ax)) == expected


def test_block_heights_track_cluster_sizes():
    stages = cuts(ks=(2, 4))
    flows, ax = tg.cluster_flow(stages, gap=0)

    first_column = [b for b in blocks(ax) if b.get_x() < 0.5]
    heights = sorted(b.get_height() for b in first_column)
    sizes = sorted(np.bincount(stages[0])[1:] / len(stages[0]))

    assert heights == pytest.approx(sizes, abs=1e-9)


def test_columns_are_equally_tall():
    stages = cuts(ks=(2, 8))
    flows, ax = tg.cluster_flow(stages, gap=0)

    for x in (0, 1):
        column = [b for b in blocks(ax) if abs(b.get_x() + 0.025 - x) < 0.1]
        assert sum(b.get_height() for b in column) == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------- missing values


def test_unassigned_observations_are_dropped_from_the_flow():
    a = np.array(["p"] * 60 + ["q"] * 60, dtype=object)
    b = a.copy()
    b[:10] = None

    flows, ax = tg.cluster_flow([a, b])

    assert sum(f["count"] for f in flows) == N - 10
    assert all(f["source"] is not None and f["target"] is not None for f in flows)


def test_unassigned_observations_do_not_get_a_block():
    a = np.array(["p"] * 60 + ["q"] * 60, dtype=object)
    b = a.copy()
    b[:10] = np.nan

    flows, ax = tg.cluster_flow([a, b])

    # p, q in the first column; p, q in the second - nothing for the NaNs
    assert len(blocks(ax)) == 4


# -------------------------------------------------------------- ordering


def test_sorting_reduces_ribbon_crossings():
    """The barycentre order should not be worse than first-appearance order."""
    stages = cuts(ks=(4, 8, 16), seed=3)

    def crossings(sort):
        flows, ax = tg.cluster_flow(stages, sort=sort)
        # Count pairs of ribbons whose ends are in opposite vertical order
        by_step = {}
        for patch, f in zip(ribbons(ax), sorted(flows, key=lambda f: f["step"])):
            xy = patch.get_xy()
            by_step.setdefault(f["step"], []).append((xy[0][1], xy[-1][1]))
        total = 0
        for pairs in by_step.values():
            for i, (a0, a1) in enumerate(pairs):
                for b0, b1 in pairs[i + 1 :]:
                    if (a0 - b0) * (a1 - b1) < 0:
                        total += 1
        return total

    assert crossings(True) <= crossings(False)


def test_unsorted_uses_first_appearance():
    a = np.array(["second"] * 60 + ["first"] * 60)
    flows, ax = tg.cluster_flow([a, a], sort=False)

    first_column = sorted(
        (b for b in blocks(ax) if b.get_x() < 0.5),
        key=lambda b: -b.get_y(),
    )
    # "second" appears first in the data, so it sits at the top
    assert first_column[0].get_height() == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------- colouring


def test_shared_colours_by_default():
    a = np.array(["p"] * 60 + ["q"] * 60)
    flows, ax = tg.cluster_flow([a, a], colors={"p": "#ff0000", "q": "#0000ff"})

    facecolors = {rgba(b.get_facecolor()) for b in blocks(ax)}
    assert facecolors == {rgba("#ff0000"), rgba("#0000ff")}


def test_unshared_colours_give_each_column_its_own_palette():
    """`fcluster` labels are arbitrary per column, so sharing them would imply
    a correspondence that is not there."""
    a = np.array([1] * 60 + [2] * 60)
    b = np.array([2] * 60 + [1] * 60)

    shared, ax1 = tg.cluster_flow([a, b], share_colors=True)
    per_column, ax2 = tg.cluster_flow([a, b], share_colors=False)

    def column_colors(ax, left):
        return [
            rgba(p.get_facecolor()) for p in blocks(ax) if (p.get_x() < 0.5) == left
        ]

    # Shared: cluster 1 is the same colour in both columns
    assert set(column_colors(ax1, True)) == set(column_colors(ax1, False))
    # Per column: the top block of each column takes the first palette colour
    assert column_colors(ax2, True)[0] == column_colors(ax2, False)[0]


def test_ribbon_colour_follows_the_source():
    a = np.array(["p"] * 60 + ["q"] * 60)
    flows, ax = tg.cluster_flow(
        [a, a], colors={"p": "#ff0000", "q": "#0000ff"}, color_by="source"
    )

    assert {rgba(r.get_facecolor())[:3] for r in ribbons(ax)} == {
        rgba("#ff0000")[:3],
        rgba("#0000ff")[:3],
    }


def test_ribbon_colour_can_follow_the_target():
    a = np.array(["p"] * 60 + ["q"] * 60)
    b = np.array(["z"] * 120)

    flows, ax = tg.cluster_flow(
        [a, b],
        colors={"p": "#ff0000", "q": "#0000ff", "z": "#00ff00"},
        color_by="target",
    )

    assert {rgba(r.get_facecolor())[:3] for r in ribbons(ax)} == {rgba("#00ff00")[:3]}


def test_cluster_labels_are_drawn():
    a = np.array(["p"] * 60 + ["q"] * 60)
    flows, ax = tg.cluster_flow([a, a])

    assert {t.get_text() for t in ax.texts} == {"p", "q"}


def test_tiny_blocks_are_left_unlabelled():
    a = np.array(["big"] * 119 + ["tiny"])
    flows, ax = tg.cluster_flow([a, a], min_label_size=0.1)

    assert "tiny" not in {t.get_text() for t in ax.texts}


def test_labels_can_be_turned_off():
    a = np.array(["p"] * 60 + ["q"] * 60)
    flows, ax = tg.cluster_flow([a, a], label_clusters=False)

    assert not ax.texts


# ------------------------------------------------------------------- input


def test_accepts_a_dataframe_and_takes_its_column_names():
    frame = pd.DataFrame({"before": ["p"] * 60 + ["q"] * 60, "after": ["p"] * 120})
    flows, ax = tg.cluster_flow(frame)

    assert [t.get_text() for t in ax.get_xticklabels()] == ["before", "after"]


def test_stage_names():
    flows, ax = tg.cluster_flow(cuts(ks=(2, 4)), names=["coarse", "fine"])
    assert [t.get_text() for t in ax.get_xticklabels()] == ["coarse", "fine"]


def test_draws_onto_a_given_axis():
    fig, ax = plt.subplots()
    flows, returned = tg.cluster_flow(cuts(ks=(2, 4)), ax=ax)

    assert returned is ax
    assert blocks(ax)


# --------------------------------------------------------------- validation


def test_needs_at_least_two_clusterings():
    with pytest.raises(ValueError, match="at least 2"):
        tg.cluster_flow([np.zeros(10)])


def test_clusterings_must_be_the_same_length():
    with pytest.raises(ValueError, match="same observations"):
        tg.cluster_flow([np.zeros(10), np.zeros(11)])


def test_bad_color_by_raises():
    with pytest.raises(ValueError, match="source.*target"):
        tg.cluster_flow(cuts(ks=(2, 4)), color_by="middle")


def test_name_count_must_match():
    with pytest.raises(ValueError, match="names"):
        tg.cluster_flow(cuts(ks=(2, 4)), names=["only one"])


def test_all_unassigned_raises():
    empty = np.array([None] * 10, dtype=object)
    with pytest.raises(ValueError, match="no assigned observations"):
        tg.cluster_flow([empty, empty])
