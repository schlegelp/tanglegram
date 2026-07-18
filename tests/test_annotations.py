"""Tests for the leaf annotation strip.

The strip gets its own axes on whichever side the leafs are and takes the leaf
labels with it, so most of these check placement and that nothing ends up
drawn twice.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import pdist

import tanglegram as tg


ORIENTATIONS = ("top", "bottom", "left", "right")
N = 12


def tree(n=N, seed=0):
    return sch.linkage(pdist(np.random.default_rng(seed).random((n, 3))), "ward")


def annot_frame(n=N):
    return pd.DataFrame(
        {
            "group": np.repeat(list("abcd"), n // 4),
            "qc": np.tile(["pass", "fail"], n // 2),
        }
    )


def strip_of(fig, ax):
    others = [a for a in fig.get_axes() if a is not ax]
    assert len(others) == 1, "expected exactly one annotation strip"
    return others[0]


def strip_image(strip):
    assert len(strip.images) == 1
    return strip.images[0].get_array()


def rgba(color):
    return tuple(np.round(mcl.to_rgba(color), 6))


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


# ------------------------------------------------------------------ layout


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_strip_is_added_on_the_leaf_side(orientation, ax):
    fig = ax.figure
    tg.dendrogram(tree(), leaf_annot=annot_frame(), orientation=orientation, ax=ax)
    strip = strip_of(fig, ax)
    fig.canvas.draw()

    tree_box, strip_box = ax.get_position(), strip.get_position()
    if orientation == "top":  # leafs at the bottom
        assert strip_box.y1 <= tree_box.y0 + 1e-9
    elif orientation == "bottom":  # leafs at the top
        assert strip_box.y0 >= tree_box.y1 - 1e-9
    elif orientation == "left":  # leafs on the right
        assert strip_box.x0 >= tree_box.x1 - 1e-9
    else:  # leafs on the left
        assert strip_box.x1 <= tree_box.x0 + 1e-9


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_leaf_labels_are_drawn_once_on_the_strip(orientation, ax):
    """The strip shares an axis with the tree, so clearing ticks on one used to
    clear them on both - and re-adding them drew the labels twice."""
    fig = ax.figure
    dn, _ = tg.dendrogram(
        tree(), leaf_annot=annot_frame(), orientation=orientation, ax=ax
    )
    strip = strip_of(fig, ax)

    params = (
        ax.xaxis.get_tick_params()
        if orientation in ("top", "bottom")
        else (ax.yaxis.get_tick_params())
    )
    drawn_by_tree = any(
        params.get(k) for k in ("labelbottom", "labeltop", "labelleft", "labelright")
    )
    assert not drawn_by_tree

    labels = (
        strip.get_xticklabels()
        if orientation in ("top", "bottom")
        else strip.get_yticklabels()
    )
    assert [t.get_text() for t in labels] == dn["ivl"]


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_labels_sit_outside_the_strip(orientation, ax):
    """Labels must not land between the tree and the strip."""
    fig = ax.figure
    tg.dendrogram(tree(), leaf_annot=annot_frame(), orientation=orientation, ax=ax)
    strip = strip_of(fig, ax)
    fig.canvas.draw()

    box = strip.get_position()
    ticklabels = (
        strip.get_xticklabels()
        if orientation in ("top", "bottom")
        else strip.get_yticklabels()
    )
    label = [t for t in ticklabels if t.get_text()][0]
    bb = label.get_window_extent().transformed(fig.transFigure.inverted())

    if orientation == "top":
        assert bb.y1 <= box.y0 + 1e-3
    elif orientation == "bottom":
        assert bb.y0 >= box.y1 - 1e-3
    elif orientation == "left":
        assert bb.x0 >= box.x1 - 1e-3
    else:
        assert bb.x1 <= box.x0 + 1e-3


def test_track_names_are_labelled(ax):
    fig = ax.figure
    tg.dendrogram(tree(), leaf_annot=annot_frame(), ax=ax)
    strip = strip_of(fig, ax)

    assert [t.get_text() for t in strip.get_yticklabels()] == ["group", "qc"]


def test_no_labels_leaves_the_strip_unlabelled(ax):
    fig = ax.figure
    tg.dendrogram(tree(), leaf_annot=annot_frame(), no_labels=True, ax=ax)
    strip = strip_of(fig, ax)

    assert not [t.get_text() for t in strip.get_xticklabels() if t.get_text()]


# ------------------------------------------------------------------ content


def test_cells_follow_leaf_order(ax):
    """Column p of the strip must describe the leaf drawn at position p."""
    fig = ax.figure
    values = np.array(["a"] * N)
    values[3] = "z"  # a single odd one out
    dn, _ = tg.dendrogram(
        tree(),
        leaf_annot={"t": values},
        annot_colors={"a": "#0000ff", "z": "#ff0000"},
        ax=ax,
    )
    img = strip_image(strip_of(fig, ax))

    odd = dn["leaves"].index(3)
    assert tuple(np.round(img[0, odd], 6)) == rgba("#ff0000")
    assert tuple(np.round(img[0, (odd + 1) % N], 6)) == rgba("#0000ff")


def test_colours_are_used_directly_when_given_as_colours(ax):
    fig = ax.figure
    colours = ["#ff0000"] * N
    dn, _ = tg.dendrogram(tree(), leaf_annot={"t": colours}, ax=ax)
    img = strip_image(strip_of(fig, ax))

    assert tuple(np.round(img[0, 0], 6)) == rgba("#ff0000")


def test_missing_values_are_transparent(ax):
    fig = ax.figure
    values = np.array(["a"] * N, dtype=object)
    values[2] = None
    dn, _ = tg.dendrogram(tree(), leaf_annot={"t": values}, ax=ax)
    img = strip_image(strip_of(fig, ax))

    blank = dn["leaves"].index(2)
    assert img[0, blank, 3] == 0  # alpha
    assert img[0, (blank + 1) % N, 3] == 1


def test_one_row_per_track(ax):
    fig = ax.figure
    tg.dendrogram(tree(), leaf_annot=annot_frame(), ax=ax)
    img = strip_image(strip_of(fig, ax))

    assert img.shape[:2] == (2, N)


def test_accepts_a_flat_sequence(ax):
    fig = ax.figure
    tg.dendrogram(tree(), leaf_annot=["a"] * N, ax=ax)
    img = strip_image(strip_of(fig, ax))

    assert img.shape[:2] == (1, N)


def test_accepts_a_dict(ax):
    fig = ax.figure
    tg.dendrogram(tree(), leaf_annot={"one": ["a"] * N, "two": ["b"] * N}, ax=ax)
    strip = strip_of(fig, ax)

    assert [t.get_text() for t in strip.get_yticklabels()] == ["one", "two"]


def test_per_track_colours(ax):
    fig = ax.figure
    dn, _ = tg.dendrogram(
        tree(),
        leaf_annot={"one": ["a"] * N, "two": ["a"] * N},
        annot_colors={"one": {"a": "#ff0000"}, "two": {"a": "#0000ff"}},
        ax=ax,
    )
    img = strip_image(strip_of(fig, ax))

    assert tuple(np.round(img[0, 0], 6)) == rgba("#ff0000")
    assert tuple(np.round(img[1, 0], 6)) == rgba("#0000ff")


def test_legend(ax):
    tg.dendrogram(
        tree(),
        leaf_annot={"qc": np.tile(["pass", "fail"], N // 2)},
        annot_legend=True,
        ax=ax,
    )
    legend = ax.get_legend()

    assert legend is not None
    assert {t.get_text() for t in legend.get_texts()} == {"qc: pass", "qc: fail"}


# --------------------------------------------------------------- validation


def test_wrong_length_raises(ax):
    with pytest.raises(ValueError, match="leaf_annot"):
        tg.dendrogram(tree(), leaf_annot=["a"] * (N - 1), ax=ax)


def test_polar_rejected(ax):
    with pytest.raises(NotImplementedError, match="polar"):
        tg.dendrogram(tree(), leaf_annot=["a"] * N, polar=True)


def test_plotly_rejected():
    pytest.importorskip("plotly")
    with pytest.raises(NotImplementedError, match="matplotlib"):
        tg.dendrogram(tree(), leaf_annot=["a"] * N, backend="plotly")


def test_three_dimensional_raises(ax):
    with pytest.raises(ValueError, match="1- or 2-dimensional"):
        tg.dendrogram(tree(), leaf_annot=np.zeros((2, 2, N)), ax=ax)
