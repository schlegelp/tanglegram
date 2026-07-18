"""Tests for the tree-similarity measures.

These score the trees rather than the drawing, so the defining property is that
rotating a dendrogram must leave them completely unchanged - which is exactly
what separates them from `entanglement` and `crossings`.
"""

import numpy as np
import pytest
import scipy.cluster.hierarchy as sch

from scipy.spatial.distance import pdist

import tanglegram as tg
from tanglegram import tangle as tt


def data(n=25, seed=0, dims=4):
    return np.random.default_rng(seed).random((n, dims))


def tree(X, method="ward"):
    return sch.linkage(pdist(X), method)


# ------------------------------------------------------------------ basics


@pytest.mark.parametrize("metric", [tg.cophenetic_correlation, tg.baker_gamma])
def test_identical_trees_score_one(metric):
    Z = tree(data())
    assert metric(Z, Z) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [tg.cophenetic_correlation, tg.baker_gamma])
def test_different_methods_score_below_one(metric):
    X = data()
    score = metric(tree(X, "ward"), tree(X, "single"))
    assert -1.0 <= score < 1.0


@pytest.mark.parametrize("metric", [tg.cophenetic_correlation, tg.baker_gamma])
def test_rotation_does_not_change_the_score(metric):
    """The whole point: these measure topology, not layout."""
    X = data()
    Z1, Z2 = tree(X, "ward"), tree(X, "single")
    before = metric(Z1, Z2)

    rotated = tt.shuffle_dendogram(Z1, copy=True, rng=np.random.default_rng(1))
    assert metric(rotated, Z2) == pytest.approx(before)


def test_rotation_does_change_entanglement():
    """Guards the premise of the test above - if rotating stopped affecting
    entanglement, the comparison would be vacuous."""
    X = data()
    Z1, Z2 = tree(X, "ward"), tree(X, "single")
    labels = [f"n{i}" for i in range(len(X))]

    rotated = tt.shuffle_dendogram(Z1, copy=True, rng=np.random.default_rng(1))
    assert tg.entanglement(rotated, Z2, labels, labels) != pytest.approx(
        tg.entanglement(Z1, Z2, labels, labels)
    )


def test_baker_gamma_ignores_monotonic_rescaling():
    """Gamma is rank-based, so cubing every merge height must not move it."""
    X = data()
    Z1, Z2 = tree(X, "ward"), tree(X, "single")

    rescaled = Z1.copy()
    rescaled[:, 2] = rescaled[:, 2] ** 3

    assert tg.baker_gamma(rescaled, Z2) == pytest.approx(tg.baker_gamma(Z1, Z2))


def test_cophenetic_correlation_does_notice_rescaling():
    X = data()
    Z1, Z2 = tree(X, "ward"), tree(X, "single")

    rescaled = Z1.copy()
    rescaled[:, 2] = rescaled[:, 2] ** 3

    assert tg.cophenetic_correlation(rescaled, Z2) != pytest.approx(
        tg.cophenetic_correlation(Z1, Z2)
    )


# ------------------------------------------------------------- label matching


def test_labels_are_used_to_match_observations():
    """Shuffling the observations and their labels together must not change
    the score."""
    X = data(n=15)
    Z1, Z2 = tree(X, "ward"), tree(X, "single")
    labels = [f"n{i}" for i in range(len(X))]

    order = np.random.default_rng(3).permutation(len(X))
    Z2_shuffled = tree(X[order], "single")
    labels2 = [labels[i] for i in order]

    assert tg.baker_gamma(Z1, Z2_shuffled, labels, labels2) == pytest.approx(
        tg.baker_gamma(Z1, Z2, labels, labels)
    )


def test_only_shared_labels_are_compared():
    X = data(n=15)
    Z1 = tree(X, "ward")
    Z2 = tree(X[:10], "ward")

    labels1 = [f"n{i}" for i in range(15)]
    labels2 = [f"n{i}" for i in range(10)]

    # The first ten observations form the same tree in both, but only as a
    # subtree of the larger one, so this is well defined without being 1.
    score = tg.baker_gamma(Z1, Z2, labels1, labels2)
    assert -1.0 <= score <= 1.0


def test_explicit_edges():
    X = data(n=12)
    Z = tree(X)
    labels = [f"n{i}" for i in range(12)]
    edges = [(lb, lb) for lb in labels[:6]]

    assert tg.baker_gamma(Z, Z, labels, labels, edges=edges) == pytest.approx(1.0)


def test_no_shared_labels_raises():
    Z = tree(data(n=10))
    with pytest.raises(ValueError, match="not a single matching label|Not a single"):
        tg.baker_gamma(Z, Z, [f"a{i}" for i in range(10)], [f"b{i}" for i in range(10)])


def test_reused_leaf_raises():
    """A many-to-one matching would compare pairs against themselves."""
    Z = tree(data(n=10))
    labels = [f"n{i}" for i in range(10)]
    edges = [("n0", "n0"), ("n1", "n0"), ("n2", "n2"), ("n3", "n3")]

    with pytest.raises(ValueError, match="more than once"):
        tg.baker_gamma(Z, Z, labels, labels, edges=edges)


def test_too_few_matched_leafs_raises():
    Z = tree(data(n=10))
    labels = [f"n{i}" for i in range(10)]

    with pytest.raises(ValueError, match="at least 3"):
        tg.baker_gamma(Z, Z, labels, labels, edges=[("n0", "n0"), ("n1", "n1")])


def test_wrong_label_count_raises():
    Z = tree(data(n=10))
    with pytest.raises(ValueError, match="labels1"):
        tg.baker_gamma(Z, Z, [f"n{i}" for i in range(9)], None)


def test_duplicate_labels_raise():
    Z = tree(data(n=10))
    labels = ["dup"] * 10
    with pytest.raises(ValueError, match="unique"):
        tg.baker_gamma(Z, Z, labels, labels)


def test_unknown_label_in_edges_raises():
    Z = tree(data(n=10))
    labels = [f"n{i}" for i in range(10)]
    with pytest.raises(ValueError, match="does not exist"):
        tg.baker_gamma(Z, Z, labels, labels, edges=[("nope", "n0")] * 3)


# ------------------------------------------------------ cophenetic distances


def test_cophenetic_distances_shape_and_symmetry():
    Z = tree(data(n=8))
    d = tg.cophenetic_distances(Z, labels=[f"n{i}" for i in range(8)])

    assert d.shape == (8, 8)
    assert list(d.columns) == [f"n{i}" for i in range(8)]
    assert np.allclose(d.values, d.values.T)
    assert np.allclose(np.diag(d.values), 0)


def test_cophenetic_distance_is_the_merge_height():
    """Two leafs joined by the very first merge sit at that merge's height."""
    Z = tree(data(n=8))
    first_merge = Z[0]
    a, b, height = int(first_merge[0]), int(first_merge[1]), first_merge[2]

    d = tg.cophenetic_distances(Z)
    assert d.iloc[a, b] == pytest.approx(height)


def test_cophenetic_distances_label_mismatch_raises():
    Z = tree(data(n=8))
    with pytest.raises(ValueError, match="labels for"):
        tg.cophenetic_distances(Z, labels=["a", "b"])
