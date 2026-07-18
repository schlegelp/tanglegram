"""Tests for the untangling algorithms.

The dynamic-programming methods claim to find the *globally* optimal set of
rotations, so wherever the tree is small enough these are checked against an
exhaustive search over all ``2**(n-1)`` rotations.
"""

from itertools import product

import numpy as np
import pytest
import scipy.cluster.hierarchy as sch

import tanglegram as tg
from tanglegram import tangle as tt


# ---------------------------------------------------------------- helpers


def random_tree(n, seed, dims=3):
    return sch.linkage(np.random.default_rng(seed).random((n, dims)), "average")


def labels_for(n, prefix="l"):
    return np.array([f"{prefix}{i}" for i in range(n)])


def identity_edges(labels):
    return [(l, l) for l in labels]


def apply_rotations(link, rows):
    link = link.copy()
    rows = np.asarray(rows, dtype=int)
    if rows.size:
        link[rows, :2] = link[rows, :2][:, ::-1]
    return link


def brute_force_1side(link1, link2, labels1, labels2, edges, L=1.5):
    """Lowest entanglement reachable by rotating link1, by exhaustive search."""
    lindex2 = tt.leaf_order(link2, labels2, as_dict=True)
    best = np.inf
    for bits in product([0, 1], repeat=link1.shape[0]):
        cand = apply_rotations(link1, np.nonzero(bits)[0])
        best = min(
            best,
            tt._entanglement(
                tt.leaf_order(cand, labels1, as_dict=True), lindex2, edges, L=L
            ),
        )
    return best


# ------------------------------------------------------- entanglement()


def test_entanglement_honours_edges():
    """Regression: the public wrapper used to silently drop `edges`."""
    labels = labels_for(12)
    link1, link2 = random_tree(12, 0), random_tree(12, 1)

    all_edges = tg.entanglement(link1, link2, labels, labels, identity_edges(labels))
    few_edges = tg.entanglement(
        link1, link2, labels, labels, identity_edges(labels)[:3]
    )

    assert not np.isclose(all_edges, few_edges)


def test_entanglement_edges_default_to_shared_labels():
    labels = labels_for(12)
    link1, link2 = random_tree(12, 0), random_tree(12, 1)

    assert np.isclose(
        tg.entanglement(link1, link2, labels, labels),
        tg.entanglement(link1, link2, labels, labels, identity_edges(labels)),
    )


def test_identical_trees_have_zero_entanglement():
    labels = labels_for(20)
    link = random_tree(20, 5)
    assert tg.entanglement(link, link, labels, labels) == 0


# ------------------------------------------------------ DP optimality


@pytest.mark.parametrize("seed", range(8))
def test_dp_1side_is_globally_optimal(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 11))
    labels = labels_for(n)
    link1, link2 = random_tree(n, seed), random_tree(n, seed + 100)
    edges = identity_edges(labels)
    L = float(rng.choice([1.0, 1.5, 2.0]))

    out1, out2 = tt.untangle_dp_1side(link1, link2, labels, labels, edges, L=L)

    assert np.isclose(
        tg.entanglement(out1, out2, labels, labels, edges, L=L),
        brute_force_1side(link1, link2, labels, labels, edges, L=L),
    )


def test_dp_1side_with_partial_edges_and_unequal_sizes():
    labels1, labels2 = labels_for(9), labels_for(13)
    link1, link2 = random_tree(9, 3), random_tree(13, 4)
    edges = identity_edges(labels1)[:6]  # only some leafs take part

    out1, out2 = tt.untangle_dp_1side(link1, link2, labels1, labels2, edges)

    assert np.isclose(
        tg.entanglement(out1, out2, labels1, labels2, edges),
        brute_force_1side(link1, link2, labels1, labels2, edges),
    )


def test_dp_1side_never_worse_than_greedy():
    labels = labels_for(120)
    rng = np.random.default_rng(0)
    X = rng.random((120, 5))
    link1 = sch.linkage(X, "average")
    link2 = sch.linkage(X + rng.normal(0, 0.8, X.shape), "average")
    edges = identity_edges(labels)

    greedy = tg.entanglement(
        *tt.untangle_step_rotate_1side(link1, link2, labels, labels, edges),
        labels,
        labels,
        edges,
    )
    exact = tg.entanglement(
        *tt.untangle_dp_1side(link1, link2, labels, labels, edges),
        labels,
        labels,
        edges,
    )
    assert exact <= greedy + 1e-12


def test_dp_2side_improves_and_is_reproducible():
    labels = labels_for(150)
    rng = np.random.default_rng(2)
    X = rng.random((150, 5))
    link1 = sch.linkage(X, "average")
    link2 = sch.linkage(X + rng.normal(0, 0.7, X.shape), "average")
    edges = identity_edges(labels)

    before = tg.entanglement(link1, link2, labels, labels, edges)
    a1, b1 = tt.untangle_dp_2side(
        link1, link2, labels, labels, edges, restarts=3, seed=42
    )
    a2, b2 = tt.untangle_dp_2side(
        link1, link2, labels, labels, edges, restarts=3, seed=42
    )

    assert tg.entanglement(a1, b1, labels, labels, edges) < before
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


# ------------------------------------------------- structural invariants


@pytest.mark.parametrize(
    "method", ["dp1side", "dp2side", "cross1side", "cross2side", "step1side", "step2side"]
)
def test_untangling_preserves_the_dendrogram(method):
    labels = labels_for(80)
    link1, link2 = random_tree(80, 11), random_tree(80, 12)
    edges = identity_edges(labels)
    original = link1.copy()

    out1, out2 = tg.untangle(link1, link2, labels, labels, edges, method=method)

    assert sch.is_valid_linkage(out1) and sch.is_valid_linkage(out2)
    # the caller's arrays must not be modified in place
    assert np.array_equal(link1, original)
    # only the order of each merge's two children may change
    assert np.array_equal(np.sort(original[:, :2], axis=1), np.sort(out1[:, :2], axis=1))
    assert np.array_equal(original[:, 2:], out1[:, 2:])
    assert sorted(sch.leaves_list(out1)) == list(range(80))


def test_step1side_visits_every_hinge():
    """Regression: hinge 0 used to be skipped."""
    labels = np.array(["a", "b", "c"])
    link = np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 3.0]])
    # differs from `link` only in the order of the very first merge
    target = np.array([[1.0, 0.0, 1.0, 2.0], [2.0, 3.0, 2.0, 3.0]])
    edges = identity_edges(labels)

    out, _ = tt.untangle_step_rotate_1side(link, target, labels, labels, edges)

    assert tg.entanglement(out, target, labels, labels, edges) == 0


# --------------------------------------------------------------- crossings


def naive_crossings(link1, link2, labels1, labels2, edges):
    """O(E^2) reference: two edges cross iff their endpoints are inverted."""
    li1 = tt.leaf_order(link1, labels1, as_dict=True)
    li2 = tt.leaf_order(link2, labels2, as_dict=True)
    pts = [(li1[a], li2[b]) for a, b in edges]
    return sum(
        1
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
        if (pts[i][0] - pts[j][0]) * (pts[i][1] - pts[j][1]) < 0
    )


@pytest.mark.parametrize("seed", range(6))
def test_crossings_matches_naive_count(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 40))
    labels = labels_for(n)
    link1, link2 = random_tree(n, seed), random_tree(n, seed + 50)
    edges = identity_edges(labels)

    assert tg.crossings(link1, link2, labels, labels, edges) == naive_crossings(
        link1, link2, labels, labels, edges
    )


def test_crossings_with_shared_and_duplicated_endpoints():
    """Edges off the same leaf never cross; edges onto the same leaf don't either."""
    labels = labels_for(10)
    link1, link2 = random_tree(10, 1), random_tree(10, 2)
    edges = identity_edges(labels) + [("l0", "l5"), ("l0", "l7"), ("l3", "l5")]

    assert tg.crossings(link1, link2, labels, labels, edges) == naive_crossings(
        link1, link2, labels, labels, edges
    )


def test_identical_trees_have_zero_crossings():
    labels = labels_for(25)
    link = random_tree(25, 8)
    assert tg.crossings(link, link, labels, labels) == 0


@pytest.mark.parametrize("seed", range(8))
def test_cross_1side_is_globally_optimal(seed):
    """The crossing DP must match exhaustive search over all rotations."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 11))
    labels = labels_for(n)
    link1, link2 = random_tree(n, seed), random_tree(n, seed + 200)
    edges = identity_edges(labels)

    brute = min(
        tg.crossings(apply_rotations(link1, np.nonzero(bits)[0]), link2, labels,
                     labels, edges)
        for bits in product([0, 1], repeat=link1.shape[0])
    )
    out1, out2 = tt.untangle_cross_1side(link1, link2, labels, labels, edges)

    assert tg.crossings(out1, out2, labels, labels, edges) == brute


def test_cross_2side_reduces_crossings():
    labels = labels_for(200)
    rng = np.random.default_rng(4)
    X = rng.random((200, 5))
    link1 = sch.linkage(X, "average")
    link2 = sch.linkage(X + rng.normal(0, 0.6, X.shape), "average")
    edges = identity_edges(labels)

    before = tg.crossings(link1, link2, labels, labels, edges)
    a, b = tt.untangle_cross_2side(
        link1, link2, labels, labels, edges, restarts=3, seed=0
    )
    assert tg.crossings(a, b, labels, labels, edges) < before


def test_cross_methods_scale_without_an_n_squared_table():
    """Mostly a smoke test: 5k leafs would be a 200 MB table for the dp methods."""
    n = 5000
    rng = np.random.default_rng(0)
    X = rng.random((n, 4))
    link1 = sch.linkage(X, "average")
    link2 = sch.linkage(X + rng.normal(0, 0.6, X.shape), "average")
    labels = labels_for(n)
    edges = identity_edges(labels)

    before = tg.crossings(link1, link2, labels, labels, edges)
    a, b = tg.untangle(link1, link2, labels, labels, edges, method="cross2side")
    assert tg.crossings(a, b, labels, labels, edges) < before


# -------------------------------------------------------- alignment quality


def test_alignment_quality_reports_a_sane_verdict():
    labels = labels_for(200)
    rng = np.random.default_rng(6)
    X = rng.random((200, 5))
    link1 = sch.linkage(X, "average")
    link2 = sch.linkage(X + rng.normal(0, 0.6, X.shape), "average")
    edges = identity_edges(labels)

    a, b = tg.untangle(link1, link2, labels, labels, edges, restarts=20, seed=0)
    q = tg.alignment_quality(a, b, labels, labels, edges, n_random=50, seed=0)

    assert q.n_leafs == 200 and q.n_edges == 200
    assert q.entanglement < q.random_entanglement      # better than a random layout
    assert q.z_score < 0 and 0 < q.improvement < 1
    assert 0 <= q.frac_well_aligned <= 1
    assert q.at_floor                                   # optimised, so nothing left
    assert "entanglement" in str(q)


def test_alignment_quality_flags_a_bad_alignment():
    """An un-untangled pair should not be reported as being at the floor."""
    labels = labels_for(150)
    rng = np.random.default_rng(7)
    X = rng.random((150, 5))
    link1 = sch.linkage(X, "average")
    link2 = sch.linkage(X + rng.normal(0, 0.5, X.shape), "average")
    edges = identity_edges(labels)

    worst = tt.shuffle_dendogram(link2, copy=True, rng=np.random.default_rng(0))
    q = tg.alignment_quality(
        link1, worst, labels, labels, edges, n_random=30, restarts=5, seed=0
    )
    assert not q.at_floor
    assert q.best_found < q.entanglement


# ------------------------------------------------------------- removals


@pytest.mark.parametrize(
    "name",
    ["untangle_permutations", "get_all_linkage", "get_all_linkage_gen",
     "CachedGenerator", "bottom_up", "refine"],
)
def test_dead_code_is_gone(name):
    assert not hasattr(tt, name)


def test_permutations_method_removed():
    labels = labels_for(6)
    link1, link2 = random_tree(6, 0), random_tree(6, 1)
    with pytest.raises(ValueError):
        tg.untangle(
            link1, link2, labels, labels, identity_edges(labels),
            method="permutations",
        )


def test_unknown_method_raises():
    labels = labels_for(6)
    link1, link2 = random_tree(6, 0), random_tree(6, 1)
    with pytest.raises(ValueError):
        tg.untangle(
            link1, link2, labels, labels, identity_edges(labels), method="nope"
        )


# ----------------------------------------------------- kwarg filtering


def test_filter_kwargs_drops_options_the_method_lacks():
    assert tt._filter_kwargs("dp1side", {"restarts": 3, "seed": 0, "L": 2}) == {"L": 2}
    assert tt._filter_kwargs("dp2side", {"restarts": 3, "seed": 0, "L": 2}) == {
        "restarts": 3,
        "seed": 0,
        "L": 2,
    }
