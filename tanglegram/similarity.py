# A Python package to plot tanglegrams
#
#    Copyright (C) 2017 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

"""Measures of how similar two dendrograms are.

`entanglement` and `crossings` in :mod:`tanglegram.tangle` score a *drawing* -
rotate a dendrogram and they change. The measures here score the *trees*: they
look at how far apart each pair of leafs sits in either hierarchy, which no
amount of rotating can alter.
"""

import numpy as np

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.stats as sstats


__all__ = ["cophenetic_correlation", "baker_gamma", "cophenetic_distances"]


def cophenetic_distances(link, labels=None):
    """Cophenetic distance between every pair of leafs.

    The cophenetic distance between two leafs is the height at which they are
    first joined, i.e. how deep in the tree you have to go to get from one to
    the other.

    Parameters
    ----------
    link :      scipy.cluster.hierarchy.linkage
    labels :    list-like, optional
                Labels for the original observations. Only used to build the
                index/columns of the returned frame.

    Returns
    -------
    pandas.DataFrame
                Square matrix of cophenetic distances.

    """
    import pandas as pd

    dists = ssd.squareform(sch.cophenet(link))

    if labels is None:
        labels = np.arange(len(dists))
    elif len(labels) != len(dists):
        raise ValueError(f"Got {len(labels)} labels for {len(dists)} observations.")

    return pd.DataFrame(dists, index=list(labels), columns=list(labels))


def cophenetic_correlation(link1, link2, labels1=None, labels2=None, edges=None):
    """Correlation between the cophenetic distances of two dendrograms.

    This is the Pearson correlation between how far apart each pair of leafs
    sits in the first tree and how far apart the same pair sits in the second.
    1 means the two hierarchies agree completely, 0 that they are unrelated.

    Unlike `entanglement` this does not care how the trees are drawn - rotating
    a dendrogram leaves it unchanged.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to compare.
    labels1,labels2 :   list-like, optional
                        Labels for link1 and link2. If not provided, the
                        observations are matched up by index.
    edges :             list of tuples, optional
                        ``[(l1, l2), ...]`` pairing a label in the first tree
                        with one in the second. If not provided, labels present
                        in both trees are paired up.

    Returns
    -------
    float
                        Correlation in [-1, 1], or NaN if either tree has no
                        variation in its cophenetic distances.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.cluster.hierarchy as sch
    >>> import tanglegram as tg
    >>> X = np.random.rand(20, 4)
    >>> Z1 = sch.linkage(X, method="ward")
    >>> Z2 = sch.linkage(X, method="single")
    >>> tg.cophenetic_correlation(Z1, Z2)   # doctest: +SKIP
    0.71...

    See Also
    --------
    baker_gamma
                        The rank-based equivalent, which ignores the actual
                        merge heights and looks only at their order.

    """
    v1, v2 = _matched_cophenetic_vectors(link1, link2, labels1, labels2, edges)

    if _is_constant(v1) or _is_constant(v2):
        return float("nan")

    return float(sstats.pearsonr(v1, v2)[0])


def baker_gamma(link1, link2, labels1=None, labels2=None, edges=None):
    """Baker's gamma index for two dendrograms.

    The Spearman (rank) correlation between the cophenetic distances of the two
    trees. Because it only looks at the *order* in which leafs are joined it is
    insensitive to the actual merge heights, which makes it the fairer measure
    when the two linkages were built with different methods or on differently
    scaled data.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to compare.
    labels1,labels2 :   list-like, optional
                        Labels for link1 and link2. If not provided, the
                        observations are matched up by index.
    edges :             list of tuples, optional
                        ``[(l1, l2), ...]`` pairing a label in the first tree
                        with one in the second. If not provided, labels present
                        in both trees are paired up.

    Returns
    -------
    float
                        Gamma in [-1, 1], or NaN if either tree has no
                        variation in its cophenetic distances.

    See Also
    --------
    cophenetic_correlation
                        The equivalent using the merge heights themselves.

    """
    v1, v2 = _matched_cophenetic_vectors(link1, link2, labels1, labels2, edges)

    if _is_constant(v1) or _is_constant(v2):
        return float("nan")

    return float(sstats.spearmanr(v1, v2)[0])


def _is_constant(v):
    return len(np.unique(v)) < 2


def _matched_cophenetic_vectors(link1, link2, labels1, labels2, edges):
    """Cophenetic distances of both trees over their shared leafs.

    Returns two equal-length vectors holding, for the same pair of leafs, the
    cophenetic distance in the first and in the second tree.
    """
    n1, n2 = len(link1) + 1, len(link2) + 1

    index1 = _label_index(labels1, n1, "labels1")
    index2 = _label_index(labels2, n2, "labels2")

    if edges is None:
        shared = [lb for lb in index1 if lb in index2]
        if not shared:
            raise ValueError("Not a single matching label in both dendrograms.")
        edges = [(lb, lb) for lb in shared]

    left, right = [], []
    for e in edges:
        if e[0] not in index1:
            raise ValueError(f"Label {e[0]} does not exist in dendrogram 1")
        if e[1] not in index2:
            raise ValueError(f"Label {e[1]} does not exist in dendrogram 2")
        left.append(index1[e[0]])
        right.append(index2[e[1]])

    # A leaf used twice would compare a pair against itself and quietly skew
    # the result, so insist on a one-to-one matching.
    for side, picked in (("first", left), ("second", right)):
        if len(set(picked)) != len(picked):
            raise ValueError(
                f"`edges` uses at least one leaf of the {side} dendrogram more "
                "than once; the two trees must be matched one-to-one."
            )

    if len(left) < 3:
        raise ValueError(
            f"Need at least 3 matched leafs to correlate, got {len(left)}."
        )

    left, right = np.asarray(left), np.asarray(right)

    # squareform() so we can pull out the matched subset in one go, then take
    # the upper triangle to get each pair exactly once
    coph1 = ssd.squareform(sch.cophenet(link1))[np.ix_(left, left)]
    coph2 = ssd.squareform(sch.cophenet(link2))[np.ix_(right, right)]

    iu = np.triu_indices(len(left), k=1)
    return coph1[iu], coph2[iu]


def _label_index(labels, n, name):
    """Map label -> observation index."""
    if labels is None:
        return {i: i for i in range(n)}

    labels = list(labels)
    if len(labels) != n:
        raise ValueError(f"Got {len(labels)} `{name}` for {n} observations.")

    if len(set(labels)) != len(labels):
        raise ValueError(f"`{name}` must be unique.")

    return {lb: i for i, lb in enumerate(labels)}
