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

import networkx as nx
import numpy as np
import pandas as pd


def make_edges(x, N=1, forward=True, reverse=True):
    """Utility function to extract edges from distance matrix.

    Parameters
    ----------
    x :         pd.DataFrame
                Distance matrix to extract edges from.
    N :         int >= 1 | float < 1
                If `int`, will the use the top `N` edges (top 1 is default). If
                float < 1 will include all edges where the
                `distance <= minimum distance + minimum distance * N`.
    forward :   bool
                Whether to include forward (rows -> columns) edges.
    reverse :   bool
                Whether to include reverse (columns -> rows) edges.

    Returns
    -------
    edges :     list
                List of edges `[(source, target), ...]` where `source` is always
                the row and `target` is always the target, no matter the
                direction. Note that this means there might be duplicate edges!

    """
    assert isinstance(x, pd.DataFrame)

    edges = []

    if forward:
        if N == 1:
            edges += [(i, c) for i, c in zip(x.index.values,
                                             x.columns.values[np.argmin(x.values, axis=1)])]
        elif N < 1:
            mn = x.min(axis=1).values
            is_below = x.values <= (mn + mn * N).reshape(-1, 1)
            edges += [(i, c) for i, c in zip(x.index[np.where(is_below)[0]],
                                             x.columns[np.where(is_below)[0]])]
        elif N > 1:
            srt = np.argsort(x.values, axis=1)[:, -N:]
            edges += [(i, c) for k, i in enumerate(x.index.values) for c in x.columns[srt[k]]]

    if reverse:
        if N == 1:
            edges += [(i, c) for c, i in zip(x.columns.values,
                                             x.index.values[np.argmin(x.values, axis=0)])]
        elif N < 1:
            mn = x.min(axis=0).values
            is_below = x.values <= (mn + mn * N).reshape(1, -1)
            edges += [(i, c) for c, i in zip(x.columns[np.where(is_below)[0]],
                                             x.index[np.where(is_below)[0]])]
        elif N > 1:
            srt = np.argsort(x.values, axis=0).T[:, -N:]
            edges += [(i, c) for k, c in enumerate(x.columns.values) for i in x.index[srt[k]]]

    return edges


def linkage_to_graph(Z, labels=None):
    """Turn linkage into a directed graph.

    Each node in the corresponds to either an original observation or a hinge
    in the dendrogram. Edges point from the root node toward the leafs.

    Parameters
    ----------
    Z :         linkage
    labels :    iterable, optional
                A label for each of the original observations in Z.

    Returns
    -------
    nx.DiGraph

    """
    # The number of original observations
    n = len(Z) + 1

    edges = []
    for i, row in enumerate(Z):
        edges.append((int(n + i), int(row[0]), row[2]))
        edges.append((int(n + i), int(row[1]), row[2]))

    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)

    nx.set_node_attributes(G, {i: i < n for i in G.nodes},
                           name='is_original')

    if labels is not None:
        if len(labels) != n:
            raise ValueError(f'Expected {n} labels, got {len(labels)}')
        nx.set_node_attributes(G, dict(zip(np.arange(n), labels)), name='label')

    return G
