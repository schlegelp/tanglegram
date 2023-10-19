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

from scipy.cluster.hierarchy import leaves_list

LETTERS = 'abcdefghijklmnopqrstuvwxyz'


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



def _is_good_cluster(v):
    if isinstance(v, dict):
        v = list(v.values())
    if len(v) < 3:
        return False
    mn = min(v)
    mx = max(v)
    if ((mx - mn) > 3) and ((mx / mn) >= 2):
        return False
    return True


# TODO: allow walking up instead
def cut_tree_dyn(Z, eval_func):
    """Find clusters that satisfy given criteria.

    We traverse the dendrogram top to bottom and at each hinge ask whether
    clusters that would result from a further split still satisfy the given
    criteria.

    Parameters
    ----------
    Z :         np.ndarray
                Linkage for dendrogram.
    eval_func : callable
                Function that evaluates whether a proposed cluster is good.
                Must accept a list of leafs (indices) as input and return either
                True (cluster good) or False (cluster no good). Think of this
                as a way to say which clusters are "forbidden".

    Returns
    -------
    clusters :  np.ndarray
                Cluster IDs.

    Examples
    --------
    Find splits such such that each clusters contains at least 10 leafs

    >>> import tanglegram as tg
    >>> cl = tg.utils.cut_tree_dyn(Z, eval_func=lambda x: len(x) > 10)

    """
    assert callable(eval_func), '`eval_func` must be a function'

    G = linkage_to_graph(Z)

    # First check if the top hinge satisfies the function
    if not eval_func(G):
        raise ValueError('`eval_func` must return True for the entire dendrogram.')

    # Walk from the root node down to the hinges
    root = [n for n in G.nodes if G.in_degree[n] == 0][0]
    clusters = {}
    i = 0
    for node in G.successors(root):
        clusters.update(_walk_down_and_eval(G,
                                            node,
                                            eval_func,
                                            cl_dict={},
                                            cl_index=i))
        i = max(clusters.values()) + 1

    return np.array([clusters[i] for i in range(len(leaves_list(Z)))])


def _walk_down_and_eval(G, node, eval_func, cl_dict, cl_index=0):
    """Recursively walk down the dendrogram."""
    keep_going = True
    is_leaf = nx.get_node_attributes(G, 'is_original')
    for n in G.successors(node):
        # Find all clusters in this subgraph
        SG = nx.dfs_tree(G, n)
        # Get leafs in the subgraph
        SG_leafs = [n for n in SG.nodes if is_leaf.get(n, False)]

        if not eval_func(SG_leafs):
            keep_going = False
            break

    if not keep_going:
        SG = nx.dfs_tree(G, node)
        G_leafs = [n for n in SG.nodes if is_leaf.get(n, False)]
        cl_dict.update({n: cl_index for n in G_leafs})
    else:
        new_index = cl_index
        for n in G.successors(node):
            _walk_down_and_eval(G, n, eval_func, cl_dict, cl_index=new_index)
            new_index = max(cl_dict.values()) + 1

    return cl_dict


def hierarchical_cluster_labels(Z, cl, prefix=''):
    """Generate hierarchical cluster labels.

    Parameters
    ----------
    Z :         linkage
    cl :        np.ndarray
                Flat clusters e.g. from `cut_tree` or `fcluster`.
    prefix :    str
                Prefix used for all labels.

    Returns
    -------
    labels :    np.ndarray
                An array with the new hierarchical labels. Order is the same as
                in the original distance vector, i.e. you can pass them straight
                to dendrogram as `labels`.

    """
    assert len(cl) == len(leaves_list(Z))

    G = linkage_to_graph(Z, labels=cl)

    # Walk from the root node down to the hinges
    root = [n for n in G.nodes if G.in_degree[n] == 0][0]

    labels = {}
    for i, node in enumerate(G.successors(root)):
        labels.update(_walk_down_and_label(G,
                                           node,
                                           labels_dict={},
                                           label=f'{prefix}{i+1}'))

    return np.array([labels[i] for i in range(len(leaves_list(Z)))])


def _walk_down_and_label(G, node, labels_dict, label=''):
    """Recursively walk down the dendrogram."""
    # Find all clusters in this subgraph
    SG = nx.dfs_tree(G, node)
    cdict = nx.get_node_attributes(G, 'label')
    SG_leafs = [n for n in SG.nodes if n in cdict]
    SG_cl = [cdict[n] for n in SG_leafs]

    # If this subgraph is homogenous we can stop
    if len(set(SG_cl)) == 1:
        labels_dict.update({n: label for n in SG_leafs})
    else:
        for i, n in enumerate(G.successors(node)):
            # Figure out whether the next label should be a number or letter
            if not label or not _is_number(label[-1]):
                new_label = f'{label}{i+1}'
            else:
                new_label = f'{label}{LETTERS[i]}'

            _walk_down_and_label(G, n, labels_dict, label=new_label)

    return labels_dict


def _is_number(x):
    """Check if string is number."""
    try:
        int(x)
        return True
    except ValueError:
        return False


def linkage_to_graph(Z, labels=None):
    """Turn linkage into a directed graph.

    Parameters
    ----------
    Z :         linkage
    labels :    iterable, optional
                A label for each of the original observations in Z.

    Returns
    -------
    nx.DiGraph
                A graph representing the dendrogram. Each node corresponds to
                either a leaf or a hinge in the dendrogram. Edges are directed
                and point from the root node (i.e. the top hinge) toward the
                leafs. Nodes representing clusters (i.e. non-leafs) have a
                "distance" property indicating the distance between the two
                downstream clusters/leafs.

    """
    # The number of original observations
    n = len(Z) + 1

    edges = []
    cl_dists = {}
    for i, row in enumerate(Z):
        edges.append((int(n + i), int(row[0])))
        edges.append((int(n + i), int(row[1])))
        cl_dists[int(n + i)] = row[2]

    G = nx.DiGraph()
    G.add_edges_from(edges)

    nx.set_node_attributes(G, {i: i < n for i in G.nodes},
                           name='is_original')
    nx.set_node_attributes(G, cl_dists, name='distance')

    if labels is not None:
        if len(labels) != n:
            raise ValueError(f'Expected {n} labels, got {len(labels)}')
        nx.set_node_attributes(G, dict(zip(np.arange(n), labels)), name='label')

    return G
