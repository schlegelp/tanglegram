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

import inspect
import math
import logging

from dataclasses import dataclass

import matplotlib.pyplot as plt
import scipy.cluster as sclust
import scipy.spatial.distance as sdist
import numpy as np
import pandas as pd

from matplotlib.patches import ConnectionPatch

from .dend import dendrogram, cluster_color_map, _to_plotly_color


__all__ = [
    "tanglegram",
    "tanglegram_many",
    "entanglement",
    "crossings",
    "alignment_quality",
    "untangle",
]

# Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
if not module_logger.handlers:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(levelname)-5s : %(message)s (%(name)s)")
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)

# TODOs:
#


def tanglegram(
    a,
    b,
    labelsA=None,
    labelsB=None,
    edges=None,
    sort=True,
    figsize=(8, 8),
    color_by_diff=False,
    link_kwargs={},
    dend_kwargs={},
    sort_kwargs={},
    edge_kwargs={},
    clusters_left=None,
    clusters_right=None,
    cluster_colors=None,
    color_edges_by=None,
    backend="matplotlib",
):
    """Plot a tanglegram from two dendrograms.

    Parameters
    ----------
    (a,b) :                 pandas.DataFrame | scipy.cluster.hierarchy.linkage
                            Dendrograms to be compared. If DataFrame, will be
                            considered a distance matrix and linkage is
                            generated (see ``link_kwargs``).
    (labelsA,labelsB) :     list of str, optional
                            If not provided and a/b are pandas Dataframes,
                            will try to extract from columns.
    edges :                 list, optional
                            Edges to plot and untangle. If not provided will
                            assume that edges are supposed to be drawn between
                            leafs with the same label. Format must be either:
                              - `[(l1, l2), ...]`, where l1 and l2 are labels in
                                `a` and `b`, respectively
                              - `[(l1, l2, c), ...]`, where `c` is the color of
                                that edge
    sort :                  bool | str
                            If True, will try rearranging the dendrograms to
                            minimize crossing of edges. You can provide the
                            exact method to use as a string; ``True`` means
                            "dp2side". See ``untangle()`` for the full list.
    figsize :               tuple
                            Size of the figure.
    link_kwargs :           dict, optional
                            Keyword arguments to be passed to
                            ``scipy.cluster.hierarchy.linkage``.
    dend_kwargs :           dict, optional
                            Keyword arguments to be passed to
                            ``scipy.cluster.hierarchy.dendrogram``.
    sort_kwargs :           dict, optional
                            Keyword arguments to be passed to
                            ``tanglegram.untangle``.
    edge_kwargs :           dict, optional
                            Keyword arguments to be passed to
                            ``ax.plot`` when generating the edges.
    clusters_left/_right:   iterable, optional
                            Cluster labels for the left and right dendrogram,
                            respectively. Both sides share a single set of
                            colors, so a cluster that appears in both is drawn
                            the same color in both.
    cluster_colors :        dict | list-like, optional
                            Colors for those clusters. Either a
                            ``{cluster: color}`` mapping or a sequence cycled in
                            order of first appearance (left side first).
    color_edges_by :        "left" | "right", optional
                            Color the connecting edges by the cluster of the
                            leaf they start from on that side. Edges whose leaf
                            has no cluster keep the default color. Ignored if
                            the corresponding ``clusters_*`` is not given.
    backend :               "matplotlib" | "plotly"
                            Which backend to use for plotting. The plotly
                            backend returns an interactive figure; hovering an
                            edge shows the pair of labels it connects.

    Returns
    -------
    fig :                   matplotlib.Figure | plotly.graph_objects.Figure
                            The figure. For matplotlib, use the .get_axes()
                            method to get the individual axes.
    dn1 :                   scipy Dendrogram
                            The left dendrogram.
    dn2 :                   scipy Dendrogram
                            The right dendrogram.

    """
    if backend not in ("matplotlib", "plotly"):
        raise ValueError(
            f'Unknown backend: "{backend}". Expected "matplotlib" or "plotly".'
        )
    # plt.style.use('ggplot')

    if isinstance(a, pd.DataFrame):
        module_logger.info("Generating linkage from distances")
        link1 = sclust.hierarchy.linkage(
            sdist.squareform(a, checks=False), **link_kwargs
        )
        if labelsA is None:
            labelsA = a.columns.tolist()
    elif isinstance(a, np.ndarray):
        link1 = a
    else:
        raise TypeError(
            "Parameter `a` needs to be either pandas DataFrame of distances "
            "or numpy array representing a linkage"
        )

    if isinstance(b, pd.DataFrame):
        module_logger.info("Generating linkage from distances")
        link2 = sclust.hierarchy.linkage(
            sdist.squareform(b, checks=False), **link_kwargs
        )
        if labelsB is None:
            labelsB = b.columns.tolist()
    elif isinstance(b, np.ndarray):
        link2 = b
    else:
        raise TypeError(
            "Parameter `b` needs to be either pandas DataFrame of distances "
            "or numpy array representing a linkage"
        )

    if len(labelsA) != len(set(labelsA)):
        module_logger.warning("Some labels in A appear to not be unique")

    if len(labelsB) != len(set(labelsB)):
        module_logger.warning("Some labels in B appear to not be unique")

    if edges is None:
        edges = [(l, l) for l in labelsA if l in labelsB]
        edges += [(l, l) for l in labelsB if l in labelsA]
        edges = list(set(edges))
    else:
        for e in edges:
            if e[0] not in labelsA:
                raise ValueError(f"Label {e[0]} does not exist in dendrogam A")
            if e[1] not in labelsB:
                raise ValueError(f"Label {e[1]} does not exist in dendrogam B")

    if not len(edges):
        raise ValueError("No matching labels between the two dendrogams")

    if sort:
        if not isinstance(sort, str):
            sort = "dp2side"
        link1, link2 = untangle(
            link1,
            link2,
            np.asarray(labelsA),
            np.asarray(labelsB),
            edges=edges,
            method=sort,
            **sort_kwargs,
        )

    # Both dendrograms draw from one palette, so a cluster present on both
    # sides gets the same color in both.
    if clusters_left is not None or clusters_right is not None:
        cluster_colors = cluster_color_map(
            clusters_left, clusters_right, colors=cluster_colors
        )

    edge_by_cluster = _edge_cluster_colors(
        edges,
        color_edges_by,
        labelsA=labelsA,
        labelsB=labelsB,
        clusters_left=clusters_left,
        clusters_right=clusters_right,
        cluster_colors=cluster_colors,
    )

    if backend == "plotly":
        return _tanglegram_plotly(
            link1,
            link2,
            labelsA=labelsA,
            labelsB=labelsB,
            edges=edges,
            figsize=figsize,
            color_by_diff=color_by_diff,
            dend_kwargs=dend_kwargs,
            edge_kwargs=edge_kwargs,
            clusters_left=clusters_left,
            clusters_right=clusters_right,
            cluster_colors=cluster_colors,
            edge_by_cluster=edge_by_cluster,
        )

    fig = plt.figure(figsize=figsize)

    # Compute and plot left dendrogram.
    no_labels = dend_kwargs.get("no_labels", False)
    if not no_labels:
        ax1 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    else:
        ax1 = fig.add_axes([0.05, 0.1, 0.35, 0.8])
    dn1, _ = dendrogram(
        link1,
        orientation="left",
        labels=labelsA,
        clusters=clusters_left,
        cluster_colors=cluster_colors,
        ax=ax1,
        **dend_kwargs,
    )

    # Compute and plot right dendrogram.
    if not no_labels:
        ax2 = fig.add_axes([0.7, 0.1, 0.25, 0.8])  # [0.3, 0.71, 0.6, 0.2])
    else:
        ax2 = fig.add_axes([0.645, 0.1, 0.35, 0.8])
    dn2, _ = dendrogram(
        link2,
        labels=labelsB,
        orientation="right",
        clusters=clusters_right,
        cluster_colors=cluster_colors,
        ax=ax2,
        **dend_kwargs,
    )

    # Generate middle plot with connecting lines
    if not no_labels:
        ax3 = fig.add_axes([0.4, 0.1, 0.2, 0.8])
    else:
        ax3 = fig.add_axes([0.405, 0.1, 0.235, 0.8])
    ax3.axis("off")
    ax3.set_xlim((0, 1))

    # Get min and max y dimensions
    max_y = max(ax1.viewLim.y1, ax2.viewLim.y1)
    min_y = min(ax1.viewLim.y0, ax2.viewLim.y0)

    # Make sure labels of both dendrograms have the same font size
    ax1.tick_params(axis="both", labelsize=8)
    ax2.tick_params(axis="both", labelsize=8)

    # Make sure all y axes have same resolution
    for _ in [ax3]:  # [ax1,ax2,ax3]:
        _.set_ylim((min_y, max_y))

    edge_kwargs["lw"] = edge_kwargs.pop("linewidth", edge_kwargs.get("lw", 1))
    edge_kwargs["ls"] = edge_kwargs.pop("linestyle", edge_kwargs.get("ls", "-"))
    # Keep the default out of `edge_kwargs`: it is the fallback for every edge,
    # so it must not be overwritten by whichever edge was drawn last.
    default_color = edge_kwargs.pop("c", edge_kwargs.pop("color", "k"))

    for i, e in enumerate(edges):
        ix_l = dn1["ivl"].index(e[0])
        ix_r = dn2["ivl"].index(e[1])

        coords_l = (
            (ax3.viewLim.y1 - ax3.viewLim.y0) / (len(dn1["leaves"])) * (ix_l + 0.5)
        )
        coords_r = (
            (ax3.viewLim.y1 - ax3.viewLim.y0) / (len(dn2["leaves"])) * (ix_r + 0.5)
        )

        color = _pick_edge_color(
            i,
            e,
            ix_l,
            ix_r,
            n_left=len(dn1["ivl"]),
            color_by_diff=color_by_diff,
            edge_by_cluster=edge_by_cluster,
            default_color=default_color,
        )

        ax3.plot([0, 1], [coords_l, coords_r], c=color, **edge_kwargs)

    module_logger.info("Done. Use matplotlib.pyplot.show() to show plot.")

    return fig, dn1, dn2


def _pick_edge_color(
    i, e, ix_l, ix_r, n_left, color_by_diff, edge_by_cluster, default_color
):
    """Color for one connecting edge.

    An explicitly given per-edge color beats the cluster coloring, and
    `color_by_diff` overrides both - it is a diagnostic view of how far each
    leaf moved, so it would be pointless to let anything else show through.
    """
    if color_by_diff:
        v = max(round(0.75 - math.fabs(ix_l - ix_r) / n_left, 2), 0)
        return (v, v, v)
    if len(e) == 3:
        return e[2]
    if edge_by_cluster is not None and edge_by_cluster[i] is not None:
        return edge_by_cluster[i]
    return default_color


def _tanglegram_plotly(
    link1,
    link2,
    labelsA,
    labelsB,
    edges,
    figsize,
    color_by_diff,
    dend_kwargs,
    edge_kwargs,
    clusters_left,
    clusters_right,
    cluster_colors,
    edge_by_cluster,
):
    """Draw a tanglegram as an interactive plotly figure.

    Three subplots side by side: the two dendrograms facing each other and a
    blank one in the middle carrying the connecting edges.
    """
    try:
        import plotly.graph_objects as go

        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "Please install the plotly library to use this backend:"
            "\n  pip3 install plotly"
        )

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.35, 0.3, 0.35],
        horizontal_spacing=0.02,
    )

    dn1, _ = dendrogram(
        link1,
        orientation="left",
        labels=labelsA,
        clusters=clusters_left,
        cluster_colors=cluster_colors,
        backend="plotly",
        fig={"fig": fig, "row": 1, "col": 1},
        **dend_kwargs,
    )
    dn2, _ = dendrogram(
        link2,
        orientation="right",
        labels=labelsB,
        clusters=clusters_right,
        cluster_colors=cluster_colors,
        backend="plotly",
        fig={"fig": fig, "row": 1, "col": 3},
        **dend_kwargs,
    )

    default_color = edge_kwargs.get("color", edge_kwargs.get("c", "black"))
    width = edge_kwargs.get("linewidth", edge_kwargs.get("lw", 1))

    # The two dendrograms may have different leaf counts, so the middle panel
    # works in fractions of its own height rather than in leaf slots.
    n1, n2 = len(dn1["leaves"]), len(dn2["leaves"])
    for i, e in enumerate(edges):
        ix_l = dn1["ivl"].index(e[0])
        ix_r = dn2["ivl"].index(e[1])

        color = _pick_edge_color(
            i,
            e,
            ix_l,
            ix_r,
            n_left=len(dn1["ivl"]),
            color_by_diff=color_by_diff,
            edge_by_cluster=edge_by_cluster,
            default_color=default_color,
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[(ix_l + 0.5) / n1, (ix_r + 0.5) / n2],
                mode="lines",
                line=dict(color=_to_plotly_color(color), width=width),
                text=f"{e[0]} - {e[1]}",
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(
        row=1,
        col=2,
        range=[0, 1],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
    )
    fig.update_yaxes(
        row=1,
        col=2,
        range=[0, 1],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
    )
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        width=figsize[0] * 100,
        height=figsize[1] * 100,
    )

    return fig, dn1, dn2


def _edge_cluster_colors(
    edges,
    color_edges_by,
    labelsA,
    labelsB,
    clusters_left,
    clusters_right,
    cluster_colors,
):
    """Color for each edge, taken from the cluster of the leaf it starts at.

    Returns None if edges should not be colored by cluster, otherwise a list
    with one color per edge (None where the leaf has no cluster).
    """
    if color_edges_by is None:
        return None

    if color_edges_by not in ("left", "right"):
        raise ValueError(
            f'`color_edges_by` must be "left" or "right", got "{color_edges_by}"'
        )

    if color_edges_by == "left":
        clusters, labels, end = clusters_left, labelsA, 0
    else:
        clusters, labels, end = clusters_right, labelsB, 1

    if clusters is None:
        module_logger.warning(
            f'`color_edges_by="{color_edges_by}"` needs `clusters_{color_edges_by}`'
            " - edges will keep their default color"
        )
        return None

    clusters = np.asarray(clusters).flatten()
    if len(clusters) != len(labels):
        raise ValueError(
            f"Got {len(clusters)} clusters for {len(labels)} labels on the "
            f"{color_edges_by} side."
        )

    # Unassigned leafs simply never make it into the color map
    label_color = {}
    for label, cl in zip(labels, clusters):
        key = cl.item() if isinstance(cl, np.generic) else cl
        if key in cluster_colors:
            label_color[label] = cluster_colors[key]

    return [label_color.get(e[end]) for e in edges]


def tanglegram_many(
    x,
    labels=None,
    edges=None,
    sort=True,
    figsize=(8, 8),
    link_kwargs={},
    dend_kwargs={},
    sort_kwargs={},
    clusters=None,
    cluster_colors=None,
):
    """Plot a tanglegram from two or more dendrograms.

    Parameters
    ----------
    x :                     iterable of pandas.DataFrame | scipy.cluster.hierarchy.linkage
                            Dendrograms to be compared. DataFrames, will be
                            considered a distance matrix and a linkage is
                            generated (see ``link_kwargs``).
    labels :                list of lists, optional
                            If not provided and ``x`` are pandas Dataframes,
                            will try to extract from columns.
    edges :                 list of list, optional
                            Edges to plot and untangle. Format must be
                            ``[edges_ab, edges_bc, .. ]`` where `edges_ab`,
                            `edges_bc`, etc. are the edges between the first and
                            second, the second the third dendrogram and so forth.
                            Each ``edges_`` must have follow these formats:
                              - `[(l1, l2), ...]`, where l1 and l2 are labels in
                                the first and the second dendrogram, respectively
                              - `[(l1, l2, c), ...]`, where `c` is the color of
                                that edge
                            If not provided will assume that edges are supposed
                            to be drawn between leafs with the same label.
    sort :                  bool | str
                            If True, will try rearranging the dendrograms to
                            minimize crossing of edges. You can provide the
                            exact method to use as a string (see ``untangle()``);
                            ``True`` means "dp2side". Every pair after the first
                            has its left dendrogram already pinned, so two-sided
                            methods drop to their one-sided variant there
                            ("dp2side" -> "dp1side", and so on).
    figsize :               tuple
                            Size of the figure.
    link_kwargs :           dict, optional
                            Keyword arguments to be passed to
                            ``scipy.cluster.hierarchy.linkage``.
    dend_kwargs :           dict, optional
                            Keyword arguments to be passed to
                            ``scipy.cluster.hierarchy.dendrogram``.
    sort_kwargs :           dict, optional
                            Keyword arguments to be passed to
                            ``tanglegram.untangle``.
    clusters :              list of list-like, optional
                            One cluster assignment per dendrogram. All of them
                            share a single set of colors, so a cluster that
                            turns up in several dendrograms is drawn the same
                            color in each.
    cluster_colors :        dict | list-like, optional
                            Colors for those clusters. Either a
                            ``{cluster: color}`` mapping or a sequence cycled in
                            order of first appearance.

    Returns
    -------
    matplotlib figure

    """
    if not isinstance(x, (list, tuple)):
        raise TypeError(f'Expected list or tuple for `x`, got "{type(x)}"')
    elif len(x) < 2:
        raise ValueError(f"Expected >= 2 dendrograms in `x`, got {len(x)}")

    if labels is not None:
        if len(labels) != len(x):
            raise ValueError("Length of labels does not match length of `x`")
        labels = list(labels)
    else:
        labels = [None] * len(x)

    linkages = []
    for i, a in enumerate(x):
        if isinstance(a, pd.DataFrame):
            module_logger.info("Generating linkage from distances")
            linkages.append(
                sclust.hierarchy.linkage(
                    sdist.squareform(a, checks=False), **link_kwargs
                )
            )
            if labels[i] is None:
                labels[i] = a.columns.tolist()
        elif isinstance(a, np.ndarray):
            linkages.append(a)
        else:
            raise TypeError(
                "Parameter `x` needs to be either pandas DataFrames or linkages (numpy arrays)"
            )

    if edges is None:
        edges = [
            [(l, l) for l in lA if l in lB] for lB, lA in zip(labels[:-1], labels[1:])
        ]
    elif isinstance(edges, (list, tuple)):
        edges = list(edges)

        if len(edges) != (len(x) - 1):
            raise ValueError(
                f"Expected {len(x) - 1} sets of edges for {len(x)} "
                f"dendrograms, got {len(edges)}"
            )

        for i, edges_ in enumerate(edges):
            for e in edges_:
                if e[0] not in labels[i]:
                    raise ValueError(
                        f"Label {e[0]} does not exist in dendrogam {i + 1}"
                    )
                if e[1] not in labels[i + 1]:
                    raise ValueError(
                        f"Label {e[1]} does not exist in dendrogam {i + 2}"
                    )
    else:
        raise TypeError(
            "Expected `edges` to be either None or list of lists, "
            f'got "{type(edges)}"'
        )

    if not len(edges):
        raise ValueError("No matching labels between the two dendrogams")

    if sort:
        for i, (l1, l2) in enumerate(zip(linkages[:-1], linkages[1:])):
            if sort == "random":
                sort_kwargs["rotate"] = "link2" if i == 0 else "link1"
                method = "random"
            else:
                sort_kwargs.pop("rotate", None)
                method = sort if isinstance(sort, str) else "dp2side"
                # Every pair after the first has its left dendrogram already
                # pinned by the preceding pair, so only the right one can move
                if i > 0:
                    method = ONE_SIDED_EQUIVALENT.get(method, method)

            # The one-sided methods take fewer options than their two-sided
            # counterparts, so drop anything the target doesn't accept
            this_kwargs = _filter_kwargs(method, sort_kwargs)

            if i == 0:
                l1, l2 = untangle(
                    l1,
                    l2,
                    np.asarray(labels[i]),
                    np.asarray(labels[i + 1]),
                    edges=edges[i],
                    method=method,
                    **this_kwargs,
                )
            else:
                l2, _ = untangle(
                    l2,
                    l1,
                    np.asarray(labels[i + 1]),
                    np.asarray(labels[i]),
                    edges=[
                        (e[1], e[0]) for e in edges[i]
                    ],  # need to reverse edges here
                    method=method,
                    **this_kwargs,
                )
            linkages[i] = l1
            linkages[i + 1] = l2

    if clusters is not None:
        if len(clusters) != len(x):
            raise ValueError(
                f"Expected {len(x)} cluster assignments for {len(x)} "
                f"dendrograms, got {len(clusters)}"
            )
        # One palette across all of them, so a cluster keeps its color
        cluster_colors = cluster_color_map(*clusters, colors=cluster_colors)

    fig, axes = plt.subplots(ncols=1, nrows=len(x), figsize=figsize)

    ix_maps = []
    for i, Z in enumerate(linkages):
        # Generate a mapping from label -> index in dendrogram
        m = dict(
            zip(
                np.asarray(labels[i])[sclust.hierarchy.leaves_list(Z)],
                np.arange(len(labels[i])),
            )
        )
        ix_maps.append(m)

        DEFAULTS = dict(above_threshold_color="slategrey")
        DEFAULTS.update(dend_kwargs)
        _ = dendrogram(
            Z,
            labels=labels[i],
            ax=axes[i],
            clusters=None if clusters is None else clusters[i],
            cluster_colors=cluster_colors,
            **DEFAULTS,
        )

    for i, edges_ in enumerate(edges):
        ax1 = axes[i]
        ax2 = axes[i + 1]
        for e in edges_:
            ix1 = ix_maps[i][e[0]]
            ix2 = ix_maps[i + 1][e[1]]

            x1 = (ix1 + 0.5) * 10
            x2 = (ix2 + 0.5) * 10

            if len(e) == 3:
                c = e[2]
            else:
                c = (0, 0, 0, 0.5)

            con = ConnectionPatch(
                xyA=(x1, 0),
                coordsA="data",
                axesA=ax1,
                xyB=(x2, 0),
                coordsB="data",
                axesB=ax2,
                color=c,
                lw=1,
                ls="--",
            )
            fig.add_artist(con)

    return fig


def entanglement(link1, link2, labels1, labels2, edges=None, L=1.5, **kwargs):
    """Calculage average displacement of leafs in dendogram 1 and 2.

    Entanglement is a measure between 1 (full entanglement) and 0 (no
    entanglement).

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to untangle.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples
                        If provided, test entanglement on these edges only.
                        Else will measure using labels that exist in both
                        dendrograms.
    L :                 any positive number
                        Distance norm to use for measuring the distance
                        between the two trees. Can be any positive number,
                        often one will want to use 0, 1, 1.5 or 2:
                        ``sum(abs(x-y)^L)``.

    Returns
    -------

    """
    # Get label indices
    lindex1 = leaf_order(link1, labels1, as_dict=True)
    lindex2 = leaf_order(link2, labels2, as_dict=True)

    # Get starting entanglement
    return _entanglement(lindex1, lindex2, edges, L=L)


def _entanglement(lindex1, lindex2, edges=None, L=1.5):
    """Low-level version of entanglement calculation. For internal use only.

    Parameters
    ----------
    lindex1,lindex2 :       dict
                            Dictionaries mapping the labels of two dendrograms
                            to their indices.
    edges :                 list of tuples, optional
                            If provided, test entanglement on these edges only.
                            Else will measure using labels that exist in both
                            dendrograms.
    L :                     any positive number
                            Distance norm to use for measuring the distance
                            between the two trees. Can be any positive number,
                            often one will want to use 0, 1, 1.5 or 2:
                            ``sum(abs(x-y)^L)``.

    """
    assert isinstance(lindex1, dict)
    assert isinstance(lindex2, dict)

    if edges is None:
        exist_in_both = list(set(lindex1) & set(lindex2))

        if not exist_in_both:
            raise ValueError("Not a single matching label in both dendrograms.")

        edges = [(l, l) for l in exist_in_both]

    # Absolute distance
    dist = np.array([lindex1[e[0]] - lindex2[e[1]] for e in edges])
    dist = np.abs(dist)
    # Absolute entanglement
    ent = np.sum(dist**L)

    # Worst case
    ix = np.arange(max(len(lindex1), len(lindex2)))
    worst = np.sum(np.abs(ix - ix[::-1]) ** L)

    # Normalized entanglement
    return ent / worst


def untangle(link1, link2, labels1, labels2, edges, method="dp2side", L=1.5, **kwargs):
    """Untangle two dendrograms using various methods.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to untangle.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples
                        If provided, test entanglement on these edges only.
                        Else will measure using labels that exist in both
                        dendrograms.
    method :            "dp2side" | "dp1side" | "cross2side" | "cross1side" |
                        "random" | "step1side" | "step2side"
                        Method to use for untangling. The first four each solve
                        their one-sided problem *exactly* - i.e. they find the
                        provably best of all ``2**(n-1)`` rotations rather than
                        searching greedily:
                          - "dp1side" minimises leaf displacement
                            (``entanglement()``) by rotating ``link1`` only.
                            ``O(n^2)`` time and memory.
                          - "dp2side" (default) alternates exact ``dp1side``
                            solves on both dendrograms.
                          - "cross1side" minimises edge *crossings*
                            (``crossings()``) by rotating ``link1`` only. Needs
                            only ``O(n)`` memory, so this is the one to use on
                            very large dendrograms.
                          - "cross2side" alternates exact ``cross1side`` solves.
                          - "random", "step1side", "step2side" are the older
                            greedy heuristics, kept as baselines. They are
                            slower *and* no better.

                        Both two-sided methods accept ``restarts`` (and
                        ``seed``), which re-runs the alternation from randomly
                        rotated copies and keeps the best result. The
                        alternation on its own can settle in a local optimum;
                        with a few dozen restarts it reliably reaches the true
                        global optimum.

                        Displacement and crossings give near-identical layouts
                        in practice, so pick "cross2side" when ``n`` is large
                        enough that an ``n x n`` table is a problem, and
                        "dp2side" otherwise.
    L :                 any positive number
                        Distance norm to use for measuring the distance
                        between the two trees. Can be any positive number,
                        often one will want to use 0, 1, 1.5 or 2:
                        ``sum(abs(x-y)^L)``.
    **kwargs
                        Passed to the respective untangling functions.

    """
    if method not in UNTANGLE_METHODS:
        raise ValueError(f'Unknown method "{method}"')

    return UNTANGLE_METHODS[method](
        link1, link2, labels1, labels2, edges, L=L, **kwargs
    )


def _filter_kwargs(method, kwargs):
    """Drop keyword arguments the given untangling method does not accept."""
    func = UNTANGLE_METHODS.get(method)
    if func is None:
        return dict(kwargs)

    params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    return {k: v for k, v in kwargs.items() if k in params}


def untangle_step_rotate_2side(
    link1, link2, labels1, labels2, edges, direction="down", L=1.5, max_n_iterations=10
):
    """Untangle by stepwise rotating around all hinges in both dendrograms.

    This is a greedy forward algorithm that rotates the first dendogram, then
    the second, then the first again and so on until a locally optimal solution
    is found. The break condition is either ``max_n_iterations`` reached or
    no improved entanglement in two consecutive iterations.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to (better) align by rotating.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    direction :         "down" | "up"
                        Whether to start at the top and move down (default) or
                        start at the leafs and move up.
    L :                 float
                        Distance norm used to calculate the entanglement.
                        Passed to ``_entanglement()``.
    max_n_iterations :  int
                        Max iterations (default = 10) to run.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    assert direction in ("down", "up")

    min_entang = float("inf")

    ol = module_logger.level
    module_logger.setLevel("WARNING")

    edges_inv = [(e[1], e[0]) for e in edges]
    for i in range(int(max_n_iterations)):
        # Rotate the first dendrogram
        link1, link2 = untangle_step_rotate_1side(
            link1, link2, labels1, labels2, edges, L=L, direction=direction
        )

        # Now rotate the second dendrogram
        link2, link1 = untangle_step_rotate_1side(
            link2, link1, labels2, labels1, edges_inv, L=L, direction=direction
        )

        # Get the new entanglement
        lindex1 = leaf_order(link1, labels1, as_dict=True)
        lindex2 = leaf_order(link2, labels2, as_dict=True)

        # Get new entanglement
        new_entang = _entanglement(lindex1, lindex2, edges, L=L)

        # Stop if there is no improvement from the last iteration
        if new_entang == min_entang:
            break
        else:
            min_entang = new_entang

        if min_entang == 0:
            break

    module_logger.setLevel(ol)
    module_logger.info(f"Finished optimising at entanglement {min_entang:.3f}")
    return link1, link2


def untangle_step_rotate_1side(
    link1, link2, labels1, labels2, edges, direction="down", L=1.5
):
    """Untangle by stepwise rotating around all hinges in one dendrogram.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to better align by rotating the first dendrogram.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    direction :         "down" | "up"
                        Whether to start at the top and move down (default) or
                        start at the leafs and move up.
    L :                 float
                        Distance norm used to calculate the entanglement.
                        Passed to ``_entanglement()``.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    assert direction in ("down", "up")

    # Get label indices
    lindex1 = leaf_order(link1, labels1, as_dict=True)
    lindex2 = leaf_order(link2, labels2, as_dict=True)

    # Get starting entanglement
    min_entang = _entanglement(lindex1, lindex2, edges, L=L)

    n_hinges = len(link1)
    for i in range(n_hinges):
        if direction == "down":
            i = n_hinges - 1 - i

        # Shuffle dendrograms
        r_link1 = rotate(link1, i, copy=True)

        # Get label indices
        r_lindex1 = leaf_order(r_link1, labels1, as_dict=True)

        # Get new entanglement
        new_entang = _entanglement(r_lindex1, lindex2, edges, L=L)

        # Check if new entanglment is better
        if new_entang < min_entang:
            min_entang = new_entang
            link1 = r_link1

        if min_entang == 0:
            break

    module_logger.info(f"Finished optimising at entanglement {min_entang:.3f}")
    return link1, link2


def untangle_random_search(
    link1, link2, labels1, labels2, edges, rotate="both", R=100, L=1.5
):
    """Untangle dendrogram using a simple random search.

    Shuffle trees and see if entanglement got better.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to (better) align by shuffling.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples
    rotate :            "both" | "link1" | link2
    R :                 int
                        Number of shuffles to perform.
    L :                 float
                        Distance norm used to calculate the entanglement.
                        Passed to ``_entanglement()``.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    assert rotate in ("both", "link1", "link2")

    # Get label indices
    lindex1 = leaf_order(link1, labels1, as_dict=True)
    lindex2 = leaf_order(link2, labels2, as_dict=True)

    # Get starting entanglement
    min_entang = _entanglement(lindex1, lindex2, edges, L=L)

    for i in range(int(R)):
        # Shuffle dendrograms
        if rotate in ("both", "link1"):
            s_link1 = shuffle_dendogram(link1)
            s_lindex1 = leaf_order(s_link1, labels1, as_dict=True)
        else:
            s_link1, s_lindex1 = link1, lindex1

        if rotate in ("both", "link2"):
            s_link2 = shuffle_dendogram(link2)
            s_lindex2 = leaf_order(s_link2, labels2, as_dict=True)
        else:
            s_link2, s_lindex2 = link2, lindex2

        # Get new entanglement
        new_entang = _entanglement(s_lindex1, s_lindex2, edges, L=L)

        # Check if new entanglment is better
        if new_entang < min_entang:
            min_entang = new_entang
            link1 = s_link1
            link2 = s_link2

        if min_entang == 0:
            break

    module_logger.info(f"Finished optimising at entanglement {min_entang:.3f}")
    return link1, link2


def _subtree_sizes(link, n):
    """Number of leafs below each node of a linkage.

    Nodes ``0..n-1`` are the leafs, node ``n+k`` is the merge in row ``k``.
    """
    size = np.ones(2 * n - 1, dtype=np.int64)
    for k in range(link.shape[0]):
        size[n + k] = size[int(link[k, 0])] + size[int(link[k, 1])]
    return size


def _dp_optimal_rotations(link, leaf_cost):
    """Exactly minimise a per-leaf placement cost over all rotations of ``link``.

    Every subtree of a dendrogram occupies a *contiguous* block of leaf slots
    and a rotation only decides in which order a node's two child blocks are
    laid down. Because the entanglement objective is a sum of independent
    per-leaf terms, this gives the recursion

        ``dp[v][s]`` = cheapest way to place subtree ``v`` into slots
        ``[s, s + size(v))``
        ``dp[v][s] = min(dp[a][s] + dp[b][s + size(a)],``
        ``            dp[b][s] + dp[a][s + size(b)])``

    which is solved bottom-up over the ``2n-1`` nodes. This searches the full
    ``2**(n-1)`` rotation space exactly in ``O(n^2)`` time.

    Parameters
    ----------
    link :      scipy.cluster.hierarchy.linkage
    leaf_cost : (n, n) array
                ``leaf_cost[i, s]`` is the cost of placing observation ``i``
                in leaf slot ``s``.

    Returns
    -------
    best :      float
                The minimal achievable cost.
    rot :       (n-1, ) bool array
                Rows of ``link`` that must be rotated to achieve ``best``.

    """
    n = leaf_cost.shape[0]
    size = _subtree_sizes(link, n)

    # dp[v] has length n - size[v] + 1: the feasible start slots for node v
    dp = {i: leaf_cost[i] for i in range(n)}
    choice = {}

    for k in range(link.shape[0]):
        v = n + k
        a, b = int(link[k, 0]), int(link[k, 1])
        sa, sb = size[a], size[b]
        m = n - size[v] + 1

        # Lay down child a first, or child b first
        c_a_first = dp[a][:m] + dp[b][sa : sa + m]
        c_b_first = dp[b][:m] + dp[a][sb : sb + m]

        flip = c_b_first < c_a_first
        choice[v] = flip
        dp[v] = np.where(flip, c_b_first, c_a_first)

        # Children are only needed to build their parent; `choice` carries
        # everything the backtrack below requires.
        del dp[a], dp[b]

    root = 2 * n - 2
    best = float(dp[root][0])

    # Backtrack iteratively - recursion would overflow on degenerate (chain
    # shaped) linkages, which are exactly the deep trees this is useful for.
    rot = np.zeros(link.shape[0], dtype=bool)
    stack = [(root, 0)]
    while stack:
        v, s = stack.pop()
        if v < n:
            continue
        a, b = int(link[v - n, 0]), int(link[v - n, 1])
        flip = bool(choice[v][s])
        rot[v - n] = flip
        if flip:
            stack.append((b, s))
            stack.append((a, s + size[b]))
        else:
            stack.append((a, s))
            stack.append((b, s + size[a]))

    return best, rot


def _leaf_cost_matrix(labels1, lindex2, edges, n1, L):
    """Cost of placing each observation of tree 1 in each of its leaf slots.

    ``lindex2`` maps tree 2's labels to their (fixed) slots. A label may take
    part in several edges - or none, in which case its row stays zero.
    """
    label_to_obs = {lab: i for i, lab in enumerate(np.asarray(labels1))}
    slots = np.arange(n1, dtype=float)

    cost = np.zeros((n1, n1), dtype=float)
    for e in edges:
        cost[label_to_obs[e[0]]] += np.abs(slots - lindex2[e[1]]) ** L
    return cost


def untangle_dp_1side(link1, link2, labels1, labels2, edges, L=1.5):
    """Untangle by optimally rotating ``link1`` while ``link2`` stays fixed.

    Unlike ``untangle_step_rotate_1side`` - which greedily visits one hinge at
    a time - this finds the *provably* best of all ``2**(n-1)`` rotations of
    ``link1`` via dynamic programming, in ``O(n^2)`` time and memory.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to better align by rotating the first.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples
    L :                 float
                        Distance norm used to calculate the entanglement.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    n1 = len(link1) + 1
    lindex2 = leaf_order(link2, labels2, as_dict=True)

    cost = _leaf_cost_matrix(labels1, lindex2, edges, n1, L)
    _, rot = _dp_optimal_rotations(link1, cost)

    link1 = link1.copy()
    to_rot = np.nonzero(rot)[0]
    link1[to_rot, :2] = link1[to_rot, :2][:, ::-1]

    lindex1 = leaf_order(link1, labels1, as_dict=True)
    module_logger.info(
        "Finished optimising at entanglement "
        f"{_entanglement(lindex1, lindex2, edges, L=L):.3f}"
    )
    return link1, link2


def untangle_dp_2side(
    link1,
    link2,
    labels1,
    labels2,
    edges,
    L=1.5,
    max_n_iterations=10,
    restarts=0,
    seed=None,
):
    """Untangle by alternating *exact* one-sided optimisations of both trees.

    Each half-step is globally optimal for the tree it rotates, but the
    alternation itself can still settle in a local optimum - solving one side
    exactly can move the other side's optimum out of reach. ``restarts``
    guards against that by re-running the alternation from randomly rotated
    copies and keeping the best result; this is affordable precisely because
    each exact solve is cheap.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to (better) align by rotating.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples
    L :                 float
                        Distance norm used to calculate the entanglement.
    max_n_iterations :  int
                        Max alternations (default = 10) per start.
    restarts :          int
                        Number of additional randomised restarts (default = 0).
    seed :              int, optional
                        Seed for the restarts, for reproducibility.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    edges_inv = [(e[1], e[0]) for e in edges]

    def score(l1, l2):
        return _entanglement(
            leaf_order(l1, labels1, as_dict=True),
            leaf_order(l2, labels2, as_dict=True),
            edges,
            L=L,
        )

    def alternate(l1, l2):
        entang = score(l1, l2)
        for _ in range(int(max_n_iterations)):
            l1, l2 = untangle_dp_1side(l1, l2, labels1, labels2, edges, L=L)
            l2, l1 = untangle_dp_1side(l2, l1, labels2, labels1, edges_inv, L=L)

            new_entang = score(l1, l2)
            # Both half-steps are exact, so this can only improve or plateau
            if new_entang >= entang - 1e-12:
                entang = new_entang
                break
            entang = new_entang

            if entang == 0:
                break
        return l1, l2, entang

    ol = module_logger.level
    module_logger.setLevel("WARNING")

    best1, best2, min_entang = alternate(link1, link2)

    if restarts:
        rng = np.random.default_rng(seed)
        for _ in range(int(restarts)):
            if min_entang == 0:
                break
            cand1, cand2, entang = alternate(
                shuffle_dendogram(link1, copy=True, rng=rng),
                shuffle_dendogram(link2, copy=True, rng=rng),
            )
            if entang < min_entang:
                best1, best2, min_entang = cand1, cand2, entang

    module_logger.setLevel(ol)
    module_logger.info(f"Finished optimising at entanglement {min_entang:.3f}")
    return best1, best2


def _random_rotation(link, rng):
    """Draw uniformly from the ``2**(n-1)`` rotations of ``link``.

    Unlike :func:`shuffle_dendogram` - which rotates a random *number* of
    hinges and so is biased towards the input ordering - this flips every hinge
    independently with probability 0.5. That makes it the right thing to
    compare an alignment against, and the wrong thing to perturb a restart with.
    """
    out = link.copy()
    idx = np.nonzero(rng.random(len(link)) < 0.5)[0]
    out[idx, :2] = out[idx, :2][:, ::-1]
    return out


def _count_inversions(values):
    """Number of index pairs ``i < j`` with ``values[i] > values[j]``."""

    def rec(a):
        if a.size < 2:
            return a, 0
        mid = a.size // 2
        left, cl = rec(a[:mid])
        right, cr = rec(a[mid:])
        # every element of `left` precedes every element of `right`, so each
        # left element greater than a right element is one inversion
        cross = int(np.sum(left.size - np.searchsorted(left, right, side="right")))
        return np.sort(np.concatenate([left, right])), cl + cr + cross

    return rec(np.asarray(values, dtype=float))[1]


def _crossings(lindex1, lindex2, edges=None):
    """Low-level crossing count. For internal use only."""
    assert isinstance(lindex1, dict)
    assert isinstance(lindex2, dict)

    if edges is None:
        exist_in_both = list(set(lindex1) & set(lindex2))
        if not exist_in_both:
            raise ValueError("Not a single matching label in both dendrograms.")
        edges = [(l, l) for l in exist_in_both]

    p1 = np.array([lindex1[e[0]] for e in edges], dtype=float)
    p2 = np.array([lindex2[e[1]] for e in edges], dtype=float)

    # Sort by position in tree 1; break ties by position in tree 2 so that
    # edges leaving the same leaf are never counted as crossing each other
    return _count_inversions(p2[np.lexsort((p2, p1))])


def crossings(link1, link2, labels1, labels2, edges=None):
    """Count how many pairs of edges cross between the two dendrograms.

    This is what a reader of the plot actually sees, as opposed to
    ``entanglement()`` which measures leaf displacement.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples, optional
                        If provided, count crossings among these edges only.
                        Else will use labels that exist in both dendrograms.

    Returns
    -------
    int

    """
    return _crossings(
        leaf_order(link1, labels1, as_dict=True),
        leaf_order(link2, labels2, as_dict=True),
        edges,
    )


def _crossing_targets(labels1, lindex2, edges, n1):
    """For each observation of tree 1, the sorted tree-2 slots of its edges."""
    label_to_obs = {lab: i for i, lab in enumerate(np.asarray(labels1))}

    targets = [[] for _ in range(n1)]
    for e in edges:
        targets[label_to_obs[e[0]]].append(lindex2[e[1]])

    return [np.sort(np.asarray(t, dtype=float)) for t in targets]


def _dp_optimal_rotations_crossings(link, targets):
    """Exactly minimise edge crossings over all rotations of ``link``.

    The crossings between a node's two child blocks depend only on *which
    child is laid down first* - not on where the block sits, since every edge
    from the first block starts above every edge from the second either way.
    Unlike the displacement objective there is therefore no start-offset
    dimension, and the optimal choice at each node is independent of every
    other node:

        ``C(v) = C(a) + C(b) + min(cross_ab, cross_ba)``

    That makes this a single bottom-up sweep in ``O(n log^2 n)`` time and, more
    importantly, ``O(n)`` memory - so it scales far past the ``O(n^2)`` table
    that :func:`_dp_optimal_rotations` needs.

    Parameters
    ----------
    link :      scipy.cluster.hierarchy.linkage
    targets :   list of arrays
                ``targets[i]`` holds the tree-2 slots of the edges leaving
                observation ``i`` (possibly none, possibly several).

    Returns
    -------
    total :     int
                The minimal achievable number of crossings.
    rot :       (n-1, ) bool array
                Rows of ``link`` that must be rotated to achieve ``total``.

    """
    n = len(targets)
    sub = {i: targets[i] for i in range(n)}
    rot = np.zeros(link.shape[0], dtype=bool)
    total = 0

    for k in range(link.shape[0]):
        a, b = int(link[k, 0]), int(link[k, 1])
        ta, tb = sub[a], sub[b]

        if ta.size and tb.size:
            # how many of `ta` sit below / at each element of `tb`
            lo = np.searchsorted(ta, tb, side="left")
            hi = np.searchsorted(ta, tb, side="right")

            cross_ab = int(np.sum(ta.size - hi))  # pairs with t_a > t_b
            ties = int(np.sum(hi - lo))  # equal targets never cross
            cross_ba = ta.size * tb.size - cross_ab - ties

            if cross_ba < cross_ab:
                rot[k] = True
                total += cross_ba
            else:
                total += cross_ab

        sub[n + k] = np.sort(np.concatenate([ta, tb]))
        del sub[a], sub[b]

    return total, rot


def untangle_cross_1side(link1, link2, labels1, labels2, edges, L=None):
    """Untangle by optimally rotating ``link1`` to minimise *crossings*.

    Finds the provably best of all ``2**(n-1)`` rotations of ``link1`` while
    ``link2`` stays fixed. Needs only ``O(n)`` memory, so this is the method to
    reach for on very large dendrograms.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples
    L :                 ignored
                        Accepted so that this method can be swapped in
                        wherever the displacement-based ones are used.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    n1 = len(link1) + 1
    lindex2 = leaf_order(link2, labels2, as_dict=True)

    targets = _crossing_targets(labels1, lindex2, edges, n1)
    total, rot = _dp_optimal_rotations_crossings(link1, targets)

    link1 = link1.copy()
    to_rot = np.nonzero(rot)[0]
    link1[to_rot, :2] = link1[to_rot, :2][:, ::-1]

    module_logger.info(f"Finished optimising at {total} crossings")
    return link1, link2


def untangle_cross_2side(
    link1,
    link2,
    labels1,
    labels2,
    edges,
    L=None,
    max_n_iterations=10,
    restarts=0,
    seed=None,
):
    """Untangle both dendrograms by alternating exact crossing minimisations.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples
    L :                 ignored
                        Accepted for interface compatibility.
    max_n_iterations :  int
                        Max alternations (default = 10) per start.
    restarts :          int
                        Number of additional randomised restarts (default = 0).
    seed :              int, optional
                        Seed for the restarts, for reproducibility.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    edges_inv = [(e[1], e[0]) for e in edges]

    def score(l1, l2):
        return _crossings(
            leaf_order(l1, labels1, as_dict=True),
            leaf_order(l2, labels2, as_dict=True),
            edges,
        )

    def alternate(l1, l2):
        current = score(l1, l2)
        for _ in range(int(max_n_iterations)):
            l1, l2 = untangle_cross_1side(l1, l2, labels1, labels2, edges)
            l2, l1 = untangle_cross_1side(l2, l1, labels2, labels1, edges_inv)

            new = score(l1, l2)
            if new >= current:
                current = new
                break
            current = new

            if current == 0:
                break
        return l1, l2, current

    ol = module_logger.level
    module_logger.setLevel("WARNING")

    best1, best2, fewest = alternate(link1, link2)

    if restarts:
        rng = np.random.default_rng(seed)
        for _ in range(int(restarts)):
            if fewest == 0:
                break
            cand1, cand2, total = alternate(
                shuffle_dendogram(link1, copy=True, rng=rng),
                shuffle_dendogram(link2, copy=True, rng=rng),
            )
            if total < fewest:
                best1, best2, fewest = cand1, cand2, total

    module_logger.setLevel(ol)
    module_logger.info(f"Finished optimising at {fewest} crossings")
    return best1, best2


@dataclass
class AlignmentQuality:
    """Report from :func:`alignment_quality`.

    The entanglement of a tanglegram is only interpretable relative to what the
    two dendrograms actually permit: a value of 0.4 can mean either a failed
    untangling or two trees that genuinely disagree. The fields below provide
    that context.
    """

    entanglement: float
    crossings: int
    random_entanglement: float
    random_std: float
    best_found: float
    median_displacement: float
    max_displacement: int
    frac_well_aligned: float
    n_leafs: int
    n_edges: int

    @property
    def z_score(self):
        """How many SDs better than a random layout this alignment is."""
        if self.random_std == 0:
            return float("nan")
        return (self.entanglement - self.random_entanglement) / self.random_std

    @property
    def improvement(self):
        """Fraction of the way from a random layout towards perfect alignment."""
        if self.random_entanglement == 0:
            return 1.0
        return 1 - self.entanglement / self.random_entanglement

    @property
    def at_floor(self):
        """Whether re-optimising failed to find anything meaningfully better.

        If True, the residual entanglement is imposed by the structure of the
        two dendrograms - no rotation will improve the plot. Differences below
        half a percent are treated as noise; restarts stop paying off well
        before that.
        """
        return self.entanglement <= self.best_found * 1.005 + 1e-12

    def __str__(self):
        near = max(1, int(0.05 * self.n_leafs))
        verdict = (
            "at the structural floor - the trees themselves disagree"
            if self.at_floor
            else f"re-optimising reached {self.best_found:.4f}; try more restarts"
        )
        return (
            f"Alignment quality ({self.n_leafs} leafs, {self.n_edges} edges)\n"
            f"  entanglement       {self.entanglement:>10.4f}   "
            f"(random layout: {self.random_entanglement:.4f} "
            f"+/- {self.random_std:.4f}, z = {self.z_score:.1f})\n"
            f"  crossings          {self.crossings:>10,}\n"
            f"  better than random {self.improvement:>9.0%}\n"
            f"  displacement       median {self.median_displacement:.0f}, "
            f"max {self.max_displacement} slots; "
            f"{self.frac_well_aligned:.0%} of edges within {near}\n"
            f"  verdict            {verdict}"
        )


def alignment_quality(
    link1,
    link2,
    labels1,
    labels2,
    edges=None,
    L=1.5,
    n_random=100,
    restarts=5,
    seed=None,
):
    """Judge an alignment against what these two dendrograms actually allow.

    Entanglement on its own is not interpretable - it has no natural scale and
    depends heavily on how similar the two trees are to begin with. This
    compares the given alignment against a random layout (to show how much the
    untangling bought) and against a fresh optimisation (to show whether
    anything is left on the table).

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        The *already untangled* linkages to judge.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    edges :             list of tuples, optional
    L :                 float
                        Distance norm used to calculate the entanglement.
    n_random :          int
                        Random layouts to sample for the baseline.
    restarts :          int
                        Restarts for the re-optimisation check. Set to 0 to
                        skip it (the ``at_floor`` verdict is then meaningless).
    seed :              int, optional

    Returns
    -------
    AlignmentQuality
                        ``print()`` it for a readable report.

    Examples
    --------
    >>> link1, link2 = tg.untangle(link1, link2, labels, labels, edges)   # doctest: +SKIP
    >>> print(tg.alignment_quality(link1, link2, labels, labels, edges))  # doctest: +SKIP

    """
    labels1, labels2 = np.asarray(labels1), np.asarray(labels2)

    lindex1 = leaf_order(link1, labels1, as_dict=True)
    lindex2 = leaf_order(link2, labels2, as_dict=True)

    if edges is None:
        exist_in_both = list(set(lindex1) & set(lindex2))
        if not exist_in_both:
            raise ValueError("Not a single matching label in both dendrograms.")
        edges = [(l, l) for l in exist_in_both]

    ent = _entanglement(lindex1, lindex2, edges, L=L)
    n_leafs = max(len(lindex1), len(lindex2))

    # --- baseline: what would a random layout give? ---
    # Note this must be a *uniform* draw from the rotation space, which is not
    # what shuffle_dendogram() does (it rotates a random *number* of hinges).
    rng = np.random.default_rng(seed)
    randoms = np.array(
        [
            _entanglement(
                leaf_order(_random_rotation(link1, rng), labels1, as_dict=True),
                leaf_order(_random_rotation(link2, rng), labels2, as_dict=True),
                edges,
                L=L,
            )
            for _ in range(int(n_random))
        ]
    )

    # --- is anything left on the table? ---
    best_found = ent
    if restarts:
        ol = module_logger.level
        module_logger.setLevel("WARNING")
        opt1, opt2 = untangle_dp_2side(
            link1, link2, labels1, labels2, edges, L=L, restarts=restarts, seed=seed
        )
        module_logger.setLevel(ol)
        best_found = min(
            ent,
            _entanglement(
                leaf_order(opt1, labels1, as_dict=True),
                leaf_order(opt2, labels2, as_dict=True),
                edges,
                L=L,
            ),
        )

    dist = np.abs(np.array([lindex1[e[0]] - lindex2[e[1]] for e in edges], dtype=float))
    near = max(1, int(0.05 * n_leafs))

    return AlignmentQuality(
        entanglement=ent,
        crossings=_crossings(lindex1, lindex2, edges),
        random_entanglement=float(randoms.mean()),
        random_std=float(randoms.std()),
        best_found=best_found,
        median_displacement=float(np.median(dist)),
        max_displacement=int(dist.max()),
        frac_well_aligned=float((dist <= near).mean()),
        n_leafs=n_leafs,
        n_edges=len(edges),
    )


def shuffle_dendogram(link, copy=True, rng=None):
    """Randomly shuffle dendrogram.

    Parameters
    ----------
    link :      scipy.cluster.hierarchy.linkage
    copy :      bool
                Whether to shuffle a copy or in place.
    rng :       numpy.random.Generator, optional
                Use for reproducible shuffles. Defaults to the global
                ``numpy.random`` state.

    Returns
    -------
    s_link :    scipy.cluster.hierarchy.linkage
                Shuffled linkage.

    """
    assert isinstance(link, np.ndarray)

    if rng is None:
        rng = np.random.default_rng()

    # How many hinges to rotate
    n_rot = rng.integers(len(link))

    # Which hinges to rotate
    to_rot = rng.choice(np.arange(len(link)), n_rot, replace=False)

    # Make a copy of the original
    if copy:
        s_link = link.copy()
    else:
        s_link = link

    # Rotate hinges
    s_link[to_rot, :2] = s_link[to_rot, :2][:, ::-1]

    return s_link


def leaf_order(link, labels=None, as_dict=True, invert=False):
    """Generate leaf label order for given linkage.

    Parameters
    ----------
    link :      scipy.cluster.hierarchy.linkage
                Linkage to get leaf label order for.
    labels :    list, optional
                If provided, return ordered labels else will return indices.
    as_dict :   bool
                If True (default), returns a dictionary mapping labels/indices
                to leaf indices.
    invert :    bool
                If True will invert the order.

    Returns
    -------
    dict
                If ``as_dict=True`` return as ``{'l1': 1, 'l2':, 5, ...}``.
    list
                If ``as_dict=False`` return as ``['l4', 'l3', 'l1', ...]``.

    """
    # This gives us the order of the original labels
    leafs_ix = sclust.hierarchy.leaves_list(link)

    if invert:
        mx = max(leafs_ix)
        leafs_ix = [mx - i for i in leafs_ix]

    if as_dict:
        if not isinstance(labels, type(None)):
            labels = np.asarray(labels)
            return dict(zip(labels[leafs_ix], np.arange(len(leafs_ix))))
        else:
            return dict(zip(np.arange(len(leafs_ix)), leafs_ix))
    else:
        if not isinstance(labels, type(None)):
            return np.asarray(labels)[leafs_ix]
        else:
            return leafs_ix


def rotate(linkage, i, copy=True):
    """Rotate linkage at given hinge."""
    # Make copy
    if copy:
        linkage = linkage.copy()

    # Rotate
    linkage[i][0], linkage[i][1] = linkage[i][1], linkage[i][0]

    return linkage


UNTANGLE_METHODS = {
    "dp1side": untangle_dp_1side,
    "dp2side": untangle_dp_2side,
    "cross1side": untangle_cross_1side,
    "cross2side": untangle_cross_2side,
    "random": untangle_random_search,
    "step1side": untangle_step_rotate_1side,
    "step2side": untangle_step_rotate_2side,
}

# Two-sided methods and the one-sided variant ``tanglegram_many()`` drops to for
# every pair after the first, where the preceding dendrogram is already pinned.
ONE_SIDED_EQUIVALENT = {
    "step2side": "step1side",
    "dp2side": "dp1side",
    "cross2side": "cross1side",
}
