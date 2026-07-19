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

import numpy as np

import matplotlib.colors as mcl
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

import scipy.cluster.hierarchy as sch


__all__ = ["dendrogram", "cluster_color_map"]

ORIENTATIONS = ("top", "bottom", "left", "right")

#: Fraction of the total tree height that leafs are lifted off the baseline
#: when ``hanging=True``.
DEFAULT_HANG = 0.05

#: Number of interpolated points used to draw a horizontal link as an arc in
#: polar projection. Straight lines would cut chords across the circle.
_ARC_RESOLUTION = 16

#: Sentinel for a node whose leafs do not all belong to the same cluster.
_MIXED = object()


def dendrogram(
    Z,
    clusters=None,
    cluster_colors=None,
    above_threshold_color="lightgrey",
    above_threshold_ls="-",
    label_dist=False,
    dist_fmt="{:.2f}",
    leaf_marker=None,
    leaf_color=None,
    leaf_size=None,
    leaf_annot=None,
    annot_colors=None,
    annot_size=0.05,
    annot_legend=False,
    hanging=False,
    polar=False,
    collapse=None,
    collapse_width=1.0,
    collapse_label=None,
    hover_info=None,
    ax=None,
    fig=None,
    backend="matplotlib",
    **kwargs,
):
    """Extends scipy's dendrogram function with extra functionality.

    On top of scipy this adds the option to:

      1. Color links by cluster assignment instead of the simple
         ``color_threshold``/``above_threshold_color`` scipy offers.
      2. Style links that do not belong to any cluster separately.
      3. Label links with their distance.
      4. Mark and color the leafs.
      5. Draw "hanging" dendrograms.
      6. Draw radial (polar) dendrograms.
      7. Use plotly instead of matplotlib as backend.

    Parameters
    ----------
    Z :             np.ndarray
                    Linkage.
    clusters :      list-like, optional
                    Cluster assignment for each of the original observations
                    (i.e. in the same order as the observations in ``Z``, not
                    in leaf order). Each cluster gets its own color. Links that
                    span more than one cluster - and leafs with a ``None``/NaN
                    assignment - are drawn in ``above_threshold_color``.
    cluster_colors : dict | list-like, optional
                    Colors to use for the clusters. Either a ``{cluster: color}``
                    dictionary or a sequence of colors that is cycled in order
                    of the clusters' first appearance in ``clusters``. Defaults
                    to matplotlib's current color cycle.
    above_threshold_color : str
                    Color for links that do not belong to a single cluster.
                    Also used by scipy's own ``color_threshold`` when
                    ``clusters`` is not provided.
    above_threshold_ls : str
                    Matplotlib line style (e.g. ``"--"``) for those same links.
    label_dist :    bool
                    Whether to label links with their distance.
    dist_fmt :      str
                    Format string used for the distance labels.
    leaf_marker :   str, optional
                    A single matplotlib marker (e.g. ``"o"``) used to mark the
                    leafs.
    leaf_color :    color | list-like, optional
                    Color(s) for the leaf markers. A single color, or one per
                    original observation (same order as ``clusters``). Defaults
                    to the color of each leaf's own link.
    leaf_size :     float, optional
                    Size of the leaf markers.
    leaf_annot :    list-like | dict | pandas.DataFrame, optional
                    One or more annotation tracks, drawn as a color strip next
                    to the leafs (like ``row_colors`` in seaborn's clustermap).
                    Values are per original observation. Accepts a flat
                    sequence for a single track, a ``{name: values}`` dict or a
                    DataFrame whose columns are the tracks. Values that already
                    look like colors are used directly; anything else is
                    treated as a category and colored automatically. Requires
                    ``backend="matplotlib"`` and is not available for
                    ``polar``.
    annot_colors :  dict | list-like, optional
                    Colors for the annotation values. Either a
                    ``{value: color}`` mapping shared by all tracks, a
                    ``{track: {value: color}}`` mapping, or a sequence to cycle.
                    A mapping only needs to cover the values you care about;
                    anything it misses is colored automatically.
    annot_size :    float
                    Thickness of each annotation track, as a fraction of the
                    dendrogram's own size.
    annot_legend :  bool
                    If True, add a legend for the annotation tracks.
    hanging :       bool | float
                    If truthy, leaf links are shortened so they hang off their
                    parent instead of reaching down to the baseline. ``True``
                    uses a drop of ``DEFAULT_HANG`` (5% of the tree height); a
                    float sets that fraction explicitly.
    polar :         bool
                    If True, plot as a radial dendrogram. Requires
                    ``backend="matplotlib"``, and ``ax`` (if given) must use a
                    polar projection.
    collapse :      callable | float | "clusters", optional
                    Draw whole clades as a single wedge instead of spelling out
                    every leaf - the usual way to make a large tree readable.
                    Only the topmost matching clade on each path is collapsed.
                      - callable: gets the list of observations under a node and
                        returns True to collapse it
                      - float: collapse every clade no taller than this
                      - ``"clusters"``: collapse each cluster that forms a clade
                        of its own (requires ``clusters``)
                    The returned dendrogram dict gains a ``"collapsed"`` entry
                    mapping each wedge to the observations it stands for, and
                    its ``"leaves"``/``"ivl"`` describe what is drawn rather
                    than every leaf. Cannot be combined with ``truncate_mode``.
    collapse_width : float
                    Width of a wedge, in leaf slots.
    collapse_label : callable, optional
                    Called as ``f(node, observations)`` to label a wedge.
                    Defaults to the cluster name and leaf count, or just the
                    leaf count.
    hover_info :    list-like, optional
                    Additional hover text for each original observation. Only
                    used by the ``plotly`` backend.
    ax :            matplotlib.axes.Axes, optional
                    Axis to plot the dendrogram on. Ignored if
                    ``backend="plotly"``.
    fig :           plotly.graph_objects.Figure | dict, optional
                    Figure to plot the dendrogram on. Ignored if
                    ``backend="matplotlib"``. Pass a dict to target a specific
                    subplot, e.g. ``{"fig": fig, "row": 1, "col": 2}``.
    backend :       "matplotlib" | "plotly"
                    Which backend to use for plotting.

    **kwargs
                    Keyword arguments are passed through to scipy's
                    ``dendrogram``. Note that ``no_plot`` and ``link_color_func``
                    are controlled by this function and cannot be set.

    Returns
    -------
    dn :            dict
                    The dendrogram as generated by scipy's dendrogram function.
    ax/fig
                    Depending on the backend, either the matplotlib axis or the
                    plotly figure.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.cluster.hierarchy as sch
    >>> import tanglegram as tg
    >>> Z = sch.linkage(np.random.rand(20, 4), method="ward")
    >>> clusters = sch.fcluster(Z, 3, criterion="maxclust")
    >>> dn, ax = tg.dendrogram(Z, clusters=clusters, hanging=True,
    ...                        leaf_marker="o", above_threshold_ls="--")

    """
    if backend not in ("matplotlib", "plotly"):
        raise ValueError(
            f'Unknown backend: "{backend}". Expected "matplotlib" or "plotly".'
        )

    for reserved in ("no_plot", "link_color_func"):
        if reserved in kwargs:
            raise ValueError(f"`{reserved}` is controlled by tanglegram.dendrogram")

    orientation = kwargs.setdefault("orientation", "top")
    if orientation not in ORIENTATIONS:
        raise ValueError(
            f'Unknown orientation: "{orientation}". Expected one of {ORIENTATIONS}.'
        )

    if polar and backend != "matplotlib":
        raise NotImplementedError(
            "Polar dendrograms are only available for the matplotlib backend."
        )

    if leaf_annot is not None:
        if backend != "matplotlib":
            raise NotImplementedError(
                "`leaf_annot` is only available for the matplotlib backend."
            )
        if polar:
            raise NotImplementedError("`leaf_annot` cannot be combined with `polar`.")
        if collapse is not None:
            raise ValueError(
                "`leaf_annot` is given per observation and cannot be combined "
                "with `collapse`, which draws whole clades as one wedge."
            )

    Z = np.asarray(Z)
    n_obs = len(Z) + 1

    dn, segments, leaf_base, wedges = _geometry(
        Z,
        clusters=clusters,
        cluster_colors=cluster_colors,
        above_threshold_color=above_threshold_color,
        above_threshold_ls=above_threshold_ls,
        hanging=hanging,
        collapse=collapse,
        collapse_width=collapse_width,
        collapse_label=collapse_label,
        **kwargs,
    )

    leaf_colors = _resolve_leaf_colors(dn, leaf_base, leaf_color, n_obs)

    if backend == "matplotlib":
        ax = _plot_matplotlib(
            dn,
            segments,
            leaf_base=leaf_base,
            leaf_colors=leaf_colors,
            leaf_marker=leaf_marker,
            leaf_size=leaf_size,
            label_dist=label_dist,
            dist_fmt=dist_fmt,
            orientation=orientation,
            polar=polar,
            wedges=wedges,
            ax=ax,
            leaf_rotation=kwargs.get("leaf_rotation", None),
            leaf_font_size=kwargs.get("leaf_font_size", None),
            no_labels=kwargs.get("no_labels", False),
        )

        if leaf_annot is not None:
            _draw_annotations(
                ax,
                dn,
                leaf_annot=leaf_annot,
                annot_colors=annot_colors,
                annot_size=annot_size,
                annot_legend=annot_legend,
                orientation=orientation,
                n_obs=n_obs,
                leaf_rotation=kwargs.get("leaf_rotation", None),
                leaf_font_size=kwargs.get("leaf_font_size", None),
                no_labels=kwargs.get("no_labels", False),
            )

        return dn, ax

    fig = _plot_plotly(
        dn,
        segments,
        leaf_base=leaf_base,
        leaf_colors=leaf_colors,
        leaf_marker=leaf_marker,
        leaf_size=leaf_size,
        label_dist=label_dist,
        dist_fmt=dist_fmt,
        orientation=orientation,
        wedges=wedges,
        hover_info=_reorder_per_leaf(hover_info, dn, n_obs, "hover_info"),
        fig=fig,
        no_labels=kwargs.get("no_labels", False),
    )
    return dn, fig


def _is_null(v):
    """Check for None/NaN/pd.NA without assuming a dtype."""
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _hashable(v):
    """Turn numpy scalars into plain Python so they can be used as dict keys."""
    return v.item() if isinstance(v, np.generic) else v


def _assign_node_clusters(Z, clusters):
    """Label every node in the linkage with its cluster.

    Returns an array of length ``2 * n_obs - 1`` where each entry is either a
    cluster label, ``None`` (no cluster) or ``_MIXED`` (spans >1 cluster).
    Index ``i < n_obs`` is an original observation, ``n_obs + i`` is row ``i``
    of ``Z``.
    """
    n = len(Z) + 1

    clusters = np.asarray(clusters).flatten()
    if len(clusters) != n:
        raise ValueError(f"Got {len(clusters)} clusters for {n} leafs.")

    node_cluster = np.empty(2 * n - 1, dtype=object)
    for i, cl in enumerate(clusters):
        node_cluster[i] = None if _is_null(cl) else _hashable(cl)

    # Single bottom-up pass: a node is "cluster c" only if every leaf below it
    # is in c. Rows of Z are ordered so both children are always resolved first.
    for i, row in enumerate(Z):
        a, b = node_cluster[int(row[0])], node_cluster[int(row[1])]
        if a is _MIXED or b is _MIXED:
            merged = _MIXED
        elif a is None and b is None:
            merged = None
        elif a is None or b is None:
            merged = _MIXED
        else:
            merged = a if a == b else _MIXED
        node_cluster[n + i] = merged

    return node_cluster


def _geometry(
    Z,
    clusters=None,
    cluster_colors=None,
    above_threshold_color="lightgrey",
    above_threshold_ls="-",
    hanging=False,
    collapse=None,
    collapse_width=1.0,
    collapse_label=None,
    **kwargs,
):
    """Work out everything needed to draw a dendrogram, without drawing it.

    Split out from `dendrogram` so that other layouts - the circular
    tanglegram, for one - can place the same geometry somewhere else entirely.

    Returns
    -------
    dn :        dict
                The dendrogram as scipy would return it, with the leaf order
                and, when collapsing, the wedge summary.
    segments :  list of dict
                Individually styled line segments, in independent/dependent
                coordinates. See `_build_segments`.
    leaf_base : dict
                Leaf position -> (baseline distance, color).
    wedges :    list of dict
                Collapsed clades to draw, empty unless `collapse` was given.

    """
    Z = np.asarray(Z)
    n_obs = len(Z) + 1
    above_threshold_color = _normalize_color(above_threshold_color)

    # Work out the color for every node in the linkage
    if clusters is not None:
        if kwargs.get("truncate_mode", None):
            raise ValueError(
                "`clusters` cannot be combined with scipy's `truncate_mode`: "
                "truncation collapses leafs, so cluster assignments no longer "
                "map onto what is drawn."
            )
        node_cluster = _assign_node_clusters(Z, clusters)
        color_map = _build_color_map(node_cluster[:n_obs], cluster_colors)

        def node_color(k):
            cl = node_cluster[k]
            if cl is _MIXED or cl is None:
                return above_threshold_color
            return color_map[cl]

    else:
        kwargs.setdefault("above_threshold_color", above_threshold_color)
        node_cluster = None
        node_color = None

    # scipy calls `link_color_func` exactly once per link and in the same order
    # as it fills icoord/dcoord, so recording the calls tells us which linkage
    # node each drawn link belongs to. That mapping is what lets us color the
    # two halves of a link independently, and tell legs apart from leafs.
    node_ids = []

    def _record(k):
        node_ids.append(k)
        return above_threshold_color if node_color is None else node_color(k)

    if node_color is None:
        # Let scipy color the links itself (i.e. honor `color_threshold`), then
        # take a second, throwaway pass purely to recover the node ids.
        dn = sch.dendrogram(Z, no_plot=True, **kwargs)
        link_colors = list(dn["color_list"])
        sch.dendrogram(Z, no_plot=True, link_color_func=_record, **kwargs)
    else:
        dn = sch.dendrogram(Z, no_plot=True, link_color_func=_record, **kwargs)
        link_colors = [node_color(k) for k in node_ids]

    if collapse is not None:
        dn, node_ids, link_colors, wedges = _collapse_layout(
            dn,
            Z,
            collapse=collapse,
            node_cluster=node_cluster,
            node_color=node_color,
            # Without clusters the colors are scipy's own, so keep them by node
            # rather than falling back to grey for everything we redraw
            original_colors=dict(zip(node_ids, link_colors)),
            collapse_width=collapse_width,
            collapse_label=collapse_label,
            above_threshold_color=above_threshold_color,
        )
    else:
        wedges = []

    segments, leaf_base = _build_segments(
        dn,
        Z,
        node_ids,
        link_colors,
        node_color=node_color,
        hanging=_hang_amount(dn, hanging),
        above_threshold_color=above_threshold_color,
        above_threshold_ls=above_threshold_ls,
    )

    return dn, segments, leaf_base, wedges


def cluster_color_map(*assignments, colors=None):
    """Build one ``{cluster: color}`` map spanning several cluster assignments.

    Each call to `dendrogram` otherwise picks colors from the clusters it is
    given, so the same cluster can end up a different color in two plots. Pass
    the map returned here as ``cluster_colors`` to keep them in step - this is
    what `tanglegram` does for its two sides.

    Parameters
    ----------
    *assignments : list-like
                    One or more cluster assignments. ``None`` assignments are
                    skipped, as are ``None``/NaN entries within them.
    colors :        dict | list-like, optional
                    Colors to use. Either a ``{cluster: color}`` mapping or a
                    sequence cycled in order of first appearance. Defaults to
                    matplotlib's current color cycle.

    Returns
    -------
    dict
                    Maps each cluster to a hex color.

    Examples
    --------
    >>> import tanglegram as tg
    >>> tg.cluster_color_map(["a", "b"], ["b", "c"])   # doctest: +SKIP
    {'a': '#1f77b4', 'b': '#ff7f0e', 'c': '#2ca02c'}

    """
    combined = []
    for assignment in assignments:
        if assignment is None:
            continue
        for cl in np.asarray(assignment).flatten():
            combined.append(None if _is_null(cl) else _hashable(cl))

    return _build_color_map(combined, colors)


def _build_color_map(leaf_clusters, cluster_colors):
    """Map cluster label -> color, deterministically."""
    # Order of first appearance, so colors don't shuffle between runs the way
    # they would if we iterated over a set.
    order = {}
    for cl in leaf_clusters:
        if cl is not None:
            order.setdefault(cl, len(order))

    if isinstance(cluster_colors, dict):
        missing = [cl for cl in order if cl not in cluster_colors]
        if missing:
            raise ValueError(f"`cluster_colors` is missing entries for: {missing}")
        return {cl: _normalize_color(cluster_colors[cl]) for cl in order}

    if cluster_colors is None:
        palette = plt.rcParams["axes.prop_cycle"].by_key().get("color")
        if not palette:
            palette = [f"C{i}" for i in range(10)]
    else:
        palette = list(cluster_colors)
        if not palette:
            raise ValueError("`cluster_colors` must not be empty.")

    return {cl: _normalize_color(palette[i % len(palette)]) for cl, i in order.items()}


def _normalize_color(c):
    """Normalize any matplotlib color spec to a hex string.

    scipy's ``link_color_func`` insists on a string, and matplotlib's color
    cycle may hand us RGB tuples, so we convert once here rather than guessing
    downstream. Note that this drops any alpha channel.
    """
    try:
        return mcl.to_hex(c)
    except (ValueError, TypeError):
        raise ValueError(f"Not a valid matplotlib color: {c!r}")


def _hang_amount(dn, hanging):
    """Translate the `hanging` parameter into an absolute distance."""
    if not hanging:
        return 0.0

    frac = DEFAULT_HANG if hanging is True else float(hanging)
    if frac < 0:
        raise ValueError(f"`hanging` must be a fraction >= 0, got {frac}")

    max_dist = max((d[1] for d in dn["dcoord"]), default=0.0)
    return frac * max_dist


def _build_segments(
    dn,
    Z,
    node_ids,
    link_colors,
    node_color,
    hanging,
    above_threshold_color,
    above_threshold_ls,
):
    """Break the dendrogram into individually styled line segments.

    scipy draws each link as one 4-point polyline in a single color. We split it
    into its two vertical legs plus the horizontal bar so that each leg can take
    the color of the child it descends to. That is what lets a cluster of a
    single leaf get its own color - previously this needed a separate pass that
    drew over the finished plot.

    Returns
    -------
    segments :  list of dicts with "i", "d", "color" and "ls" keys, where "i"
                and "d" are the independent/dependent coordinates.
    leaf_base : dict of leaf position -> (baseline distance, color)
    """
    n = len(Z) + 1
    segments = []
    leaf_base = {}

    # Whatever scipy decided to put on the independent axis - real leafs, or
    # collapsed clusters when truncating - keyed by the node it stands for.
    position = {unit: p for p, unit in enumerate(dn["leaves"])}

    for j, (ic, dc) in enumerate(zip(dn["icoord"], dn["dcoord"])):
        link_color = link_colors[j]
        height = dc[1]

        row = Z[node_ids[j] - n]
        children = (int(row[0]), int(row[1]))

        if node_color is None:
            # No cluster info - keep scipy's single color for the whole link
            leg_colors = (link_color, link_color)
        else:
            leg_colors = (node_color(children[0]), node_color(children[1]))

        # Horizontal bar
        segments.append(
            _segment(
                [ic[1], ic[2]],
                [height, height],
                link_color,
                above_threshold_color,
                above_threshold_ls,
            )
        )

        # The two vertical legs
        for leg, (x, base) in enumerate(((ic[0], dc[0]), (ic[3], dc[3]))):
            color = leg_colors[leg]
            # A leg ends on a leaf exactly when the child it descends to is one
            # of the units drawn along the independent axis. Reading that off
            # the child rather than off the coordinates keeps it right for
            # linkages that merge at distance 0, where an inner node also
            # reaches the baseline.
            p = position.get(children[leg])
            is_leaf = p is not None and base == 0

            if is_leaf and hanging:
                base = max(height - hanging, 0.0)

            if is_leaf:
                leaf_base[p] = (base, color)

            segments.append(
                _segment(
                    [x, x],
                    [base, height],
                    color,
                    above_threshold_color,
                    above_threshold_ls,
                )
            )

    return segments, leaf_base


def _leaf_spans(Z, leaves):
    """For every node, the first and last leaf slot it covers.

    scipy orders the leafs so that every subtree occupies a contiguous block,
    which is what makes a collapsed clade a single well defined wedge.
    """
    n = len(Z) + 1
    slot = np.empty(n, dtype=int)
    slot[np.asarray(leaves, dtype=int)] = np.arange(n)

    lo = np.empty(2 * n - 1, dtype=int)
    hi = np.empty(2 * n - 1, dtype=int)
    lo[:n] = hi[:n] = slot

    for i, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        lo[n + i] = min(lo[a], lo[b])
        hi[n + i] = max(hi[a], hi[b])

    return lo, hi


def _nodes_to_collapse(Z, collapse, node_cluster, leaves, lo, hi):
    """The topmost nodes matching `collapse`.

    Only the highest match on each path is collapsed - collapsing a clade
    already inside a collapsed one would have no visible effect.
    """
    n = len(Z) + 1
    leaves = np.asarray(leaves, dtype=int)

    if callable(collapse):

        def matches(node):
            return bool(collapse(leaves[lo[node] : hi[node] + 1].tolist()))

    elif isinstance(collapse, str):
        if collapse != "clusters":
            raise ValueError(
                f'`collapse` must be a callable, a number or "clusters", got '
                f'"{collapse}"'
            )
        if node_cluster is None:
            raise ValueError('`collapse="clusters"` requires `clusters`.')

        def matches(node):
            return node_cluster[node] is not _MIXED and node_cluster[node] is not None

    else:
        try:
            threshold = float(collapse)
        except (TypeError, ValueError):
            raise ValueError(
                '`collapse` must be a callable, a number or "clusters", got '
                f"{collapse!r}"
            )

        def matches(node):
            return Z[node - n, 2] <= threshold

    collapsed = set()
    stack = [2 * n - 2]
    while stack:
        node = stack.pop()
        if node < n:
            continue
        if matches(node):
            collapsed.add(node)
        else:
            stack.append(int(Z[node - n, 0]))
            stack.append(int(Z[node - n, 1]))

    return collapsed


def _collapse_layout(
    dn,
    Z,
    collapse,
    node_cluster,
    node_color,
    original_colors,
    collapse_width,
    collapse_label,
    above_threshold_color,
):
    """Re-lay out the dendrogram with some clades drawn as wedges.

    scipy cannot do this, so we compute the coordinates ourselves. The leaf
    *order* still comes from scipy, so `count_sort` and friends keep working;
    all we change is how much room each visible unit gets and where the
    remaining links end up.

    Returns a replacement ``dn``, the node id behind each link, the link colors
    and the wedges to draw.
    """
    n = len(Z) + 1
    leaves = list(dn["leaves"])
    if len(leaves) != n:
        raise ValueError("`collapse` cannot be combined with scipy's `truncate_mode`.")

    lo, hi = _leaf_spans(Z, leaves)
    collapsed = _nodes_to_collapse(Z, collapse, node_cluster, leaves, lo, hi)

    if collapse_width <= 0:
        raise ValueError(f"`collapse_width` must be > 0, got {collapse_width}")

    def children_in_draw_order(node):
        a, b = int(Z[node - n, 0]), int(Z[node - n, 1])
        return (a, b) if lo[a] <= lo[b] else (b, a)

    # Left-to-right sweep over what is actually drawn
    units = []
    stack = [2 * n - 2]
    while stack:
        node = stack.pop()
        if node < n or node in collapsed:
            units.append(node)
            continue
        a, b = children_in_draw_order(node)
        stack.append(b)
        stack.append(a)

    x = {}
    cursor = 0.0
    for unit in units:
        width = 10.0 * (collapse_width if unit in collapsed else 1.0)
        x[unit] = cursor + width / 2
        cursor += width

    # Everything above the visible units sits at the midpoint of its children.
    # Iterative so that a deep, ladder-shaped tree cannot blow the stack.
    todo = [(2 * n - 2, False)]
    while todo:
        node, ready = todo.pop()
        if node < n or node in collapsed:
            continue
        a, b = children_in_draw_order(node)
        if ready:
            x[node] = (x[a] + x[b]) / 2
        else:
            todo.append((node, True))
            todo.append((a, False))
            todo.append((b, False))

    def height_of(node):
        return 0.0 if node < n else float(Z[node - n, 2])

    def color_of(node):
        if node_color is not None:
            return node_color(node)
        return original_colors.get(node, above_threshold_color)

    icoord, dcoord, node_ids, link_colors = [], [], [], []
    for node in sorted(x, key=lambda k: (k < n, k)):
        if node < n or node in collapsed:
            continue
        a, b = children_in_draw_order(node)
        height = float(Z[node - n, 2])
        icoord.append([x[a], x[a], x[b], x[b]])
        dcoord.append([height_of(a), height, height, height_of(b)])
        node_ids.append(node)
        link_colors.append(color_of(node))

    wedges = [
        {
            "i": x[node],
            "width": 10.0 * collapse_width,
            "height": float(Z[node - n, 2]),
            "color": color_of(node),
        }
        for node in units
        if node in collapsed
    ]

    labels = _collapsed_labels(
        units, collapsed, dn, Z, leaves, lo, hi, node_cluster, collapse_label
    )

    collapsed_dn = dict(dn)
    collapsed_dn["icoord"] = icoord
    collapsed_dn["dcoord"] = dcoord
    collapsed_dn["color_list"] = list(link_colors)
    collapsed_dn["ivl"] = labels
    collapsed_dn["leaves"] = list(units)
    collapsed_dn["collapsed"] = {
        node: leaves[lo[node] : hi[node] + 1] for node in units if node in collapsed
    }
    # Wedges are wider than a leaf slot, so the renderers cannot assume the
    # usual fixed grid any more
    collapsed_dn["_unit_i"] = [x[unit] for unit in units]
    collapsed_dn["_ivw"] = cursor

    return collapsed_dn, node_ids, link_colors, wedges


def _collapsed_labels(
    units, collapsed, dn, Z, leaves, lo, hi, node_cluster, collapse_label
):
    """Label for each visible unit: the leaf's own label, or a wedge summary."""
    original = {leaf: dn["ivl"][p] for p, leaf in enumerate(leaves)}

    labels = []
    for unit in units:
        if unit not in collapsed:
            labels.append(original[unit])
            continue

        members = leaves[lo[unit] : hi[unit] + 1]
        if collapse_label is not None:
            labels.append(str(collapse_label(unit, list(members))))
        elif node_cluster is not None and node_cluster[unit] not in (None, _MIXED):
            labels.append(f"{node_cluster[unit]} ({len(members)})")
        else:
            labels.append(f"({len(members)})")

    return labels


def _segment(i, d, color, above_threshold_color, above_threshold_ls):
    above = _same_color(color, above_threshold_color)
    return {
        "i": i,
        "d": d,
        "color": color,
        "ls": above_threshold_ls if above else "-",
    }


def _same_color(a, b):
    """Compare two matplotlib colors by value rather than by name."""
    try:
        return mcl.to_rgba(a) == mcl.to_rgba(b)
    except ValueError:
        return a == b


def _normalize_annotations(leaf_annot, n_obs):
    """Bring the many accepted shapes of `leaf_annot` into one form.

    Returns ``(track_names, values)`` where values has shape
    ``(n_tracks, n_obs)``.
    """
    # Duck-typed rather than importing pandas just for this
    if hasattr(leaf_annot, "columns") and hasattr(leaf_annot, "values"):
        names = [str(c) for c in leaf_annot.columns]
        values = np.asarray(leaf_annot.values, dtype=object).T
    elif isinstance(leaf_annot, dict):
        names = [str(k) for k in leaf_annot]
        values = np.asarray(
            [np.asarray(v, dtype=object).flatten() for v in leaf_annot.values()],
            dtype=object,
        )
    else:
        values = np.asarray(leaf_annot, dtype=object)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        elif values.ndim != 2:
            raise ValueError(
                f"`leaf_annot` must be 1- or 2-dimensional, got {values.ndim} dims."
            )
        names = [""] * len(values)

    if values.shape[1] != n_obs:
        raise ValueError(
            f"Got {values.shape[1]} `leaf_annot` values for {n_obs} leafs."
        )

    return names, values


def _annotation_colors(values, colors):
    """Resolve one annotation track to a color per observation.

    Values that are already colors are used as they are; anything else is
    treated as a category and gets a color from the cycle. A ``{value: color}``
    dict only has to cover the values you care about - the rest are colored
    automatically, so one dict can be handed to every track.
    """
    present = [v for v in values if not _is_null(v)]

    if colors is None and present and all(mcl.is_color_like(v) for v in present):
        return [None if _is_null(v) else _normalize_color(v) for v in values]

    keys = [None if _is_null(v) else _hashable(v) for v in values]

    overrides = colors if isinstance(colors, dict) else {}
    palette = None if isinstance(colors, dict) else colors
    color_map = _build_color_map(
        [k for k in keys if k is not None and k not in overrides], palette
    )

    return [
        (
            None
            if k is None
            else _normalize_color(overrides[k]) if k in overrides else color_map[k]
        )
        for k in keys
    ]


def _draw_annotations(
    ax,
    dn,
    leaf_annot,
    annot_colors,
    annot_size,
    annot_legend,
    orientation,
    n_obs,
    leaf_rotation,
    leaf_font_size,
    no_labels,
):
    """Draw annotation tracks as a color strip beside the leafs.

    The strip gets its own axes on whichever side the leafs are, and takes the
    leaf labels with it so the two do not fight over the same space.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    names, values = _normalize_annotations(leaf_annot, n_obs)
    n_tracks, n_leaves = len(values), len(dn["leaves"])

    # A dict of dicts gives each track its own map; anything else is shared
    per_track = (
        isinstance(annot_colors, dict)
        and bool(annot_colors)
        and all(isinstance(v, dict) for v in annot_colors.values())
    )

    order = dn["leaves"]
    rgba = np.zeros((n_tracks, n_leaves, 4))
    legend_entries = []
    for t, track in enumerate(values):
        colors = annot_colors.get(names[t]) if per_track else annot_colors
        resolved = _annotation_colors(list(track), colors)
        seen = {}
        for p, obs in enumerate(order):
            c = resolved[obs]
            # Missing values stay transparent rather than claiming a color
            rgba[t, p] = mcl.to_rgba(c) if c is not None else (0, 0, 0, 0)
            if c is not None:
                seen.setdefault((names[t], track[obs]), c)
        legend_entries.extend(seen.items())

    ivw = n_leaves * 10
    side = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[
        orientation
    ]
    divider = make_axes_locatable(ax)
    shared = {"sharex": ax} if orientation in ("top", "bottom") else {"sharey": ax}
    cax = divider.append_axes(
        side, size=f"{annot_size * 100 * n_tracks}%", pad=0.08, **shared
    )

    if orientation in ("top", "bottom"):
        cax.imshow(
            rgba, extent=(0, ivw, n_tracks, 0), aspect="auto", interpolation="nearest"
        )
        cax.set_xlim(0, ivw)
        cax.set_ylim(n_tracks, 0)
        cax.set_yticks(np.arange(n_tracks) + 0.5)
        cax.set_yticklabels(names, fontsize="small")
        cax.set_xticks([])
    else:
        cax.imshow(
            np.transpose(rgba, (1, 0, 2)),
            extent=(0, n_tracks, 0, ivw),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        cax.set_ylim(0, ivw)
        cax.set_xlim(0, n_tracks)
        cax.set_xticks(np.arange(n_tracks) + 0.5)
        cax.set_xticklabels(names, fontsize="small", rotation=90)
        cax.set_yticks([])

    for spine in cax.spines.values():
        spine.set_visible(False)

    # The leaf labels move onto the strip, otherwise they would sit under it
    if not no_labels:
        _clear_leaf_labels(ax, orientation)
        _add_leaf_labels(cax, dn, orientation, no_labels, leaf_rotation, leaf_font_size)

    if annot_legend and legend_entries:
        handles = [
            Line2D([], [], marker="s", ls="", color=c, label=f"{name}: {value}")
            for (name, value), c in dict(legend_entries).items()
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(
                (1.02, 0.5) if orientation in ("top", "bottom") else (1.15, 0.5)
            ),
            fontsize="small",
            frameon=False,
        )

    return cax


def _clear_leaf_labels(ax, orientation):
    """Stop an axis drawing leaf labels that the strip has taken over.

    The strip shares this axis, so the tick *locations* have to stay - clearing
    them would clear them on the strip too. Only the labels are hidden.
    """
    if orientation in ("top", "bottom"):
        ax.tick_params(labelbottom=False, labeltop=False)
    else:
        ax.tick_params(labelleft=False, labelright=False)


def _reorder_per_leaf(values, dn, n_obs, name):
    """Reorder a per-observation sequence into leaf (i.e. plotting) order."""
    if values is None:
        return None

    values = np.asarray(values, dtype=object).flatten()
    if len(values) != n_obs:
        raise ValueError(f"Got {len(values)} `{name}` values for {n_obs} leafs.")

    units = np.asarray(dn["leaves"], dtype=int)
    if units.size and units.max() >= n_obs:
        # Some of what is drawn is a collapsed clade standing for many
        # observations, so a per-observation value has nothing to attach to.
        raise ValueError(
            f"`{name}` is given per observation and cannot be combined with "
            "`collapse`, which draws whole clades as one wedge."
        )

    return values[units]


def _resolve_leaf_colors(dn, leaf_base, leaf_color, n_obs):
    """Color for each leaf position; defaults to the leaf's own link color."""
    colors = [leaf_base.get(p, (0.0, "k"))[1] for p in range(len(dn["leaves"]))]

    if leaf_color is None:
        return colors

    if mcl.is_color_like(leaf_color):
        return [_normalize_color(leaf_color)] * len(colors)

    per_leaf = _reorder_per_leaf(leaf_color, dn, n_obs, "leaf_color")
    return [_normalize_color(c) for c in per_leaf]


def _unit_positions(dn):
    """Independent-axis coordinate of each visible unit.

    Normally one unit per leaf on a fixed 10-wide grid, but collapsing gives
    wedges a width of their own, so the layout records the positions instead.
    """
    if "_unit_i" in dn:
        return np.asarray(dn["_unit_i"], dtype=float)
    return np.arange(len(dn["leaves"])) * 10.0 + 5.0


def _independent_width(dn):
    """Total extent of the independent axis."""
    if "_ivw" in dn:
        return float(dn["_ivw"])
    return float(len(dn["ivl"]) * 10)


def _to_xy(i, d, orientation):
    """Map independent/dependent coordinates onto x/y for a given orientation.

    scipy only ever flips the axis *limits* for the different orientations - the
    data itself always stays positive - so this is a swap and nothing else.
    """
    if orientation in ("left", "right"):
        return d, i
    return i, d


def _max_distance(dn, wedges=()):
    """Tallest thing on the dependent axis.

    Wedges have to count: collapsing every clade can leave no links at all, and
    the axis would then have nothing to scale to.
    """
    heights = [d[1] for d in dn["dcoord"]]
    heights += [w["height"] for w in wedges]
    return max(heights, default=0.0)


def _axis_limits(dn, orientation, wedges=()):
    """Reproduce the limits scipy would set, so axes stay interchangeable."""
    ivw = _independent_width(dn)
    max_dist = _max_distance(dn, wedges)
    dvw = max_dist * 1.05 or 1.0

    if orientation == "top":
        return (0, ivw), (0, dvw)
    if orientation == "bottom":
        return (0, ivw), (dvw, 0)
    if orientation == "left":
        return (dvw, 0), (0, ivw)
    return (0, dvw), (0, ivw)


def _tick_rotation(n):
    return 0 if n <= 20 else (45 if n <= 40 else 90)


def _tick_font_size(n):
    for limit, size in ((20, 12), (30, 10), (50, 8), (85, 6)):
        if n <= limit:
            return size
    return 5


def _plot_matplotlib(
    dn,
    segments,
    leaf_base,
    leaf_colors,
    leaf_marker,
    leaf_size,
    label_dist,
    dist_fmt,
    orientation,
    polar,
    wedges,
    ax,
    leaf_rotation,
    leaf_font_size,
    no_labels,
):
    """Render the prepared segments with matplotlib."""
    if polar:
        return _plot_polar(
            dn,
            segments,
            leaf_base=leaf_base,
            leaf_colors=leaf_colors,
            leaf_marker=leaf_marker,
            leaf_size=leaf_size,
            label_dist=label_dist,
            dist_fmt=dist_fmt,
            wedges=wedges,
            ax=ax,
            leaf_font_size=leaf_font_size,
            no_labels=no_labels,
        )

    if ax is None:
        ax = plt.gca()

    # Group by style so we draw a handful of collections instead of one line
    # per segment - this matters for trees with thousands of leafs.
    for (color, ls), lines in _group_segments(segments, orientation).items():
        ax.add_collection(LineCollection(lines, colors=color, linestyles=ls))

    xlim, ylim = _axis_limits(dn, orientation, wedges)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    for wedge in wedges:
        apex = _to_xy(wedge["i"], wedge["height"], orientation)
        left = _to_xy(wedge["i"] - wedge["width"] / 2, 0.0, orientation)
        right = _to_xy(wedge["i"] + wedge["width"] / 2, 0.0, orientation)
        ax.add_patch(
            Polygon(
                [apex, left, right],
                closed=True,
                facecolor=wedge["color"],
                edgecolor=wedge["color"],
                alpha=0.65,
                zorder=2,
            )
        )

    _add_leaf_labels(ax, dn, orientation, no_labels, leaf_rotation, leaf_font_size)

    if leaf_marker:
        iv = _unit_positions(dn)
        dv = [leaf_base.get(p, (0.0, None))[0] for p in range(len(dn["leaves"]))]
        xs, ys = _to_xy(iv, np.asarray(dv), orientation)
        ax.scatter(xs, ys, marker=leaf_marker, c=leaf_colors, s=leaf_size, zorder=3)

    if label_dist:
        _add_dist_labels(ax, dn, orientation, dist_fmt)

    return ax


def _group_segments(segments, orientation):
    """Bucket segments by (color, linestyle) for LineCollection."""
    grouped = {}
    for seg in segments:
        x0, y0 = _to_xy(seg["i"][0], seg["d"][0], orientation)
        x1, y1 = _to_xy(seg["i"][1], seg["d"][1], orientation)
        grouped.setdefault((seg["color"], seg["ls"]), []).append([(x0, y0), (x1, y1)])
    return grouped


def _add_leaf_labels(ax, dn, orientation, no_labels, leaf_rotation, leaf_font_size):
    """Place the leaf labels, mirroring scipy's tick placement."""
    n = len(dn["ivl"])
    ticks = _unit_positions(dn)

    if orientation in ("top", "bottom"):
        axis = ax.xaxis
        set_ticks, set_labels = ax.set_xticks, ax.set_xticklabels
        side = "bottom" if orientation == "top" else "top"
    else:
        axis = ax.yaxis
        set_ticks, set_labels = ax.set_yticks, ax.set_yticklabels
        side = "right" if orientation == "left" else "left"

    if no_labels:
        set_ticks([])
        set_labels([])
        return

    set_ticks(ticks)
    axis.set_ticks_position(side)

    font = _tick_font_size(n) if leaf_font_size is None else leaf_font_size
    if orientation in ("top", "bottom"):
        rot = _tick_rotation(n) if leaf_rotation is None else leaf_rotation
        set_labels(dn["ivl"], rotation=rot, size=font)
    elif leaf_rotation is not None:
        set_labels(dn["ivl"], rotation=leaf_rotation, size=font)
    else:
        set_labels(dn["ivl"], size=font)

    # Tick marks would sit on top of the links
    for line in axis.get_ticklines():
        line.set_visible(False)


def _add_dist_labels(ax, dn, orientation, dist_fmt):
    """Label every link with its distance."""
    for ic, dc in zip(dn["icoord"], dn["dcoord"]):
        i = (ic[1] + ic[2]) / 2
        d = dc[1]
        x, y = _to_xy(i, d, orientation)
        ax.text(
            x,
            y,
            dist_fmt.format(d),
            ha="center" if orientation in ("top", "bottom") else "left",
            va="bottom" if orientation in ("top", "bottom") else "center",
            rotation=90 if orientation in ("left", "right") else 0,
            fontsize="x-small",
            clip_on=True,
        )


def _plot_polar(
    dn,
    segments,
    leaf_base,
    leaf_colors,
    leaf_marker,
    leaf_size,
    label_dist,
    dist_fmt,
    wedges,
    ax,
    leaf_font_size,
    no_labels,
):
    """Render the prepared segments as a radial dendrogram."""
    if ax is None:
        ax = plt.gcf().add_subplot(projection="polar")
    elif ax.name != "polar":
        raise ValueError(
            "`polar=True` requires an axis with a polar projection, e.g. "
            'plt.subplot(projection="polar"). Got a "%s" axis.' % ax.name
        )

    ivw = _independent_width(dn)
    max_dist = _max_distance(dn, wedges) or 1.0
    r_max = max_dist * 1.05

    def theta(i):
        return np.asarray(i, dtype=float) / ivw * 2 * np.pi

    def radius(d):
        # The radial axis runs inwards: leafs sit on the outer rim (where their
        # labels are) and the root sits at the centre.
        return r_max - np.asarray(d, dtype=float)

    grouped = {}
    for seg in segments:
        key = (seg["color"], seg["ls"])
        if seg["d"][0] == seg["d"][1] and seg["i"][0] != seg["i"][1]:
            # A horizontal bar spans two angles, so it has to be drawn as an
            # arc - a straight line would cut a chord across the circle.
            ts = np.linspace(seg["i"][0], seg["i"][1], _ARC_RESOLUTION)
            pts = list(zip(theta(ts), np.full(_ARC_RESOLUTION, radius(seg["d"][0]))))
        else:
            pts = list(zip(theta(seg["i"]), radius(seg["d"])))
        grouped.setdefault(key, []).append(pts)

    for (color, ls), lines in grouped.items():
        ax.add_collection(LineCollection(lines, colors=color, linestyles=ls))

    for wedge in wedges:
        # The base of the wedge follows the circle, so it needs interpolating
        # for the same reason the horizontal links do
        ts = np.linspace(
            wedge["i"] - wedge["width"] / 2,
            wedge["i"] + wedge["width"] / 2,
            _ARC_RESOLUTION,
        )
        points = [(theta(wedge["i"]), radius(wedge["height"]))]
        points += [(t, radius(0.0)) for t in theta(ts)]
        ax.add_patch(
            Polygon(
                points,
                closed=True,
                facecolor=wedge["color"],
                edgecolor=wedge["color"],
                alpha=0.65,
                zorder=2,
            )
        )

    ax.set_ylim(0, r_max)

    # Radial ticks still need to read as distances, not as raw radii
    dist_ticks = np.linspace(0, max_dist, 5)
    ax.set_yticks(radius(dist_ticks))
    ax.set_yticklabels([f"{d:.2g}" for d in dist_ticks], fontsize="x-small")
    # Leafs occupy angles 5, 15, ... ivw - 5, so the widest gap between two of
    # them straddles 0. Putting the radial labels there keeps them off the tree.
    ax.set_rlabel_position(0)
    # There is one angular tick per leaf, and hence one grid spoke per leaf,
    # which turns into a solid disc on any sizeable tree. The rings carry the
    # distance scale; the spokes say nothing.
    ax.xaxis.grid(False)

    if no_labels:
        ax.set_xticks([])
    else:
        ax.set_xticks(theta(_unit_positions(dn)))
        ax.set_xticklabels(
            dn["ivl"],
            size=(
                _tick_font_size(len(dn["ivl"]))
                if leaf_font_size is None
                else leaf_font_size
            ),
        )

    if leaf_marker:
        n_leaves = len(dn["leaves"])
        ts = theta(_unit_positions(dn))
        rs = radius([leaf_base.get(p, (0.0, None))[0] for p in range(n_leaves)])
        ax.scatter(ts, rs, marker=leaf_marker, c=leaf_colors, s=leaf_size, zorder=3)

    if label_dist:
        for ic, dc in zip(dn["icoord"], dn["dcoord"]):
            ax.text(
                theta((ic[1] + ic[2]) / 2),
                radius(dc[1]),
                dist_fmt.format(dc[1]),
                ha="center",
                va="bottom",
                fontsize="x-small",
            )

    return ax


def _plot_plotly(
    dn,
    segments,
    leaf_base,
    leaf_colors,
    leaf_marker,
    leaf_size,
    label_dist,
    dist_fmt,
    orientation,
    wedges,
    hover_info,
    fig,
    no_labels,
):
    """Render the prepared segments with plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "Please install the plotly library to use this backend:"
            "\n  pip3 install plotly"
        )

    row = col = None
    if isinstance(fig, dict):
        opts = dict(fig)
        fig = opts.pop("fig", None)
        row = opts.pop("row", None)
        col = opts.pop("col", None)
        if opts:
            raise ValueError(f"Unexpected keys in `fig` dict: {list(opts)}")
    if fig is None:
        fig = go.Figure()

    # plotly takes one color per trace, so group segments and separate the
    # individual lines within a trace using None.
    traces = []
    for (color, ls), lines in _group_segments(segments, orientation).items():
        xs, ys = [], []
        for (x0, y0), (x1, y1) in lines:
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=_to_plotly_color(color), dash=_to_plotly_dash(ls)),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for wedge in wedges:
        apex = _to_xy(wedge["i"], wedge["height"], orientation)
        left = _to_xy(wedge["i"] - wedge["width"] / 2, 0.0, orientation)
        right = _to_xy(wedge["i"] + wedge["width"] / 2, 0.0, orientation)
        traces.append(
            go.Scatter(
                x=[apex[0], left[0], right[0], apex[0]],
                y=[apex[1], left[1], right[1], apex[1]],
                mode="lines",
                fill="toself",
                fillcolor=_to_plotly_color(wedge["color"]),
                opacity=0.65,
                line=dict(color=_to_plotly_color(wedge["color"])),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if leaf_marker or hover_info is not None:
        n_leaves = len(dn["leaves"])
        iv = _unit_positions(dn)
        dv = np.asarray([leaf_base.get(p, (0.0, None))[0] for p in range(n_leaves)])
        xs, ys = _to_xy(iv, dv, orientation)
        text = list(dn["ivl"])
        if hover_info is not None:
            text = [f"{lb}<br>{hv}" for lb, hv in zip(text, hover_info)]
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    color=[_to_plotly_color(c) for c in leaf_colors],
                    size=leaf_size if leaf_size is not None else 6,
                ),
                text=text,
                hoverinfo="text",
                showlegend=False,
            )
        )

    if row is not None and col is not None:
        fig.add_traces(traces, rows=[row] * len(traces), cols=[col] * len(traces))
    else:
        fig.add_traces(traces)

    _layout_plotly(fig, dn, orientation, no_labels, row, col, wedges)

    if label_dist:
        for ic, dc in zip(dn["icoord"], dn["dcoord"]):
            x, y = _to_xy((ic[1] + ic[2]) / 2, dc[1], orientation)
            fig.add_annotation(
                x=x,
                y=y,
                text=dist_fmt.format(dc[1]),
                showarrow=False,
                font=dict(size=9),
                yshift=6,
                **({"row": row, "col": col} if row is not None else {}),
            )

    return fig


def _layout_plotly(fig, dn, orientation, no_labels, row, col, wedges=()):
    """Configure the plotly axes to match the matplotlib output."""
    xlim, ylim = _axis_limits(dn, orientation, wedges)

    base = dict(
        ticks="outside",
        zeroline=False,
        showgrid=False,
        showline=True,
    )
    # scipy signals orientation by flipping the axis limits; plotly wants an
    # explicit `autorange="reversed"` instead.
    xaxis = dict(
        base, range=sorted(xlim), autorange="reversed" if xlim[0] > xlim[1] else None
    )
    yaxis = dict(
        base, range=sorted(ylim), autorange="reversed" if ylim[0] > ylim[1] else None
    )

    if no_labels:
        (xaxis if orientation in ("top", "bottom") else yaxis)["showticklabels"] = False
    else:
        label_axis = xaxis if orientation in ("top", "bottom") else yaxis
        label_axis["tickmode"] = "array"
        label_axis["tickvals"] = _unit_positions(dn).tolist()
        label_axis["ticktext"] = list(dn["ivl"])
        if orientation == "bottom":
            xaxis["side"] = "top"
        elif orientation == "right":
            yaxis["side"] = "left"
        elif orientation == "left":
            yaxis["side"] = "right"

    if row is not None and col is not None:
        fig.update_xaxes(row=row, col=col, **xaxis)
        fig.update_yaxes(row=row, col=col, **yaxis)
    else:
        fig.update_layout(xaxis=xaxis, yaxis=yaxis)

    fig.update_layout(showlegend=False, autosize=True, hovermode="closest")


def _to_plotly_color(c):
    """Convert any matplotlib color spec into something plotly understands."""
    try:
        return mcl.to_hex(c)
    except ValueError:
        return c


def _to_plotly_dash(ls):
    """Translate a matplotlib line style into a plotly dash style."""
    return {
        "-": "solid",
        "solid": "solid",
        "--": "dash",
        "dashed": "dash",
        ":": "dot",
        "dotted": "dot",
        "-.": "dashdot",
        "dashdot": "dashdot",
    }.get(ls, "solid")
