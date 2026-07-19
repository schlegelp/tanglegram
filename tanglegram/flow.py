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

"""Alluvial view of how cluster membership changes between clusterings.

A tanglegram draws one line per leaf, which stops working somewhere around a
few hundred leafs - the lines simply fill the space. Often the question is not
"where did this leaf go" but "did this cluster survive", and that is a question
about groups, not individuals. This module answers it with ribbons whose width
is the number of observations moving from one cluster to the next.
"""

import numpy as np

import matplotlib.colors as mcl
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon, Rectangle

from .dend import cluster_color_map, _is_null, _hashable


__all__ = ["cluster_flow"]

#: Points along each side of a ribbon.
_RIBBON_RESOLUTION = 48


def cluster_flow(
    clusters,
    names=None,
    colors=None,
    share_colors=True,
    color_by="source",
    block_width=0.05,
    gap=0.02,
    alpha=0.55,
    sort=True,
    label_clusters=True,
    min_label_size=0.03,
    ax=None,
    figsize=(9, 5),
):
    """Draw how observations move between two or more clusterings.

    Each clustering becomes a column of blocks, one per cluster, with height
    proportional to the number of observations in it. Ribbons between the
    columns carry the observations that move from one cluster to another, and
    their width is how many.

    Unlike a tanglegram this does not care how many observations there are -
    only how many clusters - so it stays readable where per-leaf lines do not.

    Parameters
    ----------
    clusters :      list of list-like | pandas.DataFrame
                    Two or more cluster assignments over the *same*
                    observations, in order. A DataFrame is read column by
                    column, and its column names are used as ``names``.
                    ``None``/NaN entries are dropped from that column's blocks
                    and from any ribbon touching them.
    names :         list of str, optional
                    A name for each clustering, shown along the x axis.
    colors :        dict | list-like, optional
                    Colors for the clusters. Either a ``{cluster: color}``
                    mapping or a sequence cycled in order of first appearance.
                    All columns share one mapping, so a cluster that appears in
                    several keeps its color.
    share_colors :  bool
                    If True, a cluster label means the same thing in every
                    column and gets one color throughout. Turn this off when
                    the labels are arbitrary per column - ``fcluster`` at
                    different ``k``, say, where "cluster 1" in one column has
                    nothing to do with "cluster 1" in the next - and each
                    column will be colored on its own instead.
    color_by :      "source" | "target"
                    Whether a ribbon takes the color of the cluster it leaves
                    or the one it arrives at.
    block_width :   float
                    Width of the cluster blocks, as a fraction of the spacing
                    between two columns.
    gap :           float
                    Vertical space between the blocks of one column, as a
                    fraction of the column height.
    alpha :         float
                    Opacity of the ribbons.
    sort :          bool
                    If True, order the blocks in each column to sit near the
                    clusters they came from, which cuts down on crossings.
                    If False, order them by first appearance.
    label_clusters : bool
                    Whether to write the cluster name on each block.
    min_label_size : float
                    Skip the label on any block shorter than this fraction of
                    the column height.
    ax :            matplotlib.axes.Axes, optional
    figsize :       tuple
                    Only used when ``ax`` is not given.

    Returns
    -------
    flows :         list of dict
                    One entry per ribbon, with "step", "source", "target" and
                    "count" - the observations moving between that pair.
    ax :            matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.cluster.hierarchy as sch
    >>> import tanglegram as tg
    >>> Z = sch.linkage(np.random.rand(200, 4), method="ward")
    >>> cuts = [sch.fcluster(Z, k, criterion="maxclust") for k in (2, 4, 8)]
    >>> flows, ax = tg.cluster_flow(cuts, names=["k=2", "k=4", "k=8"])

    """
    stages, names = _normalize_stages(clusters, names)

    if color_by not in ("source", "target"):
        raise ValueError(f'`color_by` must be "source" or "target", got "{color_by}"')

    color_map = _color_map(stages, colors, share_colors)

    order = _order_blocks(stages, sort)
    blocks = _place_blocks(stages, order, gap)
    flows = _count_flows(stages)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    _draw_blocks(ax, blocks, color_map, block_width)
    _draw_ribbons(ax, blocks, flows, order, color_map, color_by, block_width, alpha)

    if label_clusters:
        _label_blocks(ax, blocks, color_map, min_label_size)

    ax.set_xlim(-0.5 * block_width - 0.05, len(stages) - 1 + 0.5 * block_width + 0.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(names)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    return flows, ax


def _color_map(stages, colors, share_colors):
    """Color for every (column, cluster) pair."""
    if share_colors:
        shared = cluster_color_map(*stages, colors=colors)
        return {
            (step, cl): shared[cl]
            for step, stage in enumerate(stages)
            for cl in shared
            if cl in set(stage)
        }

    color_map = {}
    for step, stage in enumerate(stages):
        own = cluster_color_map(stage, colors=colors)
        color_map.update({(step, cl): c for cl, c in own.items()})
    return color_map


def _normalize_stages(clusters, names):
    """Bring the accepted input shapes into a list of object arrays."""
    if hasattr(clusters, "columns") and hasattr(clusters, "values"):
        if names is None:
            names = [str(c) for c in clusters.columns]
        stages = [np.asarray(clusters[c], dtype=object) for c in clusters.columns]
    else:
        stages = [np.asarray(c, dtype=object).flatten() for c in clusters]

    if len(stages) < 2:
        raise ValueError(f"Need at least 2 clusterings to compare, got {len(stages)}.")

    sizes = {len(s) for s in stages}
    if len(sizes) != 1:
        raise ValueError(
            f"Every clustering must cover the same observations, got sizes {sorted(sizes)}."
        )

    if names is None:
        names = [str(i + 1) for i in range(len(stages))]
    elif len(names) != len(stages):
        raise ValueError(f"Got {len(names)} names for {len(stages)} clusterings.")

    # Normalise the labels themselves so they can be used as dict keys
    stages = [
        np.asarray([None if _is_null(v) else _hashable(v) for v in s], dtype=object)
        for s in stages
    ]

    return stages, list(names)


def _order_blocks(stages, sort):
    """Decide the top-to-bottom order of the clusters in each column.

    Without sorting this is just first appearance. With sorting, each column
    after the first is arranged by the average position of where its members
    came from, which is the usual way to keep a Sankey's ribbons from crossing
    more than they have to.
    """
    orders = []
    for step, stage in enumerate(stages):
        present = []
        for value in stage:
            if value is not None and value not in present:
                present.append(value)

        if not sort or step == 0:
            orders.append(present)
            continue

        previous = orders[-1]
        rank = {cl: i for i, cl in enumerate(previous)}
        prev_stage = stages[step - 1]

        def barycentre(cl, prev_stage=prev_stage, stage=stage, rank=rank):
            came_from = [
                rank[p]
                for p, c in zip(prev_stage, stage)
                if c == cl and p is not None and p in rank
            ]
            # Anything with no history goes to the bottom
            return np.mean(came_from) if came_from else len(rank)

        orders.append(sorted(present, key=barycentre))

    return orders


def _place_blocks(stages, orders, gap):
    """Work out the vertical extent of every block.

    Returns ``blocks[step][cluster] = (bottom, top)``, with each column scaled
    to fill the same height so the columns stay comparable.
    """
    blocks = []
    for step, (stage, order) in enumerate(zip(stages, orders)):
        counts = {cl: int(np.sum(stage == cl)) for cl in order}
        total = sum(counts.values())
        if not total:
            raise ValueError(f"Clustering {step + 1} has no assigned observations.")

        # Gaps eat into the column, the blocks share what is left
        spare = gap * max(len(order) - 1, 0)
        usable = max(1.0 - spare, 0.1)

        placed, cursor = {}, 1.0
        for cl in order:
            height = usable * counts[cl] / total
            placed[cl] = (cursor - height, cursor)
            cursor -= height + gap
        blocks.append(placed)

    return blocks


def _count_flows(stages):
    """How many observations move between each pair of clusters."""
    flows = []
    for step, (before, after) in enumerate(zip(stages[:-1], stages[1:])):
        counts = {}
        for a, b in zip(before, after):
            # An observation with no cluster on either side carries no flow
            if a is None or b is None:
                continue
            counts[(a, b)] = counts.get((a, b), 0) + 1

        flows.extend(
            {"step": step, "source": a, "target": b, "count": n}
            for (a, b), n in counts.items()
        )

    return flows


def _draw_blocks(ax, blocks, color_map, block_width):
    for step, placed in enumerate(blocks):
        for cl, (bottom, top) in placed.items():
            ax.add_patch(
                Rectangle(
                    (step - block_width / 2, bottom),
                    block_width,
                    top - bottom,
                    facecolor=color_map[(step, cl)],
                    edgecolor="none",
                    zorder=3,
                )
            )


def _draw_ribbons(ax, blocks, flows, orders, color_map, color_by, block_width, alpha):
    """Draw one ribbon per flow, stacked inside the blocks it joins."""
    # Within a block, the outgoing slices are stacked in the order of the
    # blocks they run to (and vice versa), which keeps the ribbons from
    # needlessly crossing each other in the gap.
    for step in range(len(blocks) - 1):
        here = [f for f in flows if f["step"] == step]
        target_rank = {cl: i for i, cl in enumerate(orders[step + 1])}
        source_rank = {cl: i for i, cl in enumerate(orders[step])}

        out_cursor = {cl: top for cl, (_, top) in blocks[step].items()}
        in_cursor = {cl: top for cl, (_, top) in blocks[step + 1].items()}

        totals_out = {}
        totals_in = {}
        for f in here:
            totals_out[f["source"]] = totals_out.get(f["source"], 0) + f["count"]
            totals_in[f["target"]] = totals_in.get(f["target"], 0) + f["count"]

        for f in sorted(
            here, key=lambda f: (source_rank[f["source"]], target_rank[f["target"]])
        ):
            src, tgt, count = f["source"], f["target"], f["count"]

            src_bottom, src_top = blocks[step][src]
            tgt_bottom, tgt_top = blocks[step + 1][tgt]

            # A block may have lost observations to NaN, so scale by what
            # actually flows rather than by the block's own size
            src_span = (src_top - src_bottom) * count / totals_out[src]
            tgt_span = (tgt_top - tgt_bottom) * count / totals_in[tgt]

            y0_top, y0_bot = out_cursor[src], out_cursor[src] - src_span
            y1_top, y1_bot = in_cursor[tgt], in_cursor[tgt] - tgt_span
            out_cursor[src] -= src_span
            in_cursor[tgt] -= tgt_span

            x0 = step + block_width / 2
            x1 = step + 1 - block_width / 2

            top_x, top_y = _sigmoid(x0, y0_top, x1, y1_top)
            bot_x, bot_y = _sigmoid(x0, y0_bot, x1, y1_bot)

            points = list(zip(top_x, top_y)) + list(zip(bot_x[::-1], bot_y[::-1]))
            ax.add_patch(
                Polygon(
                    points,
                    closed=True,
                    facecolor=color_map[
                        (step, src) if color_by == "source" else (step + 1, tgt)
                    ],
                    edgecolor="none",
                    alpha=alpha,
                    zorder=2,
                )
            )


def _sigmoid(x0, y0, x1, y1):
    """A cubic Bezier that leaves and arrives horizontally."""
    t = np.linspace(0, 1, _RIBBON_RESOLUTION)
    mid = (x0 + x1) / 2

    x = (
        (1 - t) ** 3 * x0
        + 3 * (1 - t) ** 2 * t * mid
        + 3 * (1 - t) * t**2 * mid
        + t**3 * x1
    )
    y = (
        (1 - t) ** 3 * y0
        + 3 * (1 - t) ** 2 * t * y0
        + 3 * (1 - t) * t**2 * y1
        + t**3 * y1
    )
    return x, y


def _label_blocks(ax, blocks, color_map, min_label_size):
    for step, placed in enumerate(blocks):
        for cl, (bottom, top) in placed.items():
            if top - bottom < min_label_size:
                continue
            ax.text(
                step,
                (bottom + top) / 2,
                str(cl),
                ha="center",
                va="center",
                rotation=90,
                fontsize="x-small",
                color=_readable_on(color_map[(step, cl)]),
                zorder=4,
            )


def _readable_on(color):
    """Black or white, whichever stands out on the given background."""
    r, g, b = mcl.to_rgb(color)
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"
