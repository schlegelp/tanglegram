"""Regenerate the dendrogram images used in the README.

Each block below is copied verbatim into the README, so that every example
there is known to run and to produce the image next to it. Run with:

    python docs/make_readme_images.py

"""

import pathlib
import sys

HERE = pathlib.Path(__file__).parent
IMAGES = HERE / "images"

# Document the code in this repo, not whatever happens to be pip-installed.
# The imports below deliberately come after this and after picking the Agg
# backend, hence the E402 exemptions.
sys.path.insert(0, str(HERE.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.cluster.hierarchy as sch  # noqa: E402
import scipy.spatial.distance as ssd  # noqa: E402

import tanglegram as tg  # noqa: E402


def save(fig, name, tight=True):
    IMAGES.mkdir(parents=True, exist_ok=True)
    # The annotation strip uses axes_grid1, which tight_layout cannot handle
    if tight:
        fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {IMAGES / name}")


# ------------------------------------------------------- 0. intro tanglegrams
# The three examples at the top of the README.
labelsA = ["A", "B", "C", "D"]
labelsB = ["B", "A", "C", "D"]
data = [
    [0, 0.1, 0.4, 0.3],
    [0.1, 0, 0.5, 0.6],
    [0.4, 0.5, 0, 0.2],
    [0.3, 0.6, 0.2, 0],
]

mat1 = pd.DataFrame(data, columns=labelsA, index=labelsA)
mat2 = pd.DataFrame(data, columns=labelsB, index=labelsB)

fig, dn1, dn2 = tg.tanglegram(mat1, mat2, sort=False)
save(fig, "tangle_unsorted.png", tight=False)

fig, dn1, dn2 = tg.tanglegram(mat1, mat2, sort=True)
save(fig, "tangle_sorted.png", tight=False)

fig, dn1, dn2 = tg.tanglegram(mat1, mat2, sort=True, edges=[("A", "A"), ("C", "D")])
save(fig, "tangle_edges.png", tight=False)


# --------------------------------------------------------------- shared data
# Four groups of six samples, with enough overlap that the tree does not
# recover the groups perfectly - which is what makes `clusters` interesting.
rng = np.random.default_rng(4)
centers = np.array([[0, 0], [3, 0.5], [1.5, 3], [4.5, 3.5]], dtype=float)
groups = np.repeat(np.arange(4), 6)
X = centers[groups] + rng.normal(0, 0.9, (24, 2))

Z = sch.linkage(X, method="ward")
group_names = np.array(["A", "B", "C", "D"])[groups]


# --------------------------------------------------------------- 1. clusters
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)

sch.dendrogram(Z, color_threshold=0.7 * Z[:, 2].max(), ax=axes[0])
axes[0].set_title("scipy: color_threshold")

tg.dendrogram(Z, clusters=group_names, ax=axes[1])
axes[1].set_title("tanglegram: clusters=")

save(fig, "dend_clusters.png")


# ------------------------------------------------------- 2. partial clusters
# Highlight two clades and leave the rest unassigned. Anything that is
# None/NaN is drawn in `above_threshold_color`.
cut = sch.fcluster(Z, 4, criterion="maxclust")
highlight = np.where(np.isin(cut, [1, 3]), cut, None)

fig, ax = plt.subplots(figsize=(8, 3.6))
tg.dendrogram(
    Z,
    clusters=highlight,
    cluster_colors={1: "#d62728", 3: "#1f77b4"},
    above_threshold_color="lightgrey",
    above_threshold_ls="--",
    ax=ax,
)
save(fig, "dend_partial.png")


# ------------------------------------------------------------- 3. hanging
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

tg.dendrogram(Z, clusters=group_names, hanging=True, leaf_marker="o", ax=axes[0])
axes[0].set_title("hanging=True (5% of the tree height)")

tg.dendrogram(Z, clusters=group_names, hanging=0.02, leaf_marker="o", ax=axes[1])
axes[1].set_title("hanging=0.02")

save(fig, "dend_hanging.png")


# --------------------------------------------------------- 4. link distances
small = sch.linkage(X[:8], method="ward")

fig, ax = plt.subplots(figsize=(7, 3.4))
tg.dendrogram(small, label_dist=True, dist_fmt="{:.2f}", ax=ax)
save(fig, "dend_labels.png")


# ---------------------------------------------------------------- 5. polar
fig = plt.figure(figsize=(5.5, 5.5))
tg.dendrogram(Z, clusters=group_names, polar=True, leaf_marker="o", leaf_size=20)
save(fig, "dend_polar.png")


# ------------------------------------------------------------ 6. annotations
metadata = pd.DataFrame(
    {
        "group": group_names,
        "batch": np.tile(["b1", "b2", "b3"], 8),
        "qc": np.where(rng.random(24) > 0.25, "pass", "fail"),
    }
)

fig, ax = plt.subplots(figsize=(9, 4))
tg.dendrogram(
    Z,
    clusters=group_names,
    leaf_annot=metadata,
    annot_colors={"pass": "#4caf50", "fail": "#e53935"},
    ax=ax,
)
save(fig, "dend_annot.png", tight=False)


# --------------------------------------------------------------- 7. collapse
big = sch.linkage(rng.random((150, 4)), method="ward")
big_clusters = sch.fcluster(big, 6, criterion="maxclust")

fig, axes = plt.subplots(2, 1, figsize=(9, 6))

tg.dendrogram(big, clusters=big_clusters, no_labels=True, ax=axes[0])
axes[0].set_title("150 leafs, every one drawn")

tg.dendrogram(
    big, clusters=big_clusters, collapse="clusters", collapse_width=3, ax=axes[1]
)
axes[1].set_title('collapse="clusters"')

save(fig, "dend_collapse.png")


# ----------------------------------------------------- 8. tanglegram colours
labels = [f"s{i}" for i in range(24)]
dist = pd.DataFrame(ssd.squareform(ssd.pdist(X)), index=labels, columns=labels)
dist2 = pd.DataFrame(
    ssd.squareform(ssd.pdist(X + rng.normal(0, 0.6, X.shape))),
    index=labels,
    columns=labels,
)

fig, dn1, dn2 = tg.tanglegram(
    dist,
    dist2,
    sort=True,
    figsize=(9, 6),
    clusters_left=group_names,
    clusters_right=group_names,
    color_edges_by="left",
)
save(fig, "tangle_clusters.png")


# ---------------------------------------------------------------- 9. plotly
try:
    dn, fig = tg.dendrogram(
        Z,
        clusters=group_names,
        hover_info=[f"sample {i}" for i in range(len(X))],
        leaf_marker="o",
        hanging=0.02,
        backend="plotly",
    )
    fig.update_layout(width=800, height=360, margin=dict(l=40, r=20, t=20, b=40))
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.write_image(IMAGES / "dend_plotly.png")
    print(f"wrote {IMAGES / 'dend_plotly.png'}")
except Exception as exc:  # pragma: no cover - needs plotly + kaleido
    print(f"skipped plotly image: {exc}")
