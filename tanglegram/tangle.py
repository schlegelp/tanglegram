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
#
#    You should have received a copy of the GNU General Public License
#    along


import matplotlib.pyplot as plt
import scipy.cluster as sclust
import scipy.spatial.distance as sdist
import numpy as np
import pandas as pd
import pylab
import math
import logging


__all__ = ['gen_tangle', 'get_entanglement']

# Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
if not module_logger.handlers:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
                '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)


def gen_tangle(a, b, labelsA=None, labelsB=None, optimize_order=True,
               color_by_diff=True, link_kwargs={}, dend_kwargs={}):
    """Plot a tanglegram from two dendrograms.

    Parameters
    ----------
    (a,b) :                 pandas.DataFrame | scipy.cluster.hierarchy.linkage
                            Dendrograms to be compared. If DataFrame, will be
                            considered a distance matrix and linkage is
                            generated (see ``link_kwargs``).
    (labelsA,labelsB) :     list of str
                            If not provided and a/b are pandas Dataframes,
                            will try to extract from columns.
    optimize_order :        bool
                            If True, will try rearranging dendrogram to
                            optimise pairing of similar values.
    link_kwargs :           dict, optional
                            Keyword arguments to be passed on to ``scipy.cluster.hierarchy.linkage``
    dend_kwargs :           dict, optional
                            Keyword arguments to be passed on to ``scipy.cluster.hierarchy.dendrogram``

    Returns
    -------
    matplotlib figure

    """
    plt.style.use('ggplot')

    if isinstance(a, pd.DataFrame):
        module_logger.info('Generating linkage from distances')
        link1 = sclust.hierarchy.linkage(sdist.squareform(a, checks=False), **link_kwargs)
        if not labelsA:
            labelsA = a.columns.tolist()
    elif isinstance(a, np.ndarray):
        link1 = a
    else:
        raise TypeError('Parameter `a` needs to be either pandas DataFrame or numpy array')

    if isinstance(b, pd.DataFrame):
        module_logger.info('Generating linkage from distances')
        link2 = sclust.hierarchy.linkage(sdist.squareform(b, checks=False), **link_kwargs)
        if not labelsB:
            labelsB = b.columns.tolist()
    elif isinstance(b, np.ndarray):
        link2 = b
    else:
        raise TypeError('Parameter `b` needs to be either pandas DataFrame or numpy array')

    if optimize_order:
        link1, link2, save_entang = _optimize_leaf_order(link1, link2, labelsA, labelsB)

    fig = pylab.figure(figsize=(8, 8))

    # Compute and plot left dendrogram.
    ax1 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    Z1 = sclust.hierarchy.dendrogram(link1, orientation='left', labels=labelsA, **dend_kwargs)
    #ax1.set_xticks([])
    #ax1.set_yticks([])

    # Compute and plot right dendrogram.
    ax2 = fig.add_axes([0.7, 0.1, 0.25, 0.8]) #[0.3, 0.71, 0.6, 0.2])
    Z2 = sclust.hierarchy.dendrogram(link2, labels=labelsB, orientation='right', **dend_kwargs)
    #ax2.set_xticks([])
    #ax2.set_yticks([])

    missing = list(set([l for l in Z1['ivl'] if l not in Z2['ivl']] + [l for l in Z2['ivl'] if l not in Z1['ivl']]))
    if any(missing):
        module_logger.warning('Labels {0} do not exist in both dendrograms'.format(missing))

    # Generate middle plot with connecting lines
    ax3 = fig.add_axes([0.4, 0.1, 0.2, 0.8])
    ax3.axis('off')
    ax3.set_xlim((0, 1))

    # Get min and max y dimensions
    max_y = max(ax1.viewLim.y1, ax2.viewLim.y1)
    min_y = min(ax1.viewLim.y0, ax2.viewLim.y0)

    # Make sure labels of both dendrograms have the same font size
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=8)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=8)

    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=8)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=8)

    # Make sure all y axes have same resolution
    for _ in [ax3]: #[ax1,ax2,ax3]:
        _.set_ylim((min_y, max_y))

    # Now iterate over all left leaves
    for ix_l, l in enumerate(Z1['ivl']):
        # Skip if no corresponding element
        if l not in Z2['ivl']:
            continue

        ix_r = Z2['ivl'].index(l)

        coords_l = (ax3.viewLim.y1 - ax3.viewLim.y0) / (len(Z1['leaves'])) * (ix_l + .5)
        coords_r = (ax3.viewLim.y1 - ax3.viewLim.y0) / (len(Z2['leaves'])) * (ix_r + .5)

        if not color_by_diff:
            c = 'black'
        else:
            v = max(round(.75 - math.fabs(ix_l - ix_r) / len(Z1['ivl']), 2), 0)
            c = (v, v, v)

        ax3.plot([0, 1], [coords_l, coords_r], '-', linewidth=1, c=c)

    module_logger.info('Done. Use matplotlib.pyplot.show() to show plot.')

    return fig


def rotate(linkage, i, copy=True):
    """Rotate linkage at given hinge."""
    if copy:
        linkage = linkage.copy()
    # Rotate
    linkage[i] = [linkage[i, 1], linkage[i, 0], linkage[i, 2], linkage[i, 3]]

    return linkage


def get_all_linkage(linkage, li_MID):
    """Generate all possible combinations of rotations for a given linkage.

    Parameters
    ----------
    linkage :       scipy.cluster.hierarchy.linkage
    li_MID :        int
                    Index of the linkage at which to stop rotating.

    """
    length = len(linkage)
    linkage = linkage.reshape(-1, length, 4)
    i = length - 1
    while i >= li_MID:
        for item in linkage:
            new = item.copy()
            new[i] = [new[i, 1], new[i, 0], new[i, 2], new[i, 3]]
            linkage = np.append(linkage, new)
            linkage = linkage.reshape(-1, length, 4)
        i -= 1
    return linkage


def bottom_up(li_MID, link1, link2, min_entang, labels1, labels2):
    """Rotate dendrogram from bottom to "li_MID" and find smallest entanglement."""
    # Find leafs and entanglement of start position
    leafs1 = sclust.hierarchy.leaves_list(link1)
    leafs2 = sclust.hierarchy.leaves_list(link2)
    lindex1 = dict(zip(labels1, leafs1))
    lindex2 = dict(zip(labels2, leafs2))
    min_entang = get_entanglement(lindex1, lindex2)

    # No go over each hinge/knot from bottom to "li_MID" and rotate it
    for i in range(li_MID):
        # Rotate left and right linkage
        new1 = rotate(link1, i)
        new2 = rotate(link2, i)

        # Generate leafs for the new variants
        leafsn1 = sclust.hierarchy.leaves_list(new1)
        leafsn2 = sclust.hierarchy.leaves_list(new2)
        lindexn1 = dict(zip(labels1, leafsn1))
        lindexn2 = dict(zip(labels2, leafsn2))

        # Now test pairwise entanglement
        for j, lx1 in zip([link1, new1],
                          [lindex1, lindexn1]):
            for k, lx2 in zip([link2, new2],
                              [lindex2, lindexn2]):
                new_entang = get_entanglement(lx1, lx2)

                if new_entang < min_entang:
                    min_entang = new_entang
                    link1 = j
                    link2 = k
                    lindex1 = lx1
                    lindex2 = lx2

    return link1, link2, min_entang


def refine(best_linkage1, best_linkage2, min_entang, labels1, labels2):
    """Refine rotation to maximize horizontal lines."""
    leafs1 = sclust.hierarchy.leaves_list(best_linkage1)
    leafs2 = sclust.hierarchy.leaves_list(best_linkage2)
    lindex1 = dict(zip(labels1, leafs1))
    lindex2 = dict(zip(labels2, leafs2))

    for k, z in lindex1.items():
        if z != lindex2[k]:
            find1 = lindex1[k]
            find2 = lindex2[k]
            for num, item in enumerate(best_linkage1):
                if item[0] == find1 or item[1] == find1:
                    knot1 = num
            for num, item in enumerate(best_linkage2):
                if item[0] == find2 or item[1] == find2:
                    knot2 = num
            new1 = rotate(best_linkage1, knot1)
            new2 = rotate(best_linkage2, knot2)
            all1 = np.append([best_linkage1], [new1], axis=0)
            all2 = np.append([best_linkage2], [new2], axis=0)

            all1_lindices = []
            for j in all1:
                leafs1 = sclust.hierarchy.leaves_list(j)
                all1_lindices.append(dict(zip(labels1, leafs1)))

            all2_lindices = []
            for k in all2:
                leafs2 = sclust.hierarchy.leaves_list(k)
                all2_lindices.append(dict(zip(labels2, leafs2)))

            for j, lindex1 in zip(all1, all1_lindices):
                for k, lindex2 in zip(all2, all2_lindices):
                    new_entang = get_entanglement(lindex1, lindex2)
                    if new_entang < min_entang:
                        min_entang = new_entang
                        best_linkage1 = j
                        best_linkage2 = k

    return best_linkage1, best_linkage2, min_entang


def _optimize_leaf_order(linkage_matrix1, linkage_matrix2, labels1, labels2):
    """Optimizes leaf order of linkage 1 and 2 to best match each other.

    Parameters
    ----------
    linkage_matrix1 :     scipy.cluster.hierarchy.linkage
                          First of the two linkages to be matched.
    linkage_matrix2 :     scipy.cluster.hierarchy.linkage
                          Second of the two linkages to be matched.
    labels1, labels2 :    lists of str

    Returns
    -------
    optimized linkage1
    optimized linkage2
    entanglement

    """
    final_entang = float('inf')
    cond = 0

    for li_MID in range(len(linkage_matrix1) - 1, 1, -1):
        old_entang = final_entang
        # Generate all possible combinations for rotating the dendrogram
        # at the hinges up to the given height
        all_linkage1 = get_all_linkage(linkage_matrix1, li_MID)
        all_linkage2 = get_all_linkage(linkage_matrix2, li_MID)

        # Calculate label indices for all variants
        all_lindex1 = []
        for i in all_linkage1:
            leafs1 = sclust.hierarchy.leaves_list(i)
            all_lindex1.append(dict(zip(labels1, leafs1)))

        all_lindex2 = []
        for j in all_linkage2:
            leafs2 = sclust.hierarchy.leaves_list(j)
            all_lindex2.append(dict(zip(labels2, leafs2)))

        for i, lindex1 in zip(all_linkage1, all_lindex1):
            for j, lindex2 in zip(all_linkage2, all_lindex2):
                best_linkage1 = i
                best_linkage2 = j

                min_entang = get_entanglement(lindex1, lindex2)

                if min_entang < final_entang:
                    final_linkage1 = best_linkage1
                    final_linkage2 = best_linkage2
                    final_entang = min_entang

        # Convergence condition
        if old_entang == final_entang:
            cond += 1
        old_entang = final_entang

        # Stop if entanglement has not improved twice
        if cond == 2:
            break
            
    # See if we can squeeze out even more
    old_entang = final_entang
    while True:
        # Coarse optimization
        (final_linkage1,
         final_linkage2,
         final_entang) = bottom_up(li_MID,
                                   final_linkage1, final_linkage2,
                                   final_entang,
                                   labels1, labels2)
        # Stop if no more improvement
        if final_entang >= old_entang:
            break
        old_entang = final_entang

    # Fine optimization
    (final_linkage1,
     final_linkage2,
     final_entang) = refine(final_linkage1, final_linkage2,
                            final_entang,
                            labels1, labels2)

    module_logger.info('Finished optimising at entanglement {0}'.format(final_entang))
    return final_linkage1, final_linkage2, []


def get_entanglement(lindex1, lindex2):
    """Calculage average displacement of leafs in dendogram 1 and 2.

    Ignores leafs that aren't present in both dendrograms.
    """
    assert isinstance(lindex1, dict)
    assert isinstance(lindex2, dict)

    exist_in_both = list(set(lindex1) & set(lindex2))

    if not exist_in_both:
        raise ValueError('Not a single matching label in both dendrograms.')

    return sum([math.fabs(lindex1[l] - lindex2[l]) for l in exist_in_both]) / len(exist_in_both)


if __name__ == '__main__':
    labelsA = ['A', 'B', 'C', 'D']
    labelsB = ['B', 'A', 'C', 'D']
    data = [[ 0,  .1,  .4, .3],
            [.1,   0,  .5, .6],
            [ .4, .5,   0, .2],
            [ .3, .6,  .2,  0]]

    mat1 = pd.DataFrame(data,
                        columns=labelsA,
                        index=labelsA)

    mat2 = pd.DataFrame(data,
                        columns=labelsB,
                        index=labelsB)

    # Plot tanglegram
    fig = gen_tangle(mat1, mat2)
    plt.show()
