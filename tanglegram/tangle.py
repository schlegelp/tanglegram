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

from itertools import product
from tqdm import trange, tqdm


__all__ = ['plot', 'entanglement', 'untangle']

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


def plot(a, b, labelsA=None, labelsB=None, sort=True,
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
    sort :                  bool | "random" | "step1side" | "step2side" | "permuations"
                            If True, will try rearranging dendrogram to
                            optimise pairing of similar values. You can provide
                            the exact method to use as a string. ``True``
                            defaults to "random". See ``untangle()`` for a
                            full description.
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

    if sort:
        if not isinstance(sort, str):
            sort = 'random'
        link1, link2 = untangle(link1, link2,
                                labelsA, labelsB,
                                method=sort)

    fig = pylab.figure(figsize=(8, 8))

    # Compute and plot left dendrogram.
    ax1 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    Z1 = sclust.hierarchy.dendrogram(link1, orientation='left', labels=labelsA, **dend_kwargs)

    # Compute and plot right dendrogram.
    ax2 = fig.add_axes([0.7, 0.1, 0.25, 0.8])  # [0.3, 0.71, 0.6, 0.2])
    Z2 = sclust.hierarchy.dendrogram(link2, labels=labelsB, orientation='right', **dend_kwargs)

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
    for _ in [ax3]:  # [ax1,ax2,ax3]:
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
    # Make copy
    if copy:
        linkage = linkage.copy()

    # Rotate
    linkage[i][0], linkage[i][1] = linkage[i][1], linkage[i][0]

    return linkage


def get_all_linkage(linkage, li_MID):
    """Generate all possible combinations of rotations for a given linkage.

    Parameters
    ----------
    linkage :       scipy.cluster.hierarchy.linkage
    li_MID :        int
                    Index (from the top) of the linkage at which to stop rotating.

    """
    length = len(linkage)
    permutations = linkage.reshape(-1, length, 4)
    i = length - 1
    while i >= li_MID:
        for item in permutations:
            # Make copy
            new = item.copy()
            # Rotate
            new[i][0], new[i][1] = new[i][1], new[i][0]
            # Append this permutation
            permutations = np.append(permutations, new)
            permutations = permutations.reshape(-1, length, 4)
        i -= 1
    return permutations


class CachedGenerator:
    """Caches result of generator for re-use."""
    def __init__(self, generator):
        self.generator = generator
        self._cache = []

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > (len(self._cache) - 1):
            self._cache.append(next(self.generator))
        value = self._cache[self.n]
        self.n += 1
        return value


def get_all_linkage_gen(linkage, li_MID, labels):
    """Generator for all possible combinations of rotations for a given linkage.

    Parameters
    ----------
    linkage :       scipy.cluster.hierarchy.linkage
    li_MID :        int
                    Index of the linkage at which to stop rotating.

    Yields
    ------
    new_link :      np.ndarray
                    A permutation of linkage rotations.
    lindex :        dict
                    The mapping of labels to leaf indices.

    """
    length = len(linkage)
    linkage = linkage.reshape(-1, length, 4)
    i = length - 1
    while i >= li_MID:
        for item in linkage:
            # This is the new linkage matrix
            new = item.copy()
            new[i] = [new[i, 1], new[i, 0], new[i, 2], new[i, 3]]

            # This is the leaf order
            lindex = leaf_order(new, labels, as_dict=True)

            yield new, lindex

            linkage = np.append(linkage, new)
            linkage = linkage.reshape(-1, length, 4)

        i -= 1


def bottom_up(li_MID, link1, link2, labels1, labels2, L=1.5):
    """Rotate dendrogram from bottom to "li_MID" and find smallest entanglement."""
    # Find leafs and entanglement of start position
    lindex1 = leaf_order(link1, labels1, as_dict=True)
    lindex2 = leaf_order(link2, labels2, as_dict=True)
    min_entang = entanglement(lindex1, lindex2, L=L)
    org_entang = float(min_entang)

    # No go over each hinge/knot from bottom to "li_MID" and rotate it
    for i in range(li_MID):
        # Rotate left and right linkage
        new1 = rotate(link1, i)
        new2 = rotate(link2, i)

        # Generate leafs for the new variants
        lindexn1 = leaf_order(new1, labels1, as_dict=True)
        lindexn2 = leaf_order(new2, labels2, as_dict=True)

        # Now test pairwise entanglement
        for j, lx1 in zip([link1, new1],
                          [lindex1, lindexn1]):
            for k, lx2 in zip([link2, new2],
                              [lindex2, lindexn2]):
                new_entang = entanglement(lx1, lx2, L=L)

                if new_entang < min_entang:
                    min_entang = new_entang
                    link1 = j
                    link2 = k
                    lindex1 = lx1
                    lindex2 = lx2

    improved = min_entang < org_entang
    return link1, link2, min_entang, improved


def refine(best_linkage1, best_linkage2, min_entang, labels1, labels2, L=1.5):
    """Refine rotation to maximize horizontal lines."""
    org_entang = float(min_entang)

    lindex1 = leaf_order(best_linkage1, labels1, as_dict=True)
    lindex2 = leaf_order(best_linkage2, labels2, as_dict=True)

    # For each label
    for k in list(lindex1):
        find1 = lindex1[k]
        find2 = lindex2[k]
        # If this label is not aligned between left and right dendrogram
        if find1 != find2:
            # Find the first hinges for this label
            knot1 = np.where(best_linkage1 == find1)[0][0]
            knot2 = np.where(best_linkage2 == find2)[0][0]

            # Rotate around these hinges
            new1 = rotate(best_linkage1, knot1)
            new2 = rotate(best_linkage2, knot2)
            all1 = np.append([best_linkage1], [new1], axis=0)
            all2 = np.append([best_linkage2], [new2], axis=0)

            all1_lindices = []
            for j in all1:
                all1_lindices.append(leaf_order(j, labels1, as_dict=True))

            all2_lindices = []
            for k in all2:
                all2_lindices.append(leaf_order(k, labels2, as_dict=True))

            # Check if any of the new versions are better than the old
            for j, lix1 in zip(all1, all1_lindices):
                for k, lix2 in zip(all2, all2_lindices):
                    new_entang = entanglement(lix1, lix2, L=L)
                    if new_entang < min_entang:
                        min_entang = new_entang
                        best_linkage1 = j
                        best_linkage2 = k
                        lindex1 = lix1
                        lindex2 = lix2

    improved = min_entang < org_entang
    return best_linkage1, best_linkage2, min_entang, improved


def untangle(link1, link2, labels1, labels2, method='random', L=1.5, **kwargs):
    """Untangle two dendrograms using various methods.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to untangle.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    method :            "random" | "step1side" | "step2side" | "permuations"
                        Method to use for untangling. In order of increasing
                        run-time:
                          - "random" shuffles the dendrograms ``R`` times
                          - "step1side" turns every hinge in ``link1`` and
                            keeps ``link2`` fix
                          - "step2side" turns every hinge in both dendrograms
                          - "permutations" runs permutations of rotations for
                            both dendrograms (has ``O(n^2)^2`` complexity)
    **kwargs
                        Passed to the respective untangling functions.

    See

    """
    if method == 'random':
        return untangle_random_search(link1, link2, labels1, labels2, L=L, **kwargs)
    elif method == 'step1side':
        return untangle_step_rotate_1side(link1, link2, labels1, labels2, L=L, **kwargs)
    elif method == 'step2side':
        return untangle_step_rotate_2side(link1, link2, labels1, labels2, L=L, **kwargs)
    elif method == 'permutations':
        return untangle_permutations(link1, link2, labels1, labels2, L=L, **kwargs)
    else:
        raise ValueError(f'Unknown method "{method}"')


def untangle_permutations(link1, link2, labels1, labels2, L=1.5,
                          target_ent=0, progress=True):
    """Untangle by greedily testing all possible permutations of rotations.

    This algorithm has O(n^2)^2 complexity and can run very long! In brief:
    1. Start at N = 1
    2. Find all possible permutations of rotating the top N hinges
    3. For each permutation, test the entanglement of rotating each individual
       hinge from the bottom up to the top hinge.
    4. Keep the combination of the best permutation + bottom-up rotations
    5. Increase N by +1
    6. Go back to step 2 and repeat until we reached our target entanglement.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to (better) align by rotating.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    L :                 float
                        Distance norm used to calculate the entanglement.
                        Passed to ``entanglement()``.
    target_ent :        float [0-1]
                        Target entanglement.
    progress :          bool
                        Whether to show a progress bar.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    # Keep track of past entanglements
    entang = [float('inf')]

    bar_format = ("{l_bar}{bar}| [{elapsed}<{remaining}, "
                  "{rate_fmt}, N {postfix[0]}, entangl {postfix[1]:.4f}]")
    with tqdm(desc='Searching', leave=False, total=2, postfix=[1, 1],
              bar_format=bar_format, disable=not progress) as pbar:
        for i in range(len(link1) - 1):
            li_MID = len(link1) - 1 - i
            # Keep track of minimal entanglement this round
            min_entang = entang[-1]

            if progress:
                pbar.total = (2**(i+1))**2
                pbar.n = 0
                pbar.postfix[0] = i

            # Now test these combinations
            for i, lindex1 in get_all_linkage_gen(link1, li_MID, labels1):
                for j, lindex2 in get_all_linkage_gen(link2, li_MID, labels2):
                    best_linkage1 = i
                    best_linkage2 = j

                    # Now optimize from the bottom up to li_MID
                    # Coarse optimization
                    (best_linkage1,
                     best_linkage2,
                     this_entang,
                     improved1) = bottom_up(li_MID,
                                            best_linkage1, best_linkage2,
                                            labels1, labels2, L=L)

                    # Fine optimization
                    (best_linkage1,
                     best_linkage2,
                     this_entang,
                     improved2) = refine(best_linkage1, best_linkage2,
                                         this_entang,
                                         labels1, labels2, L=L)

                    # Keep this iteration if it's better than the previous
                    if this_entang < min_entang:
                        final_linkage1 = best_linkage1
                        final_linkage2 = best_linkage2
                        min_entang = this_entang

                        if progress:
                            pbar.postfix[1] = this_entang

                    if progress:
                        pbar.update()

                    # Stop if optimal entangle found
                    if min_entang <= target_ent:
                        break

                # Stop if optimal entangle found
                if min_entang <= target_ent:
                    break

            # Track how entanglment evolves
            entang.append(min_entang)

            # Convergence condition:
            # If entanglement is optimal
            if entang[-1] <= target_ent:
                break

    module_logger.info(f'Finished optimising at entanglement {entang[-1]:.2f}')
    return final_linkage1, final_linkage2


def untangle_step_rotate_2side(link1, link2, labels1, labels2,
                               direction='down', L=1.5, max_n_iterations=10):
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
                        Passed to ``entanglement()``.
    max_n_iterations :  int
                        Max iterations (default = 10) to run.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    assert direction in ('down', 'up')

    min_entang = float('inf')

    for i in range(int(max_n_iterations)):
        # Rotate the first dendrogram
        link1, link2 = untangle_step_rotate_1side(link1, link2,
                                                  labels1, labels2,
                                                  L=L, direction=direction)

        # Now rotate the second dendrogram
        link2, link1 = untangle_step_rotate_1side(link2, link1,
                                                  labels2, labels1,
                                                  L=L, direction=direction)

        # Get the new entanglement
        lindex1 = leaf_order(link1, labels1, as_dict=True)
        lindex2 = leaf_order(link2, labels2, as_dict=True)

        # Get new entanglement
        new_entang = entanglement(lindex1, lindex2, L=L)

        # Stop if there is no improvement from the last iteration
        if new_entang == min_entang:
            break
        else:
            min_entang = new_entang

        if min_entang == 0:
            break

    return link1, link2


def untangle_step_rotate_1side(link1, link2, labels1, labels2,
                               direction='down', L=1.5):
    """Untangle by stepwise rotating around all hinges in one dendrogram.

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
                        Passed to ``entanglement()``.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    assert direction in ('down', 'up')

    # Get label indices
    lindex1 = leaf_order(link1, labels1, as_dict=True)
    lindex2 = leaf_order(link2, labels2, as_dict=True)

    # Get starting entanglement
    min_entang = entanglement(lindex1, lindex2, L=L)

    n_hinges = len(link1) - 1
    for i in range(n_hinges):
        if direction == 'down':
            i = n_hinges - i

        # Shuffle dendrograms
        r_link1 = rotate(link1, i, copy=True)

        # Get label indices
        r_lindex1 = leaf_order(r_link1, labels1, as_dict=True)

        # Get new entanglement
        new_entang = entanglement(r_lindex1, lindex2, L=L)

        # Check if new entanglment is better
        if new_entang < min_entang:
            min_entang = new_entang
            link1 = r_link1

        if min_entang == 0:
            break

    return link1, link2


def untangle_random_search(link1, link2, labels1, labels2, R=100, L=1.5):
    """Untangle dendrogram using a simple random search.

    Shuffle trees and see if entanglement got better.

    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to (better) align by shuffling.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    R :                 int
                        Number of shuffles to perform.
    L :                 float
                        Distance norm used to calculate the entanglement.
                        Passed to ``entanglement()``.

    Returns
    -------
    link1,link2
                        Reordered linkages.

    """
    # Get label indices
    lindex1 = leaf_order(link1, labels1, as_dict=True)
    lindex2 = leaf_order(link2, labels2, as_dict=True)

    # Get starting entanglement
    min_entang = entanglement(lindex1, lindex2, L=L)

    for i in range(int(R)):
        # Shuffle dendrograms
        s_link1 = shuffle_dendogram(link1)
        s_link2 = shuffle_dendogram(link2)

        # Get label indices
        s_lindex1 = leaf_order(s_link1, labels1, as_dict=True)
        s_lindex2 = leaf_order(s_link2, labels2, as_dict=True)

        # Get new entanglement
        new_entang = entanglement(s_lindex1, s_lindex2, L=L)

        # Check if new entanglment is better
        if new_entang < min_entang:
            min_entang = new_entang
            link1 = s_link1
            link2 = s_link2

        if min_entang == 0:
            break

    return link1, link2


def shuffle_dendogram(link, copy=True):
    """Randomly shuffle dendrogram.

    Parameters
    ----------
    link :      scipy.cluster.hierarchy.linkage

    Returns
    -------
    s_link :    scipy.cluster.hierarchy.linkage
                Shuffled linkage.

    """
    assert isinstance(link, np.ndarray)

    # How many hinges to rotate
    n_rot = np.random.randint(len(link))

    # Which hinges to rotate
    to_rot = np.random.choice(np.arange(len(link)), n_rot, replace=False)

    # Make a copy of the original
    if copy:
        s_link = link.copy()
    else:
        s_link = link

    # Rotate hinges
    s_link[to_rot, :2] = s_link[to_rot, :2][:, ::-1]

    return s_link


def leaf_order(link, labels=None, as_dict=True):
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

    Returns
    -------
    dict
                If ``as_dict=True`` return as ``{'l1': 1, 'l2':, 5, ...}``.
    list
                If ``as_dict=False`` return as ``['l4', 'l3', 'l1', ...]``.

    """
    # This gives us the order of the original labels
    leafs_ix = sclust.hierarchy.leaves_list(link)

    if as_dict:
        if not isinstance(labels, type(None)):
            return dict(zip(labels, leafs_ix))
        else:
            return dict(zip(np.arange(len(leafs_ix)), leafs_ix))
    else:
        if not isinstance(labels, type(None)):
            return np.asarray(labels)[leafs_ix]
        else:
            return leafs_ix


def entanglement(lindex1, lindex2, L=1.5):
    """Calculage average displacement of leafs in dendogram 1 and 2.

    Entanglement is a measure between 1 (full entanglement) and 0 (no
    entanglement). Ignores leafs that aren't present in both dendrograms.

    Parameters
    ----------
    lindex1,lindex2 :       dict
                            Dictionaries mapping the labels of two dendrograms
                            to their indices.
    L :                     any positive number
                            Distance norm to use for measuring the distance
                            between the two trees. Can be any positive number,
                            often one will want to use 0, 1, 1.5 or 2:
                            ``sum(abs(x-y)^L)``.

    """
    assert isinstance(lindex1, dict)
    assert isinstance(lindex2, dict)

    exist_in_both = list(set(lindex1) & set(lindex2))

    if not exist_in_both:
        raise ValueError('Not a single matching label in both dendrograms.')

    # Absolute distance
    dist = np.array([lindex1[l] - lindex2[l] for l in exist_in_both])
    dist = np.abs(dist)
    # Absolute entanglement
    ent = np.sum(dist ** L)

    # Worst case
    ix = np.arange(max(len(lindex1), len(lindex2)))
    worst = np.sum(np.abs(ix - ix[::-1]) ** L)

    # Normalized entanglemtn
    return ent / worst


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
