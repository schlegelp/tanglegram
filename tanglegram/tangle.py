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


__all__ = ['entanglement', 'untangle']

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

def draw_tanglegram(linkage_1, linkage_2, labels1, labels2, color_by_diff=True, dend_kwargs={}):
    plt.clf()
    fig = pylab.figure(figsize=(10, 8))

    # Compute and plot left dendrogram.
    ax1 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    Z1 = sclust.hierarchy.dendrogram(linkage_1, orientation='left', labels=labels1, **dend_kwargs)
    # ax1.set_xticks([])
    # ax1.set_yticks([])

    # Compute and plot right dendrogram.
    ax2 = fig.add_axes([0.7, 0.1, 0.25, 0.8])  # [0.3, 0.71, 0.6, 0.2])
    Z2 = sclust.hierarchy.dendrogram(linkage_2, labels=labels2, orientation='right', **dend_kwargs)
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    if True in [l not in Z2['ivl'] for l in Z1['ivl']]:
        # plt.clf()
        # raise ValueError('Mismatch of dendrogram labels - unable to compare')
        module_logger.warning('Labels {0} do not exist in both dendrograms'.format(
            set([l for l in Z1['ivl'] if l not in Z2['ivl']] + [l for l in Z2['ivl'] if l not in Z1['ivl']])))

    # Generate middle plot with connecting lines
    ax3 = fig.add_axes([0.41, 0.1, 0.18, 0.8])
    ax3.axis('off')
    ax3.set_xlim((0, 1))

    # Get min and max y dimensions
    max_y = max(ax1.viewLim.y1, ax2.viewLim.y1)
    min_y = min(ax1.viewLim.y0, ax2.viewLim.y0)

    # Make sure labels of both dendrograms have the same font size
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=10)

    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=10)

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
    if i < li_MID:
        return linkage
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


def get_all_linkage_gen(linkage, stop, labels, start=0):
    """Generator for all possible combinations of rotations for a given linkage.
    Parameters
    ----------
    linkage :       scipy.cluster.hierarchy.linkage
    start :         int
                    Index of the linkage at which to stop rotating. Counts from
                    top.
    labels :        list
                    Labels for given linkage.
    start :         int
                    At what hinge to start returning permutations.
    Yields
    ------
    new_link :      np.ndarray
                    A permutation of linkage rotations.
    lindex :        dict
                    The mapping of labels to leaf indices.
    """
    length = len(linkage)
    linkage = linkage.reshape(-1, length, 4)

    # Invert to/from
    start = length - 1 - start
    stop = length - 1 - stop

    i = length - 1
    while i > stop:
        # Use range because linkage will change in size as we edit it
        for j in range(len(linkage)):
            # This is the new linkage matrix
            new = linkage[j].copy()
            new[i][0], new[i][1] = new[i][1], new[i][0]

            if i <= start:
                # This is the leaf order
                lindex = leaf_order(new, labels, as_dict=True)

                yield new, lindex

            linkage = np.append(linkage, new)
            linkage = linkage.reshape(-1, length, 4)
        i -= 1


def bottom_up(stop, link1, link2, labels1, labels2, L=1.0):
    """Rotate dendrogram from bottom to "stop" and find smallest entanglement."""
    # Find entanglement of start position
    min_entang = entanglement(link1, link2)
    org_entang = float(min_entang)

    # Now go over each hinge/knot from bottom to "stop" and rotate it
    for i in range(stop):
        # Rotate left and right linkage
        new1 = rotate(link1, i)
        new2 = rotate(link2, i)
        all1 = np.append([link1], [new1], axis=0)
        all2 = np.append([link2], [new2], axis=0)

        # Now test pairwise entanglement
        for j in all1:
            for k in all2:
                new_entang = entanglement(j, k)

                if new_entang < min_entang:
                    min_entang = new_entang
                    link1 = j
                    link2 = k

    improved = org_entang - min_entang
    return link1, link2, min_entang, improved

def refine(best_linkage1, best_linkage2, labels1, labels2):
    """Rotate dendrogram by going through all objects to find smallest entanglement."""
    dend1 = dendrogram(best_linkage1, labels=labels1, no_plot=True)
    dend2 = dendrogram(best_linkage2, labels=labels2, no_plot=True)
    old_entang = entanglement(best_linkage1, best_linkage2)
    min_entang = old_entang

    # If an object does not appear at the same position on x-axis in both dendrograms
    # Find the first interior vertex containing such object and rotate dendrograms
    for i in range(len(best_linkage1)):
        if dend1["ivl"][i] != dend2["ivl"][i]:
            for num, item in enumerate(best_linkage1):
                if item[0] == i or item[1] == i:
                    knot1 = num
            for num, item in enumerate(best_linkage2):
                if item[0] == i or item[1] == i:
                    knot2 = num
            new1 = rotate(best_linkage1, knot1)
            new2 = rotate(best_linkage2, knot2)
            all1 = np.append([best_linkage1], [new1], axis=0)
            all2 = np.append([best_linkage2], [new2], axis=0)
            save = []
            for j in all1:
                for k in all2:
                    new_entang = entanglement(j, k)
                    if new_entang < min_entang:
                        min_entang = new_entang
                        best_linkage1 = j
                        best_linkage2 = k
            save.append(min_entang)
    improved = old_entang - min_entang
    return best_linkage1, best_linkage2, min_entang, improved


def untangle(link1, link2, labels1, labels2, method='random', L=2.0, **kwargs):
    """Untangle two dendrograms using various methods.
    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to untangle.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    method :            "random" | "step1side" | "step2side" | "ShUntan"
                        Method to use for untangling. In order of increasing
                        run-time:
                          - "random" shuffles the dendrograms ``R`` times
                          - "step1side" turns every hinge in ``link1`` and
                            keeps ``link2`` fix
                          - "step2side" turns every hinge in both dendrograms
                          - "ShUnTan" runs permutations of rotations for
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
    elif method == 'ShUnTan':
        return untangle_ShUnTan(link1, link2, labels1, labels2, L=L, **kwargs)
    else:
        raise ValueError(f'Unknown method "{method}"')


def untangle_ShUnTan(link1, link2, labels1, labels2, L=2.0, n_permute=-1, target_ent=0, progress=True):
    """Untangle by greedily testing all possible permutations of rotations.
    This algorithm has O(n^2)^2 complexity and can run very long! In brief:
    1. Start at N = 1
    2. Shuffling: Find all possible permutations of rotating the top N hinges
    3. Untangling: For each permutation, test the entanglement of rotating each individual
       hinge from the bottom up to the top hinge and hinge containing singleton cluster.
    4. Keep the combination of the best permutation
    5. Increase N by +1
    6. Go back to step 2 and repeat until we reach our target entanglement.
    Parameters
    ----------
    link1,link2 :       scipy.cluster.hierarchy.linkage
                        Linkages to (better) align by rotating.
    labels1,labels2 :   list
                        Labels for link1 and link2, respectively.
    L :                 float
                        Distance norm used to calculate the entanglement.
                        Passed to ``entanglement()``.
    n_permute :         int
                        Number of hinges from to permute. Positive values count
                        from the top, negative from the bottom. The default of
                        -1 means that permutations will be run for all hinges.
    target_ent :        float [0-1]
                        Target entanglement.
    progress :          bool
                        Whether to show a progress bar.
    Returns
    -------
    link1,link2
                        Reordered linkages.
    """
    # TODO:
    # - once all possible permutations are computed, we could test them using
    #   parallel processes

    # Keep track of past entanglements
    entang = [entanglement(link1, link2)]
    min_entang = entang[-1]
    bar_format = ("{l_bar}{bar}| [{elapsed}<{remaining}, "
                  "{rate_fmt}, N {postfix[0]}/{postfix[1]}, "
                  "entanglement {postfix[2]:.4f}]")

    if n_permute == 0:
        raise ValueError('`n_permute` must not be zero')
    elif n_permute < 0:
        # Translate to count from top
        n_permute = len(link1) - (n_permute + 1)
    elif n_permute > len(link1):
        raise ValueError('`n_permute` must not be great than number of hinges')

    with tqdm(desc='Searching', leave=False, total=2,
              postfix=[1, n_permute, entang[-1]],
              bar_format=bar_format, disable=not progress) as pbar:

        ix = 1

        # Keep track of minimal entanglement this round
        all1 = get_all_linkage(link1, len(link1) - ix)
        all2 = get_all_linkage(link2, len(link2) - ix)
        if progress:
            pbar.total = 4
            pbar.n = 0
            pbar.postfix[0] = 1

        for i in all1:
            for j in all2:
                best_linkage1 = i
                best_linkage2 = j

                # Now optimize from the bottom up to li_MID = Coarse optimization
                improved = 1
                while improved != 0:
                    (best_linkage1, best_linkage2, this_entang, improved) = bottom_up(len(link1) - ix, best_linkage1, best_linkage2, labels1, labels2, L=L)

                # Fine optimization
                (best_linkage1, best_linkage2, this_entang, improved) = refine(best_linkage1, best_linkage2, labels1, labels2)

                # Keep this iteration if it's better than the previous
                if this_entang < min_entang:
                    final_linkage1 = best_linkage1
                    final_linkage2 = best_linkage2
                    min_entang = this_entang
                    original1 = i
                    original2 = j
                    if progress:
                        pbar.postfix[2] = this_entang
                entang.append(min_entang)

                if progress:
                    pbar.update()

                # Stop if optimal entangle found
                if min_entang <= target_ent:
                    return final_linkage1, final_linkage2, min_entang
                    break
        # Track how entanglement evolves
        old_entang = min_entang

        for ix in range(2, n_permute):
            all1 = get_all_linkage(link1, len(link1) - ix)
            all2 = get_all_linkage(link2, len(link2) - ix)
            if progress:
                pbar.total = (2**ix)**2 -(2**(ix-1))**2
                pbar.n = 0
                pbar.postfix[0] = ix
            for index1, i in enumerate(all1):
                for index2, j in enumerate(all2):
                    test = 2 ** (ix - 1)
                    if not(index1 < test and index2 < test):
                        best_linkage1 = i
                        best_linkage2 = j
                        start = timer()

                        # Now optimize from the bottom up to li_MID = Coarse optimization
                        improved = 1
                        while improved != 0:
                            (best_linkage1, best_linkage2, this_entang, improved) = bottom_up(len(link1) - ix, best_linkage1, best_linkage2, labels1, labels2, L=L)

                        # Fine optimization
                        (best_linkage1, best_linkage2, this_entang, improved) = refine(best_linkage1, best_linkage2, labels1, labels2)

                        # Keep this iteration if it's better than the previous
                        if this_entang < min_entang:
                            final_linkage1 = best_linkage1
                            final_linkage2 = best_linkage2
                            min_entang = this_entang
                            original1 = i
                            original2 = j
                            if progress:
                                pbar.postfix[2] = this_entang
                        entang.append(min_entang)

                        if progress:
                            pbar.update()

                        # Stop if optimal entangle found
                        if min_entang <= target_ent:
                            return final_linkage1, final_linkage2, min_entang
                            break

            if min_entang == old_entang:
                break
            old_entang = min_entang

    module_logger.info(f'Finished optimising at entanglement {entang[-1]:.3f}')
    return final_linkage1, final_linkage2, min_entang



def untangle_step_rotate_2side(link1, link2, labels1, labels2,
                               direction='down', L=2.0, max_n_iterations=10):
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
    entang = [entanglement(link1, link2)]
    min_entang = entang[-1]
    for i in range(int(max_n_iterations)):
        # Rotate the first dendrogram
        link1, link2, new_entang = untangle_step_rotate_1side(link1, link2, labels1, labels2, direction=direction, L=L)
        entang.append(new_entang)

        # Now rotate the second dendrogram
        link2, link1, new_entang = untangle_step_rotate_1side(link2, link1, labels2, labels1, direction=direction, L=L)
        entang.append(new_entang)

        # Stop if there is no improvement from the last iteration
        if new_entang == min_entang:
            break
        else:
            min_entang = new_entang

        if min_entang == 0:
            break

    return link1, link2, min_entang


def untangle_step_rotate_1side(link1, link2, labels1, labels2, direction='down', L=2.0):
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

    # Get starting entanglement
    min_entang = entanglement(link1, link2)

    n_hinges = len(link1) - 1
    for i in range(n_hinges):
        if direction == 'down':
            i = n_hinges - i

        # Shuffle dendrograms
        r_link1 = rotate(link1, i, copy=True)

        # Get new entanglement
        new_entang = entanglement(r_link1, link2)

        # Check if new entanglment is better
        if new_entang < min_entang:
            min_entang = new_entang
            link1 = r_link1

        if min_entang == 0:
            break

    return link1, link2, min_entang

def untangle_random_search(link1, link2, labels1, labels2, R=1000, L=1.0):
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
    # Get starting entanglement
    min_entang = entanglement(link1, link2)

    for i in range(int(R)):
        # Shuffle dendrograms
        s_link1 = shuffle_dendogram(link1)
        s_link2 = shuffle_dendogram(link2)

        # Get new entanglement
        new_entang = entanglement(s_link1, s_link2)

        # Check if new entanglment is better
        if new_entang < min_entang:
            min_entang = new_entang
            link1 = s_link1
            link2 = s_link2

        if min_entang == 0:
            break

    return link1, link2, min_entang


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


def entanglement(link1, link2):
    """ Entanglement is measured by giving the left tree’s labels the values of 1 till tree size,
    matching these numbers with the right tree, and then dividing the sum of the square difference
    between elements of these two vectors (sum(abs(x-y)^2)) by that number of the "worst case" entanglement
    (e.g: when the right tree is the complete reverse of the left tree).
    Skips leafs that aren't present in both dendrograms.
    """
    lindex1 = leaves_list(link1)
    lindex2 = leaves_list(link2)

    exist_in_both = list(set(lindex1) & set(lindex2))
    ix = np.arange(max(len(lindex1), len(lindex2)))

    if not exist_in_both:
        raise ValueError('Not a single matching label in both dendrograms.')

    # Mapping the "number" (1 til tree size) in the left tree with the right tree
    matching_leaf_vector = np.zeros(max(len(lindex1), len(lindex2)))
    for i in lindex2:
        k = np.where(lindex2 == i)
        matching_leaf_vector[k] = np.where(lindex1 == i)
    ix = ix + 1
    matching_leaf_vector = matching_leaf_vector + 1

    return np.sum(np.abs(ix - matching_leaf_vector) ** 2) / np.sum(np.abs(ix - ix[::-1]) ** 2)
