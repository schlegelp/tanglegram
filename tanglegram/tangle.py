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
import random

from tqdm import tqdm, trange
try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook, tnrange
        tqdm = tqdm_notebook
        trange = tnrange
except:
    pass

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


def gen_tangle(a, b, labelsA=None, labelsB=None, optimize_order=10000,
               color_by_diff=True, link_kwargs={}, dend_kwargs={}):
    """ Plots a tanglegram from two dendrograms.

    Parameters
    ----------
    (a,b) :                 {pandas.DataFrame, scipy.cluster.hierarchy.linkage}
                            Dendrograms to be compared. 
    (labelsA,labelsB) :     list of str
                            If not provided and a/b pandas Dataframe, will try 
                            to extract from columns.
    optimize_method :       {'random','greedy',None}, optional
                            Sets method to use for aligning left and right dendrogram
                            'random' uses brute force approach -> might give varying 
                            results on each iteration.
    p :                     int, optional
                            Paramter to be passed to optimize_method. For 'random',
                            this determines the number of iterations.
    color_by_diff :         bool, optional
                            If True, straight edges will have lighter color, 
                            emphasizing differences between dendrograms.
    **kwargs
                            _kwargs_ to be passed on to scipy.cluster.hierarchy.linkage

    Returns
    -------
    matplotlib figure
    """

    plt.style.use('ggplot')

    if isinstance(a, pd.DataFrame):
        module_logger.info('Generating linkage from dataframe')
        linkage1 = sclust.hierarchy.linkage(sdist.squareform(a, checks=False), **link_kwargs)
        if not labelsA:
            labelsA = a.columns.tolist()
    elif isinstance(a, np.ndarray):
        linkage1 = a
    else:
        raise TypeError('Parameter <a> needs to be either pandas DataFrame or numpy array')

    if isinstance(b, pd.DataFrame):
        module_logger.info('Generating linkage from dataframe')
        linkage2 = sclust.hierarchy.linkage(sdist.squareform(b, checks=False), **link_kwargs)
        if not labelsB:
            labelsB = b.columns.tolist()
    elif isinstance(b, np.ndarray):
        linkage2 = b
    else:
        raise TypeError('Parameter <b> needs to be either pandas DataFrame or numpy array')

    if optimize_method == 'random':
        linkage1, linkage2 = _random_optimize_leaf_order(
            linkage1, linkage2, labelsA, labelsB, max_iter=p)
    else:
        module_logger.warning('Unknown optmizing method.')

    fig = pylab.figure(figsize=(8, 8))

    # Compute and plot left dendrogram.
    ax1 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    Z1 = sclust.hierarchy.dendrogram(
        linkage1, orientation='left', labels=labelsA)
    # ax1.set_xticks([])
    # ax1.set_yticks([])

    # Compute and plot right dendrogram.
    ax2 = fig.add_axes([0.7, 0.1, 0.25, 0.8])  # [0.3, 0.71, 0.6, 0.2])
    Z2 = sclust.hierarchy.dendrogram(
        linkage2, labels=labelsB, orientation='right')
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    if True in [l not in Z2['ivl'] for l in Z1['ivl']]:
        plt.clf()
        raise ValueError('Mismatch of dendrogram labels - unable to compare')

    # Generate middle plot with connecting lines
    ax3 = fig.add_axes([0.4, 0.1, 0.2, 0.8])
    ax3.axis('off')
    ax3.set_ylim((ax1.viewLim.y0, ax1.viewLim.y1))
    ax3.set_xlim((0, 1))

    # Now iterate over all left leaves
    for ix_l, l in enumerate(Z1['ivl']):
        coords_l = (ax1.viewLim.y1 - ax1.viewLim.y0) / \
            (len(Z1['leaves'])) * (ix_l + .5)

        try:
            ix_r = Z2['ivl'].index(l)
        except:
            continue

        coords_r = (ax2.viewLim.y1 - ax2.viewLim.y0) / \
            (len(Z2['leaves'])) * (ix_r + .5)

        if not color_by_diff:
            c = 'black'
        else:
            v = max(round(.75 - math.fabs(ix_l - ix_r) / len(Z1['ivl']), 2), 0)
            c = (v, v, v)

        ax3.plot([0, 1], [coords_l, coords_r], '-', linewidth=1.5, c=c)

    module_logger.info('Done. Use matplotlib.pyplot.show() to show plot.')

    return fig


def _optimize_leaf_order(link1, link2, labels1, labels2, max_iter):
    """ Optimizes leaf order of linkage 1 and 2 to best match each other.

    Parameters
    ----------
    (link1,link2) :     scipy.cluster.hierarchy.linkage
                        linkages to be matched
    (labels1,labels2) : list of str


    Returns
    -------
    optimized linkage
    """

    dend1 = sclust.hierarchy.dendrogram(link1, orientation = 'left',
                                        no_plot=True, labels=labels1)
    dend2 = sclust.hierarchy.dendrogram(link2, orientation = 'right',
                                        no_plot=True, labels=labels2)

    entangles =[get_entanglement(dend1, dend2)]

    pbar = trange(max_iter)

    for i in pbar:
        temp_link1 = link1.copy()
        temp_link2 = link2.copy()

        # Go over all "hinges" of the dendrogram
        for k in range(len(link2)):
            # In 50% of the cases turn the hinge
            if random.randint(0, 100) > 50:
                temp_link2[k] = [temp_link2[k][1], temp_link2[k][0], temp_link2[k][2], temp_link2[k][3]]

        # Do the same for other dendrogram
        for k in range(len(link1)):
            if random.randint(0,100) > 50:
                temp_link1[k] = [temp_link1[k][1], temp_link1[k][0], temp_link1[k][2], temp_link1[k][3]]

        # Generate actual dendrogram to get leaf order
        dend1 = sclust.hierarchy.dendrogram(temp_link1, orientation = 'left',
                                            no_plot=True, labels=labels1)
        dend2 = sclust.hierarchy.dendrogram(temp_link2, orientation = 'right',
                                            no_plot=True, labels=labels2)

        # Test new entanglement -> keep if better then previous
        new_entangle = get_entanglement(dend1, dend2)
        if entangles[-1] >= new_entangle:
            entangles.append(new_entangle)
            link1 = temp_link1
            link2 = temp_link2
        else:
            entangles.append( entangles[-1] )

        if i > 0 and i % 100 == 0:
            slope = round(entangles[-1] - np.mean(entangles[-100:-1]), 2)
            pbar.set_description('Optimising {0}'.format(slope))

    pbar.close()

    module_logger.info('Finished optimising at entanglement {0}'.format(round(entangles[-1], 4)))


    return link1,link2


def get_entanglement(dend1, dend2):
    """ Returns average displacement of leafs in dendogram 1 and 2. Skips
    leafs that aren't present in both dendrograms.
    """
    exist_in_both = [l for l in dend1['ivl'] if l in dend2['ivl']]

    if not exist_in_both:
        raise ValueError('No matching labels in dendrograms.')

    return sum([math.fabs(dend1['ivl'].index(l) - dend2['ivl'].index(l)) for l in exist_in_both]) / len(exist_in_both)


if __name__ == '__main__':
    labelsA= ['A', 'B', 'C', 'D']
    labelsB= ['B', 'A', 'C', 'D']
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
    fig = gen_tangle(mat1, mat2, optimize_order=False)
    plt.show()

