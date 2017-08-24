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
import numpy as np
import pandas as pd
import pylab

def plot(x, y, labelsA=None, labelsB=None, **kwargs):
    """ Plots a tanglegram from two dendrograms

    Parameters
    ----------
    (x,y) :     {np.ndarray, pd.DataFrame, scipy.cluster.hierarchy.linkage}
                Dendrograms to be compared. Can be either original distance
                matrix or a scipy linkage.
    (labelsA,labelsB) : list of str
                        If not provided and x/y pandas Dataframe, will try to 
                        extract from columns.
    **kwargs
                _kwargs_ to be passed on to scipy.cluster.hiearchy if x or y
                is distance matrix.

    Returns
    -------
    matplotlib figure
    """

    if isinstance(x, (pd.DataFrame,np.ndarray)):
        linkage1 = sclust.hierarchy.linkage(x)
    else:
        linkage1 = x

    if isinstance(y, (pd.DataFrame,np.ndarray)):
        linkage2 = sclust.hierarchy.linkage(y)
    else:
        linkage2 = y

    if not labelsA:
        if isinstance(x, pd.DataFrame):
            labelsA = list( x.columns )

    if not labelsB:
        if isinstance(y, pd.DataFrame):
            labelsB = list( y.columns )

    fig = pylab.figure(figsize=(8, 8))

    # Compute and plot left dendrogram. 
    ax1 = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    Z1 = sclust.hierarchy.dendrogram(
        linkage1, orientation='left', labels=labelsA, **kwargs)
    #ax1.set_xticks([])
    #ax1.set_yticks([])

    # Compute and plot right dendrogram.
    ax2 = fig.add_axes([0.7, 0.1, 0.25, 0.8])#[0.3, 0.71, 0.6, 0.2])
    Z2 = sclust.hierarchy.dendrogram(
        linkage2, labels=labelsB, orientation='right', **kwargs)
    #ax2.set_xticks([])
    #ax2.set_yticks([])

    if True in [l not in Z2['ivl'] for l in Z1['ivl']]:
        plt.clf()
        raise ValueError('Mismatch of dendrogram labels - unable to compare')

    # Generate middle plot with connecting lines
    ax3 = fig.add_axes([0.4, 0.1, 0.25, 0.8])    
    ax3.axis('off')
    ax3.set_ylim((ax1.viewLim.y0,ax1.viewLim.y1))
    ax3.set_xlim((0,1))    

    # Now iterate over all left leaves
    for i,l in enumerate(Z1['ivl']):        
        coords_l = ( ax1.viewLim.y1 - ax1.viewLim.y0 ) / ( len( Z1['leaves'] )  ) * (i+.5)
        
        ix_r = Z2['ivl'].index(l)
        coords_r = ( ax2.viewLim.y1 - ax2.viewLim.y0 ) / ( len( Z2['leaves'] ) ) * (ix_r+.5) 

        ax3.plot( [0,1], [coords_l,coords_r], '-', linewidth=1.5, c='black')        
    
    plt.show()

    return fig

if __name__ == '__main__':
    labelsA= ['A','B','C','D']
    labelsB= ['B','A','C','D']
    mat = pd.DataFrame([[1,.1,0,0],[.1,1,.5,0],[0,.5,1,0],[0,0,0,1]],
                        columns=labelsA,
                        index=labelsA)

    mat2 = pd.DataFrame([[1,.1,0,0],[.1,1,.5,0],[0,.5,1,0],[0,0,0,1]],
                        columns=labelsB,
                        index=labelsB)

    fig = plot(mat,mat2)

