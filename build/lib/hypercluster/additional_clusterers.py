"""
Additonal clustering classes can be added here, as long as they have a 'fit' method.


Attributes:
    HDBSCAN (clustering class): See `hdbscan`_

.. _hdbscan:
    https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html#the-simple-case/
"""
from typing import Optional, Iterable
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN
import networkx as nx
import community
from .constants import pdist_adjacency_methods
from itertools import product


class NMFCluster:
    """Uses non-negative factorization from sklearn to assign clusters to samples, based on the
    maximum membership score of the sample per component.

    Args:
        n_clusters: The number of clusters to find. Used as n_components when fitting.
        **nmf_kwargs:
    """
    def __init__(self, n_clusters: int = 8, **nmf_kwargs):

        nmf_kwargs['n_components'] = n_clusters

        self.NMF = NMF(**nmf_kwargs)
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, data):
        """If negative numbers are present, creates one data matrix with all negative numbers
        zeroed. Create another data matrix with all positive numbers zeroed and the signs of all
        negative numbers reversed. Concatenate both matrices resulting in a data matrix twice as
        large as the original, but with positive values only and zeros and hence appropriate for
        NMF. Uses decomposed matrix H, which is nxk (with n=number of samples and k=number of
        components) to assign cluster membership. Each sample is assigned to the cluster for
        which it has the highest membership score. See `sklearn.decomposition.NMF`_  

        Args: 
            data (DataFrame): Data to fit with samples as rows and features as columns.  

        Returns: 
            self with labels\_ attribute.  

        .. _sklearn.decomposition.NMF: 
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        """

        if np.any(data<0):
            positive = data.copy()
            positive[positive < 0] = 0
            negative = data.copy()
            negative[negative > 0] = 0
            negative = -negative
            data = pd.concat([positive, negative], axis=1, join='outer')

        self.labels_ = pd.DataFrame(self.NMF.fit_transform(data)).idxmax(axis=1)
        return self


class LouvainCluster:
    """Louvain clustering using a specified adjacency graph, including shared nearest neighbor \
    (SNN) and `python-louvain`_ package

    Args: 
        adjacency_method (str): Method to use for adjacency matrix. To match seurat, specify \
        "SNN" otherwise choose a metric that can be passed to `scipy.spatial.distance.pdist`_  
        k (int): Number of nearest neighbors to use for shared nearest neighbors adjacency matrix.  
        resolution (float): Resolution to use in louvain clustering.  
        adjacency_kwargs (dict): Additional kwargs for either \
        `sklearn.neighbors.NearestNeighbors`_ or `scipy.spatial.distance.pdist`_  
        **louvain_kwargs: Additional kwargs for python-louvain `community.best_partition`_  

    .. _python-louvain:
        https://python-louvain.readthedocs.io/en/latest/
    .. _sklearn.neighbors.NearestNeighbors:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    .. _scipy.spatial.distance.pdist:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    .. _community.best_partition:
        https://python-louvain.readthedocs.io/en/latest/api.html#community.best_partition
    """

    def __init__(
            self,
            adjacency_method: str = 'SNN',
            k: int = 20,
            resolution: float = 0.8,
            adjacency_kwargs: Optional[dict] = None,
            louvain_kwargs: Optional[dict] = None,
    ):

        self.adjacency_method = adjacency_method
        self.k = int(k)
        self.resolution = resolution
        self.adjacency_kwargs = adjacency_kwargs
        self.louvain_kwargs = louvain_kwargs

    def fit(
            self,
            data: pd.DataFrame,
    ):

        adjacency_method = self.adjacency_method
        k = self.k
        resolution = self.resolution
        adjacency_kwargs = self.adjacency_kwargs
        louvain_kwargs = self.louvain_kwargs
        if k >= len(data):
            logging.warning(
                'k was set to %s, with only %s samples. Changing to k to %s-1'
                % (k, len(data), len(data))
            )
            k = len(data) - 1
        if adjacency_method == 'SNN':
            if adjacency_kwargs is None:
                adjacency_kwargs = {}
            adjacency_kwargs['n_neighbors'] = adjacency_kwargs.get('n_neighbors', k)
            nns = NearestNeighbors(**adjacency_kwargs)
            nns.fit(data)
            adjacency_mat = nns.kneighbors_graph(data).toarray()
            adjacency_mat = {
                (i, j): sum((row1+row2) == 2) for i, row1 in enumerate(list(adjacency_mat))
                for j, row2 in enumerate(list(adjacency_mat))
            }
            adjacency_mat = pd.Series(adjacency_mat)
            adjacency_mat.index = pd.MultiIndex.from_tuples(adjacency_mat.index)
            adjacency_mat = adjacency_mat.unstack().values
        elif adjacency_method in pdist_adjacency_methods:
            adjacency_mat = pdist(data, metric=adjacency_method, **adjacency_kwargs)
        else:
            raise ValueError(
                'Adjacency method %s invalid. Must be "SNN" or a valid metric for '
                'scipy.spatial.distance.pdist.' % adjacency_method
            )
        if louvain_kwargs is None:
            louvain_kwargs = {}
        g = nx.from_numpy_array(adjacency_mat)
        print('starting louvain')
        louvain_kwargs['resolution'] = louvain_kwargs.get('resolution', resolution)
        labels = pd.Series(community.best_partition(g, **louvain_kwargs)).sort_index()
        print('louvain done')
        if labels.is_unique or (len(labels.unique()) == 1):
            labels = pd.Series([-1 for i in range(len(labels))])
        labels = labels.values
        self.labels_ = labels
        return self

