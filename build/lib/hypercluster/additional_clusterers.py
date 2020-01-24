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
from .constants import pdist_adjacency_methods, valid_partition_types
import igraph as ig
import louvain
import leidenalg


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
    def __init__(
            self,
            adjacency_method: str = 'SNN',
            k: int = 20,
            resolution: float = 0.8,
            adjacency_kwargs: Optional[dict] = None,
            louvain_kwargs: Optional[dict] = None,
            partition_type: str = 'RBConfigurationVertexPartition'
    ):
        if adjacency_method not in ['SNN', 'CNN'] + pdist_adjacency_methods:
            raise ValueError(
                'Adjacency method %s invalid. Must be "SNN", "CNN" or a valid metric for '
                'scipy.spatial.distance.pdist.' % adjacency_method
            )
        if partition_type not in valid_partition_types:
            raise ValueError(
                'Partition type %s not valid, must be in constants.valid_partition_types' %
                partition_type
            )
        self.adjacency_method = adjacency_method
        self.k = int(k)
        self.resolution = resolution
        self.adjacency_kwargs = adjacency_kwargs
        self.louvain_kwargs = louvain_kwargs
        self.partition_type = partition_type

    def fit(
            self,
            data: pd.DataFrame,
    ):
        adjacency_method = self.adjacency_method
        k = self.k
        resolution = self.resolution
        adjacency_kwargs = self.adjacency_kwargs
        louvain_kwargs = self.louvain_kwargs
        partition_type = self.partition_type
        if k >= len(data):
            logging.warning(
                'k was set to %s, with only %s samples. Changing to k to %s-1'
                % (k, len(data), len(data))
            )
            k = len(data) - 1
        if (adjacency_method == 'SNN') | (adjacency_method == 'CNN'):
            if adjacency_kwargs is None:
                adjacency_kwargs = {}
            adjacency_kwargs['n_neighbors'] = adjacency_kwargs.get('n_neighbors', k)
            nns = NearestNeighbors(algorithm='ball_tree', **adjacency_kwargs)
            nns.fit(data)
            adjacency_mat = nns.kneighbors_graph(data)
            if adjacency_method == 'SNN':
                adjacency_mat = adjacency_mat.multiply(adjacency_mat.transpose())
            if adjacency_method == 'CNN':
                adjacency_mat = adjacency_mat*adjacency_mat.transpose()
        elif adjacency_method in pdist_adjacency_methods:
            adjacency_mat = pdist(data, metric=adjacency_method, **adjacency_kwargs)

        if louvain_kwargs is None:
            louvain_kwargs = {}
        g = ig.Graph.Adjacency(adjacency_mat.toarray().tolist())

        if partition_type in ['RBConfigurationVertexPartition', 'CPMVertexPartition']:
            louvain_kwargs['resolution_parameter'] = louvain_kwargs.get(
                'resolution_parameter', resolution
            )

        labels = eval('louvain.find_partition(g, louvain.%s, **louvain_kwargs)' % partition_type)
        labels = pd.Series({v: i for i in range(len(labels)) for v in labels[i]}).sort_index()
        if labels.is_unique or (len(labels.unique()) == 1):
            labels = pd.Series([-1 for i in range(len(labels))])
        labels = labels.values
        self.labels_ = labels
        return self

