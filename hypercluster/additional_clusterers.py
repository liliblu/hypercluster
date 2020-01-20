"""
Additonal clustering classes can be added here, as long as they have a 'fit' method.


Attributes:
    HDBSCAN (clustering class): See `hdbscan`_

.. _hdbscan:
    https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html#the-simple-case/
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from hdbscan import HDBSCAN


class NMFCluster:
    """Uses non-negative factorization from sklearn to assign clusters to samples, based on the
    maximum membership score of the sample per component.

    Args:
        n_clusters: The number of clusters to find. Used as n_components when fitting.
        n_inits: The number of initializations to use for voting which cluster
        **nmf_kwargs:
    """
    def __init__(self, n_clusters: int = 8, **nmf_kwargs):

        nmf_kwargs['n_components'] = n_clusters

        self.NMF = NMF(**nmf_kwargs)
        self.n_clusters = n_clusters
        self.n_inits = None
        self.labels_ = None

    def fit(self, data):
        """If negative numbers are present, creates one data matrix with all negative numbers
        zeroed. Create another data matrix with all positive numbers zeroed and the signs of all
        negative numbers reversed. Concatenate both matrices resulting in a data matrix twice as
        large as the original, but with positive values only and zeros and hence appropriate for
        NMF. Uses decomposed matrix H, which is nxk (with n=number of samples and k=number of
        components) to assign cluster membership. Each sample is assigned to the cluster for
        which it has the highest membership score. See sklearn.decomposition.NMF
        (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

        Args:
            data (DataFrame): Data to fit with samples as rows and features as columns.

        Returns:
            self with labels_ attribute.
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
