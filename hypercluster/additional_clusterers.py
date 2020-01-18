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
    def __init__(self, n_clusters: int = 8, n_inits: Optional[int] = None, **nmfkwargs):
        kwargs = {**nmfkwargs}
        kwargs['n_components'] = n_clusters

        self.NMF = NMF(**kwargs)
        self.n_clusters = n_clusters
        self.n_inits = None
        self.labels_ = None

    def fit(self, data, n_inits: Optional[int] = None):
        """Create one data matrix with all negative numbers zeroed. Create another data matrix 
        with all positive numbers zeroed and the signs of all negative numbers removed. 
        Concatenate both matrices resulting in a data matrix twice as large as the original, 
        but with positive values only and zeros and hence appropriate for NMF.

        Args: 
            data: 

        Returns: 

        """
        if n_inits is None:
            if self.n_inits is None:
                n_inits = 50
            else:
                n_inits = self.n_inits
        self.n_inits = n_inits

        if np.any(data<0):
            positive = data.copy()
            positive[positive < 0] = 0
            negative = data.copy()
            negative[negative > 0] = 0
            negative = -negative
            data = pd.concat([positive, negative], axis=1, join='outer')

        self.NMF = self.NMF.fit(data)
        votes = pd.DataFrame(index=data.index)
        for i in range(n_inits):
            votes[i] = pd.DataFrame(self.NMF.transform(data)).idxmax(axis=1)

        self.labels_ = votes.mode(axis=1)
        return self
