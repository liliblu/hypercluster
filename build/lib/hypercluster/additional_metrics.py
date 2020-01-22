from typing import Iterable, Optional
from collections import Counter
from pandas import DataFrame
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist

__doc__ = (
    "More functions for evaluating clustering results. Additional metric evaluations can "
    "be added here, as long as the second argument is the labels to evaluate"
)


def number_clustered(_, labels: Iterable) -> float:
    """Returns the number of clustered samples. 

    Args: 
        _: Dummy, pass anything or None.  
        labels (Iterable): Vector of sample labels.  

    Returns (int): 
        The number of clustered labels.  

    """
    return (labels != -1).sum()


def smallest_largest_clusters_ratio(_, labels: Iterable) -> float:
    """Number in the smallest cluster over the number in the largest cluster.  

    Args: 
        _: Dummy, pass anything or None.  
        labels (Iterable): Vector of sample labels.  

    Returns (float): 
        Ratio of number of members in smallest over largest cluster.  

    """
    counts = Counter(labels)
    counts.pop(-1, None)
    return min(counts.values()) / max(counts.values())


def smallest_cluster_ratio(_, labels: Iterable) -> float:
    """Number in the smallest cluster over the total samples. 

    Args: 
        _: Dummy, pass anything or None.  
        labels (Iterable): Vector of sample labels.  

    Returns (float): 
        Ratio of number of members in smallest over all samples.  

    """
    counts = Counter(labels)
    counts.pop(-1, None)
    return min(counts.values()) / len(labels)


def number_of_clusters(_, labels: Iterable) -> float:
    """Number of total clusters. 

    Args: 
        _: Dummy, pass anything or None  
        labels (Iterable): Vector of sample labels.  

    Returns (int): 
        Number of clusters.  

    """
    return len(Counter(labels))


def smallest_cluster_size(_, labels: Iterable) -> float:
    """Number in smallest cluster 

    Args: 
        _: Dummy, pass anything or None  
        labels (Iterable): Vector of sample labels.  

    Returns (int): 
        Number of samples in smallest cluster. 

    """
    return min(Counter(labels).values())


def largest_cluster_size(_, labels: Iterable) -> float:
    """Number in largest cluster 

    Args: 
        _: Dummy, pass anything or None  
        labels (Iterable): Vector of sample labels.  

    Returns (int): 
        Number of samples in largest cluster. 

    """
    return max(Counter(labels).values())
