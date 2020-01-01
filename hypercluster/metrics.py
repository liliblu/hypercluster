from typing import Iterable
from collections import Counter

__doc__ = "More functions for evaluating clustering results."


def number_clustered(_, labels: Iterable) -> float:
    return len(labels)


def smallest_largest_clusters_ratio(_, labels: Iterable) -> float:
    counts = Counter(labels)
    counts.pop(-1, None)
    smallest = min(counts.values())
    largest = max(counts.values())
    return smallest / largest


def smallest_cluster_ratio(_, labels: Iterable) -> float:
    counts = Counter(labels)
    counts.pop(-1, None)
    return min(counts.values()) / len(labels)
