from typing import Iterable
__doc__ = "More functions for evaluating clustering results."


def fraction_clustered(_, labels: Iterable) -> float:
    return (sum(labels != -1)/len(labels))


def smallest_largest_clusters_ratio(_, labels:Iterable) -> float:
    counts = labels.value_counts()
    smallest = min(counts.keys(), key=(lambda k: counts[k]))
    largest = max(counts.keys(), key=(lambda k: counts[k]))
    return smallest/largest


def smallest_cluster_ratio(_, labels: Iterable) -> float:
    counts = labels.value_counts()
    return min(counts.keys(), key=(lambda k: counts[k])) / len(labels)
