param_delim = ";"
val_delim = "-"
slow = ["AffinityPropagation", "MeanShift"]
fast = ["KMeans", "OPTICS", "HDBSCAN"]
fastest = ["MiniBatchKMeans"]
partitioners = ["AffinityPropagation", "MeanShift", "KMeans", "MiniBatchKMeans"]
clusterers = ["OPTICS", "HDBSCAN"]
categories = {
    "slow": slow,
    "fast": fast,
    "fastest": fastest,
    "partitioning": partitioners,
    "clustering": clusterers,
}

min_cluster_size = [i for i in range(2, 17, 2)]
n_clusters = [i for i in range(2, 41)]
damping = [i / 100 for i in range(55, 95, 5)]


variables_to_optimize = {
    "HDBSCAN": dict(min_cluster_size=min_cluster_size),
    "KMeans": dict(n_clusters=n_clusters),
    "MiniBatchKMeans": dict(n_clusters=n_clusters),
    "AffinityPropagation": dict(damping=damping),
    "MeanShift": dict(cluster_all=[False]),
    "OPTICS": dict(min_samples=min_cluster_size),
}


need_ground_truth = [
    "adjusted_rand_score",
    "adjusted_mutual_info_score",
    "homogeneity_score",
    "completeness_score",
    "fowlkes_mallows_score",
    "mutual_info_score",
    "v_measure_score",
]

inherent_metrics = [
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
    "number_clustered",
    "smallest_largest_clusters_ratio",
    "smallest_cluster_ratio",
]

min_or_max = {
    "adjusted_rand_score": max,
    "adjusted_mutual_info_score": max,
    "homogeneity_score": max,
    "completeness_score": max,
    "fowlkes_mallows_score": max,
    "silhouette_score": max,
    "calinski_harabasz_score": max,
    "davies_bouldin_score": min,
    "mutual_info_score": max,
    "v_measure_score": max,
}
