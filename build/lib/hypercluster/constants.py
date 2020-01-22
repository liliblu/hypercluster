import numpy as np


__doc__ = """
Attributes: 
    param_delim: delimiter between hyperparameters for snakemake file labels and labels DataFrame \
    columns.  
    val_delim: delimiter between hyperparameter label and value for snakemake file labels and \
    labels DataFrame columns.  
    categories: Convenient groups of clusterers to use. If all samples need to be clustered, \
    'partitioners' is a good choice. If there are millions of samples, 'fastest' might be a good \
    choice.    
    variables_to_optimize: Some default hyperparameters to optimize and value ranges for a \
    selection of commonly used clustering algoirthms from sklearn. Used as deafults for \
    clustering.AutoClusterer and clustering.optimize_clustering.    
    need_ground_truth: list of sklearn metrics that need ground truth labeling. \
    "adjusted_rand_score", "adjusted_mutual_info_score", "homogeneity_score", \
    "completeness_score", "fowlkes_mallows_score", "mutual_info_score", "v_measure_score"    
    inherent_metrics: list of sklearn metrics that need original data for calculation. \
    "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", \
    "smallest_largest_clusters_ratio", "number_of_clusters", "smallest_cluster_size", \
    "largest_cluster_size"  
    min_or_max: establishing whether each sklearn metric is better when minimized or maximized for \
    clustering.pick_best_labels.  
"""
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
resolutions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
k_snn = [20, 30, 60]


variables_to_optimize = {
    "HDBSCAN": dict(min_cluster_size=min_cluster_size),
    "KMeans": dict(n_clusters=n_clusters),
    "MiniBatchKMeans": dict(n_clusters=n_clusters),
    "AffinityPropagation": dict(damping=damping),
    "MeanShift": dict(cluster_all=[False]),
    "OPTICS": dict(min_samples=min_cluster_size),
    "NMFCluster": dict(n_clusters=n_clusters),
    "LouvainCluster": dict(resolution=resolutions, k=k_snn)
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
    "smallest_largest_clusters_ratio",
    "number_of_clusters",
    "smallest_cluster_size",
    "largest_cluster_size"
]

min_or_max = {
    "adjusted_rand_score": 'max',
    "adjusted_mutual_info_score": 'max',
    "homogeneity_score": 'max',
    "completeness_score": 'max',
    "fowlkes_mallows_score": 'max',
    "silhouette_score": 'max',
    "calinski_harabasz_score": 'max',
    "davies_bouldin_score": 'min',
    "mutual_info_score": 'max',
    "v_measure_score": 'max',
}

pdist_adjacency_methods = [
    'braycurtis',
    'canberra',
    'chebyshev',
    'cityblock',
    'correlation',
    'cosine',
    'dice',
    'euclidean',
    'hamming',
    'jaccard',
    'jensenshannon',
    'kulsinski',
    'mahalanobis',
    'matching',
    'minkowski',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean',
    'yule'
]

