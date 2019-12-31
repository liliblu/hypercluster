# from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, OPTICS
# from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, \
#     completeness_score, fowlkes_mallows_score, silhouette_score, calinski_harabasz_score,  \
#     davies_bouldin_score
# from hdbscan import HDBSCAN


# clusterers = {
#     'HDBSCAN': HDBSCAN,
#     'kmeans': KMeans,
#     'minibatchkmeans': MiniBatchKMeans,
#     'affinitypropagation': AffinityPropagation,
#     'meanshift': MeanShift,
#     'optics': OPTICS,
# }
#TODO add all other clusterers
slow = ['AffinityPropagation', 'MeanShift']
fast = ['KMeans', 'OPTICS', 'HDBSCAN']
fastest = ['MiniBatchKMeans']
speeds = {
    'slow':slow,
    'fast':fast,
    'fastest':fastest
}
#TODO change speed to categories, and have partitioning vs clustering in there too.

min_cluster_size = [i for i in range(2, 17, 2)]
n_clusters = [i for i in range(2, 41)]
damping = [i/100 for i in range(55, 95, 5)]

variables_to_optimize = {
    'HDBSCAN':dict(min_cluster_size=min_cluster_size),
    'KMeans':dict(n_clusters=n_clusters),
    'MiniBatchKMeans':dict(n_clusters=n_clusters),
    'AffinityPropagation':dict(damping=damping),
    'MeanShift':dict(cluster_all=[False]), #TODO add something to optimize here
    'OPTICS':dict(min_samples=min_cluster_size),
}

# evaluations = {
#     'adjrand': adjusted_rand_score,
#     'adjmutualinfo':adjusted_mutual_info_score,
#     'homogeneity':homogeneity_score,
#     'completeness':completeness_score,
#     'fowlkesmallows':fowlkes_mallows_score,
#     'silhouette':silhouette_score,
#     'calinskiharabasz':calinski_harabasz_score,
#     'daviesbouldin': davies_bouldin_score,
# }

#TODO add all other evaluations
need_ground_truth = [
    'adjusted_rand_score',
    'adjusted_mutual_info_score',
    'homogeneity_score',
    'completeness_score',
    'fowlkes_mallows_score'
]


inherent_metric = [
    'silhouette_score',
    'calinski_harabasz_score',
    'davies_bouldin_score'
]

min_or_max = {
    'adjusted_rand_score':max,
    'adjusted_mutual_info_score':max,
    'homogeneity_score':max,
    'completeness_score':max,
    'fowlkes_mallows_score':max,
    'silhouette_score':max,
    'calinski_harabasz_score':max,
    'davies_bouldin_score':min
}
