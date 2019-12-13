from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, OPTICS
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, \
    completeness_score, fowlkes_mallows_score, silhouette_score, calinski_harabasz_score,  \
    davies_bouldin_score
from hdbscan import HDBSCAN
import logging
from typing import Optional, Iterable, Dict
from itertools import product


clusterers = {
    'hdbscan':HDBSCAN,
    'kmeans':KMeans,
    'minibatchkmeans':MiniBatchKMeans,
    'affinitypropagation':AffinityPropagation,
    'meanshift':MeanShift,
    'optics':OPTICS,
}


slow = ['affinitypropagation', 'meanshift']
fast = ['kmeans', 'optics', 'hdbscan']
fastest = ['minibatchkmeans']
speeds = {
    'slow':slow,
    'fast':fast,
    'fastest':fastest
}


min_cluster_size = [i for i in range(2, 17, 2)]
n_clusters = [i for i in range(2, 41)]
damping = [i/10 for i in range(5, 11)]


variables_to_optimize = {
    'hdbscan':dict(min_cluster_size=min_cluster_size),
    'kmeans':dict(n_clusters=n_clusters),
    'minibatchkmeans':dict(n_clusters=n_clusters),
    'affinitypropagation':dict(damping=damping),
    'meanshift':dict(cluster_all=[False]),
    'optics':dict(min_samples=min_cluster_size),
}


evaluations = {
    'adjrand': adjusted_rand_score,
    'adjmutualinfo':adjusted_mutual_info_score,
    'homogeneity':homogeneity_score,
    'completeness':completeness_score,
    'fowlkesmallows':fowlkes_mallows_score,
    'silhouette':silhouette_score,
    'calinskiharabasz':calinski_harabasz_score,
    'daviesbouldin': davies_bouldin_score,
}


need_ground_truth = [
    'adjrand',
    'adjmutualinfo',
    'homogeneity',
    'completeness',
    'fowlkesmallows'
]
inherent_metric = [
    'silhouette',
    'calinskiharabasz',
    'daviesbouldin'
]


min_or_max = {
    'adjrand':max,
    'adjmutualinfo':max,
    'homogeneity':max,
    'completeness':max,
    'fowlkesmallows':max,
    'silhouette':max,
    'calinskiharabasz':max,
    'daviesbouldin':min
}


def cluster(clusterer_name, data, **params):
    clusterer = clusterers[clusterer_name](**params)
    return clusterer.fit(data)


def run_conditions_one_algorithm(
        data: DataFrame,
        clusterer_name: Optional[str] = 'hdbscan',
        params: Optional[dict] = None,
        random_search: bool = True,
        random_search_fraction: float = 0.5,
        param_weights: Optional[dict] = None,
        **clus_kwargs
) -> DataFrame:
    #TODO make sure none of the parameters that can be fed into clusterers are iterables
    if params is None:
        params = variables_to_optimize[clusterer_name]

    clus_kwargs.update(params)
    conditions = 1
    vars_to_optimize = {}
    static_vars = {}
    for k, v in clus_kwargs.items():
        if len(v) > 1:
            vars_to_optimize[k] = v
            conditions *= len(v)
        else:
            static_vars[k] = v

    parameters = pd.DataFrame(columns=list(vars_to_optimize.keys()))
    if random_search:
        will_search = int(conditions*random_search_fraction)
        logging.INFO(
            'For clusterer %s, %s total conditions. Random search will test %s '
            'conditions. ' %(clusterer_name, conditions, will_search)
        )
        for i in range(will_search):
            temp_params = dict()
            for k, v in vars_to_optimize.items():
                weights = param_weights.get(k, None)
                temp_params[k] = np.random.choice(v, p=weights)
            parameters = parameters.append(temp_params)

    elif not random_search:
        logging.INFO(
            'For clusterer %s, %s total conditions. Grid search will test %s '
            'conditions. ' % (clusterer_name, conditions, conditions)
        )

        for row in iter(product(*vars_to_optimize.values())):
            parameters = parameters.append(dict(zip(vars_to_optimize.keys(), row)))

    label_results = pd.DataFrame(columns=parameters.columns.union(data.index))
    for i, row in parameters.iterrows():
        single_params = row.to_dict()
        single_params.update(static_vars)

        labels = cluster(clusterer_name, data, **single_params).labels_

        label_row = dict(zip(labels, data.index))
        label_row.update(single_params)
        label_results = label_results.append(label_row)

        logging.INFO('%s - %s of conditions done' % (i, (i / len(parameters))))
    label_results = label_results.set_index(parameters.columns).transpose()
    return label_results


def evaluate_results(
        label_df: DataFrame,
        method: str = 'silhouette',
        data: Optional[DataFrame] = None,
        gold_standard: Optional[Iterable] = None,
        **metric_kwargs
) -> dict:

    evaluation_results = {}
    if method in need_ground_truth:
        if not gold_standard:
            raise ValueError('Chosen evaluation metric requires gold standard set.')
        for col in label_df.columns:
            evaluation_results[col] = evaluations[method](gold_standard, label_df[col])
        return evaluation_results

    elif method in inherent_metric:
        if not data:
            raise ValueError('Chosen evaluation metric requires data input.')
        for col in label_df.columns:
            evaluation_results[col] = evaluations[method](data, label_df[col], **metric_kwargs)
        return evaluation_results

    else:
        raise ValueError('Evaluation metric %s not valid' % method)


def optimize_clustering(
        data,
        algorithm_names: clusterers.keys(),
        algorithm_parameters: Dict[dict] = {},
        random_search: bool = True,
        random_search_fraction: float = 0.3,
        algorithm_param_weights: Optional[dict] = None,
        algorithm_clus_kwargs: Optional[dict] = {},
        evaluation_method: Optional[str] = 'silhouette',
        gold_standard: Optional[Iterable] = None,
        metric_kwargs: Optional[dict] = {},
):
    if algorithm_names in speeds.keys():
        algorithm_names = speeds[algorithm_names]

    clustering_labels = {}
    clustering_evaluations = {}
    for clusterer_name in algorithm_names:
        if clusterer_name not in clusterers.keys():
            logging.ERROR('Algorithm %s not available, skipping. '% clusterer_name)
            continue

        label_df = run_conditions_one_algorithm(
            data,
            clusterer_name=clusterer_name,
            params=algorithm_parameters.get(clusterer_name, None),
            random_search=random_search,
            random_search_fraction=random_search_fraction,
            param_weights=algorithm_param_weights[clusterer_name],
            **algorithm_clus_kwargs.get(clusterer_name, None)
        )
        clustering_labels[clusterer_name] = label_df

        evaluation_results = evaluate_results(
            label_df,
            method=evaluation_method,
            data=data,
            gold_standard=gold_standard,
            **metric_kwargs
        )

        clustering_evaluations.update({
            tuple([clusterer_name]+[i for i in params_key]): value for params_key, value in evaluation_results.items()
        })

    top_choice = min_or_max[clusterer_name](clustering_evaluations, key=lambda k: clustering_evaluations[k])
    best_labels = clustering_labels[top_choice[0]][top_choice[1:]]

    return best_labels, clustering_evaluations, clustering_labels





