from pandas import DataFrame
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, OPTICS
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, \
    completeness_score, fowlkes_mallows_score, silhouette_score, calinski_harabasz_score,  \
    davies_bouldin_score
from hdbscan import HDBSCAN
import logging
from typing import Optional, Iterable, Dict, Union, Any
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
damping = [i/100 for i in range(55, 95, 5)]


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


def cluster(clusterer_name, data, params: dict = {}):
    clusterer = clusterers[clusterer_name](**params)
    return clusterer.fit(data)


def run_conditions_one_algorithm(
        data: DataFrame,
        clusterer_name: Optional[str] = 'hdbscan',
        params: Optional[dict] = None,
        random_search: bool = True,
        random_search_fraction: float = 0.5,
        param_weights: dict = {},
        clus_kwargs: Optional[dict] = None,
        return_parameters: bool = False
) -> Optional[DataFrame]:
    #TODO make sure none of the parameters that can be fed into clusterers are iterables
    if params is None:
        params = variables_to_optimize[clusterer_name]
    if clus_kwargs is None:
        clus_kwargs = {}

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
    if conditions == 1:
        logging.error(
            'Clusterer %s was only given one set of parameters, nothing to optimize.'
            % clusterer_name
        )
        return None

    parameters = pd.DataFrame(columns=list(vars_to_optimize.keys()))

    for row in iter(product(*vars_to_optimize.values())):
        parameters = parameters.append(
            dict(zip(vars_to_optimize.keys(), row)), ignore_index=True
        )
    if random_search:
        will_search = int(conditions * random_search_fraction)
        # calculates probability of getting a particular set of parameters, given the probs of
        # all the individual params. If a prob isn't set, give uniform probability to each
        # parameter.
        if param_weights:
            weights = parameters.apply(
                lambda row: np.prod(
                    [param_weights.get(
                        i, len(vars_to_optimize[i])*[1/len(vars_to_optimize[i])]
                    )[vars_to_optimize[i].index(val)]
                     for var_lab, val in row.to_dict().items()]
                )
            )
        else:
            weights = None
        parameters = parameters.sample(will_search, weights=weights)

    logging.info(
        'For clusterer %s, testing %s out of %s possible conditions'
        % (clusterer_name, len(parameters), conditions)
    )

    if return_parameters:
        return parameters

    label_results = pd.DataFrame(columns=parameters.columns.union(data.index))
    for i, row in parameters.iterrows():
        single_params = row.to_dict()
        #TODO why did kmeans n_clusters==30 show up twice???
        single_params.update(static_vars)
        labels = cluster(clusterer_name, data, single_params).labels_

        label_row = dict(zip(data.index, labels))
        label_row.update(single_params)
        label_results = label_results.append(label_row, ignore_index=True)

        logging.info('%s - %s of conditions done' % (i, (i / len(parameters))))

    label_results = label_results.set_index(list(parameters.columns)).transpose()
    label_results.index = pd.MultiIndex.from_tuples(label_results.index)
    label_results = label_results[label_results.columns[~(label_results==-1).all()]]

    if label_results.shape[1] == 0:
        return None
    return label_results

#TODO write a fn that evaluates the above fn too

def evaluate_results(
        label_df: DataFrame,
        method: str = 'silhouette',
        data: Optional[DataFrame] = None,
        gold_standard: Optional[Iterable] = None,
        metric_kwargs: Optional[dict] = None
) -> dict:
    if metric_kwargs is None:
        metric_kwargs = {}

    evaluation_results = {}
    if method in need_ground_truth:
        if gold_standard is None:
            raise ValueError('Chosen evaluation metric %s requires gold standard set.' % method)
    elif method in inherent_metric:
        if data is None:
            raise ValueError('Chosen evaluation metric %s requires data input.' % method)
    else:
        raise ValueError('Evaluation metric %s not valid' % method)

    for col in label_df.columns:
        if method in need_ground_truth:
            clustered = (gold_standard != -1) & (label_df[col] != -1)
            compare_to = gold_standard[clustered]
        elif method in inherent_metric:
            clustered = (label_df[col] != -1)
            compare_to = data.loc[clustered]

        if len(label_df[col][clustered].value_counts()) <= 2:
            logging.warning('Condition %s does not have at least two clusters, skipping' %col)
            continue
        evaluation_results[col] = evaluations[method](
            compare_to, label_df[col][clustered], **metric_kwargs
        )

    return evaluation_results


def optimize_clustering(
        data,
        algorithm_names: Union[Iterable, str] = clusterers.keys(),
        algorithm_parameters: Optional[Dict[str, dict]] = None,
        random_search: bool = True,
        random_search_fraction: float = 0.3,
        algorithm_param_weights: Optional[dict] = None,
        algorithm_clus_kwargs: Optional[dict] = None,
        evaluation_method: Optional[str] = 'silhouette',
        gold_standard: Optional[Iterable] = None,
        metric_kwargs: Optional[dict] = None,
):

    if algorithm_param_weights is None:
        algorithm_param_weights = {}
    if algorithm_clus_kwargs is None:
        algorithm_clus_kwargs = {}
    if algorithm_parameters is None:
        algorithm_parameters = {}
    if metric_kwargs is None:
        metric_kwargs = {}

    if algorithm_names in list(speeds.keys()):
        algorithm_names = speeds[algorithm_names]

    clustering_labels = {}
    clustering_evaluations = {}
    for clusterer_name in algorithm_names:
        if clusterer_name not in clusterers.keys():
            logging.error('Algorithm %s not available, skipping. '% clusterer_name)
            continue

        label_df = run_conditions_one_algorithm(
            data,
            clusterer_name=clusterer_name,
            params=algorithm_parameters.get(clusterer_name, None),
            random_search=random_search,
            random_search_fraction=random_search_fraction,
            param_weights=algorithm_param_weights.get(clusterer_name, None),
            clus_kwargs=algorithm_clus_kwargs.get(clusterer_name, {})
        )
        if label_df is None:
            logging.warning(
                'Clusterer %s had no labeling results, skipping evaluation' % clusterer_name
                )
            continue
        clustering_labels[clusterer_name] = label_df

        evaluation_results = evaluate_results(
            label_df,
            method=evaluation_method,
            data=data,
            gold_standard=gold_standard,
            metric_kwargs=metric_kwargs
        )

        clustering_evaluations.update({
            tuple([clusterer_name, params_key]): value for params_key, value in evaluation_results.items()
        })

    top_choice = min_or_max[evaluation_method](clustering_evaluations, key=lambda k: clustering_evaluations[k])
    best_labels = clustering_labels[top_choice[0]][top_choice[1]]

    return best_labels, clustering_evaluations, clustering_labels

