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
    'hdbscan': HDBSCAN,
    'kmeans': KMeans,
    'minibatchkmeans': MiniBatchKMeans,
    'affinitypropagation': AffinityPropagation,
    'meanshift': MeanShift,
    'optics': OPTICS,
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


def calculate_row_weights(row, param_weights, vars_to_optimize):
    weights = []
    for var_lab, val in row.to_dict().items():
        weights.append(
            param_weights.get(var_lab, {}).get(val, (1/len(vars_to_optimize[var_lab])))
        )
    return np.prod(weights)


def cluster(clusterer_name, data, params: dict = {}):
    clusterer = clusterers[clusterer_name](**params)
    return clusterer.fit(data)


class AutoClusterer:
    def __init__(
            self,
            clusterer_name: Optional[str] = 'hdbscan',
            params_to_optimize: Optional[dict] = None,
            random_search: bool = True,
            random_search_fraction: float = 0.5,
            param_weights: dict = {},
            clus_kwargs: Optional[dict] = None
    ):
        self.clusterer_name = clusterer_name
        self.params_to_optimize = params_to_optimize
        self.random_search = random_search
        self.random_search_fraction = random_search_fraction
        self.param_weights = param_weights
        self.clus_kwargs = clus_kwargs

        if self.params_to_optimize is None:
            self.params_to_optimize = variables_to_optimize[clusterer_name]
        if self.clus_kwargs is None:
            self.clus_kwargs = {}

        self.labels_ = None
        self.static_kwargs = None
        self.total_possible_conditions = None
        self.param_sets = None
        self.generate_param_sets()
        self.labels_ = None

    def generate_param_sets(self):
        total_kwargs = self.clus_kwargs
        total_kwargs.update(self.params_to_optimize)

        conditions = 1
        vars_to_optimize = {}
        static_kwargs = {}
        for parameter_name, possible_values in self.clus_kwargs.items():
            if len(possible_values) > 1:
                vars_to_optimize[parameter_name] = possible_values
                conditions *= len(possible_values)
            else:
                static_kwargs[parameter_name] = possible_values
        if conditions == 1:
            logging.error(
                'Clusterer %s was only given one set of parameters, nothing to optimize.'
                % self.clusterer_name
            )
            self.param_sets = None
            return self

        self.static_kwargs = static_kwargs
        self.total_possible_conditions = conditions

        parameters = pd.DataFrame(columns=list(vars_to_optimize.keys()))
        for row in iter(product(*vars_to_optimize.values())):
            parameters = parameters.append(
                dict(zip(vars_to_optimize.keys(), row)), ignore_index=True
            )
        if self.random_search:
            will_search = int(conditions * self.random_search_fraction)
            # calculates probability of getting a particular set of parameters, given the probs of
            # all the individual params. If a prob isn't set, give uniform probability to each
            # parameter.
            if self.param_weights:
                weights = parameters.apply(
                    lambda row: calculate_row_weights(row, self.param_weights, vars_to_optimize),
                    axis=1
                )
            else:
                weights = None
            parameters = parameters.sample(will_search, weights=weights)

        logging.info(
            'For clusterer %s, testing %s out of %s possible conditions'
            % (self.clusterer_name, len(parameters), conditions)
        )
        self.param_sets = parameters
        return self

    def fit(self, data: DataFrame):
        if self.param_sets is None:
            logging.error('No parameters to optimize for %s, cannot fit' % self.clusterer_name)
            return self

        label_results = pd.DataFrame(columns=self.param_sets.columns.union(data.index))
        for i, row in self.param_sets.iterrows():
            single_params = row.to_dict()
            single_params.update(self.static_kwargs)
            labels = cluster(self.clusterer_name, data, single_params).labels_

            label_row = dict(zip(data.index, labels))
            label_row.update(single_params)
            label_results = label_results.append(label_row, ignore_index=True)
            logging.info('%s - %s of conditions done' % (i, (i / self.total_possible_conditions)))

        label_results = label_results.set_index(list(self.param_sets.columns)).transpose()
        self.labels_ = label_results
        return self


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

        label_df = AutoClusterer(
            clusterer_name=clusterer_name,
            params_to_optimize=algorithm_parameters.get(clusterer_name, None),
            random_search=random_search,
            random_search_fraction=random_search_fraction,
            param_weights=algorithm_param_weights.get(clusterer_name, None),
            clus_kwargs=algorithm_clus_kwargs.get(clusterer_name, {})
        ).fit(data).labels_
        if label_df is None:
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
            (clusterer_name, params_key): value for params_key, value in
            evaluation_results.items()
        })

    top_choice = min_or_max[evaluation_method](
        clustering_evaluations,
        key=lambda k: clustering_evaluations[k]
    )
    best_labels = clustering_labels[top_choice[0]][top_choice[1]]

    return best_labels, clustering_evaluations, clustering_labels
