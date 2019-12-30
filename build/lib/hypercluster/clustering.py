from pandas import DataFrame
import pandas as pd, numpy as np
import logging
from typing import Optional, Iterable, Dict, Union
from itertools import product
from .constants import *


def calculate_row_weights(
        row: Iterable,
        param_weights: dict,
        vars_to_optimize: dict
) -> float:
    """
    Used to select random rows of parameter combinations using individual parameter weights.  

    Args:
        row:  Series of parameters, with parameter names as index.
        param_weights: Dictionary of str: dictionaries. Ex format - {'parameter_name':{'param_option_1':0.5, 'param_option_2':0.5}}.
        vars_to_optimize: Dictionary with possibilities for different parameters. Ex format - {'parameter_name':[1, 2, 3, 4, 5]}.

    Returns:
        Float representing the probability of seeing that combination of parameters,
        given their individual weights.

    """
    weights = []
    for var_lab, val in row.to_dict().items():
        weights.append(
            param_weights.get(var_lab, {}).get(val, (1/len(vars_to_optimize[var_lab])))
        )
    #TODO if probs are given to some options and not other, split the remaining probability,
    # don't just give equal prob.
    return np.prod(weights)


def cluster(clusterer_name: str, data: DataFrame, params: dict = {}):
    """
    Runs a given clusterer with a given set of parameters.

    Args:
        clusterer_name: String name of clusterer, for options see hypercluster.categories.clusterers.
        data: Dataframe with elements to cluster as index and examples as columns.  
        params: Dictionary of parameter names and values to feed into clusterer. Default {}.  

    Returns: 
        Instance of the clusterer fit with the data provided.  
    """
    clusterer = clusterers[clusterer_name](**params)
    return clusterer.fit(data)


class AutoClusterer:
    """
    Main hypercluster object.
    Args:
            clusterer_name: String name of clustererm for options see
        hypercluster.categories.clusterers..
            params_to_optimize: Dictionary with possibilities for different parameters. Ex format - {
        'parameter_name':[1, 2, 3, 4, 5]}. If None, will optimize default selection, given in
        hypercluster.constants.variables_to_optimize. Default None.
            random_search: Whether to search a random selection of possible parameters or all
        possibilites. Default True.
            random_search_fraction: If random_search is True, what fraction of the possible
            parameters to search. Default 0.5.
            param_weights: Dictionary of str: dictionaries. Ex format - {'parameter_name':{
        'param_option_1':0.5, 'param_option_2':0.5}}.
            clus_kwargs: Additional kwargs to pass into given clusterer, but not to be optimized.
        Default None.
        """
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
        """
        Uses info from init to make a Dataframe of all parameter sets that will be tried.
        Returns:
            self
        """
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
            #TODO change so that it makes a df with 1 row
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
        """
        Fits clusterer to data with each parameter set.
        Args:
            data: Dataframe with elements to cluster as index and examples as columns.

        Returns:
            self with self.labels_ assigned
        """
        if self.param_sets is None:
            logging.warning('No parameters to optimize for %s, cannot fit' % self.clusterer_name)
            self.labels_ = None
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
    """
    Uses a given metric to evaluate clustering results.  
    Args:
        label_df: Dataframe with elements to cluster as index and different labelings as columns
        method: Str of name of evaluation to use. For options see hypercluster.categories.evaluations. Default is silhouette.
        data: If using an inherent metric, must provide Dataframe of original data used to
        cluster. For options see hypercluster.constants.inherent_metric.
        gold_standard: If using a metric that compares to ground truth, must provide a set of
        gold standard labels. For options see hypercluster.constants.need_ground_truth.
        metric_kwargs: Additional kwargs to use in evaluation.

    Returns:
        Dictionary where every column from the label_df is a key and its evaluation is the value.
    """

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
        random_search_fraction: float = 0.5,
        algorithm_param_weights: Optional[dict] = None,
        algorithm_clus_kwargs: Optional[dict] = None,
        evaluation_method: Optional[str] = 'silhouette',
        gold_standard: Optional[Iterable] = None,
        metric_kwargs: Optional[dict] = None,
) -> tuple:
    """
    Runs through many clusterers and parameters to get best clustering labels.
    Args:
        data: Dataframe with elements to cluster as index and examples as columns.
        algorithm_names: Which clusterers to try. Default is all. For options see
        hypercluster.constants.clusterers. Can also put 'slow', 'fast' or 'fastest' for subset of clusterers. See hypercluster.constants.speeds.
        algorithm_parameters: Dictionary of str:dict, with parameters to optimize for each clusterer. Ex. structure:: {'clusterer1':{'param1':['opt1', 'opt2', 'opt3']}}.
        random_search: Whether to search a random selection of possible parameters or all possibilities. Default True.
        random_search_fraction: If random_search is True, what fraction of the possible parameters to search, applied to all clusterers. Default 0.5.
        algorithm_param_weights: Dictionary of str: dictionaries. Ex format - {'clusterer_name': {'parameter_name':{'param_option_1':0.5, 'param_option_2':0.5}}}.
        algorithm_clus_kwargs: Dictionary of additional kwargs per clusterer.
        evaluation_method: Str name of evaluation metric to use. For options see hypercluster.categories.evaluations. Default silhouette.
        gold_standard: If using a evaluation needs ground truth, must provide ground truth labels. For options see hypercluster.constants.need_ground_truth.
        metric_kwargs: Additional evaluation metric kwargs.  

    Returns:
        Best labels, dictionary of clustering evaluations, dictionary of all clustering labels
    """

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
        #TODO allow more evaluations, change output into a df like the smk
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
