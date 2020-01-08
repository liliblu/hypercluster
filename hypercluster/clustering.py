from sklearn.cluster import *
from sklearn.metrics import *
from .additional_clusterers import *
from .metrics import *
from pandas import DataFrame
import pandas as pd
import numpy as np
import logging
from typing import Optional, Iterable, Dict, Union
from itertools import product
from .constants import *


def calculate_row_weights(
    row: Iterable, param_weights: dict, vars_to_optimize: dict
) -> float:
    """
    Used to select random rows of parameter combinations using individual parameter weights.  

    Args:
        row (Iterable):  Series of parameters, with parameter names as index.  
        param_weights (dict): Dictionary of str: dictionaries. Ex format - {'parameter_name':{
        'param_option_1':0.5, 'param_option_2':0.5}}.  
        vars_to_optimize (Iterable): Dictionary with possibilities for different parameters. Ex 
        format - {'parameter_name':[1, 2, 3, 4, 5]}.  

    Returns:
        Float representing the probability of seeing that combination of parameters, given their
        individual weights.  

    """
    weights = []
    for var_lab, val in row.to_dict().items():
        weights.append(
            param_weights.get(var_lab, {}).get(
                val, (1 / len(vars_to_optimize[var_lab]))
            )
        )
    # TODO if probs are given to some options and not other, split the remaining probability,
    # don't just give equal prob.
    return np.prod(weights)


def cluster(clusterer_name: str, data: DataFrame, params: dict = {}):
    """
    Runs a given clusterer with a given set of parameters.  

    Args:
        clusterer_name (str): String name of clusterer.  
        data (DataFrame): Dataframe with elements to cluster as index and examples as columns.  
        params (dict): Dictionary of parameter names and values to feed into clusterer. Default {}  

    Returns: 
        Instance of the clusterer fit with the data provided.   
    """
    clusterer = eval(clusterer_name)(**params)
    return clusterer.fit(data)


class AutoClusterer:
    """
    Main hypercluster object.  
    Args:
        clusterer_name (str): String name of clusterer  
        params_to_optimize (dict): Dictionary with possibilities for different parameters. Ex
        format - {'parameter_name':[1, 2, 3, 4, 5]}. If None, will optimize default
        selection, given in hypercluster.constants.variables_to_optimize. Default None.  
        random_search (bool): Whether to search a random selection of possible parameters or
        all possibilites. Default True.  
        random_search_fraction (float): If random_search is True, what fraction of the
        possible parameters to search. Default 0.5.  
        param_weights (dict): Dictionary of str: dictionaries. Ex format - {
        'parameter_name':{'param_option_1':0.5, 'param_option_2':0.5}}.  
        clus_kwargs (dict): Additional kwargs to pass into given clusterer, but not to be
        optimized. Default None.  
    """

    def __init__(
        self,
        clusterer_name: Optional[str] = "hdbscan",
        params_to_optimize: Optional[dict] = None,
        random_search: bool = True,
        random_search_fraction: float = 0.5,
        param_weights: dict = {},
        clus_kwargs: Optional[dict] = None,
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
        conditions = 1
        vars_to_optimize = {}
        static_kwargs = {}
        for parameter_name, possible_values in self.params_to_optimize.items():
            if len(possible_values) == 1:
                static_kwargs[parameter_name] = possible_values
            elif len(possible_values) > 1:
                vars_to_optimize[parameter_name] = possible_values
                conditions *= conditions * len(possible_values)
            else:
                logging.error(
                    "Parameter %s was given no possibilities. Will continue with default "
                    "parameters."
                    % parameter_name
                )

        self.static_kwargs = static_kwargs
        self.total_possible_conditions = conditions

        parameters = pd.DataFrame(columns=list(vars_to_optimize.keys()))
        for row in iter(product(*vars_to_optimize.values())):
            parameters = parameters.append(
                dict(zip(vars_to_optimize.keys(), row)), ignore_index=True
            )

        if self.random_search and len(parameters) > 1:
            will_search = int(conditions * self.random_search_fraction)

            # calculates probability of getting a particular set of parameters, given the probs of
            # all the individual params. If a prob isn't set, give uniform probability to each
            # parameter.
            if self.param_weights:
                weights = parameters.apply(
                    lambda param_set: calculate_row_weights(
                        param_set, self.param_weights, vars_to_optimize
                    ),
                    axis=1,
                )
            else:
                weights = None
            parameters = parameters.sample(will_search, weights=weights)

        for col in static_kwargs.keys():
            parameters[col] = static_kwargs[col]

        logging.info(
            "For clusterer %s, testing %s out of %s possible conditions"
            % (self.clusterer_name, len(parameters), conditions)
        )

        self.param_sets = parameters
        return self

    def fit(self, data: DataFrame):
        """
        Fits clusterer to data with each parameter set.  
        Args:
            data (DataFrame): Dataframe with elements to cluster as index and features as columns.  

        Returns:
            self with self.labels_ assigned  
        """

        if self.param_sets.shape == (0, 0):
            labels_results = pd.DataFrame(
                cluster(self.clusterer_name, data).labels_,
                columns=["default_parameters"],
                index=data.index,
            )

        label_results = pd.DataFrame(columns=self.param_sets.columns.union(data.index))
        for i, row in self.param_sets.iterrows():
            single_params = row.to_dict()
            labels = cluster(self.clusterer_name, data, single_params).labels_

            label_row = dict(zip(data.index, labels))
            label_row.update(single_params)
            label_results = label_results.append(label_row, ignore_index=True)
            logging.info(
                "%s - %s of conditions done" % (i, (i / self.total_possible_conditions))
            )
        if len(self.param_sets.columns) > 0:
            label_results = label_results.set_index(
                list(self.param_sets.columns)
            ).transpose()

        if isinstance(data.index, pd.MultiIndex):
            label_results.index = pd.MultiIndex.from_tuples(label_results.index)
        self.labels_ = label_results
        return self


def evaluate_results(
    labels: Iterable,
    method: str = "silhouette_score",
    data: Optional[DataFrame] = None,
    gold_standard: Optional[Iterable] = None,
    metric_kwargs: Optional[dict] = None,
) -> dict:
    """
    Uses a given metric to evaluate clustering results.  
    Args:
        labels (Iterable): Series of labels.  
        method (str): Str of name of evaluation to use. Default is silhouette.  
        data (DataFrame): If using an inherent metric, must provide Dataframe with which to
        calculate the metric.  
        gold_standard (Iterable): If using a metric that compares to ground truth, must provide a set of
        gold standard labels.  
        metric_kwargs (dict): Additional kwargs to use in evaluation.  

    Returns:  
        Metric value   
    """
    if isinstance(labels, pd.Series) is False:
        labels = pd.Series(labels)
    if len(labels[labels != -1].value_counts()) < 2:
        logging.error(
            "Condition %s does not have at least two clusters, skipping" % labels.name
        )
        return np.nan

    if metric_kwargs is None:
        metric_kwargs = {}

    if method in need_ground_truth:
        if gold_standard is None:
            raise ValueError(
                "Chosen evaluation metric %s requires gold standard set." % method
            )
        clustered = (gold_standard != -1) & (labels != -1)
        compare_to = gold_standard[clustered]

    elif method in inherent_metrics:
        if data is None:
            raise ValueError(
                "Chosen evaluation metric %s requires data input." % method
            )
        clustered = labels != -1
        compare_to = data.loc[clustered]
    else:
        compare_to = None
        clustered = labels.index

    return eval(method)(compare_to, labels[clustered], **metric_kwargs)


def optimize_clustering(
    data,
    algorithm_names: Union[Iterable, str] = variables_to_optimize.keys(),
    algorithm_parameters: Optional[Dict[str, dict]] = None,
    random_search: bool = True,
    random_search_fraction: float = 0.5,
    algorithm_param_weights: Optional[dict] = None,
    algorithm_clus_kwargs: Optional[dict] = None,
    evaluation_methods: Optional[list] = None,
    gold_standard: Optional[Iterable] = None,
    metric_kwargs: Optional[dict] = None,
) -> tuple:
    """
    Runs through many clusterers and parameters to get best clustering labels.  
    Args:
        data (DataFrame): Dataframe with elements to cluster as index and examples as columns.
        algorithm_names (Iterable[str]): Which clusterers to try. Default is in
        variables_to_optimize. Can also  put 'slow', 'fast' or 'fastest' for subset of
        clusterers. See hypercluster.constants.categories.  
        algorithm_parameters (dict): Dictionary of str:dict, with parameters to optimize for each
        clusterer. Ex. structure:: {'clusterer1':{'param1':['opt1', 'opt2', 'opt3']}}.  
        random_search (bool): Whether to search a random selection of possible parameters or all
        possibilities. Default True.  
        random_search_fraction (float): If random_search is True, what fraction of the possible
        parameters to search, applied to all clusterers. Must be between 0 and 1. Default 0.5.  
        algorithm_param_weights (dict): Dictionary of str: dictionaries. Ex format - {
        'clusterer_name': {'parameter_name':{'param_option_1':0.5, 'param_option_2':0.5}}}.  
        algorithm_clus_kwargs (dict): Dictionary of additional kwargs per clusterer.
        evaluation_methods (Iterable[str]): Str name of evaluation metric to use. Default
        all inherent metrics.  
        gold_standard (Iterable): If using a evaluation needs ground truth, must provide ground
        truth labels. For options see hypercluster.constants.need_ground_truth.  
        metric_kwargs (dict): Additional evaluation metric kwargs.  

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
    if evaluation_methods is None:
        evaluation_methods = inherent_metrics

    if algorithm_names in list(categories.keys()):
        algorithm_names = categories[algorithm_names]

    clustering_labels = {}
    clustering_labels_df = pd.DataFrame()
    for clusterer_name in algorithm_names:
        label_df = (
            AutoClusterer(
                clusterer_name=clusterer_name,
                params_to_optimize=algorithm_parameters.get(clusterer_name, None),
                random_search=random_search,
                random_search_fraction=random_search_fraction,
                param_weights=algorithm_param_weights.get(clusterer_name, None),
                clus_kwargs=algorithm_clus_kwargs.get(clusterer_name, None),
            )
            .fit(data)
            .labels_
        )

        clustering_labels[clusterer_name] = label_df

        # Put all parameter labels into 1 for a big df
        label_df = label_df.transpose()
        cols_for_labels = label_df.index.to_frame()

        inds = cols_for_labels.apply(
            lambda row: param_delim.join(
                [clusterer_name]
                + ["%s%s%s" % (k, val_delim, v) for k, v in row.to_dict().items()]
            ),
            axis=1,
        )

        label_df.index = inds
        label_df = label_df.transpose()
        clustering_labels_df = pd.concat(
            [clustering_labels_df, label_df], join="outer", axis=1
        )

    evaluation_results_df = pd.DataFrame({"methods": evaluation_methods})
    for col in clustering_labels_df.columns:
        evaluation_results_df[col] = evaluation_results_df.apply(
            lambda row: evaluate_results(
                clustering_labels_df[col],
                method=row["methods"],
                data=data,
                gold_standard=gold_standard,
                metric_kwargs=metric_kwargs.get(row["methods"], None),
            ),
            axis=1,
        )
    evaluation_results_df = evaluation_results_df.set_index('methods')
    return evaluation_results_df, clustering_labels_df, clustering_labels


def pick_best_labels(
        evaluation_results_df: DataFrame,
        clustering_labels_df: DataFrame,
        method: str = "silhouette_score",
        min_or_max: str = 'max'
) -> Iterable:
    """
    From evaluations and a metric to minimize or maximize, return all labels with top pick.  
    Args:
        evaluation_results_df (DataFrame): Evaluations DataFrame from optimize_clustering  
        clustering_labels_df (DataFrame: Labels DataFrame from optimize_clustering   
        method (str): Method with which to choose the best labels.  
        min_or_max (str): Whether to minimize or maximize the metric. Must be 'min' or 'max'  

    Returns:  
        DataFrame of All top labels.  
    """
    best_labels = evaluation_results_df.loc[method, :]
    if min_or_max == 'min':
        best_labels = best_labels.index[best_labels==best_labels.min()]
        return clustering_labels_df[best_labels]
    elif min_or_max == 'max':
        best_labels = best_labels.index[best_labels == best_labels.max()]
        return clustering_labels_df[best_labels]
    logging.error('min_or_max must be either min or max, %s invalid choice' % min_or_max)
