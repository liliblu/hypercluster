from sklearn.cluster import *
from sklearn.metrics import *
from .additional_clusterers import *
from .additional_metrics import *
from pandas import DataFrame
import pandas as pd
import numpy as np
import logging
from typing import Optional, Iterable, Dict
from .constants import *


def calculate_row_weights(
    row: Iterable, param_weights: dict, vars_to_optimize: dict
) -> float:
    """Used to select random rows of parameter combinations using individual parameter weights.  

    Args: 
        row (Iterable):  Series of parameters, with parameter names as index.  
        param_weights (dict): Dictionary of str: dictionaries. Ex format - {'parameter_name':{ \
        'param_option_1':0.5, 'param_option_2':0.5}}.  
        vars_to_optimize (Iterable): Dictionary with possibilities for different parameters. Ex \
        format - {'parameter_name':[1, 2, 3, 4, 5]}.  

    Returns (float): 
        Float representing the probability of seeing that combination of parameters, given their \
        individual weights.  

    """
    param_weights.update({
        param: {
            val: param_weights.get(param, {}).get(
                val, (1-sum(param_weights.get(param, {}).values()))/len([
                    notweighted for notweighted in vars_to_optimize.get(param,  {}) 
                    if notweighted not in param_weights.get(param, {}).keys()
                ])
            )
        } for param, vals in vars_to_optimize.items() for val in vals
    })

    return np.prod([param_weights[param][val] for param, val in row.to_dict().items()])


def cluster(clusterer_name: str, data: DataFrame, params: dict = {}):
    """Runs a given clusterer with a given set of parameters.  

    Args: 
        clusterer_name (str): String name of clusterer.  
        data (DataFrame): Dataframe with elements to cluster as index and examples as columns.  
        params (dict): Dictionary of parameter names and values to feed into clusterer. Default {}  

    Returns: 
        Instance of the clusterer fit with the data provided.   
    """
    clusterer = eval(clusterer_name)(**params)
    return clusterer.fit(data)


def evaluate_one(
    labels: Iterable,
    method: str = "silhouette_score",
    data: Optional[DataFrame] = None,
    gold_standard: Optional[Iterable] = None,
    metric_kwargs: Optional[dict] = None,
) -> dict:
    """Uses a given metric to evaluate clustering results.  

    Args: 
        labels (Iterable): Series of labels.  
        method (str): Str of name of evaluation to use. Default is silhouette.  
        data (DataFrame): If using an inherent metric, must provide DataFrame with which to \
        calculate the metric.  
        gold_standard (Iterable): If using a metric that compares to ground truth, must provide a \
        set of gold standard labels.  
        metric_kwargs (dict): Additional kwargs to use in evaluation.  

    Returns (float):  
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


def generate_flattened_df(df_dict: Dict[str, DataFrame]) -> DataFrame:
    """Takes dictionary of results from many clusterers and makes 1 dataframe. 

    Args: 
        df_dict (Dict[str, DataFrame]): Dictionary of dataframes to flatten. Can be .labels_ or \
        .evaluations_ from MultiAutoClusterer.  

    Returns: 
        Flattened DataFrame with all data. 
    """
    merged_df = pd.DataFrame()
    for clus_name, df in df_dict.items():
        df = df.transpose()
        cols_for_labels = df.index.to_frame()
        inds = cols_for_labels.apply(
            lambda row: param_delim.join(
                [clus_name] + ["%s%s%s" % (k, val_delim, v) for k, v in row.to_dict().items()]
            ),
            axis=1,
        )
        df.index = inds
        df = df.transpose()

        merged_df = pd.concat(
            [merged_df, df], join="outer", axis=1
        )
    return merged_df


def pick_best_labels(
        evaluation_results_df: DataFrame,
        clustering_labels_df: DataFrame,
        method: Optional[str] = None,
        min_or_max: Optional[str] = None
) -> Iterable:
    """From evaluations and a metric to minimize or maximize, return all labels with top pick.  

    Args: 
        evaluation_results_df (DataFrame): Evaluations DataFrame from optimize_clustering.  
        clustering_labels_df (DataFrame: Labels DataFrame from optimize_clustering.  
        method (str): Method with which to choose the best labels.  
        min_or_max (str): Whether to minimize or maximize the metric. Must be 'min' or 'max'  

    Returns (DataFrame): 
        DataFrame of all top labels.  
    """
    if method is None:
        method = "silhouette_score"
    if min_or_max is None:
        min_or_max = 'max'

    best_labels = evaluation_results_df.loc[method, :]
    if min_or_max == 'min':
        best_labels = best_labels.index[best_labels == best_labels.min()]
        return clustering_labels_df[best_labels]
    elif min_or_max == 'max':
        best_labels = best_labels.index[best_labels == best_labels.max()]
        return clustering_labels_df[best_labels]
    logging.error('min_or_max must be either min or max, %s invalid choice' % min_or_max)
