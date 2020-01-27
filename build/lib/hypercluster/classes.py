from typing import List, Union
from pandas import DataFrame
from .utilities import *
from .visualize import *
from itertools import product
from .constants import *


class Clusterer:
    """Meta class for shared methods for both AutoClusterer and MultiAutoClusterer.  
    """
    def pick_best_labels(self, method: Optional[str] = None, min_or_max: Optional[str] = None):
        return pick_best_labels(self.evaluation_df, self.labels_df, method, min_or_max)

    def visualize_evaluations(
            self,
            savefig: bool = False,
            output_prefix: str = "evaluations",
            **heatmap_kws
    ) -> List[matplotlib.axes.Axes]:
        return visualize_evaluations(self.evaluation_df, savefig, output_prefix, **heatmap_kws)

    def visualize_sample_label_consistency(
            self,
            savefig: bool = False,
            output_prefix: Optional[str] = None,
            **heatmap_kws
    ) -> List[matplotlib.axes.Axes]:
        return visualize_sample_label_consistency(
            self.labels_df,
            savefig,
            output_prefix,
            **heatmap_kws
        )

    def visualize_label_agreement(
            self,
            method: Optional[str] = None,
            savefig: bool = False,
            output_prefix: Optional[str] = None,
            **heatmap_kws
    ) -> List[matplotlib.axes.Axes]:
        return visualize_label_agreement(
            self.labels_df,
            method,
            savefig,
            output_prefix,
            **heatmap_kws
        )

    def visualize_for_picking_labels(
            self,
            method: Optional[str] = None,
            savefig_prefix: Optional[str] = None
    ):
        return visualize_for_picking_labels(self.evaluation_df, method, savefig_prefix)

    def fit_predict(self, data: Optional[DataFrame], parameter_set_name, method, min_of_max):
        pass


class AutoClusterer (Clusterer):
    """Main hypercluster object.  

    Attributes: 
        clusterer_name (str): String name of clusterer.  
        params_to_optimize (dict): Dictionary with possibilities for different parameters. Ex \
        format - {'parameter_name':[1, 2, 3, 4, 5]}. If None, will optimize default \
        selection, given in hypercluster.constants.variables_to_optimize. Default None.  
        random_search (bool): Whether to search a random selection of possible parameters or \
        all possibilities. Default True.  
        random_search_fraction (float): If random_search is True, what fraction of the \
        possible parameters to search. Default 0.5.  
        param_weights (dict): Dictionary of str: dictionaries. Ex format - { \
        'parameter_name':{'param_option_1':0.5, 'param_option_2':0.5}}.  
        clus_kwargs (dict): Additional kwargs to pass into given clusterer, but not to be \
        optimized. Default None.  
        labels_ (Optional[DataFrame]): If already fit, labels DataFrame fit to data.  
        evaluation_ (Optional[DataFrame]): If already fit and evalute, evaluations per label.  
        data (Optional[DataFrame]): Data to fit, will not fit by default even if passed data.  
    """

    def __init__(
        self,
        clusterer_name: Optional[str] = "KMeans",
        params_to_optimize: Optional[dict] = None,
        random_search: bool = False,
        random_search_fraction: Optional[float] = 0.5,
        param_weights: dict = {},
        clus_kwargs: Optional[dict] = None,
        labels_: Optional[DataFrame] = None,
        evaluation_: Optional[DataFrame] = None,
        data: Optional[DataFrame] = None,
        labels_df: Optional[DataFrame] = None,
        evaluation_df: Optional[DataFrame] = None
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

        if labels_df is None and labels_ is not None:
            labels_df = generate_flattened_df(labels_)
        self.labels_df = labels_df

        if evaluation_df is None and evaluation_ is not None:
            evaluation_df = generate_flattened_df(evaluation_)
        self.evaluation_df = evaluation_df

        self.labels_ = labels_
        self.evaluation_ = evaluation_
        self.data = data

        self.static_kwargs = None
        self.total_possible_conditions = None
        self.param_sets = None
        self.generate_param_sets()

    def generate_param_sets(self):
        """Uses info from init to make a Dataframe of all parameter sets that will be tried. 

        Returns (AutoClusterer): 
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
        """Fits clusterer to data with each parameter set. 

        Args: 
            data (DataFrame): DataFrame with elements to cluster as index and features as columns.  

        Returns (AutoClusterer):  
            self
        """
        self.data = data
        if self.param_sets.shape == (0, 0):
            label_results = pd.DataFrame(
                cluster(self.clusterer_name, data).labels_,
                columns=["default_parameters"],
                index=data.index,
            )
            self.labels_ = label_results
            self.labels_df = generate_flattened_df({self.clusterer_name: label_results})
            return self

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
        self.labels_df = generate_flattened_df({self.clusterer_name: label_results})
        return self

    def evaluate(
            self,
            methods: Optional[Iterable[str]] = None,
            metric_kwargs: Optional[dict] = None,
            gold_standard: Optional[Iterable] = None
    ):
        """Evaluate labels with given metrics. 

        Args: 
            methods (Optional[Iterable[str]]): List of evaluation methods to use.  
            metric_kwargs (Optional[dict]): Additional kwargs per evaluation metric. Structure of \
            {'metric_name':{'param1':value, 'param2':val2}.  
            gold_standard (Optional[Iterable]): Gold standard labels, if available. Only needed \
            if using a metric that needs ground truth.  

        Returns (AutoClusterer):  
            self with attribute .evaluation_; a DataFrame with all eval values per labels.  

        """
        if self.labels_ is None:
            logging.error('Cannot evaluate model, need to fit first.')
        if methods is None:
            methods = inherent_metrics
        if metric_kwargs is None:
            metric_kwargs = {}

        evaluation_df = pd.DataFrame({"methods": methods})
        for col in self.labels_.columns:
            evaluation_df[col] = evaluation_df.apply(
                lambda row: evaluate_one(
                    self.labels_[col],
                    method=row["methods"],
                    data=self.data,
                    gold_standard=gold_standard,
                    metric_kwargs=metric_kwargs.get(row["methods"], None),
                ),
                axis=1,
            )
        evaluation_df = evaluation_df.set_index('methods')
        evaluation_df.columns = self.labels_.columns
        self.evaluation_ = evaluation_df
        self.evaluation_df = generate_flattened_df({self.clusterer_name: evaluation_df})
        return self


class MultiAutoClusterer (Clusterer):
    """Object for training multiple clustering algorithms.  

    Attributes: 
        algorithm_names (Optional[Union[Iterable, str]]): List of algorithm names to test OR \
        name of category of clusterers from hypercluster.constants.categories, OR None. If None, \
        default is hypercluster.constants.variables_to_optimize.keys().  
        algorithm_parameters (Optional[Dict[str, dict]]):  Dictionary of hyperparameters to \
        optimize. Example format: {'clusterer_name1':{'hyperparam1':[val1, val2]}}.  
        random_search (bool): Whether to search a random subsample of possible conditions.  
        random_search_fraction (float): If random_search, what fraction of conditions to search.  
        algorithm_param_weights (Dict[str, Dict[str, dict]]): If random_search, and you want to \
        give probability weights to certain parameters, dictionary of probability weights. \
        Example format: {'clusterer1': {'hyperparam1':{val1:probability1, val2:probability2}}}.  
        algorithm_clus_kwargs (Dict[str, dict]): Dictionary of additional keyword args for any \
        clusterer. Example format: {'clusterer1':{'param1':val1}}.  
        data (Optional[DataFrame]): Optional, data to fit. Will not fit even if passed, \
        need to call fit method.  
        evaluation_methods (Optional[List[str]]): List of metrics with which to evaluate. If \
        None, will use hypercluster.constants.inherent_metrics. Default is None.  
        metric_kwargs (Optional[Dict[str, dict]]): Additional keyword args for any metric \
        function. Example format: {'metric1':{'param1':value}}.  
        gold_standard (Optional[Iterable]): If using methods that need ground truth, vector of \
        correct labels. Can also pass in during evaluate.  
        autoclusterers (Iterable[AutoClusterer]): If building from initialized AutoClusterer \
        objects, can give a list of them here. If these are given, it will override anything
        passed to labels\_ and evaluation\_.  
        labels_ (Optional[Dict[str, DataFrame]]): Dictionary of label DataFrames per clusterer, \
        if already fit.  Example format: {'clusterer1': labels_df}.  
        evaluation_ (Optional[Dict[str, DataFrame]]): Dictionary of evaluation DataFrames per \
        clusterer, if already fit and evaluated.  Example format: {'clusterer1': evaluation_df}.  
        labels_df (Optional[DataFrame]): Combined DataFrame of all labeling results.  
        evaluation_df (Optional[DataFrame]): Combined DataFrame of all evaluation results.  
    """
    def __init__(
            self,
            algorithm_names: Optional[Union[Iterable, str]] = None,
            algorithm_parameters: Optional[Dict[str, dict]] = None,
            random_search: bool = False,
            random_search_fraction: Optional[float] = 0.5,
            algorithm_param_weights: Optional[dict] = None,
            algorithm_clus_kwargs: Optional[dict] = None,
            data: Optional[DataFrame] = None,
            evaluation_methods: Optional[List[str]] = None,
            metric_kwargs: Optional[Dict[str, dict]] = None,
            gold_standard: Optional[Iterable] = None,
            autoclusterers: Iterable[AutoClusterer] = None,
            labels_: Dict[str, AutoClusterer] = None,
            evaluation_: Dict[str, AutoClusterer] = None,
            labels_df: Optional[DataFrame] = None,
            evaluation_df: Optional[DataFrame] = None
    ):

        self.random_search = random_search
        self.random_search_fraction = random_search_fraction

        if autoclusterers is None:
            if algorithm_names in list(categories.keys()):
                algorithm_names = categories[algorithm_names]
            elif algorithm_names is None:
                algorithm_names = variables_to_optimize.keys()
            self.algorithm_names = algorithm_names

            if algorithm_parameters is None:
                algorithm_parameters = {
                    clus_name: variables_to_optimize[clus_name] for clus_name in
                    self.algorithm_names
                }
            self.algorithm_parameters = algorithm_parameters

            if algorithm_param_weights is None:
                algorithm_param_weights = {}
            self.algorithm_param_weights = algorithm_param_weights

            if algorithm_clus_kwargs is None:
                self.algorithm_clus_kwargs = {}

            if labels_ is None:
                labels_ = {}
            else:
                labels_df = generate_flattened_df(labels_)
            self.labels_ = labels_
            self.labels_df = labels_df

            if evaluation_ is None:
                evaluation_ = {}
            else:
                evaluation_df = generate_flattened_df(evaluation_)
            self.evaluation_ = evaluation_
            self.evaluation_df = evaluation_df

            autoclusterers = []
            for clus_name in self.algorithm_names:
                autoclusterers.append(AutoClusterer(
                    clus_name,
                    params_to_optimize=self.algorithm_parameters.get(clus_name, {}),
                    random_search = self.random_search,
                    random_search_fraction = self.random_search_fraction,
                    param_weights=self.algorithm_param_weights.get(clus_name, {}),
                    clus_kwargs=self.algorithm_clus_kwargs.get(clus_name, {}),
                    labels_=self.labels_.get(clus_name, None),
                    evaluation_=self.evaluation_.get(clus_name, None)
                ))
        else:
            self.algorithm_names = [ac.clusterer_name for ac in autoclusterers]
            self.algorithm_parameters = {
                ac.clusterer_name: ac.params_to_optimize for ac in autoclusterers
            }
            self.algorithm_param_weights = {
                ac.clusterer_name: ac.param_weights for ac in autoclusterers
            }
            self.algorithm_clus_kwargs = {
                ac.clusterer_name: ac.clus_kwargs for ac in autoclusterers
            }
            self.labels_ = {
                ac.clusterer_name: ac.labels_ for ac in autoclusterers if ac.labels_ is not None
            }
            self.evaluation_ = {
                ac.clusterer_name: ac.evaluation_
                for ac in autoclusterers if ac.evaluation_ is not None
            }

            self.labels_df = generate_flattened_df(self.labels_)
            self.evaluation_df = generate_flattened_df(self.evaluation_)

        self.autoclusterers = autoclusterers
        self.data = data
        self.evaluation_methods = evaluation_methods
        self.metric_kwargs = metric_kwargs
        self.gold_standard = gold_standard

    def fit(self, data: Optional[DataFrame] = None):
        if data is None:
            data = self.data
        if data is None:
            raise ValueError('Must initialize with data or pass data in function to fit.')
        self.data = data

        fitted_clusterers = []
        for clusterer in self.autoclusterers:
            fitted_clusterers.append(clusterer.fit(data))
        self.autoclusterers = fitted_clusterers
        self.labels_ = {
            ac.clusterer_name: ac.labels_ for ac in self.autoclusterers
        }
        self.labels_df = generate_flattened_df(self.labels_)
        return self

    def evaluate(
            self,
            evaluation_methods: Optional[list] = None,
            metric_kwargs: Optional[dict] = None,
            gold_standard: Optional[Iterable] = None
    ):
        if evaluation_methods is None and self.evaluation_methods is None:
            evaluation_methods = inherent_metrics
        elif evaluation_methods is None:
            evaluation_methods = self.evaluation_methods

        if metric_kwargs is None and self.metric_kwargs is None:
            metric_kwargs = {}
        elif metric_kwargs is None:
            metric_kwargs = self.metric_kwargs

        if gold_standard is None:
            gold_standard = self.gold_standard

        evaluated_clusterers = []
        for ac in self.autoclusterers:
            evaluated_clusterers.append(ac.evaluate(
                methods=evaluation_methods,
                metric_kwargs=metric_kwargs,
                gold_standard=gold_standard
            ))

        self.gold_standard = gold_standard
        self.metric_kwargs = metric_kwargs
        self.evaluation_methods = evaluation_methods

        self.autoclusterers = evaluated_clusterers
        self.evaluation_ = {
            ac.clusterer_name: ac.evaluation_ for ac in self.autoclusterers
        }
        self.evaluation_df = generate_flattened_df(self.evaluation_)
        return self
