from hypercluster import utilities
from hypercluster.constants import *
import hypercluster
import pandas as pd
import numpy as np


test_data = pd.DataFrame(
        np.array(
            [[1, 2], [-1.8, 4], [1, -0.5],
             [10, 2], [-10, 4], [10, 0],
             [np.nan, 5], [3.2, np.nan], [0, 14],
             [-16.4, 3.67], [13.22, -3], [3.3, np.nan],
             [42, np.nan], [-8, 2], [1.2, 12],
             [np.nan, 2.1], [0.25, np.nan], [0.1, 1.11],
             [-44, 0], [-0.22, -0.11], [2.34, 6.7],
             [-10, np.nan], [-2.3, -2.5], [np.nan, 0],
             [np.nan, 22], [8.6, -7.5], [0, 14],
             [-6.4, 23.67], [-3.22, 3], [np.nan, np.nan],
             [-20, 2.01], [0.25, -.25], [0.455, 0.233],
             [np.nan, -0.89], [19, np.nan], [np.nan, np.nan],
             [-29, 3.6], [-13, -3], [3.3, np.nan],
             [-4, np.nan], [-0.2, -0.1], [0.34, 0.7]]
        )
)


test_data['ind1'] = 'a'
test_data['ind2'] = range(len(test_data))
test_data = test_data.set_index(['ind1', 'ind2'])
test_data = test_data.fillna(test_data.median())

test_ground_truth = pd.Series(
    np.random.randint(0, 2, size=(len(test_data), )),
    index=test_data.index
)


def test_cluster_one():
    # Test all clusterers are working with default params
    for clus_name in variables_to_optimize.keys():
        utilities.cluster(clus_name, test_data)

    # Test with putting extra params in there
    for clus_name in variables_to_optimize.keys():
        vars = variables_to_optimize[clus_name]
        key = list(vars.keys())[0]
        params = {key: vars[key][0]}
        # grabbing a variable and making sure var passing works
        utilities.cluster(clus_name, test_data, params)


def test_autoclusterer():
    for clus_name in variables_to_optimize.keys():
        hypercluster.AutoClusterer(clus_name).fit(test_data)
    for clus_name in variables_to_optimize.keys():
        hypercluster.AutoClusterer(clus_name, random_search=False).fit(test_data)


def test_param_weights():
    for clus_name in variables_to_optimize.keys():
        weights = {
            param: {value: (1/len(values)) for value in values} for param, values in
            variables_to_optimize[
                clus_name
            ].items()
        }
        hypercluster.AutoClusterer(clus_name, param_weights=weights).fit(
            test_data
        )
    for clus_name in variables_to_optimize.keys():
        hypercluster.AutoClusterer(clus_name, random_search=False).fit(test_data)


def test_passing_kwargs_for_a_clusterer():
    clus_name = 'KMeans'

    hypercluster.AutoClusterer(clus_name, clus_kwargs={'max_iter': 50}).fit(
        test_data
    )


def test_evaluate_results():
    labs = hypercluster.AutoClusterer('KMeans').fit(test_data).labels_
    for metric in inherent_metrics + need_ground_truth:
        utilities.evaluate_one(
            labs[labs.columns[0]], metric, data=test_data, gold_standard=test_ground_truth
        )


def test_multiauto():
    hypercluster.MultiAutoClusterer().fit(test_data).evaluate()
