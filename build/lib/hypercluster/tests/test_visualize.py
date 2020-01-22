from hypercluster import visualize
import hypercluster
import numpy as np
import pandas as pd


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


def test_vis_eval():
    clusterer = hypercluster.MultiAutoClusterer().fit(test_data).evaluate()
    visualize.visualize_evaluations(clusterer.evaluation_df)
    clusterer.visualize_evaluations(savefig=True)
# 
    clusterer = hypercluster.AutoClusterer().fit(test_data).evaluate()
    visualize.visualize_evaluations(clusterer.evaluation_df)
    clusterer.visualize_evaluations()


def test_vis_sample():
    clusterer = hypercluster.MultiAutoClusterer().fit(test_data).evaluate()
    visualize.visualize_sample_label_consistency(clusterer.labels_df)
    clusterer.visualize_sample_label_consistency()

    clusterer = hypercluster.AutoClusterer().fit(test_data).evaluate()
    visualize.visualize_sample_label_consistency(clusterer.labels_df)
    clusterer.visualize_sample_label_consistency()


def test_vis_labels():
    clusterer = hypercluster.MultiAutoClusterer().fit(test_data).evaluate()
    visualize.visualize_label_agreement(clusterer.labels_df)
    clusterer.visualize_label_agreement(
        savefig=True,
    )

    clusterer = hypercluster.AutoClusterer().fit(test_data).evaluate()
    visualize.visualize_label_agreement(clusterer.labels_df)
    clusterer.visualize_label_agreement()