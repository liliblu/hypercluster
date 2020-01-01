from hypercluster import visualize
import pandas as pd


def test_vis():
    evaluations = pd.read_csv('test_input/clustering/test_input_evaluations.txt', index_col=0)
    visualize.visualize_evaluations(
        evaluations, output_prefix='test_input/eval.vis', savefig=True
    )
