hypercluster SnakeMake pipeline
===============================

Line-by-line explanation of config.yml
--------------------------------------

.. list-table:: Explanation for config.yml
   :widths: 33 33 33
   :header-rows: 1

   * - config.yml parameter
     - Explanation
     - Example from `scRNA-seq workflow <https://github.com/liliblu/hypercluster/tree/dev/examples/snakemake_scRNA_example>`_
   * - ``input_data_folder``
     - Path to folder in which input data can be found. No / at the end.
     - ``/input_data``
   * - ``input_data_files``
     - | List of prefixes of data files. Exclude extension, .csv, .tsv and .txt
       | allowed.
     - ``['input_data1', 'input_data2']``
   * - ``gold_standard_file``
     - | File name of gold_standard_file. Must have same pandas.read_csv kwargs
       | as the corresponding input file. Must be in input_data_folder.
     - ``{'input_data': 'gold_standard_file.txt'}``
   * - ``read_csv_kwargs``
     - | Per input data file, keyword args to put into `pandas.read_csv <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>`_.
       | **If specifying multiindex, also put the same in output_kwargs['labels']**
     - ``{'test_input': {'index_col':[0]}}``
   * - ``output_folder``
     - Path to folder in which results will be written. No / at the end.
     - ``/hypercluster_results``
   * - ``intermediates_folder``
     - | Name of the folder within the output_folder to put intermediate results,
       | such as labels and evaluations per condition. No need to change this usually.
     - ``clustering_intermediates``
   * - ``clustering_results``
     - | Name of the folder within the output_folder to put final results.
       | No need to change this usually.
     - ``clustering``
   * - ``clusterer_kwargs``
     - | Additional static keyword arguments to pass to individual clusterers.
       | Not to optimize.
     - ``KMeans: {'random_state':8}}``
   * - ``generate_parameters_addtl_kwargs``
     - Additonal keyword arguments for the hypercluster.AutoClusterer class.
     - ``{'KMeans': {'random_search':true, 'param_weights': {'n_clusters': {5: 0.25, 6:0.75}}}``
   * - ``evaluations``
     - | Names of evaluation metrics to use. See
       | hypercluster.constants.inherent_metrics or
       | hypercluster.constants.need_ground_truth
     - ``['silhouette_score', 'number_clustered']``
   * - ``eval_kwargs``
     - Additional kwargs per evaluation metric function.
     - ``{'silhouette_score': {'random_state': 8}}``
   * - ``metric_to_choose_best``
     - | If picking best labels, which metric to maximize to choose the labels. If not choosing
       | best labels, leave as empty string ('').
     - ``silhouette_score``
   * - ``metric_to_compare_labels``
     - | If comparing labeling result pairwise similarity, which metric to use. To not generate
       | this comparison, leave blank/or empty string.
     - ``adjusted_rand_score``
   * - ``compare_samples``
     - | Whether to made a table and figure with counts of how often two samples are in the same
       | cluster.
     - ``true``
   * - ``output_kwargs``
     - | pandas.to_csv and pandas.read_csv kwargs per output type. Generally,
       | don't need to change the evaluations kwargs, but labels index_col have to
       | match index_col like in the read_csv_kwargs.
     - ``{'evaluations': {'index_col':[0]},  'labels': {'index_col':[0]}}``
   * - ``heatmap_kwargs``
     - Additional kwargs for `seaborn.heatmap <https://seaborn.pydata.org/generated/seaborn.heatmap.html>`_ for visualizations.
     - ``{'vmin':-2, 'vmax':2}``
   * - ``optimization_parameters``
     - Fun part! This is where you put which hyperparameters per algorithm to try.
     - ``{'KMeans': {'n_clusters': [5, 6, 7]}}``

**Note: Formatting of lists and dictionaries can be in python syntax (like above) or yaml syntax, or a mixture, like below. **

config.yml example from `scRNA-seq workflow <https://github.com/liliblu/hypercluster/tree/dev/examples/snakemake_scRNA_example>`_
----------------------------------

.. code-block:: yaml

    input_data_folder: '.'
    input_data_files:
      - sc_data
    gold_standards:
      test_input: 'gold_standard.csv'
    read_csv_kwargs:
      test_input: {'index_col':[0]}

    output_folder: 'results'
    intermediates_folder: 'clustering_intermediates'
    clustering_results: 'clustering'

    clusterer_kwargs: {}
    generate_parameters_addtl_kwargs: {}

    evaluations:
      - silhouette_score
      - calinski_harabasz_score
      - davies_bouldin_score
      - number_clustered
      - smallest_largest_clusters_ratio
      - smallest_cluster_ratio
    eval_kwargs: {}

    metric_to_choose_best: silhouette_score
    metric_to_compare_labels: adjusted_rand_score
    compare_samples: true

    output_kwargs:
      evaluations:
        index_col: [0]
      labels:
        index_col: [0]
    heatmap_kwargs: {}

    optimization_parameters:
      HDBSCAN:
        min_cluster_size: &id002
        - 2
        - 3
        - 4
        - 5
      KMeans:
        n_clusters: &id001
        - 5
        - 6
        - 7
      MiniBatchKMeans:
        n_clusters: *id001
      OPTICS:
        min_samples: *id002
