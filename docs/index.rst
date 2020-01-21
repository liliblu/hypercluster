.. Hypercluster documentation master file, created by
   sphinx-quickstart on Mon Dec 30 16:45:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for hypercluster
==============================

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   hypercluster
   snakemake


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Installation and logistics
==========================

************
Installation
************

Available via pip::

  pip install hypercluster

Or bioconda::

  conda install hypercluster
   # or
  conda install -c conda-forge -c bioconda hypercluster

If you are having problems installing with conda, try changing your channel priority. Priority of
conda-forge > bioconda > defaults is recommended.

To check channel priority: :code:`conda config --get channels`


It should look like::

   --add channels 'defaults'   # lowest priority
   --add channels 'bioconda'
   --add channels 'conda-forge'   # highest priority

If it doesn't look like that, try::

   conda config --add channels bioconda
   conda config --add channels conda-forge

*********************************************
Quick reference for clustering and evaluation
*********************************************

.. list-table:: Clustering algorithms
   :widths: 50 50
   :header-rows: 1

   * - Clusterer
     - Type
   * - KMeans/MiniBatch KMeans
     - Partitioner
   * - Affinity Propagation
     - Partitioner
   * - Mean Shift
     - Partitioner
   * - DBSCAN
     - Clusterer
   * - OPTICS
     - Clusterer
   * - Birch
     - Partitioner
   * - OPTICS
     - Clusterer
   * - HDBSCAN
     - Clusterer
   * - NMF
     - Partitioner


.. list-table:: Evaluations
   :widths: 50 50
   :header-rows: 1

   * - Metric
     - Type
   * - adjusted_rand_score
     - Needs ground truth
   * - adjusted_mutual_info_score
     - Needs ground truth
   * - homogeneity_score
     - Needs ground truth
   * - completeness_score
     - Needs ground truth
   * - fowlkes_mallows_score
     - Needs ground truth
   * - mutual_info_score
     - Needs ground truth
   * - v_measure_score
     - Needs ground truth
   * - silhouette_score
     - Inherent metric
   * - calinski_harabasz_score
     - Inherent metric
   * - davies_bouldin_score
     - Inherent metric
   * - smallest_largest_clusters_ratio
     - Inherent metric
   * - number_of_clusters
     - Inherent metric
   * - smallest_cluster_size
     - Inherent metric
   * - largest_cluster_size
     - Inherent metric


***********************
Quickstart and examples
***********************

With snakemake:
--------------
.. code-block::

   snakemake -s hypercluster.smk --configfile config.yml --config input_data_files=test_data input_data_folder=.

With python:
-----------
.. code-block:: python

   import pandas as pd
   from sklearn.datasets import make_blobs
   import hypercluster

   data, labels = make_blobs()
   data = pd.DataFrame(data)
   labels = pd.Series(labels, index=data.index, name='labels')

   # With a single clustering algorithm
   clusterer = hypercluster.AutoClusterer()
   clusterer.fit(data).evaluate(
     methods = hypercluster.constants.need_ground_truth+hypercluster.constants.inherent_metrics,
     gold_standard = labels
     )

   clusterer.visualize_evaluations()

   # With a range of algorithms

   clusterer = hypercluster.MultiAutoClusterer()
   clusterer.fit(data).evaluate(
     methods = hypercluster.constants.need_ground_truth+hypercluster.constants.inherent_metrics,
     gold_standard = labels
     )

   clusterer.visualize_evaluations()

Example work flows for both python and snakemake are
`here <https://github.com/liliblu/hypercluster/tree/dev/examples/>`_

Source code is available `here <https://github.com/liliblu/hypercluster/>`_
