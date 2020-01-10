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

***********************
Quickstart and examples
***********************
Quickstart: 

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
