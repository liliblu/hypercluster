# Hypercluster
A package for clustering optimization with sklearn. 

### Requirements:  
pandas  
numpy  
scipy  
matplotlib  
seaborn  
scikit-learn  
hdbscan  

Optional:
snakemake


### Install  
With pip:
```
pip install hypercluster
```

or with conda:
```
conda install hypercluster
# or
conda create -c conda-forge -c bioconda hypercluster
```
If you are having problems installing with conda, try changing your channel priority. Priority of conda-forge > bioconda > defaults is recommended. 
To check channel priority: `conda config --get channels`
It should look like:
```
--add channels 'defaults'   # lowest priority
--add channels 'bioconda'
--add channels 'conda-forge'   # highest priority
```

If it doesn't look like that, try:
```
conda config --add channels bioconda
conda config --add channels conda-forge
```

### Docs 
https://hypercluster.readthedocs.io/en/latest/index.html   

### Examples
https://github.com/liliblu/hypercluster/tree/dev/examples


### Quickstart example
```python
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
```

