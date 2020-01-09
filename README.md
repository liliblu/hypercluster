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
```
pip install hypercluster
```
or
```
conda install -c bioconda hypercluster
```
Right now there are issue with the bioconda install on linux. Try the pip, if you are having problems. 

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
clusterer = hypercluster.utilities.AutoClusterer()
clusterer.fit(data).evaluate(
  methods = hypercluster.constants.need_ground_truth+hypercluster.constants.inherent_metrics, 
  gold_standard = labels
  )

hypercluster.visualize.visualize_evaluations(clusterer.evaluation_, multiple_clusterers=False)

# With a range of algorithms

evals, labels_df, labels_dict = optimize_clustering(data)

hypercluster.visualize.visualize_evaluations(evals)

```

