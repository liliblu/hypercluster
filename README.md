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
from sklearn.datasets import make_blobs
import hypercluster

data, labels = make_blobs()

clusterer = hypercluster.AutoClusterer()
clusterer.fit(data).evaluate(
  methods = hypercluster.constants.need_ground_truth+hypercluster.constants.inherent_metric, 
  gold_standard = labels
  )

hypercluster.visualize.visualize_evaluations(clusterer.evaluations_)
```

