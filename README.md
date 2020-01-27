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
conda install -c conda-forge -c bioconda hypercluster
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

It will also be useful to check out sklearn's page on [clustering](https://scikit-learn.org/stable/modules/clustering.html) 
and [evaluation metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) 

### Examples
https://github.com/liliblu/hypercluster/tree/dev/examples

### Quickstart with SnakeMake

Default `config.yml` and `hypercluster.smk` are in the snakemake repo above.  
Edit the `config.yml` file or arguments.
```bash
snakemake -s hypercluster.smk --configfile config.yml --config input_data_files=test_data input_data_folder=. 
```

Example editing with python:
```python
import yaml

with open('config.yml', 'r') as fh:
    config = yaml.load(fh)
    
input_data_prefix = 'test_data'
config['input_data_folder'] = os.path.abspath('.')
config['input_data_files'] = [input_data_prefix]
config['read_csv_kwargs'] = {input_data_prefix:{'index_col': [0]}}

with open('config.yml', 'w') as fh:
    yaml.dump(config, stream=fh)
```

Then call snakemake. 
```bash
snakemake -s hypercluster.smk
```

Or submit the snakemake scheduler as an sbatch job e.g. with BigPurple Slurm:
```bash
module add slurm
sbatch snakemake_submit.sh
```
Examples for `snakemake_submit.sh` and `cluster.json` is in the scRNA-seq example. 

### Quickstart with python
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
