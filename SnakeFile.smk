import pandas as pd
from autocluster import clustering
from autocluster.clustering import (variables_to_optimize, evaluations)
import yaml
import subprocess
from datetime import datetime

configfile: 'config.yml'

input_data = config['input_data_file']
run_label = config['run_label']
optimization_parameters = config['optimization_parameters']
read_csv_kwargs = config['read_csv_kwargs']
evaluations = config['evaluations_to_calculate']
clustering_addtl_kwargs = config['clustering_addtl_kwargs']
generate_parameters_addtl_kwargs = config['generate_parameters_addtl_kwargs']

# now = datetime.now()
now = 'test'
intermediates_folder = 'clustering_intermediates.%s' % now
clustering_results = 'clustering.%s' % now

subprocess.run(['mkdir', '-p', clustering_results])

def concat_dfs(df_list):
    df = pd.DataFrame()
    for fil in df_list:
        temp = pd.read_csv(fil, index_col=0)
        df = pd.concat(df, temp, join='outer', axis=1)
    return df


def update_config(config, another_object, label):
    config[label] = another_object
    return config

def generate_parameters(config):
    kwargs = config['generate_parameters_addtl_kwargs']
    clusterer_kwargs = config['clusterer_kwargs']
    parameters = config['optimization_parameters']
    all_params_to_test = []
    for clusterer, params in parameters.items():
        clus_kwargs = clusterer_kwargs.get(clusterer, {})
        df = clustering.run_conditions_one_algorithm(
            return_parameters=True, params=params, clus_kwargs=clus_kwargs, **kwargs
        )
        df['clusterer'] = clusterer
        all_params_to_test.extend(df.to_dict('records'))

    final_param_sets = {}
    for param_set in all_params_to_test:
        clusterer = param_set['clusterer']
        lab = '.'.join([clusterer]+[
            '%s%s' % (k, v) for k, v in param_set.items() if k != 'clusterer'
        ])
        final_param_sets.update({lab:param_set})
    config = update_config(config, final_param_sets, 'param_sets')
    config = update_config(config, final_param_sets.keys(), 'param_sets_labels')
    with open('%s/params_to_test.yml' % clustering_results, 'w') as fh:
        yaml.dump(final_param_sets, fh)

generate_parameters(config)

rule all:
    input:
        # '{clustering_results}/{run_label}_labels.csv'
        '%s/%s_labels.csv' % (clustering_results, run_label)


rule run_clusterer:
    input:
        '%s/params_to_test.yml' % clustering_results,
        infile = input_data,
    output:
        '%s/{params_label}_labels.txt' % intermediates_folder
    params:
        kwargs = config['param_sets'],
        readkwargs = read_csv_kwargs,
        cluskwargs = clustering_addtl_kwargs
    run:
        df = pd.read_csv({input.infile}, **{params.readkwargs})

        params_label = {input.kwargs}.keys()[0]
        params = {input.kwargs}.values()
        clusterer = params.pop('clusterer')

        cls = clustering.cluster(clusterer, df, **params)
        labs = pd.DataFrame(cls.labels_, index=df.index, columns=[params_label])
        labs.to_csv('{intermediates_folder}/{params_label}_labels.txt')


rule collect_labels:
    input:
        expand(
            "%s/{labs}_labels.txt" % intermediates_folder,
            labs=config['param_sets_labels']
        )
    output:
        '%s/%s_labels.csv' % (clustering_results, run_label)
    run:
        files = expand(
            '{params_label}_labels.txt',
            input_data=inputs,
            params_labels=config['param_labels']
        )
        df = concat_dfs(files)
        df.to_csv('%s/%s_labels.csv' % (clustering_results, run_label))


# rule collect_evaluations:
#     input:
#         expand(
#             '{input_data}_{params_label}_evaluations.txt',
#             input_data=inputs,
#             params_labels=param_labels
#         ),
#         '{run_label}'
#     output:
#         "{clustering_results}/{run_label}_evaluations.csv"
#     run:
#         files = expand(
#             '{input_data}_{params_label}_evaluations.txt',
#             input_data=inputs,
#             params_labels=param_labels
#         )
#         df = concat_dfs(files)
#         df.to_csv('{run_label}_evaluations.csv')


#TODO how to remove intermediate files?
#TODO how to toggle removing intermediates?

#snakemake -pn -s SnakeFile.smk