import pandas as pd, numpy as np
from autocluster import clustering
import os
import yaml
import subprocess

configfile: 'config.yml'

input_data_folder = config['input_data_folder']
input_files = config['input_data_file']
optimization_parameters = config['optimization_parameters']
read_csv_kwargs = config['read_csv_kwargs']
clustering_addtl_kwargs = config['clustering_addtl_kwargs']
generate_parameters_addtl_kwargs = config['generate_parameters_addtl_kwargs']


intermediates_folder = config['intermediates_folder']
clustering_results = config['clustering_results']
gold_standard_file = config['gold_standard_file']


def concat_dfs(df_list, **read_csv_kwargs):
    df = pd.DataFrame()
    for fil in df_list:
        temp = pd.read_csv(fil, **read_csv_kwargs)
        df = pd.concat([df, temp], join='outer', axis=1)
    return df


def generate_parameters(config):
    kwargs = config['generate_parameters_addtl_kwargs']
    clusterer_kwargs = config['clusterer_kwargs']
    parameters = config['optimization_parameters']
    all_params_to_test = []
    for clusterer, params in parameters.items():
        clus_kwargs = clusterer_kwargs.get(clusterer, {})
        df = clustering.AutoClusterer(
            clusterer_name=clusterer,
            params_to_optimize=params,
            clus_kwargs=clus_kwargs,
            **kwargs
        ).param_sets
        df['clusterer'] = clusterer
        all_params_to_test.extend(df.to_dict('records'))
    #TODO why is random search not working? getting key not found errors
    final_param_sets = {}
    for param_set in all_params_to_test:
        clusterer = param_set['clusterer']
        lab = '.'.join([clusterer]+[
            '%s-%s' % (k, v) for k, v in param_set.items() if k != 'clusterer'
        ])
        final_param_sets.update({lab:param_set})
    config['param_sets'] = final_param_sets
    config['param_sets_labels'] = list(final_param_sets.keys())

    with open('params_to_test.yml', 'w') as fh:
        yaml.dump(final_param_sets, fh)


generate_parameters(config)


rule all:
    input:
        expand(
            '{input_file}/%s/{input_file}_{targets}.csv' % clustering_results,
            input_file=input_files,
            targets=config['targets']
        )
        # 'brca-combined-v4.0-phosphoproteome-ratio-norm-unfiltered-dedup-prot-normed-top10000std/%s/%s.csv' % (intermediates_folder, 'minibatchkmeans.n_clusters37_evaluations')

rule run_clusterer:
    input:
        infile = '%s/{input_file}.csv' % input_data_folder,
    output:
        "{input_file}/%s/{labs}_labels.csv" % intermediates_folder
    params:
        kwargs = lambda wildcards: config["param_sets"][wildcards.labs],
        readkwargs = read_csv_kwargs,
        cluskwargs = clustering_addtl_kwargs
    run:
        df = pd.read_csv(input.infile, **params.readkwargs)

        clusterer = params.kwargs.pop('clusterer')
        cls = clustering.cluster(clusterer, df, params.kwargs)

        labs = pd.DataFrame(cls.labels_, index=df.index, columns=[wildcards.labs])
        labs.to_csv(
            '%s/%s/%s_labels.csv'% (wildcards.input_file, intermediates_folder, wildcards.labs)
        )


rule run_evaluation:
    input:
        "{input_file}/%s/{labs}_labels.csv" % intermediates_folder
    output:
        "{input_file}/%s/{labs}_evaluations.csv" % intermediates_folder
    params:
        gold_standard_file = gold_standard_file,
        input_data = '%s/{input_file}.csv' % input_data_folder,
        readkwargs = read_csv_kwargs,
        evals = config["evaluations"],
        evalkwargs = config["eval_kwargs"]
    run:
        test_labels = pd.read_csv(input[0], **params.readkwargs)
        if os.path.exists(params.gold_standard_file):
            gold_standard = pd.read_csv(params.gold_standard_file, **params.readkwargs)
        else:
            gold_standard = None
        data = pd.read_csv(params.input_data, **params.readkwargs)
        res = pd.DataFrame({'methods':params.evals})

        res[wildcards.labs] = res.apply(
            lambda row: clustering.evaluate_results(
                test_labels,
                method=row['methods'],
                data=data,
                gold_standard=gold_standard,
                metric_kwargs=params.evalkwargs.get(row['methods'], {})
            ).get(wildcards.labs, np.nan), axis=1
        )
        res = res.set_index('methods')
        res.to_csv(
            '%s/%s/%s_evaluations.csv'% (wildcards.input_file, intermediates_folder, wildcards.labs)
        )


rule collect_dfs:
    input:
        files = expand(
            '{{input_file}}/%s/{params_label}_{{targets}}.csv' % intermediates_folder,
            params_label = config['param_sets_labels'],
        )
    params:
        readkwargs = config['output_kwargs']
    output:
        '{input_file}/%s/{input_file}_{targets}.csv' % (clustering_results)
    run:
        df = concat_dfs(input.files, **params.readkwargs[wildcards.targets])
        df.to_csv(
            '%s/%s/%s_%s.csv' % (
                wildcards.input_file, clustering_results,wildcards.input_file, wildcards.targets
            )
        )


#TODO add pairwise comparison bn results
#TODO add pick best labels
#TODO add heatmaps for vis
#TODO add example where opt 2 things at once for 1 clusterer
