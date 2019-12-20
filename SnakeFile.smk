import pandas as pd
from autocluster import clustering
import os
import yaml
import subprocess

configfile: 'config.yml'

input_data = config['input_data_file']
run_label = config['run_label']
optimization_parameters = config['optimization_parameters']
read_csv_kwargs = config['read_csv_kwargs']
clustering_addtl_kwargs = config['clustering_addtl_kwargs']
generate_parameters_addtl_kwargs = config['generate_parameters_addtl_kwargs']


intermediates_folder = config['intermediates_folder']
clustering_results = config['clustering_results']
addtl_files_folder = config['addtl_files_folder']

subprocess.run(['mkdir', '-p', clustering_results])

def concat_dfs(df_list):
    df = pd.DataFrame()
    for fil in df_list:
        temp = pd.read_csv(fil, index_col=0)
        df = pd.concat([df, temp], join='outer', axis=1)
    return df


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
    #TODO why is random search not working? getting key not found errors
    final_param_sets = {}
    for param_set in all_params_to_test:
        clusterer = param_set['clusterer']
        lab = '.'.join([clusterer]+[
            '%s%s' % (k, v) for k, v in param_set.items() if k != 'clusterer'
        ])
        final_param_sets.update({lab:param_set})
    config['param_sets'] = final_param_sets
    config['param_sets_labels'] = list(final_param_sets.keys())

    with open('%s/params_to_test.yml' % clustering_results, 'w') as fh:
        yaml.dump(final_param_sets, fh)

generate_parameters(config)


rule all:
    input:
        '%s/%s_labels.csv' % (clustering_results, run_label),
         '%s/%s_evaluations.csv' % (clustering_results, run_label)


rule run_clusterer:
    input:
        infile = input_data,
    output:
        "%s/{labs}_labels.txt" % intermediates_folder
    params:
        kwargs = lambda wildcards: config["param_sets"][wildcards.labs],
        readkwargs = read_csv_kwargs,
        cluskwargs = clustering_addtl_kwargs
    run:
        df = pd.read_csv(input.infile, **params.readkwargs)

        params_label = wildcards.labs
        params = params.kwargs
        clusterer = params.pop('clusterer')

        cls = clustering.cluster(clusterer, df, params)

        labs = pd.DataFrame(cls.labels_, index=df.index, columns=[params_label])
        labs.to_csv('%s/%s_labels.txt'% (intermediates_folder, params_label))


rule run_evaluation:
    input:
        "%s/{labs}_labels.txt" % intermediates_folder
    output:
        "%s/{labs}_evaluations.txt" % intermediates_folder
    params:
        gold_standard_file = "%s/gold_standard.txt" % addtl_files_folder,
        input_data = input_data,
        readkwargs = read_csv_kwargs,
        evals = config["evaluations"],
        evalkwargs = config["eval_kwargs"]
    run:
        test_labels = pd.read_csv(input[0], index_col=0)
        if os.path.exists(params.gold_standard_file):
            gold_standard = pd.read_csv(params.gold_standard_file, **params.readkwargs)
        else:
            gold_standard = None
        data = pd.read_csv(params.input_data, **params.readkwargs)
        res = pd.DataFrame({'methods':params.evals})
        print(res)
        res[wildcards.labs] = res.apply(
            lambda row: clustering.evaluate_results(
                test_labels,
                method=row['methods'],
                data=data,
                gold_standard=gold_standard,
                metric_kwargs=params.evalkwargs.get(row['methods'], {})
            ), axis=1
        )
        res = res.set_index('methods')
        res.to_csv('%s/%s_evaluations.txt'% (intermediates_folder, wildcards.labs))


rule collect_dfs:
    input:
        files = expand(
            '%s/{params_label}_{{targets}}.txt' % intermediates_folder,
            params_label = config['param_sets_labels'],
        )
    output:
        '%s/{run_label}_{targets}.csv' % (clustering_results)
    run:
        df = concat_dfs(input.files)
        df.to_csv('%s/%s_%s.csv' % (clustering_results, run_label, wildcards.targets))


#TODO Snakemake provides experimental support for dynamic files. Dynamic files can be used whenever
# one has a rule, for which the number of output files is unknown before the rule was executed. This is useful for example with certain clustering algorithms:
#snakemake -pn -s SnakeFile.smk