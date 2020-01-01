import pandas as pd
from hypercluster import clustering, visualize
from hypercluster.constants import param_delim, val_delim
import os
import yaml


configfile: 'config.yml'

input_data_folder = config['input_data_folder']
input_files = config['input_data_files']
optimization_parameters = config['optimization_parameters']
read_csv_kwargs = config['read_csv_kwargs']
clustering_addtl_kwargs = config['clustering_addtl_kwargs']
generate_parameters_addtl_kwargs = config['generate_parameters_addtl_kwargs']


intermediates_folder = config['intermediates_folder']
clustering_results = config['clustering_results']
gold_standard_file = config['gold_standard_file']


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
        lab = param_delim.join([clusterer]+[
            '%s%s%s' % (k, val_delim, v) for k, v in param_set.items() if k != 'clusterer'
        ])
        final_param_sets.update({lab:param_set})
    config['param_sets'] = final_param_sets
    config['param_sets_labels'] = list(final_param_sets.keys())

    with open('params_to_test.yml', 'w') as fh:
        yaml.dump(final_param_sets, fh)


def handle_ext(wildcards):
    base = wildcards.input_file
    files = []
    for file_ext in [".csv", ".tsv", ".txt"]:
        file = '%s/%s%s' % (input_data_folder, base, file_ext)
        if os.path.exists(file):
            files.append(file)
    if len(files) == 1:
        return files[0]
    if len(files) > 1:
        raise ValueError(
        'Multiple files with prefix %s/%s can be found, must be unique' % (input_data_folder, base)
    )
    raise FileNotFoundError(
        'No .txt, .csv or .tsv files with prefix %s/%s can be found' % (input_data_folder, base)
    )


def concat_dfs(df_list, kwargs):
    results = pd.DataFrame()
    for fil in df_list:
        temp = pd.read_csv(fil, **kwargs)
        results = pd.concat([results, temp], join='outer', axis=1)
    return results


generate_parameters(config)


rule all:
    input:
        expand(
            '{input_file}/%s/{input_file}_{targets}.txt' % clustering_results,
            input_file=input_files,
            targets=config['targets']
        ) +
        expand(
            "{input_file}/%s/{labs}_labels.txt" % intermediates_folder,
            input_file=input_files,
            labs=config["param_sets_labels"]
         ) +
        expand(
            '{input_file}/%s/{input_file}_evaluations.pdf' % clustering_results,
            input_file=input_files
        )


rule run_clusterer:
    input:
        infile = handle_ext
    output:
        "{input_file}/%s/{labs}_labels.txt" % intermediates_folder
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
            '%s/%s/%s_labels.txt'% (wildcards.input_file, intermediates_folder, wildcards.labs),
            sep = read_csv_kwargs.get('sep', ',')
        )


rule run_evaluation:
    input:
        "{input_file}/%s/{labs}_labels.txt" % intermediates_folder
    output:
        "{input_file}/%s/{labs}_evaluations.txt" % intermediates_folder
    params:
        gold_standard_file = gold_standard_file,
        input_data = handle_ext,
        readkwargs = {
            'index_col':read_csv_kwargs.get('index_col', 0),
            'sep':read_csv_kwargs.get('sep', ',')
        },
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
                test_labels[test_labels.columns[0]],
                method=row['methods'],
                data=data,
                gold_standard=gold_standard,
                metric_kwargs=params.evalkwargs.get(row['methods'], None)
            ), axis=1
        )
        res = res.set_index('methods')
        res.to_csv(
            '%s/%s/%s_evaluations.txt'% (wildcards.input_file, intermediates_folder,
                                         wildcards.labs),
            sep=params.readkwargs['sep']
        )


rule collect_dfs:
    input:
        files = expand(
            '{{input_file}}/%s/{params_label}_{{targets}}.txt' % intermediates_folder,
            params_label = config['param_sets_labels'],
        )
    params:
        outputkwargs = config['output_kwargs']
    output:
        '{input_file}/%s/{input_file}_{targets}.txt' % clustering_results
    run:
        df = concat_dfs(input.files, params.outputkwargs[wildcards.targets])
        df.to_csv(
            '%s/%s/%s_%s.txt' % (
                wildcards.input_file, clustering_results,wildcards.input_file, wildcards.targets
            ), sep = read_csv_kwargs.get('sep', ',')
        )


rule visualize_evaluations:
    input:
        files = '{input_file}/%s/{input_file}_evaluations.txt' % clustering_results,
    output:
        output_file = '{input_file}/%s/{input_file}_evaluations.pdf' % clustering_results
    params:
        heatmap_kwargs = config['heatmap_kwargs']

    run:
        df = pd.read_csv(input.files, sep=read_csv_kwargs.get('sep', ','), index_col=0)
        visualize.visualize_evaluations(
            df, output_prefix=output.output_file.rsplit('.', 1)[0], savefig=True,
            **params.heatmap_kwargs
        )

#TODO add pairwise comparison bn results
#TODO add example where opt 2 things at once for 1 clusterer
