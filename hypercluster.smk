import pandas as pd,numpy as np
from hypercluster import clustering, visualize
from hypercluster.constants import param_delim, val_delim
import os, subprocess
from shutil import copyfile
import yaml


configfile: 'config.yml'

input_data_folder = config['input_data_folder']
input_files = config['input_data_files']
optimization_parameters = config['optimization_parameters']
read_csv_kwargs = config['read_csv_kwargs']
clusterer_kwargs = config['clusterer_kwargs']
generate_parameters_addtl_kwargs = config['generate_parameters_addtl_kwargs']


intermediates_folder = config['intermediates_folder']
clustering_results = config['clustering_results']
gold_standards = config['gold_standards']


def generate_parameters(config):
    kwargs = config['generate_parameters_addtl_kwargs']
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
        )
        + expand(
            "{input_file}/%s/{labs}_{targets}.txt" % intermediates_folder,
            input_file=input_files,
            labs=config["param_sets_labels"],
            targets=config['targets']
         )
        + expand(
            '{input_file}/%s/{input_file}_evaluations.pdf' % clustering_results,
            input_file=input_files
        )
        + expand(
            "{input_file}/%s/best_parameters.txt" % clustering_results,
            input_file=input_files
        )
        + expand(
            '{input_file}/%s/%s_label_comparison.txt' % (
                clustering_results, config['metric_to_compare_labels']
            ),
            input_file=input_files
        )
        # + expand(
        #     '{input_file}/%s/sample_label_agreement.txt' % clustering_results,
        #     input_file=input_files
        # )


rule run_clusterer:
    input:
        infile = handle_ext
    output:
        "{input_file}/%s/{labs}_labels.txt" % intermediates_folder
    params:
        kwargs = lambda wildcards: config["param_sets"][wildcards.labs],
        readkwargs = lambda wildcards: read_csv_kwargs.get(wildcards.input_file, {}),
        cluskwargs = clusterer_kwargs
    run:
        df = pd.read_csv(input.infile, **params.readkwargs)
        kwargs = params.kwargs
        clusterer = kwargs.pop('clusterer')

        kwargs.update(clusterer_kwargs.get(clusterer, {}))
        cls = clustering.cluster(clusterer, df, kwargs)

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
        gold_standards = lambda wildcards: gold_standards.get(wildcards.input_file, ''),
        input_data = handle_ext,
        readkwargs = lambda wildcards: read_csv_kwargs.get(wildcards.input_file, {}),
        evals = config["evaluations"],
        evalkwargs = config["eval_kwargs"]
    run:
        readcsv_kwargs = {
            'index_col':params.readkwargs.get('index_col', 0),
            'sep':params.readkwargs.get('sep', ',')
        }
        test_labels = pd.read_csv(input[0], **params.readkwargs)
        if os.path.exists(params.gold_standards):
            gold_standard = pd.read_csv(
                '%s/%s' %(input_data_folder, params.gold_standards),
                **readcsv_kwargs
            )
        else:
            gold_standard = None

        data = pd.read_csv(params.input_data, **readcsv_kwargs)
        res = pd.DataFrame({'methods':params.evals})

        res[wildcards.labs] = res.apply(
            lambda row: clustering.evaluate_results(
                test_labels[test_labels.columns[0]],
                method=row['methods'],
                data=data,
                gold_standard=gold_standard[gold_standard.columns[0]],
                metric_kwargs=params.evalkwargs.get(row['methods'], None)
            ), axis=1
        )
        res = res.set_index('methods')
        res.to_csv(
            '%s/%s/%s_evaluations.txt'% (
                wildcards.input_file,
                intermediates_folder,
                wildcards.labs
            ), sep=readcsv_kwargs['sep']
        )


rule collect_dfs:
    input:
        files = expand(
            '{{input_file}}/%s/{params_label}_{{targets}}.txt' % intermediates_folder,
            params_label = config['param_sets_labels'],
        )
    params:
        outputkwargs = lambda wildcards: config['output_kwargs'].get(wildcards.targets)
    output:
        '{input_file}/%s/{input_file}_{targets}.txt' % clustering_results
    run:
        kwargs = {
            'index_col':params.outputkwargs.get('index_col', 0),
            'sep':params.outputkwargs.get('sep', ',')
        }

        df = concat_dfs(input.files, kwargs)
        df.to_csv(
            '%s/%s/%s_%s.txt' % (
                wildcards.input_file,
                clustering_results,
                wildcards.input_file,
                wildcards.targets
            ), sep = kwargs['sep']
        )


rule visualize_evaluations:
    input:
        files = '{input_file}/%s/{input_file}_evaluations.txt' % clustering_results
    output:
        output_file = '{input_file}/%s/{input_file}_evaluations.pdf' % clustering_results
    params:
        heatmap_kwargs = config['heatmap_kwargs']

    run:
        df = pd.read_csv(
            input.files, sep=read_csv_kwargs.get(
                wildcards.input_file, {}
            ).get('sep', ','), index_col=0
        )

        visualize.visualize_evaluations(
            df, output_prefix=output.output_file.rsplit('.', 1)[0], savefig=True,
            **params.heatmap_kwargs
        )



rule pick_best_clusters:
    input:
        evals = '{input_file}/%s/{input_file}_evaluations.txt' % clustering_results
    output:
        "{input_file}/%s/best_parameters.txt" % clustering_results
    params:
        metric = config['metric_to_choose_best']
    run:
        subprocess.run([
            'touch',
            "%s/%s/best_labels.txt" % (
                wildcards.input_file,
                clustering_results,
            )
        ])

        df = pd.read_csv(
            input.evals, sep=read_csv_kwargs.get(
                wildcards.input_file, {}
            ).get('sep', ','), index_col=0
        ).transpose()
        labs = list(df[df[params.metric]==df[params.metric].max()].index)
        for lab in labs:
            copyfile(
                "%s/%s/%s_labels.txt" % (
                    wildcards.input_file,
                    intermediates_folder,
                    lab
                ),
                "%s/%s/%s_labels.txt" % (
                    wildcards.input_file,
                    clustering_results,
                    lab
                )
            )
            with open(
                    "%s/%s/best_labels.txt" % (wildcards.input_file,clustering_results), 'a'
            ) as fh:
                fh.write('%s\n' % lab)

rule compare_labels:
    input:
         labels = '{input_file}/%s/{input_file}_labels.txt' % clustering_results
    output:
        table = '{input_file}/%s/%s_label_comparison.txt' % (
            clustering_results, config['metric_to_compare_labels']
        )
    params:
        make_label_fig = config['make_label_fig'],
        metric = config['metric_to_compare_labels']
    run:
        df = pd.read_csv(input.labels, **read_csv_kwargs.get(wildcards.input_file, {}))
        df = df.corr(lambda x, y: clustering.evaluate_results(
            x, method=params.metric, gold_standard=y
        ))
        df.to_csv(
            '%s/%s/%s_label_comparison.txt' % (
                wildcards.input_file, clustering_results, params.metric
            )
        )
        if params.make_label_fig:
            visualize.visualize_pairwise(
                df,
                config['heatmap_kwargs'],
                output_prefix='%s/%s/%s_label_comparison' % (
                    wildcards.input_file, clustering_results, params.metric
                )
            )


rule compare_samples:
    input:
         labels = '{input_file}/%s/{input_file}_labels.txt' % clustering_results
    output:
          table = '{input_file}/%s/sample_label_agreement.txt' % clustering_results
    params:
          make_sample_fig = config['make_sample_fig']
    run:
        df = pd.read_csv(input.labels, **read_csv_kwargs.get(wildcards.input_file, {})).transpose()
        df = df.corr(lambda x, y: sum(np.equal(x, y)))
        df.to_csv('%s/%s/sample_label_agreement.txt' % (wildcards.input_file, clustering_results))
        if params.make_sample_fig:
            visualize.visualize_pairwise(
                df,
                config['heatmap_kwargs'],
                output_prefix='%s/%s/sample_agreement' % (wildcards.input_file, clustering_results)
            )
