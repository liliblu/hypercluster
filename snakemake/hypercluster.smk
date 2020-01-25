import pandas as pd,numpy as np
from hypercluster import utilities, visualize
import hypercluster
from hypercluster.constants import param_delim, val_delim
import os, subprocess
from shutil import copyfile
import yaml

subprocess.run(['mkdir', '-p', 'logs'])
targets = ['labels', 'evaluations']

configfile: 'config.yml'

input_data_folder = config['input_data_folder']
input_files = config['input_data_files']

output_folder = config['output_folder']
subprocess.run(['mkdir', '-p', output_folder])

intermediates_folder = config['intermediates_folder']
clustering_results = config['clustering_results']


def generate_parameters(config):
    parameters = config['optimization_parameters']
    all_params_to_test = []
    for clusterer, params in parameters.items():
        clus_kwargs = config['clusterer_kwargs'].get(clusterer, {})
        kwargs = config['generate_parameters_addtl_kwargs'].get(clusterer, {})
        df = hypercluster.AutoClusterer(
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

    with open('%s/params_to_test.yml' % output_folder, 'w') as fh:
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


def get_target_files(config):
    target_files = expand(
        '%s/{input_file}/%s/{targets}.txt' % (output_folder, clustering_results),
        input_file=input_files,
        targets=targets
    ) + expand(

        "%s/{input_file}/%s/{labs}_{targets}.txt" % (output_folder, intermediates_folder),
        input_file=input_files,
        labs=config["param_sets_labels"],
        targets=targets
    ) + expand(
            '%s/{input_file}/%s/evaluations.pdf' % (output_folder, clustering_results),
            input_file=input_files
    )

    if config['metric_to_choose_best']:
        target_files.append(
            expand(
                "%s/{input_file}/%s/best_parameters.txt" % (output_folder, clustering_results),
                input_file=input_files
            )
        )
    if config['metric_to_compare_labels']:
        target_files.append(
            expand(
                '%s/{input_file}/%s/%s_label_comparison.txt' % (
                    output_folder, clustering_results, config['metric_to_compare_labels']
                ),
                input_file=input_files
            )
        )
    if config['compare_samples']:
        target_files.append(
            expand(
                '%s/{input_file}/%s/sample_label_agreement.txt' % (output_folder, clustering_results),
                input_file=input_files
            )
        )
    if config['screeplot_evals']:
        target_files.append(
            expand(
                '%s/{input_file}/%s/scree_plots.{eval}.pdf' % (output_folder, clustering_results),
                input_file=input_files,
                eval=config['screeplot_evals']
            )
        )

    return target_files


generate_parameters(config)
files_to_generate = get_target_files(config)

rule all:
    input:
         files_to_generate


rule run_clusterer:
    input:
        infile = handle_ext
    output:
        "%s/{input_file}/%s/{labs}_labels.txt" % (output_folder, intermediates_folder)
    params:
        kwargs = lambda wildcards: config["param_sets"][wildcards.labs],
        readkwargs = lambda wildcards: config['read_csv_kwargs'].get(wildcards.input_file, {}),
        cluskwargs = config['clusterer_kwargs']
    run:
        df = pd.read_csv(input.infile, **params.readkwargs)
        kwargs = params.kwargs
        clusterer = kwargs.pop('clusterer')

        kwargs.update(params.cluskwargs.get(clusterer, {}))
        print(kwargs)
        cls = utilities.cluster(clusterer, df, kwargs)

        labs = pd.DataFrame(cls.labels_, index=df.index, columns=[wildcards.labs])
        labs.to_csv(output[0], sep = params.readkwargs.get('sep', ','))


rule run_evaluation:
    input:
        "%s/{input_file}/%s/{labs}_labels.txt" % (output_folder, intermediates_folder)
    output:
        "%s/{input_file}/%s/{labs}_evaluations.txt" % (output_folder, intermediates_folder)
    params:
        gold_standards = lambda wildcards: config['gold_standards'].get(wildcards.input_file, ''),
        input_data = handle_ext,
        readkwargs = lambda wildcards: config['read_csv_kwargs'].get(wildcards.input_file, {}),
        evals = config["evaluations"],
        evalkwargs = config["eval_kwargs"]
    run:
        readkwargs = {
            'index_col':params.readkwargs.get('index_col', 0),
            'sep':params.readkwargs.get('sep', ',')
        }
        test_labels = pd.read_csv(input[0], **params.readkwargs)
        if os.path.exists(params.gold_standards):
            gold_standard = pd.read_csv(
                '%s/%s' %(input_data_folder, params.gold_standards),
                **readkwargs
            )
            gold_standard = gold_standard[gold_standard.columns[0]]
        else:
            gold_standard = None

        data = pd.read_csv(params.input_data, **readkwargs)
        res = pd.DataFrame({'methods':params.evals})

        res[wildcards.labs] = res.apply(
            lambda row: utilities.evaluate_one(
                test_labels[test_labels.columns[0]],
                method=row['methods'],
                data=data,
                gold_standard=gold_standard,
                metric_kwargs=params.evalkwargs.get(row['methods'], None)
            ), axis=1
        )
        res = res.set_index('methods')
        res.to_csv(output[0], sep=readkwargs['sep'])


rule collect_dfs:
    input:
        files = expand(
            '%s/{{input_file}}/%s/{params_label}_{{targets}}.txt' % (
                output_folder, intermediates_folder
            ), params_label = config['param_sets_labels']
        )
    params:
        outputkwargs = lambda wildcards: config['output_kwargs'].get(wildcards.targets)
    output:
        '%s/{input_file}/%s/{targets}.txt' % (output_folder, clustering_results)
    run:
        kwargs = {
            'index_col':params.outputkwargs.get('index_col', 0),
            'sep':params.outputkwargs.get('sep', ',')
        }

        df = concat_dfs(input.files, kwargs)
        df.to_csv(
            output[0], sep = kwargs['sep'] # TODO see if this works for the rest
        )


rule visualize_evaluations:
    input:
        files = '%s/{input_file}/%s/evaluations.txt' % (
            output_folder, clustering_results
        )
    output:
        output_file = '%s/{input_file}/%s/evaluations.pdf' % (
            output_folder, clustering_results
        )
    params:
        heatmap_kwargs = config['heatmap_kwargs'],
        readkwargs = lambda wildcards: config['read_csv_kwargs'].get(wildcards.input_file, {})
    run:
        df = pd.read_csv(input.files, sep=params.readkwargs.get('sep', ','), index_col=0)

        visualize.visualize_evaluations(
            df, output_prefix=output.output_file.rsplit('.', 1)[0], savefig=True,
            **params.heatmap_kwargs
        )


rule pick_best_clusters:
    input:
        evals = '%s/{input_file}/%s/evaluations.txt' % (output_folder, clustering_results)
    output:
        "%s/{input_file}/%s/best_parameters.txt" % (output_folder, clustering_results),
    params:
        metric = config['metric_to_choose_best'],
        sep = lambda wcs: config['read_csv_kwargs'].get(wcs.input_file, {}).get('sep', ',')
    run:
        df = pd.read_csv(input.evals, sep=params.sep, index_col=0).transpose()
        labs = list(df[df[params.metric]==df[params.metric].max()].index)
        for lab in labs:
            copyfile(
                "%s/%s/%s/%s_labels.txt" % (
                    output_folder,
                    wildcards.input_file,
                    intermediates_folder,
                    lab
                ),
                "%s/%s/%s/%s_labels.txt" % (
                    output_folder,
                    wildcards.input_file,
                    clustering_results,
                    lab
                )
            )
            with open(output[0], 'a') as fh:
                fh.write('%s\n' % lab)

        visualize.visualize_for_picking_labels(
            df.transpose(),
            method=params.metric,
            savefig_prefix='%s/scree_plots.%s' % (
                output[0].rsplit('/', 1)[0], params.metric
            )
        )

rule compare_labels:
    input:
         labels = '%s/{input_file}/%s/labels.txt' % (output_folder, clustering_results)
    output:
        table = '%s/{input_file}/%s/%s_label_comparison.txt' % (
            output_folder, clustering_results, config['metric_to_compare_labels']
        )
    params:
        metric = config['metric_to_compare_labels'],
        readkwargs = lambda wildcards: config['read_csv_kwargs'].get(wildcards.input_file, {})
    run:
        kwargs = {
            'index_col':params.readkwargs.get('index_col', 0),
            'sep':params.readkwargs.get('sep', ',')
        }
        df = pd.read_csv(input.labels, **kwargs)
        df = df.corr(lambda x, y: utilities.evaluate_one(
            x, method=params.metric, gold_standard=y
        ))
        df.to_csv(output.table)

        visualize.visualize_pairwise(
            df,
            savefig=True,
            output_prefix=output.table.rsplit('.', 1)[0],
            method = params.metric,
            **config['heatmap_kwargs']
        )


rule compare_samples:
    input:
         labels = '%s/{input_file}/%s/labels.txt' % (output_folder, clustering_results)
    output:
         table = '%s/{input_file}/%s/sample_label_agreement.txt' % (output_folder,
                                                                     clustering_results)
    params:
          readkwargs = lambda wildcards: config['read_csv_kwargs'].get(wildcards.input_file, {})
    run:
        kwargs = {
            'index_col':params.readkwargs.get('index_col', 0),
            'sep':params.readkwargs.get('sep', ',')
        }
        df = pd.read_csv(input.labels, **kwargs).transpose()
        df = df.corr(
            lambda x, y: sum(np.equal(x[((x != -1) | (y != -1))], y[((x != -1) | (y != -1))]))
        )

        df.to_csv(output.table, sep = kwargs['sep'])

        visualize.visualize_pairwise(
            df,
            savefig=True,
            output_prefix=output.table.rsplit('.', 1)[0],
            method = '# same label',
            **config['heatmap_kwargs']
        )


rule draw_scree_plots:
    input:
         eval_df = '%s/{input_file}/%s/evaluations.txt' % (output_folder, clustering_results)
    output:
         pdfs = expand(
             '%s/{{input_file}}/%s/scree_plots.{eval}.pdf' % (output_folder, clustering_results),
             eval=config['screeplot_evals']
         )
    params:
        sep = lambda wcs: config['read_csv_kwargs'].get(wcs.input_file, {}).get('sep', ',')
    run:
        df = pd.read_csv(input.eval_df, sep=params.sep, index_col=0)
        for metric in config['screeplot_evals']:
            visualize.visualize_for_picking_labels(
                df,
                method=metric,
                savefig_prefix='%s/%s/%s/scree_plots.%s' % (
                    output_folder, wildcards.input_file, clustering_results, metric
                )
            )


