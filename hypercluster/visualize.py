from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from typing import List, Optional
from hypercluster.constants import param_delim, val_delim
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from hypercluster import clustering

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set(font="arial", style="white", color_codes=True, font_scale=1.3)
matplotlib.rcParams.update({"savefig.bbox": "tight"})
cmap = sns.cubehelix_palette(
    start=0,
    rot=0.4,
    gamma=1.0,
    hue=0.82,
    light=1,
    dark=0,
    reverse=False,
    as_cmap=True
)
cmap.set_over('black')
cmap.set_under('white')
cmap.set_bad("#DAE0E6")


def zscore(df):
    return df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)


def compute_order(
        df,
        dist_method="euclidean",
        cluster_method="average"
):
    dist_mat = pdist(df, metric=dist_method)
    link_mat = hierarchy.linkage(dist_mat, method=cluster_method)

    return df.index[hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link_mat, dist_mat))]


def visualize_evaluations(
    evaluations_df: DataFrame,
    output_prefix: str = "evaluations",
    savefig: bool = False,
    **heatmap_kws
) -> List[matplotlib.axes.Axes]:

    clusterers = sorted(
        list(set([i.split(param_delim, 1)[0] for i in evaluations_df.columns]))
    )
    evaluations_df = zscore(evaluations_df)
    width = 0.18 * (len(evaluations_df.columns) + 2 + (0.01 * (len(clusterers) - 1)))
    height = 0.22 * (len(evaluations_df))

    fig, axs = plt.subplots(
        figsize=(width, height),
        nrows=1,
        ncols=(len(clusterers) + 1),
        gridspec_kw=dict(
            width_ratios=[
                dict(
                    Counter(
                        [i.split(param_delim, 1)[0] for i in evaluations_df.columns]
                    )
                )[clus]
                for clus in clusterers
            ]
            + [2],
            wspace=0.01,
            left=0,
            right=1,
            top=1,
            bottom=0,
        ),
    )
    vmin = np.nanquantile(evaluations_df, 0.1)
    vmax = np.nanquantile(evaluations_df, 0.9)

    heatmap_kws['cmap'] = heatmap_kws.get('cmap', cmap)
    heatmap_kws['vmin'] = heatmap_kws.get('vmin', vmin)
    heatmap_kws['vmax'] = heatmap_kws.get('vmax', vmax)

    for i, clus in enumerate(clusterers):
        temp = evaluations_df[
            [
                col
                for col in evaluations_df.columns
                if clus == col.split(param_delim, 1)[0]
            ]
        ].transpose()

        temp.index = pd.MultiIndex.from_frame(
            pd.DataFrame(
                [
                    {
                        kv.split(val_delim)[0]: kv.split(val_delim)[1]
                        for kv in i.split(param_delim, 1)[1:]
                    }
                    for i in temp.index
                ]
            ).astype(float, errors='ignore')
        )
        temp = temp.sort_index()
        temp = temp.transpose()

        ax = axs[i]
        sns.heatmap(
            temp,
            ax=ax,
            yticklabels=temp.index,
            xticklabels=["-".join([str(i) for i in col]) for col in temp.columns],
            cbar_ax=axs[-1],
            cbar_kws=dict(label="z-score"),
            **heatmap_kws
        )
        ax.set_ylabel("")
        ax.set_title(clus)
        ax.set_yticklabels([])

    axs[0].set_ylabel("evaluation method")
    axs[0].set_yticklabels(temp.index, rotation=0)
    if savefig:
        plt.savefig("%s.pdf" % output_prefix)
    return axs


def visualize_pairwise(
        df: DataFrame,
        heatmap_kws: Optional[dict] = None,
        savefig: bool = True,
        output_prefix: str = "heatmap.pairwise",
        method: Optional[str] = None
):
    if heatmap_kws is None:
        heatmap_kws = {}

    vmin = np.nanquantile(df, 0.1)
    vmax = np.nanquantile(df, 0.9)

    heatmap_kws['cmap'] = heatmap_kws.get('cmap', cmap)
    heatmap_kws['vmin'] = heatmap_kws.get('vmin', vmin)
    heatmap_kws['vmax'] = heatmap_kws.get('vmax', vmax)
    cbar_kws = heatmap_kws.get('cbar_kws', {})
    cbar_kws['label'] = cbar_kws.get('label', method)
    heatmap_kws['cbar_kws'] = cbar_kws

    cbar_ratio = 2
    wspace = 0.01
    height = 0.18 * len(df)
    width = 0.18 * (len(df.columns)+cbar_ratio+wspace)
    fig, axs = plt.subplots(
        figsize=(width, height),
        nrows=1,
        ncols=2,
        gridspec_kw=dict(
            width_ratios=[len(df.columns), cbar_ratio],
            wspace=wspace,
            left=0,
            right=1,
            top=1,
            bottom=0,
        )
    )
    try:
        order = compute_order(df.fillna(df.median()))
    except ValueError:
        order = df.index
    df = df.loc[order, order]
    sns.heatmap(
        df,
        xticklabels=order,
        yticklabels=order,
        ax=axs[0],
        cbar_ax=axs[1],
        **heatmap_kws
    )
    if savefig:
        plt.savefig('%s.pdf' % output_prefix)

    return axs


def visualize_label_agreement_pairwise(
        labels: DataFrame,
        method: 'adjusted_rand_score',
        heatmap_kws: Optional[dict] = None,
        savefig: bool = True,
        output_prefix: str = "heatmap.labels.pairwise"
):
    labels = labels.corr(
        lambda x, y: clustering.evaluate_results(x, method=method, gold_standard=y)
    )
    return visualize_pairwise(labels, heatmap_kws, savefig, output_prefix, method=method)


def visualize_sample_labeling_pairwise(
        labels: DataFrame,
        heatmap_kws: Optional[dict] = None,
        savefig: bool = True,
        output_prefix: str = "heatmap.sample.pairwise"
):
    labels = labels.corr(lambda x, y: sum(np.equal(x, y)))
    return visualize_pairwise(labels, heatmap_kws, savefig, output_prefix, method='# same cluster')