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
from hypercluster import utilities

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
    """Row zscores a DataFrame, ignores np.nan   

    Args: 
        df (DataFrame): DataFrame to z-score  

    Returns (DataFrame): 
        Row-zscored DataFrame.  
    """
    return df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)


def compute_order(
        df,
        dist_method: str = "euclidean",
        cluster_method: str = "average"
):
    """Gives hierarchical clustering order for the rows of a DataFrame  

    Args: 
        df (DataFrame): DataFrame with rows to order.  
        dist_method (str):  Distance method to pass to scipy.cluster.hierarchy.linkage.    
        cluster_method (str): Clustering method to pass to scipy.spatial.distance.pdist.  

    Returns (pandas.Index): 
        Ordered row index.  

    """
    dist_mat = pdist(df, metric=dist_method)
    link_mat = hierarchy.linkage(dist_mat, method=cluster_method)

    return df.index[hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link_mat, dist_mat))]


def visualize_evaluations(
    evaluations_df: DataFrame,
    savefig: bool = False,
    output_prefix: str = "evaluations",
    **heatmap_kws
) -> List[matplotlib.axes.Axes]:
    """Makes a z-scored visualization of all evaluations.  

    Args: 
        evaluations_df (DataFrame): Evaluations dataframe from clustering.optimize_clustering  
        output_prefix (str): If saving a figure, file prefix to use.  
        savefig (bool): Whether to save a pdf
        **heatmap_kws: Additional keyword arguments to pass to seaborn.heatmap.  

    Returns (List[matplotlib.axes.Axes]): 
        List of all matplotlib axes.  

    """
    clusterers = sorted(
        list(set([i.split(param_delim, 1)[0] for i in evaluations_df.columns]))
    )
    width_ratios = [
            dict(
                Counter(
                    [i.split(param_delim, 1)[0] for i in evaluations_df.columns]
                )
            )[clus]
            for clus in clusterers
        ]
    clus_cols = {
        clus: [col for col in evaluations_df.columns if clus == col.split(param_delim, 1)[0]] 
        for clus in clusterers
    }

    evaluations_df = zscore(evaluations_df)
    width = 0.18 * (len(evaluations_df.columns) + 2 + (0.01 * (len(clusterers) - 1)))
    height = 0.22 * (len(evaluations_df))

    fig, axs = plt.subplots(
        figsize=(width, height),
        nrows=1,
        ncols=(len(clusterers) + 1),
        gridspec_kw=dict(
            width_ratios=width_ratios + [2],
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
            clus_cols[clus]
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
        savefig: bool = False,
        output_prefix: Optional[str] = None,
        method: Optional[str] = None, 
        **heatmap_kws
) -> List[matplotlib.axes.Axes]:
    """Visualize symmetrical square DataFrames.  

    Args: 
        df (DataFrame): DataFrame to visualize.    
        savefig (bool): Whether to save a pdf.  
        output_prefix (str): If saving a pdf, file prefix to use.  
        method (str): Label for cbar, if relevant.   
        **heatmap_kws: Additional keywords to pass to `seaborn.heatmap`_   

    Returns (List[matplotlib.axes.Axes]): 
        List of matplotlib axes for figure.  

    .. _seaborn.heatmap:
        https://seaborn.pydata.org/generated/seaborn.heatmap.html
    """
    heatmap_kws = {**heatmap_kws}

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
        if output_prefix is None:
            output_prefix = "heatmap.pairwise"
        plt.savefig('%s.pdf' % output_prefix)

    return axs


def visualize_label_agreement(
        labels: DataFrame,
        method: Optional[str] = None,
        savefig: bool = False,
        output_prefix: Optional[str] = None,
        **heatmap_kws
) -> List[matplotlib.axes.Axes]:
    """Visualize similarity between clustering results given an evaluation metric.  

    Args: 
        labels (DataFrame): Labels DataFrame, e.g. from optimize_clustering or 
        AutoClusterer.labels_  
        method (str): Method with which to compare labels. Must be a metric like the ones in 
        constants.need_ground_truth, which takes two sets of labels.  
        savefig (bool): Whether to save a pdf.  
        output_prefix (str): If saving a pdf, file prefix to use.  
        **heatmap_kws: Additional keywords to pass to `seaborn.heatmap`_   

    Returns (List[matplotlib.axes.Axes]): 
        List of matplotlib axes  

    .. _seaborn.heatmap:
        https://seaborn.pydata.org/generated/seaborn.heatmap.html
    """
    if savefig and output_prefix is None:
        output_prefix = 'heatmap.labels.pairwise'
    if method is None:
        method = 'adjusted_rand_score'

    labels = labels.corr(
        lambda x, y: utilities.evaluate_one(x, method=method, gold_standard=y)
    )
    return visualize_pairwise(labels, savefig, output_prefix, method=method, **heatmap_kws)


def visualize_sample_label_consistency(
        labels: DataFrame,
        savefig: bool = False,
        output_prefix: Optional[str] = None,
        **heatmap_kws
) -> List[matplotlib.axes.Axes]:
    """Visualize how often two samples are labeled in the same group across conditions. Interpret 
    with care--if you use more conditions for some type of clusterers, e.g. more n_clusters for 
    KMeans, those cluster more similarly across conditions than between clusterers. This means 
    that more agreement in labeling could be due to the choice of clusterers rather than true 
    similarity between samples.  

    Args: 
        labels (DataFrame): Labels DataFrame, e.g. from optimize_clustering or 
        AutoClusterer.labels_  
        savefig (bool): Whether to save a pdf.  
        output_prefix (str): If saving a pdf, file prefix to use.  
        **heatmap_kws: Additional keywords to pass to `seaborn.heatmap`_  

    Returns (List[matplotlib.axes.Axes]): 
        List of matplotlib axes  

    .. _seaborn.heatmap:
        https://seaborn.pydata.org/generated/seaborn.heatmap.html

    """
    if savefig and output_prefix is None:
        output_prefix = "heatmap.sample.pairwise"
    labels = labels.transpose().corr(lambda x, y: sum(
        np.equal(x[((x != -1) | (y != -1))], y[((x != -1) | (y != -1))])
    ))
    return visualize_pairwise(labels, savefig, output_prefix, method='# same label', **heatmap_kws)
