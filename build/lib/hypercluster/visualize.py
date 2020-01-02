from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from typing import List
from hypercluster.constants import param_delim, val_delim

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
sns.set(font="arial", style="white", color_codes=True, font_scale=1.3)
matplotlib.rcParams.update({"savefig.bbox": "tight"})
cmap = sns.cubehelix_palette(8, as_cmap=True)
cmap.set_bad('#DAE0E6')


def zscore(df):
    return df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)


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
            ] + [2],
            wspace=0.01,
            left=0,
            right=1,
            top=1,
            bottom=0,
        ),
    )
    vmin = np.nanquantile(evaluations_df, 0.1)
    vmax = np.nanquantile(evaluations_df, 0.9)
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
            ).astype(float)
        )
        temp = temp.sort_index()
        temp = temp.transpose()

        ax = axs[i]
        sns.heatmap(
            temp,
            ax=ax,
            yticklabels=temp.index,
            xticklabels=["-".join([str(i) for i in col]) for col in temp.columns],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
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
