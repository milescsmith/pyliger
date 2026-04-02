import numpy as np
from plotnine import (
    aes,
    geom_point,
    geom_text,
    ggplot,
    ggtitle,
    guide_legend,
    guides,
    scale_color_hue,
    theme,
    theme_classic,
    xlab,
    ylab,
)


def plot_by_dataset_and_cluster(
    liger_object,
    clusters=None,
    title: str | None=None,
    pt_size: float=0.3,
    text_size: int=10,
    do_shuffle: bool=True,
    rand_seed: int=1,
    axis_labels: tuple[str, str] | None=None,
    do_legend: bool=True,
    legend_size: int=7,
    return_plots: bool=False,
    legend_text_size: int=12,
):
    """Plot t-SNE coordinates of cells across datasets

    Generates two plots of all cells across datasets, one colored by dataset and one colored by
    cluster. These are useful for visually examining the alignment and cluster distributions,
    respectively. If clusters have not been set yet (quantileAlignSNF not called), will plot by
    single color for second plot. It is also possible to pass in another clustering (as long as
    names match those of cells).

    Parameters
    ----------
    liger_object : TYPE
        DESCRIPTION.
    clusters : TYPE, optional
        DESCRIPTION
    title : str, optional
        DESCRIPTION.
    pt_size : float, default=0.3
        DESCRIPTION.
    text_size : int, default=3
        DESCRIPTION.
    do_shuffle : bool, default=True
        DESCRIPTION.
    rand_seed : int, default=1
        DESCRIPTION.
    axis_labels : tuple[str, str], optional
        DESCRIPTION.
    do_legend : bool, default=True
        DESCRIPTION.
    legend_size : int, default=5
        DESCRIPTION.
    legend_text_size : int, default=12
        DESCRIPTION.
    return_plots : bool, default=Fakse
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # tsne_coords = [adata.obs['tsne_coords'] for adata in liger_object.adata_list]
    tsne_df = liger_object.tsne_coords
    tsne_df["Cluster"] = np.asarray(
        np.concatenate(
            [adata.obs["cluster"].to_numpy() for adata in liger_object.adata_list]
        )
    )
    tsne_df["Cluster"] = tsne_df["Cluster"].astype("category")
    tsne_df["Dataset"] = np.concatenate(
        [
            np.repeat(adata.uns["sample_name"], adata.shape[0])
            for adata in liger_object.adata_list
        ]
    )

    if do_shuffle:
        tsne_df = tsne_df.sample(frac=1, random_state=rand_seed)

    p1 = (
        ggplot(data=tsne_df, mapping=aes(x="tsne1", y="tsne2", color="Dataset"))
        + geom_point(size=pt_size)
        + guides(color=guide_legend(override_aes={"size": legend_size}))
        + scale_color_hue(h=15 / 360.0, l=0.65, s=1.0, color_space="husl")
    )

    centers = (
        tsne_df.groupby("Cluster")
        .agg(tsne1=("tsne1", "median"), tsne2=("tsne2", "median"))
        .reset_index()
    )

    p2 = (
        ggplot(data=tsne_df, mapping=aes(x="tsne1", y="tsne2", color="Cluster"))
        + geom_point(size=pt_size)
        + geom_text(
            data=centers, mapping=aes(label="Cluster"), color="black", size=text_size
        )
        + guides(color=guide_legend(override_aes={"size": legend_size}))
        + scale_color_hue(h=15 / 360.0, l=0.65, s=1.0, color_space="husl")
    )

    if title:
        p1 = p1 + ggtitle(title[0])
        p2 = p2 + ggtitle(title[1])

    if axis_labels:
        p1 = p1 + xlab(axis_labels[0]) + ylab(axis_labels[1])
        p2 = p2 + xlab(axis_labels[0]) + ylab(axis_labels[1])

    p1 = p1 + theme_classic(legend_text_size)
    p2 = p2 + theme_classic(legend_text_size)

    if not do_legend:
        p1 = p1 + theme(legend_position="none")
        p2 = p2 + theme(legend_position="none")

    if return_plots:
        return [p1, p2]
    else:
        return None
