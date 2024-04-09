import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from pandora.custom_errors import PandoraException
from pandora.embedding import Embedding
from pandora.embedding_comparison import EmbeddingComparison


def get_distinct_colors(n_colors: int) -> List[str]:
    """Returns a list of n HSV colors evenly spaced in the HSV colorspace.

    Parameters
    ----------
    n_colors : int
        Number of colors to return

    Returns
    -------
    List[str]
        List of n plotly HSV color strings.
    """
    hue_values = np.linspace(0, 100, n_colors, endpoint=False)
    hue_values = np.clip(hue_values, 0, 100)
    return [f"hsv({v}%, 100%, 80%)" for v in hue_values]


def get_rdylgr_color_scale() -> List[Tuple[float, str]]:
    """Returns a continuous hex color scale from red ``(#d60000)`` to green ``(#209c05)``.

    Returns
    -------
    List[Tuple[float, str]]
        A list of ``(float, str)`` tuples that can be used as continuous color scale in plotly figures.
    """
    colors = ["#d60000", "#f2ce02", "#ebff0a", "#85e62c", "#209c05"]
    steps = np.linspace(0, 1, num=len(colors), endpoint=True)

    return list(zip(steps, colors))


def improve_plotly_text_position(x_values: pd.Series) -> List[str]:
    """Returns improved text positions for sample annotations in plotly figures based on the x-values of the samples.

    Parameters
    ----------
    x_values : pd.Series
        x values of the samples to plot

    Returns
    -------
    List[str]
        A list of text positions, one position for each sample in ``x_values``.
    """
    positions = [
        "top left",
        "top center",
        "top right",
        "bottom left",
        "bottom center",
        "bottom right",
    ]
    return [positions[i % len(positions)] for i in range(len(x_values))]


def _get_hovertemplate(hover_params):
    return "<br>".join(
        [param + f": %{{customdata[{i}]}}" for i, param in enumerate(hover_params)]
    )


def _check_plot_dimensions(embedding: Embedding, dim_x: int, dim_y: int) -> None:
    """Checks whether the dimensions requested for plotting on the x- and y-axis are present in the embedding data.

    Parameters
    ----------
    embedding : Embedding
        Embedding data to plot.
    dim_x : int
        Index of dimension to plot on the x-axis.
    dim_y : int
        Index of dimension to plot on the y-axis.

    Returns
    -------
    None

    Raises
    ------
    PandoraException
        - if ``dim_x == dim_y``
        - if either ``dim_x`` or ``dim_y`` does not exist in the Embedding data.
    """
    if dim_x == dim_y:
        raise PandoraException("dim_x and dim_y cannot be identical.")
    dim_x = f"D{dim_x}"
    if dim_x not in embedding.embedding.columns:
        raise PandoraException(f"Requested plot PC {dim_x} for x-axis does not exist.")
    dim_y = f"D{dim_y}"
    if dim_y not in embedding.embedding.columns:
        raise PandoraException(f"Requested plot PC {dim_y} for x-axis does not exist.")


def _update_fig(
    embedding: Embedding,
    fig: go.Figure,
    dim_x: int,
    dim_y: int,
    show_variance_in_axes: bool,
) -> go.Figure:
    """Updates a figure depicting an Embedding plot (x-axis, y-axis, layout).

    Parameters
    ----------
    embedding : Embedding
        Embedding object to plot.
    fig : go.Figure
        Plotly figure to update.
    dim_x : int
        Index of the dimension plotted on the x-axis (zero-indexed).
    dim_y : int
        Index of the dimension plotted on the y-axis (zero-indexed).
    show_variance_in_axes : bool
        If true, also includes embedding.explained_variance data in the x-, and y-axis titles.

    Returns
    -------
    go.Figure
        Figure with updated x- and y-axes as well as an updated layout.
    """
    xtitle = f"PC {dim_x + 1}"
    ytitle = f"PC {dim_y + 1}"

    if show_variance_in_axes:
        xtitle += f" ({round(embedding.explained_variances[dim_x] * 100, 1)}%)"
        ytitle += f" ({round(embedding.explained_variances[dim_y] * 100, 1)}%)"

    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)
    fig.update_layout(template="plotly_white", height=1000, width=1000)

    return fig


def plot_populations(
    embedding: Embedding,
    dim_x: int = 0,
    dim_y: int = 1,
    fig: Optional[go.Figure] = None,
    **kwargs,
) -> go.Figure:
    """Plots the data for the provided Embedding data using the given dimension indices and colors all populations as
    provided by Embedding using distinct colors.

    Parameters
    ----------
    embedding : Embedding
        Embedding data to plot.
    dim_x : int
        Index of the dimension plotted on the x-axis (zero-indexed).
    dim_y : int
        Index of the dimension plotted on the y-axis (zero-indexed).
    fig : go.Figure, default=None
        Optional figure containing previous plotting data (e.g. another Embedding plot).
    **kwargs
        Optional plot arguments passed to go.Scatter. Refer to the plotly documentation for options.
        The following settings are not allowed: [``x``, ``y``, ``mode``, ``marker``, ``marker_color``, ``name``]

    Returns
    -------
    go.Figure
        Plotly figure depicting the Embedding data

    Raises
    ------
    PandoraException
        If there are no populations associated with the embedding data.
    """
    _check_plot_dimensions(embedding, dim_x, dim_y)
    show_variance_in_axes = fig is None
    fig = go.Figure() if fig is None else fig

    if embedding.embedding.population.isna().all():
        raise PandoraException(
            "Cannot plot populations: no populations associated with Embedding data."
        )

    populations = embedding.embedding.population.unique()
    colors = get_distinct_colors(len(populations))

    xcol = f"D{dim_x}"
    ycol = f"D{dim_y}"

    hover_cols = ["sample_id", "population", xcol, ycol]
    hover_cols_display = ["Sample", "Population", f"D{dim_x + 1}", f"D{dim_y + 1}"]

    for i, population in enumerate(populations):
        _data = embedding.embedding.loc[embedding.embedding.population == population]
        fig.add_trace(
            go.Scatter(
                x=_data[xcol],
                y=_data[ycol],
                mode="markers",
                marker_color=colors[i],
                name=population,
                customdata=_data[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
                **kwargs,
            )
        )

    return _update_fig(embedding, fig, dim_x, dim_y, show_variance_in_axes)


def plot_projections(
    embedding: Embedding,
    embedding_populations: pd.Series,
    dim_x: int = 0,
    dim_y: int = 1,
    fig: Optional[go.Figure] = None,
    **kwargs,
):
    """Plots the data for the provided Embedding data using the given dimension indices. Only samples with populations
    *not* in embedding_populations are color-coded according to their population. All other samples are colored in
    lightgray.

    Use this plotting function if you want to highlight only projected samples in an Embedding plot.

    Parameters
    ----------
    embedding : Embedding
        Embedding data to plot.
    embedding_populations : pd.Series
        Pandas Series of population names used to compute the Embedding. Samples belonging to
        these populations are plotted in lightgray.
    dim_x : int
        Index of the dimension plotted on the x-axis (zero-indexed).
    dim_y : int
        Index of the dimension plotted on the y-axis (zero-indexed).
    fig : go.Figure
        Optional figure containing previous plotting data (e.g. another Embedding plot).
    **kwargs
        Optional plot arguments passed to go.Scatter. Refer to the plotly documentation for options.
        The following settings are not allowed: [``x``, ``y``, ``mode``, ``marker``, ``marker_color``, ``name``]

    Returns
    -------
    go.Figure
        Plotly figure depicting the Embedding data.

    Raises
    ------
    PandoraException
        If ``embedding_populations`` is empty. In this case there are not projected samples to plot.
    """
    _check_plot_dimensions(embedding, dim_x, dim_y)
    show_variance_in_axes = fig is None
    fig = go.Figure() if fig is None else fig

    if embedding_populations.empty:
        raise PandoraException(
            "It appears that all populations were used for the Embedding. "
            "To plot projections provide a non-empty list of populations with which the Embedding was performed!"
        )

    populations = embedding.embedding.population.unique()
    projection_colors = get_distinct_colors(populations.shape[0])

    xcol = f"D{dim_x}"
    ycol = f"D{dim_y}"

    hover_cols = ["sample_id", "population", xcol, ycol]
    hover_cols_display = ["Sample", "Population", f"D{dim_x + 1}", f"D{dim_y + 1}"]

    for i, population in enumerate(populations):
        _data = embedding.embedding.loc[lambda x: x.population == population]
        marker_color = (
            projection_colors[i]
            if population not in embedding_populations.values
            else "lightgray"
        )
        fig.add_trace(
            go.Scatter(
                x=_data[xcol],
                y=_data[ycol],
                mode="markers",
                marker_color=marker_color,
                name=population,
                showlegend=population not in embedding_populations.values,
                customdata=_data[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
                **kwargs,
            )
        )
    return _update_fig(embedding, fig, dim_x, dim_y, show_variance_in_axes)


def plot_clusters(
    embedding: Embedding,
    dim_x: int = 0,
    dim_y: int = 1,
    kmeans_k: Optional[int] = None,
    fig: Optional[go.Figure] = None,
    **kwargs,
) -> go.Figure:
    """Plots the data for the provided Embedding data using the given dimension indices and color-codes samples
    according to their cluster label as inferred by using K-Means clustering.

    Parameters
    ----------
    embedding : Embedding
        Embedding data to plot.
    dim_x : int
        Index of the dimension plotted on the x-axis (zero-indexed).
    dim_y : int
        Index of the dimension plotted on the y-axis (zero-indexed).
    kmeans_k : int
        Optional k to use for K-Means clustering. If not set, the optimal number of clusters k
        is automatically determined based on the data provided by Embedding.
    fig : go.Figure
        Optional figure containing previous plotting data (e.g. another Embedding plot).
    **kwargs
        Optional plot arguments passed to go.Scatter. Refer to the plotly documentation for options.
        The following settings are not allowed: [``x``, ``y``, ``mode``, ``marker``, ``marker_color``, ``name``]

    Returns
    -------
    go.Figure
        Plotly figure depicting the Embedding data.
    """
    _check_plot_dimensions(embedding, dim_x, dim_y)
    show_variance_in_axes = fig is None
    fig = go.Figure() if fig is None else fig

    if kmeans_k is None:
        kmeans_k = embedding.get_optimal_kmeans_k()

    cluster_labels = embedding.cluster(kmeans_k=kmeans_k).labels_

    _embedding_data = embedding.embedding.copy()
    _embedding_data["cluster"] = cluster_labels

    colors = get_distinct_colors(kmeans_k)

    xcol = f"D{dim_x}"
    ycol = f"D{dim_y}"

    hover_cols = ["sample_id", "population", xcol, ycol]
    hover_cols_display = ["Sample", "Population", f"D{dim_x + 1}", f"D{dim_y + 1}"]

    for i in range(kmeans_k):
        _data = _embedding_data.loc[_embedding_data.cluster == i]
        fig.add_trace(
            go.Scatter(
                x=_data[xcol],
                y=_data[ycol],
                mode="markers",
                marker_color=colors[i],
                name=f"Cluster {i + 1}",
                customdata=_data[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
                **kwargs,
            )
        )

    return _update_fig(embedding, fig, dim_x, dim_y, show_variance_in_axes)


def plot_support_values(
    embedding: Embedding,
    sample_support_values: pd.Series,
    support_value_rogue_cutoff: float = 0.5,
    dim_x: int = 0,
    dim_y: int = 1,
    projected_samples: Optional[pd.Series] = None,
    **kwargs,
) -> go.Figure:
    """Plots the data for the provided Embedding data using the given dimension indices, color-coding the support value
    for each sample.

    The colors range from red (low support) to green (high support).
    If projected_samples is set, only samples in projected_samples are color-coded according to their support value,
    all other samples are shown in lightgray.

    Parameters
    ----------
    embedding : Embedding
        Embedding data to plot.
    sample_support_values : pd.Series
        Pandora support value for each sample in ``embedding.embedding``.
        Note: The index of the Series is expected to contain the sample IDs and
        to be identical to ``embedding.sample_ids``.
    support_value_rogue_cutoff : float
        Samples with a support value below this threshold are annotated with
        the sample ID and the support value. All other samples are only color-coded.
    dim_x : int
        Index of the dimension plotted on the x-axis (zero-indexed).
    dim_y : int
        Index of the dimension plotted on the y-axis (zero-indexed).
    projected_samples : pd.Series
        Pandas series of sample IDs. If set, only samples in this list are color-coded according
        to their support value. All other samples are shown in gray.
    **kwargs
        Optional plot arguments passed to go.Scatter. Refer to the plotly documentation for options.
        The following settings are not allowed: [``x``, ``y``, ``mode``, ``marker``, ``marker_color``, ``name``,
        ``text``, ``textposition``]

    Returns
    -------
    go.Figure
        Plotly figure depicting the Embedding data.

    Warns
    -----
    UserWarning
        - If not all samples in ``sample_support_values`` are present in the embedding data. This is most likely due to outlier detection during the computation of the embedding.
        - If not all samples in the embedding data have a support value associated.

    Raises
    ------
    PandoraException
        If no samples are left to plot after filtering samples present in both embedding data and ``sample_support_values``.
    """
    _check_plot_dimensions(embedding, dim_x, dim_y)

    # next,we have to filter out samples that are either not present in embedding.embedding (detected as outlier)
    # or not present in sample_support_values (detected as outlier in all replicates)
    # if there are such samples, we issue a warning containing the affected IDs and continue to plot only samples
    # present in both embedding.embedding and sample_support_values
    embedding_ids = set(embedding.sample_ids)
    support_ids = set(sample_support_values.index)

    not_in_embedding = support_ids - embedding_ids
    not_in_support = embedding_ids - support_ids
    present_in_both = support_ids & embedding_ids

    if len(not_in_embedding) > 0:
        warnings.warn(
            "Some of the provided sample_support_values sample IDs are not present in the embedding.embedding data. "
            "This is most likely due to outlier detecting during the computation of embedding.embedding. "
            f"Affected samples are: {not_in_embedding}. Note that these samples will not show up in the plot."
        )

    if len(not_in_support) > 0:
        warnings.warn(
            "Not all samples in embedding.embedding data have a support value in sample_support_values. "
            "This is most likely due to outlier detecting during the computation of support_values. "
            f"Affected samples are: {not_in_support}. Note that these samples will not show up in the plot."
        )

    if len(present_in_both) == 0:
        raise PandoraException(
            "No samples left to plot after filtering. "
            "Make sure the provided sample IDs in sample_support_values match the sample IDs in embedding.embedding."
        )

    embedding = embedding.embedding.loc[lambda x: x.sample_id.isin(present_in_both)]
    sample_support_values = sample_support_values.loc[
        lambda x: x.index.isin(present_in_both)
    ]

    # to make sure we are annotating the correct support values for the correct embedding vectors, we explicitly sort
    # the embedding and support value data
    embedding = embedding.sort_values(by="sample_id").reset_index(drop=True)
    sample_support_values = sample_support_values.sort_index()
    embedding = embedding.copy()
    embedding["support"] = sample_support_values.values

    xcol = f"D{dim_x}"
    ycol = f"D{dim_y}"

    hover_cols = ["sample_id", "population", "support", xcol, ycol]
    hover_cols_display = [
        "Sample",
        "Population",
        "Support Value",
        f"D{dim_x + 1}",
        f"D{dim_y + 1}",
    ]

    if projected_samples is not None:
        marker_colors = []
        marker_text = []

        for idx, row in embedding.iterrows():
            if row.sample_id in projected_samples.values:
                # check if the sample is projected, if so the marker color should be according to it's support
                support = sample_support_values.loc[
                    lambda x: x.index == row.sample_id
                ].item()
                marker_colors.append(support)
                if support < support_value_rogue_cutoff:
                    # if the support is worse than the threshold, annotate the projected sample
                    marker_text.append(f"{round(support, 2)}<br>({row.sample_id})")
                else:
                    # do not annotate
                    marker_text.append("")
            else:
                # otherwise print an unlabeled, gray marker
                marker_colors.append("lightgray")
                marker_text.append("")

    else:
        marker_colors = list(sample_support_values.values)

        # annotate only samples with support below support_value_rogue_cutoff
        marker_text = [
            f"{round(support, 2)}<br>({sample})"
            if support < support_value_rogue_cutoff
            else ""
            for (sample, support) in sorted(sample_support_values.items())
        ]

    fig = go.Figure(
        go.Scatter(
            x=embedding[xcol],
            y=embedding[ycol],
            mode="markers+text",
            text=marker_text,
            textposition="bottom center",
            marker=dict(
                color=marker_colors,
                colorscale=get_rdylgr_color_scale(),
                colorbar=dict(title="Pandora support value"),
                showscale=True,
                cmin=0,
                cmax=1,
            ),
            customdata=embedding[hover_cols].values,
            hovertemplate=_get_hovertemplate(hover_cols_display),
            **kwargs,
        )
    )
    fig.update_traces(textposition=improve_plotly_text_position(embedding[xcol]))

    fig.update_xaxes(title=f"PC {dim_x + 1}")
    fig.update_yaxes(title=f"PC {dim_y + 1}")
    fig.update_layout(template="plotly_white", height=1000, width=1000)
    return fig


def plot_embedding_comparison(
    embedding_comparison: EmbeddingComparison, dim_x: int = 0, dim_y: int = 1, **kwargs
) -> go.Figure:
    """Method to plot the closest match between two Embeddings. Plots the transformed Embeddings based on the
    EmbeddingComparison object.

    Parameters
    ----------
    embedding_comparison : EmbeddingComparison
        EmbeddingComparison object containing the two Embeddings to plot.
    dim_x : int
        Index of the dimension plotted on the x-axis (zero-indexed).
    dim_y : int
        Index of the dimension plotted on the y-axis (zero-indexed).
    **kwargs
        Optional plot arguments passed to go.Scatter. Refer to the plotly documentation for options.
        The following settings are not allowed: [``x``, ``y``, ``mode``, ``marker``, ``marker_color``, ``name``,
        ``marker_symbol``]

    Returns
    -------
    go.Figure
        Plotly figure depicting both Embeddings in ``embedding_comparison``.
    """
    _check_plot_dimensions(embedding_comparison.reference, dim_x, dim_y)
    _check_plot_dimensions(embedding_comparison.comparable, dim_x, dim_y)

    xcol = f"D{dim_x}"
    ycol = f"D{dim_y}"

    hover_cols = ["sample_id", "population", xcol, ycol]
    hover_cols_display = ["Sample", "Population", f"D{dim_x + 1}", f"D{dim_y + 1}"]

    fig = go.Figure(
        [
            go.Scatter(
                x=embedding_comparison.reference.embedding[xcol],
                y=embedding_comparison.reference.embedding[ycol],
                marker_color="darkblue",
                name="Standardized reference Embedding",
                mode="markers",
                customdata=embedding_comparison.reference.embedding[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
                **kwargs,
            ),
            go.Scatter(
                x=embedding_comparison.comparable.embedding[xcol],
                y=embedding_comparison.comparable.embedding[ycol],
                marker_color="orange",
                marker_symbol="star",
                name="Transformed comparable Embedding",
                mode="markers",
                customdata=embedding_comparison.comparable.embedding[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
                **kwargs,
            ),
        ]
    )

    fig.update_xaxes(title=f"PC {dim_x + 1}")
    fig.update_yaxes(title=f"PC {dim_y + 1}")

    fig.update_layout(template="plotly_white", height=1000, width=1000)

    return fig


def plot_embedding_comparison_rogue_samples(
    embedding_comparison: EmbeddingComparison,
    support_values: pd.Series,
    support_value_rogue_cutoff: float = 0.5,
    dim_x: int = 0,
    dim_y: int = 1,
    **kwargs,
) -> go.Figure:
    """Method to plot the closest match between two Embeddings. Plots the transformed Embeddings based on the
    EmbeddingComparison object. Additionally, all samples with a support value below ``support_value_rogue_cutoff`` are
    highlighted.

    Parameters
    ----------
    embedding_comparison : EmbeddingComparison
        EmbeddingComparison object containing the two Embeddings to plot.
    support_values : pd.Series
        Pandora support values for all samples in the embedding data. The series' indices should be the sample IDs and
        the values the respective support value.
    support_value_rogue_cutoff : float
        Samples with a support value below this threshold are considered rogue and
        are highlighted by color, their sample ID and support value.
    dim_x : int
        Index of the dimension plotted on the x-axis (zero-indexed).
    dim_y : int
        Index of the dimension plotted on the y-axis (zero-indexed).
    **kwargs
        Optional plot arguments passed to go.Scatter. Refer to the plotly documentation for options.
        The following settings are not allowed: [``x``, ``y``, ``mode``, ``marker``, ``marker_color``, ``name``, `
        `text``, ``textposition``, ``marker_symbol``]

    Returns
    -------
    go.Figure
        Plotly figure depicting both Embeddings in ``embedding_comparison`` and highlighting rouge samples.
    """
    _check_plot_dimensions(embedding_comparison.reference, dim_x, dim_y)
    _check_plot_dimensions(embedding_comparison.comparable, dim_x, dim_y)

    xcol = f"D{dim_x}"
    ycol = f"D{dim_y}"

    hover_cols = ["sample_id", "population", xcol, ycol]
    hover_cols_display = ["Sample", "Population", f"D{dim_x + 1}", f"D{dim_y + 1}"]

    rogue_samples = support_values.loc[lambda x: x < support_value_rogue_cutoff]

    rogue_colors = get_distinct_colors(rogue_samples.shape[0])
    rogue_text = [
        f"{sample_id}<br>({round(support, 2)})"
        for sample_id, support in rogue_samples.items()
    ]

    rogue_reference = embedding_comparison.reference.embedding.loc[
        lambda x: x.sample_id.isin(rogue_samples.index)
    ]
    rogue_comparable = embedding_comparison.comparable.embedding.loc[
        lambda x: x.sample_id.isin(rogue_samples.index)
    ]

    fig = go.Figure(
        [
            # all samples
            go.Scatter(
                x=embedding_comparison.reference.embedding[xcol],
                y=embedding_comparison.reference.embedding[ycol],
                marker_color="lightgray",
                name="Standardized reference Embedding",
                mode="markers",
                customdata=embedding_comparison.reference.embedding[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
                **kwargs,
            ),
            go.Scatter(
                x=embedding_comparison.comparable.embedding[xcol],
                y=embedding_comparison.comparable.embedding[ycol],
                marker_color="lightgray",
                marker_symbol="star",
                name="Transformed comparable Embedding",
                mode="markers",
                customdata=embedding_comparison.comparable.embedding[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
                **kwargs,
            ),
            # Rogue samples
            go.Scatter(
                x=rogue_reference[xcol],
                y=rogue_reference[ycol],
                marker_color=rogue_colors,
                text=rogue_text,
                textposition="bottom center",
                mode="markers+text",
                showlegend=False,
                customdata=rogue_reference[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
            ),
            go.Scatter(
                x=rogue_comparable[xcol],
                y=rogue_comparable[ycol],
                marker_color=rogue_colors,
                marker_symbol="star",
                text=rogue_text,
                textposition="bottom center",
                mode="markers+text",
                showlegend=False,
                customdata=rogue_comparable[hover_cols].values,
                hovertemplate=_get_hovertemplate(hover_cols_display),
            ),
        ]
    )

    fig.update_xaxes(title=f"PC {dim_x + 1}")
    fig.update_yaxes(title=f"PC {dim_y + 1}")

    fig.update_layout(template="plotly_white", height=1000, width=1000)

    return fig
