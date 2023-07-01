from plotly import graph_objects as go

from pandora.custom_types import *
from pandora.custom_errors import *
from pandora.pca import PCA
from pandora.pca_comparison import PCAComparison


def get_distinct_colors(n_colors: int) -> List[str]:
    """Returns a list of n HSV colors evenly spaced in the HSV colorspace.

    Args:
        n_colors (int): Number of colors to return

    Returns:
        List[str]: List of n plotly HSV color strings.

    """
    hue_values = np.linspace(0, 100, n_colors, endpoint=False)
    hue_values = np.clip(hue_values, 0, 100)
    return [f"hsv({v}%, 100%, 80%)" for v in hue_values]


def get_rdylgr_color_scale():
    colors = ["#d60000", "#f2ce02", "#ebff0a", "#85e62c", "#209c05"]
    steps = np.linspace(0, 1, num=len(colors), endpoint=True)

    return list(zip(steps, colors))


def improve_plotly_text_position(x_values) -> List[str]:
    positions = [
        "top left",
        "top center",
        "top right",
        "bottom left",
        "bottom center",
        "bottom right",
    ]
    return [positions[i % len(positions)] for i in range(len(x_values))]


def _update_fig(
    pca: PCA, fig: go.Figure, pcx: int, pcy: int, show_variance_in_axes: bool
):
    xtitle = f"PC {pcx + 1}"
    ytitle = f"PC {pcy + 1}"

    if show_variance_in_axes:
        xtitle += f" ({round(pca.explained_variances[pcx] * 100, 1)}%)"
        ytitle += f" ({round(pca.explained_variances[pcy] * 100, 1)}%)"

    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)
    fig.update_layout(template="plotly_white", height=1000, width=1000)

    return fig


def plot_pca_populations(pca: PCA, pcx: int = 0, pcy: int = 1, fig: go.Figure = None, **kwargs):
    show_variance_in_axes = fig is None
    fig = go.Figure() if fig is None else fig

    if pca.pca_data.population.isna().all():
        raise PandoraException(
            "Cannot plot populations: no populations associated with PCA data."
        )

    populations = pca.pca_data.population.unique()
    colors = get_distinct_colors(len(populations))

    for i, population in enumerate(populations):
        _data = pca.pca_data.loc[pca.pca_data.population == population]
        fig.add_trace(
            go.Scatter(
                x=_data[f"PC{pcx}"],
                y=_data[f"PC{pcy}"],
                mode="markers",
                marker_color=colors[i],
                name=population,
                **kwargs,
            )
        )

    return _update_fig(pca, fig, pcx, pcy, show_variance_in_axes)


def plot_pca_projections(
    pca: PCA,
    pca_populations: List[str],
    pcx: int = 0,
    pcy: int = 1,
    fig: go.Figure = None,
    **kwargs,
):
    """
    TODO: Docstring
    """
    show_variance_in_axes = fig is None
    fig = go.Figure() if fig is None else fig

    if len(pca_populations) == 0:
        raise PandoraException(
            "It appears that all populations were used for the PCA. "
            "To plot projections provide a non-empty list of populations with which the PCA was performed!"
        )

    if not all(p in pca.pca_data.population.unique() for p in pca_populations):
        raise PandoraException("Not all of the passed pca_populations seem to be present in self.pca_data.")

    populations = pca.pca_data.population.unique()
    projection_colors = get_distinct_colors(populations.shape[0])

    for i, population in enumerate(populations):
        _data = pca.pca_data.loc[lambda x: x.population == population]
        marker_color = (
            projection_colors[i]
            if population not in pca_populations
            else "lightgray"
        )
        fig.add_trace(
            go.Scatter(
                x=_data[f"PC{pcx}"],
                y=_data[f"PC{pcy}"],
                mode="markers",
                marker_color=marker_color,
                name=population,
                **kwargs,
            )
        )
    return _update_fig(pca, fig, pcx, pcy, show_variance_in_axes)


def plot_pca_clusters(
    pca: PCA,
    pcx: int = 0,
    pcy: int = 1,
    kmeans_k: int = None,
    fig: go.Figure = None,
    **kwargs,
) -> go.Figure:
    """
    TODO: Docstring
    """
    show_variance_in_axes = fig is None
    fig = go.Figure() if fig is None else fig

    if kmeans_k is None:
        kmeans_k = pca.get_optimal_kmeans_k()

    cluster_labels = pca.cluster(kmeans_k=kmeans_k).labels_

    _pca_data = pca.pca_data.copy()
    _pca_data["cluster"] = cluster_labels

    colors = get_distinct_colors(kmeans_k)

    for i in range(kmeans_k):
        _data = _pca_data.loc[_pca_data.cluster == i]
        fig.add_trace(
            go.Scatter(
                x=_data[f"PC{pcx}"],
                y=_data[f"PC{pcy}"],
                mode="markers",
                marker_color=colors[i],
                name=f"Cluster {i + 1}",
                **kwargs,
            )
        )

    return _update_fig(pca, fig, pcx, pcy, show_variance_in_axes)


def plot_support_values(
    pca: PCA,
    sample_support_values: pd.Series,
    support_value_rogue_cutoff: float = 0.5,
    pcx: int = 0,
    pcy: int = 1,
    projected_samples: Optional[List[str]] = None,
    **kwargs
) -> go.Figure:
    # check that the number of support values matches the number of samples in the PCA data
    if len(sample_support_values) != pca.pca_data.shape[0]:
        # TODO: statt striktem check einfach lightgray falls das sample nicht vertreten ist (= support value NaN)
        raise PandoraException(
            f"Provide exactly one support value for each sample. "
            f"Got {len(sample_support_values)} support values, "
            f"but {pca.pca_data.shape[0]} samples in the PCA."
        )

    # check that the provided support values match the sample IDs in the PCA data
    if not all(s in pca.pca_data.sample_id.tolist() for s in sample_support_values.index):
        raise PandoraException(
            "Sample IDs of provided support values don't match the sample IDs in the PCA data."
        )

    # to make sure we are annotating the correct support values for the correct PC vectors, we explicitly sort
    # the x-, y-, and support value data
    pca_data = pca.pca_data.sort_values(by="sample_id").reset_index(drop=True)
    x_data = pca_data[f"PC{pcx}"]
    y_data = pca_data[f"PC{pcy}"]

    if projected_samples is not None:
        marker_colors = []
        marker_text = []

        for idx, row in pca_data.iterrows():
            if row.sample_id in projected_samples:
                # check if the sample is projected, if so the marker color should be according to it's support
                support = sample_support_values.loc[lambda x: x.index == row.sample_id].item()
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
            f"{round(support, 2)}<br>({sample})" if support < support_value_rogue_cutoff else ""
            for (sample, support) in sorted(sample_support_values.items())
        ]

    fig = go.Figure(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers+text",
            text=marker_text,
            textposition="bottom center",
            marker=dict(
                color=marker_colors,
                colorscale=get_rdylgr_color_scale(),
                colorbar=dict(title="Bootstrap support"),
                showscale=True,
                cmin=0,
                cmax=1,
            ),
            **kwargs
        )
    )
    fig.update_traces(textposition=improve_plotly_text_position(x_data))

    fig.update_xaxes(title=f"PC {pcx + 1}")
    fig.update_yaxes(title=f"PC {pcy + 1}")
    fig.update_layout(template="plotly_white", height=1000, width=1000)
    return fig


def plot_pca_comparison(pca_comparison: PCAComparison, pcx: int = 0, pcy: int = 1, **kwargs) -> go.Figure:
    fig = go.Figure(
        [
            go.Scatter(
                x=pca_comparison.reference.pca_data[f"PC{pcx}"],
                y=pca_comparison.reference.pca_data[f"PC{pcy}"],
                marker_color="darkblue",
                name="Standardized reference PCA",
                mode="markers",
                **kwargs,
            ),
            go.Scatter(
                x=pca_comparison.comparable.pca_data[f"PC{pcx}"],
                y=pca_comparison.comparable.pca_data[f"PC{pcy}"],
                marker_color="orange",
                marker_symbol="star",
                name="Transformed comparable PCA",
                mode="markers",
                **kwargs,
            ),
        ]
    )

    fig.update_xaxes(title=f"PC {pcx + 1}")
    fig.update_yaxes(title=f"PC {pcy + 1}")

    fig.update_layout(template="plotly_white", height=1000, width=1000)

    return fig


def plot_pca_comparison_rogue_samples(pca_comparison: PCAComparison, support_value_rogue_cutoff: float = 0.5, pcx: int = 0, pcy: int = 1, **kwargs) -> go.Figure:
    rogue_samples = pca_comparison.detect_rogue_samples(support_value_rogue_cutoff=support_value_rogue_cutoff)
    rogue_samples["color"] = get_distinct_colors(rogue_samples.shape[0])
    rogue_samples["text"] = [f"{row.sample_id}<br>({round(row.support, 2)})" for idx, row in rogue_samples.iterrows()]

    fig = go.Figure(
        [
            # all samples
            go.Scatter(
                x=pca_comparison.reference.pca_data[f"PC{pcx}"],
                y=pca_comparison.reference.pca_data[f"PC{pcy}"],
                marker_color="lightgray",
                name="Standardized reference PCA",
                mode="markers",
                **kwargs,
            ),
            go.Scatter(
                x=pca_comparison.comparable.pca_data[f"PC{pcx}"],
                y=pca_comparison.comparable.pca_data[f"PC{pcy}"],
                marker_color="lightgray",
                marker_symbol="star",
                name="Transformed comparable PCA",
                mode="markers",
                **kwargs,
            ),
            # Rogue samples
            go.Scatter(
                x=pca_comparison.reference.pca_data.loc[lambda x: x.sample_id.isin(rogue_samples.sample_id)][f"PC{pcx}"],
                y=pca_comparison.reference.pca_data.loc[lambda x: x.sample_id.isin(rogue_samples.sample_id)][f"PC{pcy}"],
                marker_color=rogue_samples.color,
                text=rogue_samples.text,
                textposition="bottom center",
                mode="markers+text",
                showlegend=False
            ),
            go.Scatter(
                x=pca_comparison.comparable.pca_data.loc[lambda x: x.sample_id.isin(rogue_samples.sample_id)][f"PC{pcx}"],
                y=pca_comparison.comparable.pca_data.loc[lambda x: x.sample_id.isin(rogue_samples.sample_id)][f"PC{pcy}"],
                marker_color=rogue_samples.color,
                marker_symbol="star",
                text=rogue_samples.text,
                textposition="bottom center",
                mode="markers+text",
                showlegend=False
            )
        ]
    )

    fig.update_xaxes(title=f"PC {pcx + 1}")
    fig.update_yaxes(title=f"PC {pcy + 1}")

    fig.update_layout(template="plotly_white", height=1000, width=1000)

    return fig
