from plotly import graph_objects as go

from pandora.custom_types import *

def get_distinct_colors(n_colors: int) -> List[str]:
    """Returns a list of n HSV colors evenly spaced in the HSV colorspace.

    Args:
        n_colors (int): Number of colors to return

    Returns:
        List[str]: List of n plotly HSV color strings.

    """
    return [f"hsv({v}%, 100%, 80%)" for v in np.linspace(0, 100, n_colors, endpoint=False)]


def get_rdylgr_color_scale():
    colors = ["#d60000", "#f2ce02", "#ebff0a", "#85e62c", "#209c05"]
    steps = np.linspace(0, 1, num=len(colors), endpoint=True)

    return list(zip(steps, colors))


def improve_plotly_text_position(x_values) -> List[str]:
    positions = ["top left", "top center", "top right", "bottom left", "bottom center", "bottom right"]
    return [positions[i % len(positions)] for i in range(len(x_values))]


def plot_support_values(
        x_data: List[float],
        y_data: List[float],
        support_values: List[Tuple[str, float]],
        x_title: str,
        y_title: str
):
    support_values_annotations = [
        f"{round(support, 2)}<br>({sample})"
        # annotate only samples with a support value < 0.9 otherwise things are going to get messy
        if support < 0.9 else ""
        for (sample, support) in support_values
    ]

    fig = go.Figure(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers+text",
            text=support_values_annotations,
            textposition="bottom center",
            marker=dict(
                color=[v for _, v in support_values],
                colorscale=get_rdylgr_color_scale(),
                showscale=True,
                cmin=0,
                cmax=1
            )
        )
    )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    fig.update_layout(template="plotly_white", height=1000, width=1000)
    fig.update_traces(textposition=improve_plotly_text_position(x_data))
    return fig
