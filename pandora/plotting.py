from pandora.custom_types import *


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
    positions = ["top left", "top center", "top right", "bottom left", "bottom center", "bottom right"]
    return [positions[i % len(positions)] for i in range(len(x_values))]
