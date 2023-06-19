from pandora.custom_types import *


def get_distinct_colors(n_colors: int) -> List[str]:
    """Returns a list of n HSV colors evenly spaced in the HSV colorspace.

    Args:
        n_colors (int): Number of colors to return

    Returns:
        List[str]: List of n plotly HSV color strings.

    """
    return [f"hsv({v}%, 100%, 80%)" for v in np.linspace(0, 100, n_colors, endpoint=False)]


def get_color_scale():
    pass


def improve_plotly_text_position(x_values) -> List[str]:
    positions = ["top left", "top center", "top right", "bottom left", "bottom center", "bottom right"]
    return [positions[i % len(positions)] for i in range(len(x_values))]