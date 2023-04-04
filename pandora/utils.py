from pandora.custom_types import *


def get_colors(n: int) -> List[str]:
    """Returns a list of n HSV colors evenly spaced in the HSV colorspace.

    Args:
        n (int): Number of colors to return

    Returns:
        List[str]: List of n plotly HSV color strings.

    """
    return [f"hsv({v}%, 100%, 80%)" for v in np.linspace(0, 100, n, endpoint=False)]
