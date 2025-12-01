"""Shared configuration for plotting scripts."""

import matplotlib.pyplot as plt
import seaborn as sns

_palette = sns.color_palette("husl", 10)

WINDOW_COLORS = {
    "full": "black",
    "coarse": _palette[0],
    "fine": _palette[5],
    "slide_1_to_5": _palette[1],
    "slide_2_to_6": _palette[2],
    "slide_3_to_7": _palette[3],
    "slide_4_to_8": _palette[4],
    "slide_5_to_9": _palette[6],
    "slide_6_to_10": _palette[7],
    "slide_7_to_11": _palette[8],
    "slide_8_to_12": _palette[9],
}


def get_window_color(window_name: str) -> str:
    """Get color for a window, generating one if not in preset."""
    if window_name in WINDOW_COLORS:
        return WINDOW_COLORS[window_name]
    return "gray"


def get_window_label(window_name: str) -> str:
    """Get formatted label for a window."""
    if window_name.startswith("slide_"):
        parts = window_name.split("_")
        return f"Slide {parts[1]}-{parts[3]}"
    return window_name.capitalize()
