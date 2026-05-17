from __future__ import annotations

from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd


def render_result(result: Any) -> tuple[list[pd.DataFrame], str]:
    if result is None:
        return [], "No result produced."
    if isinstance(result, pd.DataFrame):
        return [result], "DataFrame result"
    if isinstance(result, pd.Series):
        return [result.to_frame(name=result.name or "value")], "Series result"
    if isinstance(result, (list, tuple)):
        try:
            frame = pd.DataFrame(result)
            return [frame], "Collection result"
        except Exception:
            return [], str(result)
    return [], str(result)


def format_figure_title(fig: matplotlib.figure.Figure, fallback: str = "Chart") -> str:
    if fig._suptitle is not None and fig._suptitle.get_text():
        return fig._suptitle.get_text()
    axes = fig.get_axes()
    if axes and axes[0].get_title():
        return axes[0].get_title()
    return fallback


def clear_matplotlib() -> None:
    plt.close("all")
