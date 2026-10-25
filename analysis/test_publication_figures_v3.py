"""Layout regressions for the two compact main-text comparison figures."""
import json

import matplotlib.pyplot as plt

from plot_quality_depth_and_size_unified_v3 import build_figure as quality_figure
from plot_optimizer_comparison_n12p2_v3 import build_figure as optimizer_figure
from plot_optimizer_comparison_n12p2_v2 import SUMMARY


def test_quality_panels_are_side_by_side_with_shared_scale():
    fig = quality_figure()
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_position().x0 < fig.axes[1].get_position().x0
        assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
        assert fig.get_size_inches()[0] > fig.get_size_inches()[1]
    finally:
        plt.close(fig)


def test_optimizer_main_figure_has_only_a_and_b():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    fig = optimizer_figure(summary)
    try:
        assert len(fig.axes) == 2
        titles = [axis.get_title() for axis in fig.axes]
        assert titles[0].startswith("(a)")
        assert titles[1].startswith("(b)")
        assert all("(c)" not in title for title in titles)
        assert fig.get_size_inches()[0] > fig.get_size_inches()[1]
    finally:
        plt.close(fig)
