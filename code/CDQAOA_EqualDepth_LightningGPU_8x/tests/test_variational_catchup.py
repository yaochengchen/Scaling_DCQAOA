import importlib
import json

import numpy as np
import pytest

from cdtransfer.instances import IsingInstance
from cdtransfer.simulator import StatevectorSimulator
from scripts.variational_catchup_helpers import (
    cumulative_best,
    first_crossing,
    mean_trajectory,
    pad_best,
)

pilot = importlib.import_module("scripts.16_variational_catchup_pilot")


def test_cumulative_best_keeps_running_maximum():
    assert cumulative_best([0.2, 0.7, 0.4, 0.8]) == [0.2, 0.7, 0.7, 0.8]


def test_cumulative_best_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        cumulative_best([0.2, float("nan")])


def test_pad_best_carries_last_value_to_requested_length():
    assert pad_best([0.2, 0.6], 4) == [0.2, 0.6, 0.6, 0.6]


def test_mean_trajectory_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        mean_trajectory([[0.2, 0.4], [0.3]])


def test_mean_trajectory_averages_aligned_runs():
    assert mean_trajectory([[0.2, 0.4], [0.4, 0.8]]) == pytest.approx([0.3, 0.6])


def test_first_crossing_counts_from_one_and_includes_equality():
    assert first_crossing([0.5, 0.7, 0.8], 0.7) == 2


def test_first_crossing_returns_none_when_unreached():
    assert first_crossing([0.2, 0.4], 0.5) is None


@pytest.mark.parametrize(("method", "dimensions"), [("qaoa", 2), ("cd_qaoa", 3)])
def test_run_trajectory_records_every_bounded_evaluator_call(method, dimensions):
    instance = IsingInstance(2, np.array([[0, 1]]), np.array([1.0]))
    simulator = StatevectorSimulator(instance)
    calls = []

    def evaluator(angles):
        calls.append(angles.pack(include_cd=(method == "cd_qaoa")).tolist())
        return simulator.evaluate(angles)

    result = pilot.run_trajectory(
        instance_id="tiny",
        fingerprint=instance.fingerprint,
        instance=instance,
        optimum=1.0,
        method=method,
        start=0,
        seed=41,
        max_evaluations=8,
        evaluator=evaluator,
    )

    assert len(calls) == len(result["evaluations"]) <= 8
    assert len(result["initial_point"]) == dimensions
    assert all(len(item["angles"]) == dimensions for item in result["evaluations"])


def test_run_trajectory_initial_point_is_reproducible_for_a_seed():
    instance = IsingInstance(2, np.array([[0, 1]]), np.array([1.0]))
    simulator = StatevectorSimulator(instance)
    kwargs = dict(
        instance_id="tiny",
        fingerprint=instance.fingerprint,
        instance=instance,
        optimum=1.0,
        method="qaoa",
        start=0,
        seed=20261012,
        max_evaluations=5,
        evaluator=simulator.evaluate,
    )
    first = pilot.run_trajectory(**kwargs)
    second = pilot.run_trajectory(**kwargs)
    assert first["initial_point"] == second["initial_point"]


def test_summary_counts_fixed_instances_once_and_variational_trajectories_equally():
    instances = [
        {"id": f"i{index}", "fixed_ratio": index / 10, "fingerprint": f"fp{index}"}
        for index in range(1, 6)
    ]
    trajectories = []
    for method, ratio in (("qaoa", 0.6), ("cd_qaoa", 0.2)):
        for instance in instances:
            for start in range(3):
                trajectories.append({
                    "instance_id": instance["id"],
                    "method": method,
                    "start": start,
                    "evaluations": [{"best_so_far": ratio}],
                })

    result = pilot.summarize(trajectories, instances, cap=3)

    assert result["fixed_cd_qaoa_target"] == pytest.approx(0.3)
    assert result["methods"]["qaoa"]["trajectory_count"] == 15
    assert result["methods"]["qaoa"]["mean_best_so_far"] == pytest.approx([0.6] * 3)
    assert result["methods"]["qaoa"]["final_mean_at_max_evaluations"] == pytest.approx(0.6)
    assert result["methods"]["qaoa"]["first_crossing_evaluation"] == 1
    assert result["methods"]["cd_qaoa"]["first_crossing_evaluation"] is None


@pytest.mark.parametrize(
    ("method", "evaluation", "expected"),
    [("qaoa", 12, "QAOA: 12"), ("cd_qaoa", None, "CD-QAOA: not reached")],
)
def test_crossing_annotation_labels_reached_and_unreached(method, evaluation, expected):
    assert pilot.crossing_annotation(method, evaluation) == expected


def test_report_uses_the_configured_instance_and_evaluation_counts(tmp_path):
    summary = {
        "fixed_cd_qaoa_target": 0.5,
        "evaluation_counts": [1, 2, 3],
        "methods": {
            method: {
                "first_crossing_evaluation": None,
                "final_mean_at_max_evaluations": 0.4,
                "trajectory_count": 1,
            }
            for method in ("qaoa", "cd_qaoa")
        },
    }
    config = {"p": 1, "instances": 1, "starts": 1, "max_evaluations": 3,
              "optimizer": "Nelder-Mead", "base_seed": 7}
    report_path = tmp_path / "REPORT.md"

    pilot.write_report(summary, config, 1.25, report_path)
    report = report_path.read_text(encoding="utf-8")

    assert "evaluation 3" in report
    assert "evaluation 100" not in report
    assert "selected instances" in report
    assert "Simulation and validation time" in report
    assert "Total wall time" not in report


def test_unreached_cd_qaoa_annotation_sits_below_its_endpoint():
    assert pilot.crossing_annotation_offset("cd_qaoa", None) == (-4, -18)
    assert pilot.crossing_annotation_offset("qaoa", None) == (-4, -18)


def test_plot_uses_focused_y_range_and_distinct_fixed_method_colour():
    summary = {
        "fixed_cd_qaoa_target": 0.8759047772,
        "methods": {
            "qaoa": {"mean_best_so_far": [0.5859905, 0.7730841]},
            "cd_qaoa": {"mean_best_so_far": [0.5587225, 0.8465069]},
        },
    }

    assert pilot.plot_y_limits(summary) == pytest.approx((0.52, 0.90))
    assert pilot.PLOT_STYLES["fixed_cd_qaoa"]["color"] == "#B5179E"
    assert pilot.PLOT_STYLES["qaoa"]["color"] != "tab:blue"
    assert pilot.PLOT_STYLES["cd_qaoa"]["color"] != "tab:orange"


@pytest.mark.parametrize(
    "artifact_name",
    ["trajectories.json", "summary.json", "variational_catchup.png", "REPORT.md"],
)
def test_run_pilot_refuses_to_modify_existing_pilot_artifacts(tmp_path, artifact_name):
    output_dir = tmp_path / "pilot"
    output_dir.mkdir()
    sentinel = b"existing immutable artifact\n"
    artifact_path = output_dir / artifact_name
    artifact_path.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="overwrite"):
        pilot.run_pilot(tmp_path / "missing-config.json", output_dir)

    assert artifact_path.read_bytes() == sentinel
    assert sorted(path.name for path in output_dir.iterdir()) == [artifact_name]


def test_run_pilot_writes_all_artifacts_to_a_fresh_directory(tmp_path):
    output_dir = tmp_path / "fresh-pilot"

    summary = pilot.run_pilot(
        pilot.PROJECT_ROOT / "configs" / "variational_catchup_pilot.json",
        output_dir,
        instances_count=1,
        starts=1,
        max_evaluations=3,
    )

    assert sorted(path.name for path in output_dir.iterdir()) == sorted(pilot.OUTPUT_ARTIFACTS)
    assert len(summary["evaluation_counts"]) == 3
    trajectories = json.loads((output_dir / "trajectories.json").read_text(encoding="utf-8"))
    assert len(trajectories["trajectories"]) == 2
    assert max(row["actual_evaluations"] for row in trajectories["trajectories"]) <= 3
