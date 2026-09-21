"""Layer 3c: scientific validation, not code-green or loss-curve checks."""

from __future__ import annotations

from .state import CycleState, PlanTask


def _current_task(state: CycleState) -> PlanTask:
    return state["tasks"][state.get("task_index", 0)]


def validate_science(state: CycleState) -> tuple[bool, str]:
    task = _current_task(state)
    experiment = state.get("experiment_result") or {}
    if experiment.get("status") == "blocked":
        return False, str(experiment.get("reason", "blocked"))

    criterion = (task["validation_criterion"] + " " + task["subtask"]).lower()
    flux = experiment.get("flux_ratio")
    if flux and any(token in criterion for token in ("flux", "ratio", "asinh", "normaliz")):
        asinh_ok = bool(flux.get("linear_within_tolerance"))
        percentile_worse = bool(flux.get("percentile_destroys_ratios"))
        passed = asinh_ok and percentile_worse
        notes = (
            f"shared linear median rel error={flux['shared_linear_median_rel_error']:.4f}; "
            f"shared asinh={flux['shared_asinh_median_rel_error']:.4f}; "
            f"independent percentile={flux['independent_percentile_median_rel_error']:.4f}; "
            f"tolerance={flux['tolerance']}"
        )
        return passed, notes

    if experiment.get("status") == "completed":
        return True, "Local hardcoded execution completed; no physical probe matched this criterion."
    return False, f"Experiment did not complete: {experiment}"
