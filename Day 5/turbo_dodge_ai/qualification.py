"""Evaluation summaries and qualification-manifest helpers.

This module intentionally depends only on the Python standard library.  The
menu can therefore determine whether *Human vs AI* is available even on a
machine that has not installed Stable-Baselines3 yet.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
QUALIFIED_MANIFEST = ARTIFACTS_DIR / "qualified_model.json"

SOLO_THRESHOLDS = {
    "median_survival_seconds": 30.0,
    "survival_rate_30_seconds": 0.70,
    "mean_abs_steering_change": 0.15,
}

MULTIPLAYER_THRESHOLDS = {
    "win_or_draw_rate": 0.70,
    "win_rate": 0.35,
    "contact_draw_rate": 0.10,
}


def utc_now() -> str:
    """Return a portable ISO-8601 timestamp in UTC."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _finite_float(value: Any, default: float = 0.0) -> float:
    """Convert a metric value without allowing NaN/Infinity into JSON."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _as_jsonable(value: Any) -> Any:
    """Turn Path/numpy scalar-like values into safely serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    # numpy scalar types expose ``item`` but standard types such as str do not.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _as_jsonable(item())
        except (TypeError, ValueError):
            pass
    return value


def _env_dt(environment: Any) -> float:
    """Best-effort fixed timestep lookup for an environment or wrapper."""

    candidate = getattr(environment, "dt", None)
    if candidate is None:
        candidate = getattr(getattr(environment, "unwrapped", None), "dt", None)
    dt = _finite_float(candidate, 1.0 / 30.0)
    return dt if dt > 0.0 else 1.0 / 30.0


def _action_scalar(action: Any) -> float:
    """Extract the single steering value from SB3's one-dimensional action."""

    try:
        return float(action[0])
    except (IndexError, KeyError, TypeError):
        return _finite_float(action)


def rollout_policy(
    model: Any,
    environment_factory: Callable[..., Any],
    *,
    phase: str,
    episodes: int,
    seed_start: int,
    deterministic: bool = True,
    render: bool = False,
    max_steps: int | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Run a model on fixed seeds and return machine-readable episode records.

    ``environment_factory`` is deliberately injected so this file has no
    import-time Gymnasium or Stable-Baselines3 dependency.  The core
    environment's optional info fields are used when present and have safe
    fallbacks for smoke-test environments.
    """

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    records: list[dict[str, Any]] = []
    for index in range(episodes):
        seed = seed_start + index
        environment = environment_factory(phase=phase, seed=seed, render_mode="human" if render else None)
        try:
            observation, reset_info = environment.reset(seed=seed)
            info: Mapping[str, Any] = reset_info if isinstance(reset_info, Mapping) else {}
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            actions: list[float] = []

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=deterministic)
                actions.append(_action_scalar(action))
                observation, reward, terminated, truncated, step_info = environment.step(action)
                info = step_info if isinstance(step_info, Mapping) else {}
                total_reward += _finite_float(reward)
                steps += 1
                if render:
                    environment.render()
                if max_steps is not None and steps >= max_steps:
                    truncated = True

            steering_changes = [abs(current - previous) for previous, current in zip(actions, actions[1:])]
            elapsed = _finite_float(info.get("elapsed_time"), steps * _env_dt(environment))
            outcome = str(
                info.get("match_result", info.get("outcome", info.get("result", "")))
            ).lower()
            collision_reason = str(info.get("collision_reason", ""))
            contact_draw = (
                outcome == "draw"
                and ("contact" in collision_reason.lower() or bool(info.get("car_contact", False)))
            )
            records.append(
                {
                    "seed": seed,
                    "phase": phase,
                    "return": total_reward,
                    "steps": steps,
                    "survival_seconds": elapsed,
                    "passed_groups": int(_finite_float(info.get("passed_groups"), 0.0)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "collision_reason": collision_reason,
                    "outcome": outcome,
                    "contact_draw": contact_draw,
                    "mean_abs_steering_change": _mean(steering_changes),
                }
            )
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()

    return summarize_records(records), records


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Produce comparable solo and multiplayer metrics from rollout records."""

    survivals = [_finite_float(record.get("survival_seconds")) for record in records]
    returns = [_finite_float(record.get("return")) for record in records]
    passed = [_finite_float(record.get("passed_groups")) for record in records]
    steering = [_finite_float(record.get("mean_abs_steering_change")) for record in records]
    count = len(records)
    outcomes = [str(record.get("outcome", "")).lower() for record in records]
    wins = sum(outcome == "win" for outcome in outcomes)
    draws = sum(outcome == "draw" for outcome in outcomes)
    losses = sum(outcome == "loss" for outcome in outcomes)
    contact_draws = sum(bool(record.get("contact_draw", False)) for record in records)
    known_results = wins + draws + losses

    summary: dict[str, float | int] = {
        "episode_count": count,
        "mean_return": _mean(returns),
        "median_survival_seconds": _median(survivals),
        "mean_survival_seconds": _mean(survivals),
        "survival_rate_30_seconds": (sum(value >= 30.0 for value in survivals) / count) if count else 0.0,
        "mean_passed_groups": _mean(passed),
        "mean_abs_steering_change": _mean(steering),
        "match_count": known_results,
        "win_rate": (wins / known_results) if known_results else 0.0,
        "draw_rate": (draws / known_results) if known_results else 0.0,
        "win_or_draw_rate": ((wins + draws) / known_results) if known_results else 0.0,
        "contact_draw_rate": (contact_draws / known_results) if known_results else 0.0,
    }
    return summary


def assess_qualification(
    solo_metrics: Mapping[str, Any] | None,
    multiplayer_metrics: Mapping[str, Any] | None,
    *,
    require_multiplayer: bool = True,
) -> dict[str, Any]:
    """Evaluate the published unlock thresholds without making policy choices."""

    checks: dict[str, dict[str, Any]] = {}
    solo_metrics = solo_metrics or {}
    multiplayer_metrics = multiplayer_metrics or {}
    for metric, threshold in SOLO_THRESHOLDS.items():
        actual = _finite_float(solo_metrics.get(metric), -1.0)
        # Steering change has an upper bound; all other solo metrics have lower bounds.
        passed = actual <= threshold if metric == "mean_abs_steering_change" else actual >= threshold
        checks[f"solo.{metric}"] = {"actual": actual, "threshold": threshold, "passed": passed}

    if require_multiplayer:
        for metric, threshold in MULTIPLAYER_THRESHOLDS.items():
            actual = _finite_float(multiplayer_metrics.get(metric), -1.0)
            passed = actual <= threshold if metric == "contact_draw_rate" else actual >= threshold
            checks[f"multiplayer.{metric}"] = {"actual": actual, "threshold": threshold, "passed": passed}

    return {
        "qualified": bool(checks) and all(check["passed"] for check in checks.values()),
        "require_multiplayer": require_multiplayer,
        "checks": checks,
    }


def load_manifest(path: Path = QUALIFIED_MANIFEST) -> dict[str, Any] | None:
    """Load the menu-facing manifest, returning ``None`` for absent/corrupt data."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def relative_to_package(path: str | Path) -> str:
    """Return a model path relative to ``turbo_dodge_ai`` when possible."""

    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        return str(candidate)


def publish_manifest(
    *,
    model_path: str | Path,
    phase: str,
    solo_metrics: Mapping[str, Any] | None,
    multiplayer_metrics: Mapping[str, Any] | None,
    qualification: Mapping[str, Any] | None = None,
    path: Path = QUALIFIED_MANIFEST,
    require_multiplayer: bool = True,
) -> dict[str, Any]:
    """Write the stable UI manifest atomically.

    An already qualified model is retained when a later, unqualified run is
    evaluated.  This avoids locking the versus menu after a user experiments
    with a fresh training run.
    """

    assessment = dict(
        qualification
        or assess_qualification(
            solo_metrics, multiplayer_metrics, require_multiplayer=require_multiplayer
        )
    )
    candidate = {
        "qualified": bool(assessment.get("qualified", False)),
        "model_path": relative_to_package(model_path),
        "phase": phase,
        "timestamp": utc_now(),
        "metrics": {
            "solo": _as_jsonable(dict(solo_metrics or {})),
            "multiplayer": _as_jsonable(dict(multiplayer_metrics or {})),
        },
        "qualification": _as_jsonable(assessment),
    }

    existing = load_manifest(path)
    if existing and existing.get("qualified") and not candidate["qualified"]:
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return candidate


def format_qualification_status(assessment: Mapping[str, Any]) -> str:
    """Return a compact, UI/terminal-friendly explanation of the unlock state."""

    state = "QUALIFIED" if assessment.get("qualified") else "NOT QUALIFIED"
    failed = [
        f"{name}: {check.get('actual', 0):.3f} (target {check.get('threshold', 0):.3f})"
        for name, check in dict(assessment.get("checks", {})).items()
        if not check.get("passed")
    ]
    return state if not failed else f"{state} — " + "; ".join(failed)
