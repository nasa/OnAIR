# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"
#
# ---------------------------------------------------------------------------
# OnAIR Plugin: Anomaly Guardrail (with Severity Classification)
# ---------------------------------------------------------------------------
# Conforms to AIPlugin:
#   - class Plugin(AIPlugin)
#   - update(low_level_data=[], high_level_data={})
#   - render_reasoning()
#
# Core idea:
#   - Keep a sliding window per selected columns
#   - Detect outliers by Z-score or IQR
#   - Detect stuck sensors by repeated identical values
#   - Emit a recommendation with events when anomalies occur
#   - Respect a cooldown to avoid repeated emissions
#   - Optionally append audit rows to a CSV
#   - Classify each anomaly event as "minor" | "moderate" | "critical"
#          and derive an overall recommended action from severity.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import csv
import math
import time
from collections import deque
from pathlib import Path
import os
import configparser



from onair.src.ai_components.ai_plugin_abstract.ai_plugin import AIPlugin

Number = Union[int, float]
OptionalNumber = Optional[Number]


# --------------------------- Configuration ----------------------------------

@dataclass(frozen=True)
class GuardrailConfig:
    """
    Immutable configuration for the guardrail.

    - columns: indices of numeric fields to monitor in low_level_data
    - window: sliding window length
    - min_samples: minimum samples before testing
    - method: "zscore" or "iqr"
    - z_threshold: Z-score threshold to register an outlier event
    - iqr_k: IQR multiplier for outlier bounds
    - stuck_threshold: consecutive equal readings to flag "stuck"
    - cooldown_s: minimum seconds between recommendations
    - action: default action if no severity_actions mapping is used
    - log_csv_path: optional CSV audit file path
    - epsilon: small value to avoid division by zero

    Severity classification (added):
    - severity_moderate_score: score at/above which an event is "moderate"
    - severity_critical_score: score at/above which an event is "critical"
      * For Z-score events: score == z
      * For IQR events:    score == normalized distance beyond bound (|Δ| / IQR)
      * For stuck events:  score == consecutive_count / stuck_threshold
    - stuck_moderate_factor, stuck_critical_factor: factors applied to stuck_threshold
      to compute the normalized score for stuck events (count / stuck_threshold).
    - severity_actions: action per overall severity (fallback to `action` if absent)
    """
    columns: Sequence[int]
    window: int = 60
    min_samples: int = 20
    method: str = "zscore"               # "zscore" | "iqr"
    z_threshold: float = 3.0
    iqr_k: float = 1.5
    stuck_threshold: int = 50
    cooldown_s: float = 15.0
    action: str = "SAFE_MODE"
    log_csv_path: Optional[str] = None
    epsilon: float = 1e-9

    # --- Severity classification parameters (minimal, no external deps) ---
    severity_moderate_score: float = 2.5
    severity_critical_score: float = 4.0
    stuck_moderate_factor: float = 2.0
    stuck_critical_factor: float = 3.0
    severity_actions: Dict[str, str] = field(
        default_factory=lambda: {
            "minor": "LOG_ONLY",
            "moderate": "REDUCE_LOAD",
            "critical": "SAFE_MODE",
        }
    )

    def __post_init__(self) -> None:
        # Defensive validation; keep strict to avoid silent misconfiguration.
        if not self.columns or any((not isinstance(c, int) or c < 0) for c in self.columns):
            raise ValueError("`columns` must be a non-empty sequence of non-negative integers")
        if self.window <= 0:
            raise ValueError("`window` must be positive")
        if self.min_samples <= 0 or self.min_samples > self.window:
            raise ValueError("`min_samples` must be in (0, window]")
        if self.method not in ("zscore", "iqr"):
            raise ValueError("`method` must be 'zscore' or 'iqr'")
        if self.z_threshold <= 0.0:
            raise ValueError("`z_threshold` must be > 0")
        if self.iqr_k <= 0.0:
            raise ValueError("`iqr_k` must be > 0")
        if self.stuck_threshold < 1:
            raise ValueError("`stuck_threshold` must be >= 1")
        if self.cooldown_s < 0.0:
            raise ValueError("`cooldown_s` must be >= 0")
        if self.epsilon <= 0.0:
            raise ValueError("`epsilon` must be > 0")
        if self.severity_moderate_score <= 0.0 or self.severity_critical_score <= 0.0:
            raise ValueError("severity thresholds must be > 0")
        if self.severity_critical_score < self.severity_moderate_score:
            raise ValueError("critical score must be >= moderate score")
        if self.stuck_moderate_factor <= 0.0 or self.stuck_critical_factor <= 0.0:
            raise ValueError("stuck factors must be > 0")
        if self.stuck_critical_factor < self.stuck_moderate_factor:
            raise ValueError("stuck critical factor must be >= stuck moderate factor")


@dataclass(frozen=True)
class AnomalyEvent:
    """Single anomaly record."""
    step: int
    column: int
    kind: str                # "zscore" | "iqr" | "stuck"
    value: float
    score: Optional[float] = None
    severity: Optional[str] = None       # "minor" | "moderate" | "critical"
    context: Dict[str, Union[int, float, str]] = field(default_factory=dict)


# ------------------------ Numeric Sliding Window ----------------------------

class _SlidingWindow:
    """
    Fixed-capacity numeric sliding window for one column.

    - Accepts finite floats/ints; ignores None and non-finite values
    - Provides mean/std (Bessel-corrected) and quartiles
    """
    __slots__ = ("_capacity", "_data")

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = int(capacity)
        self._data: deque[float] = deque(maxlen=self._capacity)

    def push(self, v: OptionalNumber) -> None:
        if v is None:
            return
        if isinstance(v, (int, float)) and math.isfinite(v):
            self._data.append(float(v))

    def is_ready(self, min_samples: int) -> bool:
        return len(self._data) >= min_samples

    def mean_std(self, eps: float) -> Tuple[float, float]:
        # Sample std with Bessel correction; clamp by eps
        n = len(self._data)
        if n == 0:
            return 0.0, eps
        mu = sum(self._data) / n
        var = sum((x - mu) ** 2 for x in self._data) / max(1, n - 1)
        sigma = var ** 0.5
        return mu, max(sigma, eps)

    def quartiles(self) -> Tuple[float, float, float]:
        # Linear interpolation on sorted data
        if not self._data:
            return 0.0, 0.0, 0.0
        xs = sorted(self._data)
        n = len(xs)

        def q(p: float) -> float:
            idx = p * (n - 1)
            lo, hi = int(math.floor(idx)), int(math.ceil(idx))
            if lo == hi:
                return xs[lo]
            w = idx - lo
            return xs[lo] * (1 - w) + xs[hi] * w

        return q(0.25), q(0.5), q(0.75)


# ------------------------- Guardrail Core Logic -----------------------------

class AnomalyGuardrail:
    """
    Core anomaly detector with a simple feed API.

    - Call feed(step, row) with numeric row
    - Returns a recommendation dict when events occur and cooldown allows
    - Each event is now classified by severity; overall severity selects action.
    """

    def __init__(self, config: GuardrailConfig) -> None:
        self._cfg = config
        self._windows: Dict[int, _SlidingWindow] = {c: _SlidingWindow(config.window) for c in config.columns}
        self._last_values: Dict[int, Optional[float]] = {c: None for c in config.columns}
        self._stuck_counts: Dict[int, int] = {c: 0 for c in config.columns}
        self._last_action_ts: float = 0.0

        if self._cfg.log_csv_path:
            self._init_csv(self._cfg.log_csv_path)

    def feed(self, step: int, row: Sequence[Number]) -> Optional[Dict[str, Union[str, float, int, dict, list]]]:
        """
        Process one timestep.

        - step: non-negative int
        - row: sequence with numeric entries at configured columns
        - returns a recommendation dict or None
        """
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if not isinstance(row, Sequence):
            raise TypeError("row must be a sequence")

        events: List[AnomalyEvent] = []

        # Ingest and test each selected column
        for col in self._cfg.columns:
            val = self._safe_float(row, col)
            self._update_stuck(col, val, step, events)
            self._windows[col].push(val)
            if self._windows[col].is_ready(self._cfg.min_samples):
                self._check_statistical_anomaly(step, col, val, events)

        if not events:
            return None

        # Respect cooldown between actions (but still audit events)
        now = time.monotonic()
        if (now - self._last_action_ts) < self._cfg.cooldown_s:
            self._log_events(events)
            return None

        self._last_action_ts = now
        self._log_events(events)
        return self._make_recommendation(events)

    # --------------------------- Internals -----------------------------------

    def _check_statistical_anomaly(self, step: int, col: int, val: Optional[float], events: List[AnomalyEvent]) -> None:
        """Apply Z-score or IQR tests; classify severity on creation."""
        if val is None or not math.isfinite(val):
            return

        if self._cfg.method == "zscore":
            mu, sigma = self._windows[col].mean_std(self._cfg.epsilon)
            z = abs((val - mu) / sigma)
            if z >= self._cfg.z_threshold:
                sev = self._classify_continuous_score(z)
                events.append(
                    AnomalyEvent(
                        step=step, column=col, kind="zscore", value=val, score=z, severity=sev,
                        context={
                            "mu": mu,
                            "sigma": sigma,
                            "z_threshold": self._cfg.z_threshold,
                            "window": self._cfg.window,
                        },
                    )
                )
        elif self._cfg.method == "iqr":
            q1, median, q3 = self._windows[col].quartiles()
            iqr = max(q3 - q1, self._cfg.epsilon)
            lo = q1 - self._cfg.iqr_k * iqr
            hi = q3 + self._cfg.iqr_k * iqr
            if val < lo or val > hi:
                # Normalized distance beyond bound; higher => more severe
                score = (lo - val) / iqr if val < lo else (val - hi) / iqr
                sev = self._classify_continuous_score(abs(score))
                events.append(
                    AnomalyEvent(
                        step=step, column=col, kind="iqr", value=val, score=abs(score), severity=sev,
                        context={
                            "q1": q1,
                            "median": median,
                            "q3": q3,
                            "iqr_k": self._cfg.iqr_k,
                            "window": self._cfg.window,
                        },
                    )
                )
        else:
            raise ValueError(f"unknown method: {self._cfg.method}")

    def _update_stuck(self, col: int, val: Optional[float], step: int, events: List[AnomalyEvent]) -> None:
        """Track consecutive identical values and raise 'stuck' when threshold is exceeded."""
        prev = self._last_values[col]
        if val is None or not math.isfinite(val):
            self._last_values[col] = val
            self._stuck_counts[col] = 0
            return

        # Same-value detection with epsilon tolerance
        if prev is not None and math.isfinite(prev) and abs(val - prev) < self._cfg.epsilon:
            self._stuck_counts[col] += 1
            if self._stuck_counts[col] >= self._cfg.stuck_threshold:
                # Normalize count by threshold to compare against factor-based severities
                ratio = self._stuck_counts[col] / max(1, self._cfg.stuck_threshold)
                sev = self._classify_stuck_ratio(ratio)
                events.append(
                    AnomalyEvent(
                        step=step,
                        column=col,
                        kind="stuck",
                        value=val,
                        score=float(self._stuck_counts[col]),
                        severity=sev,
                        context={
                            "stuck_threshold": self._cfg.stuck_threshold,
                            "ratio": ratio,
                        },
                    )
                )
        else:
            self._stuck_counts[col] = 0

        self._last_values[col] = val

    def _classify_continuous_score(self, score: float) -> str:
        """
        Classify severity for Z-score/IQR-based events.
        - score: z for Z-score, |Δ|/IQR for IQR
        """
        if score >= self._cfg.severity_critical_score:
            return "critical"
        if score >= self._cfg.severity_moderate_score:
            return "moderate"
        return "minor"

    def _classify_stuck_ratio(self, ratio: float) -> str:
        """
        Classify severity for 'stuck' based on consecutive-count ratio:
          ratio = (consecutive_count / stuck_threshold).
        """
        if ratio >= self._cfg.stuck_critical_factor:
            return "critical"
        if ratio >= self._cfg.stuck_moderate_factor:
            return "moderate"
        return "minor"

    def _make_recommendation(self, events: List[AnomalyEvent]) -> Dict[str, Union[str, float, int, dict, list]]:
        """
        Compose recommendation payload:
        - Picks an overall severity (highest among events).
        - Selects an action based on severity_actions mapping, falling back to `action`.
        """
        severity_rank = {"minor": 0, "moderate": 1, "critical": 2}
        overall = "minor"
        for e in events:
            sev = e.severity or "minor"
            if severity_rank[sev] > severity_rank[overall]:
                overall = sev

        action = self._cfg.severity_actions.get(overall, self._cfg.action)

        return {
            "reason": "ANOMALY_DETECTED",
            "overall_severity": overall,
            "action": action,
            "events": [self._event_to_dict(e) for e in events],
        }

    @staticmethod
    def _event_to_dict(e: AnomalyEvent) -> Dict[str, Union[str, int, float, dict]]:
        """Serialize an event to a JSON-friendly dict."""
        d: Dict[str, Union[str, int, float, dict]] = {
            "step": e.step,
            "col": e.column,
            "kind": e.kind,
            "val": e.value,
        }
        if e.score is not None:
            d["score"] = float(e.score)
        if e.severity is not None:
            d["severity"] = e.severity
        if e.context:
            d["context"] = e.context
        return d

    @staticmethod
    def _safe_float(row: Sequence[Number], col: int) -> Optional[float]:
        """Best-effort extraction of a finite float from row[col]."""
        if col < 0 or col >= len(row):
            return None
        v = row[col]
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return fv if math.isfinite(fv) else None

    # ---------------------------- CSV Audit ----------------------------------

    def _init_csv(self, path: str) -> None:
        """Create CSV with header if missing."""
        try:
            with open(path, "x", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "step", "column", "kind", "value", "score", "severity", "context"])
        except FileExistsError:
            return

    def _log_events(self, events: Iterable[AnomalyEvent]) -> None:
        """Append anomaly events to CSV if configured."""
        if not self._cfg.log_csv_path:
            return
        ts = time.time()
        with open(self._cfg.log_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for e in events:
                w.writerow([
                    f"{ts:.3f}",
                    e.step,
                    e.column,
                    e.kind,
                    f"{e.value:.9g}",
                    "" if e.score is None else f"{float(e.score):.6g}",
                    "" if e.severity is None else e.severity,
                    e.context,
                ])


# --------------------------- OnAIR AIPlugin ---------------------------------
class Plugin(AIPlugin):
    """
    Anomaly Guardrail plugin for OnAIR.

    Responsibilities:
      - Load configuration from headers, INI file, or defaults.
      - Build a GuardrailConfig and the underlying AnomalyGuardrail engine.
      - Clean the full telemetry frame before feeding into the guardrail.
      - Deduplicate repeated anomaly recommendations.
      - Provide human-readable reasoning for dashboards.
    """

    def __init__(self, construct_name: str, headers: Union[dict, Sequence[str]]):
        super().__init__(construct_name, headers if isinstance(headers, list) else [])
        self.name = construct_name
        self.component_name = construct_name

        # Store headers
        self.header_names: List[str] = list(headers) if isinstance(headers, list) else []
        self.headers: Dict[str, object] = headers if isinstance(headers, dict) else {}

        # Load configuration
        cfg_dict: Dict[str, object] = dict(self.headers.get("config") or {})

        if not cfg_dict:
            ini_path: Optional[str] = None
            if isinstance(self.headers, dict):
                ini_path = self.headers.get("configfile")
            if not ini_path:
                ini_path = os.environ.get("ONAIR_CONFIG_FILE") or os.environ.get("ONAIR_INI")
            if not ini_path:
                ini_path = str(
                    Path(__file__).resolve().parents[2]
                    / "onair" / "config" / "config_anomaly_guardrail.ini"
                )

            try:
                cp = configparser.ConfigParser()
                cp.read(ini_path)
                if cp.has_section("anomaly_guardrail"):
                    s = cp["anomaly_guardrail"]

                    def as_int_list(text: str) -> List[int]:
                        return [int(t.strip()) for t in text.split(",") if t.strip()]

                    def as_int(key: str, default: Optional[int] = None) -> Optional[int]:
                        return int(s.get(key, default)) if s.get(key) is not None else default

                    def as_float(key: str, default: Optional[float] = None) -> Optional[float]:
                        return float(s.get(key, default)) if s.get(key) is not None else default

                    # Core configuration
                    if s.get("columns"):
                        cfg_dict["columns"] = as_int_list(s.get("columns"))
                    if s.get("window"):
                        cfg_dict["window"] = as_int("window", 60)
                    if s.get("min_samples"):
                        cfg_dict["min_samples"] = as_int("min_samples", 20)
                    if s.get("method"):
                        cfg_dict["method"] = s.get("method", "zscore").lower()
                    if s.get("z_threshold"):
                        cfg_dict["z_threshold"] = as_float("z_threshold", 3.0)
                    if s.get("iqr_k"):
                        cfg_dict["iqr_k"] = as_float("iqr_k", 1.5)
                    if s.get("stuck_threshold"):
                        cfg_dict["stuck_threshold"] = as_int("stuck_threshold", 50)
                    if s.get("cooldown_s"):
                        cfg_dict["cooldown_s"] = as_float("cooldown_s", 15.0)
                    if s.get("action"):
                        cfg_dict["action"] = s.get("action", "SAFE_MODE")
                    if s.get("log_csv"):
                        cfg_dict["log_csv_path"] = s.get("log_csv")
                    if s.get("epsilon"):
                        cfg_dict["epsilon"] = as_float("epsilon", 1e-9)

                    # Severity thresholds
                    if s.get("severity_moderate_score"):
                        cfg_dict["severity_moderate_score"] = as_float("severity_moderate_score", 2.5)
                    if s.get("severity_critical_score"):
                        cfg_dict["severity_critical_score"] = as_float("severity_critical_score", 4.0)
                    if s.get("stuck_moderate_factor"):
                        cfg_dict["stuck_moderate_factor"] = as_float("stuck_moderate_factor", 2.0)
                    if s.get("stuck_critical_factor"):
                        cfg_dict["stuck_critical_factor"] = as_float("stuck_critical_factor", 3.0)

                    # Severity actions mapping
                    if s.get("severity_actions"):
                        pairs = [p.strip() for p in s.get("severity_actions").split(",") if p.strip()]
                        mapping = {}
                        for p in pairs:
                            if ":" in p:
                                k, v = p.split(":", 1)
                                mapping[k.strip()] = v.strip()
                        cfg_dict["severity_actions"] = mapping

            except Exception:
                pass

        # Default: monitor all columns except index 0
        cfg_dict.setdefault("columns", list(range(1, len(self.header_names))))

        # Normalize column names to indices if necessary
        names = list(self.header_names)
        cols = cfg_dict.get("columns")
        if cols and isinstance(cols, list) and len(cols) > 0 and isinstance(cols[0], str):
            mapped = []
            for name in cols:
                if name not in names:
                    raise ValueError(f"[Guardrail] Unknown column name: {name}")
                mapped.append(names.index(name))
            cfg_dict["columns"] = mapped

        # Validate indices
        cols = cfg_dict.get("columns")
        if cols is not None:
            for c in cols:
                if not isinstance(c, int) or c < 1 or (len(names) and c >= len(names)):
                    raise ValueError(
                        f"[Guardrail] Invalid column index: {c} (header size={len(names)})"
                    )

        # Validate numeric parameters
        for key in ("window", "min_samples", "stuck_threshold"):
            v = cfg_dict.get(key)
            if v is not None and (not isinstance(v, int) or v <= 0):
                raise ValueError(f"[Guardrail] {key} must be positive int, got {v}")

        method = str(cfg_dict.get("method", "zscore")).lower()
        if method not in ("zscore", "iqr"):
            raise ValueError("[Guardrail] method must be one of: zscore | iqr")
        cfg_dict["method"] = method

        # Ensure log path exists
        try:
            logs_dir = Path(__file__).resolve().parents[2] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            cfg_dict.setdefault("log_csv_path", str(logs_dir / "anomaly_events.csv"))
        except Exception:
            cfg_dict.setdefault("log_csv_path", "logs/anomaly_events.csv")

        # Build configuration and guardrail engine
        self._cfg = GuardrailConfig(**cfg_dict)
        self._guardrail = AnomalyGuardrail(self._cfg)

        # State for deduplication and stats
        self._step = -1
        self._last_recommendation: Optional[dict] = None
        self._last_event_key: Optional[str] = None
        self._last_event_step: int = -10**9
        self._dedup_steps: int = 5
        self._n_frames: int = 0
        self._n_events: int = 0

    def _clean_full_frame(self, data: Sequence[Number]) -> List[Optional[float]]:
        """
        Convert the full telemetry row to floats, replacing invalid entries with None.
        Keeps original length and order so global indices remain valid.
        """
        cleaned: List[Optional[float]] = []
        for v in data:
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    cleaned.append(None)
                else:
                    cleaned.append(fv)
            except Exception:
                cleaned.append(None)
        return cleaned

    def update(self, low_level_data: Sequence[Number] = [], high_level_data: dict = {}) -> Optional[dict]:
        """
        Increment the step counter, clean the full row, and feed it to the guardrail.
        Deduplicate repeated recommendations for a short window.
        """
        self._step += 1
        self._n_frames += 1

        rec = None
        try:
            if isinstance(low_level_data, Sequence) and len(low_level_data) > 0:
                cleaned_full = self._clean_full_frame(low_level_data)
                rec = self._guardrail.feed(self._step, cleaned_full)
        except Exception as e:
            rec = {
                "reason": "FEED_ERROR",
                "action": "NOOP",
                "overall_severity": "minor",
                "error": str(e),
                "events": [],
            }

        if rec and rec.get("reason") and rec.get("overall_severity"):
            key = f'{rec.get("reason")}:{rec.get("overall_severity")}'
            if key == self._last_event_key and (self._step - self._last_event_step) <= self._dedup_steps:
                rec["suppressed"] = True
            else:
                self._last_event_key = key
                self._last_event_step = self._step

        if rec and isinstance(rec.get("events"), list):
            self._n_events += len(rec["events"])

        self._last_recommendation = rec
        return rec

    def render_reasoning(self) -> List[str]:
        """
        Produce a single-line reasoning string.
        Returns "Nominal" when no anomalies are present.
        """
        rec = self._last_recommendation
        if not rec:
            return ["Nominal"]

        reason = rec.get("reason", "ANOMALY_DETECTED")
        action = rec.get("action", "SAFE_MODE")
        overall = rec.get("overall_severity", "minor")
        suffix = " [suppressed]" if rec.get("suppressed") else ""

        events = rec.get("events", [])
        if not events:
            return ["Nominal"]

        counts: Dict[str, int] = {}
        for e in events:
            k = e.get("kind", "unknown")
            counts[k] = counts.get(k, 0) + 1
        kinds_summary = ", ".join(f"{k}:{v}" for k, v in counts.items())

        return [f"{reason} -> severity={overall}; action={action}{suffix}; events={len(events)} ({kinds_summary})"]
