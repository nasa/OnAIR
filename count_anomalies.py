
"""
count_anomalies.py

Goal:
- Read the anomaly CSV produced by the AnomalyGuardrail plugin.
- Optionally "collapse" long runs of repeated 'stuck' events into a single entry per run
  (edge mode), and optionally emit a closing 'stuck_end' entry.
- Write a cleaned CSV and print a concise summary.

Logic notes:
- "level" mode: keep all rows as-is (current behavior).
- "edge" mode: for each column, detect transitions:
    inactive -> active  : emit the first 'stuck' row (start edge)
    active   -> inactive: optionally emit 'stuck_end' (end edge)
- A run is considered continuous if consecutive 'stuck' rows for the same column have
  step differences <= max_step_gap (default 1). Larger gaps open a new run.
- Non-'stuck' rows are always passed through unchanged. When we see a non-'stuck'
  row we also close any open runs (this makes end edges align better with mixed streams).
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path) -> List[Dict[str, str]]:
    """Load CSV rows with DictReader. If the file is empty, return []."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    """Write CSV dict rows with a header. Create parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[Dict[str, str]]) -> Dict[str, int]:
    """Count rows per 'kind' for a quick console summary."""
    out: Dict[str, int] = {}
    for r in rows:
        k = r.get("kind", "")
        out[k] = out.get(k, 0) + 1
    return out


def collapse_stuck(
    rows: List[Dict[str, str]],
    emit_end: bool = False,
    max_step_gap: int = 1,
) -> List[Dict[str, str]]:
    """
    Collapse consecutive 'stuck' runs per column into a single start edge (and optional end edge).

    State per column:
      last_step: last seen step of the current run (int)
      open_row : the first row (dict) of the current run or None if closed

    Rules:
      - If next stuck step is within max_step_gap of last_step => same run (update last_step, no emit)
      - Else => new run (emit this row as the start edge, store as open_row)
      - On non-stuck rows => close any open runs (emit 'stuck_end' if enabled), pass row through
      - At EOF => close any remaining open runs (optional 'stuck_end')
    """
    clean: List[Dict[str, str]] = []
    # Per-column run cache: col -> {"last_step": int, "open_row": dict or None}
    run: Dict[int, Dict[str, object]] = {}

    def parse_int(s: str, default: int) -> int:
        try:
            return int(s)
        except Exception:
            return default

    def col_key(r: Dict[str, str]) -> int:
        return parse_int(r.get("column", "-1"), -1)

    def step_num(r: Dict[str, str]) -> int:
        return parse_int(r.get("step", "-1"), -1)

    def close_run(col: int, row_after: Dict[str, str] = None) -> None:
        """
        Emit an end edge for an open run if requested. The end step is (row_after.step - 1)
        when available; otherwise we reuse the open row's step. We duplicate the open row's
        fields to keep the CSV schema stable and only adjust 'kind', 'score', and 'step'.
        """
        if not emit_end:
            run[col]["open_row"] = None
            return
        open_row = run[col]["open_row"]
        if not open_row:
            return
        end_row = dict(open_row)  # copy the opening row as a template
        end_row["kind"] = "stuck_end"
        end_row["score"] = "0"
        if row_after is not None:
            end_row["step"] = str(max(0, step_num(row_after) - 1))
        # else keep the original start step (best effort if no context)
        clean.append(end_row)
        run[col]["open_row"] = None

    # Main pass
    for r in rows:
        if r.get("kind", "") != "stuck":
            # Any non-stuck row closes all open runs before passing through
            for col in list(run.keys()):
                if run[col].get("open_row"):
                    close_run(col, r)
            clean.append(r)
            continue

        # 'stuck' row: decide if we are in the same run or starting a new one
        col = col_key(r)
        stp = step_num(r)

        # Initialize per-column state on demand
        if col not in run:
            run[col] = {"last_step": None, "open_row": None}

        last_step = run[col]["last_step"]  # may be None at the beginning
        open_row = run[col]["open_row"]

        # Same run if last_step exists and step gap is small enough
        if (isinstance(last_step, int)) and (stp - last_step <= max_step_gap):
            run[col]["last_step"] = stp
            # Do NOT emit (we are collapsing repeated stucks)
            continue

        # Otherwise we are starting a new run: emit this as the start edge
        clean.append(r)
        run[col]["open_row"] = r
        run[col]["last_step"] = stp

    # End of file: close any still-open runs
    for col in list(run.keys()):
        if run[col].get("open_row"):
            close_run(col, None)

    return clean


def main():
    # CLI so you can switch behaviors without touching code
    ap = argparse.ArgumentParser(description="Summarize and optionally collapse anomaly CSV.")
    ap.add_argument("--in", dest="inp", default="logs/anomaly_events.csv", help="input CSV path")
    ap.add_argument("--out", dest="out", default="logs/anomaly_events_clean.csv", help="output CSV path")
    ap.add_argument("--mode", choices=["level", "edge"], default="edge",
                    help="level: keep all rows; edge: collapse consecutive 'stuck' rows")
    ap.add_argument("--emit-end", action="store_true",
                    help="when in edge mode, also emit a 'stuck_end' row at run termination")
    ap.add_argument("--max-gap", type=int, default=1,
                    help="max allowed step gap to consider a 'stuck' row part of the same run")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        print(f"[!] Not found: {inp}")
        return

    rows = load_rows(inp)
    if not rows:
        print("[i] Empty input.")
        return

    fieldnames = list(rows[0].keys())
    print(f"[i] Loaded rows: {len(rows)}")

    if args.mode == "edge":
        cleaned = collapse_stuck(rows, emit_end=args.emit_end, max_step_gap=args.max_gap)
    else:
        cleaned = rows

    write_rows(Path(args.out), cleaned, fieldnames)
    print(f"[i] Wrote cleaned rows: {len(cleaned)} → {args.out}")

    # Print a small before/after summary by kind
    by_kind_in = summarize(rows)
    by_kind_out = summarize(cleaned)
    print("[i] Summary (input) :", by_kind_in)
    print("[i] Summary (output):", by_kind_out)

    # Convenience: show last input and last output rows for quick sanity check
    print("[i] Last input row :", rows[-1])
    print("[i] Last output row:", cleaned[-1] if cleaned else "<none>")


if __name__ == "__main__":
    main()