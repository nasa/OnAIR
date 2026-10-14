# OnAIR Telemetry Ingestion Hardening — Final Report

Repo root for all commands: `OnAIR-main/`. The root `conftest.py` was used for every
test invocation and every mutmut run (it injects `pytest.gen`, mocks `simdkalman` /
`sbn_client` / `message_headers`); it was never bypassed, modified, or removed.

Environment note: the host had no `python` launcher, no pip, and no `venv`/`ensurepip`.
A virtual environment was bootstrapped at `OnAIR-main/.venv` (Python 3.12.3, pip
bootstrapped via `get-pip.py`), and `requirements.txt` (superset: run + unittest) plus
`mutmut 2.5.1` and `hypothesis 6.169.1` were installed into it. All `python` /
`mutmut` commands below ran from `.venv` (`PATH="$PWD/.venv/bin:$PATH"`). Because
mutmut executes its runner via `subprocess` **without a shell**, the Pass-2 runner's
leading `MUTMUT_HYPOTHESIS_FAST=1` was supplied by exporting the variable in the
invoking shell (semantically identical: the pytest child process inherits it).

## 1. Baseline evidence (Step 0)

```
python -m pip install -r requirements-unittest.txt
python -m pip install "mutmut>=2.4,<3" hypothesis   # mutmut 2.5.1, hypothesis 6.169.1
python -m pytest test/ -q
→ 371 passed in 4.02s        # 100% green before any change (conftest seed printed)
```

## 2. Pre-patch mutation proof (Step 1) — the blind-spot catalog

Runner (both scoped passes): `python -m pytest -x -q
test/onair/data_handling/test_parser_util.py test/plugins/kalman/test_kalman_plugin.py`.

### 2a. `onair/data_handling/parser_util.py` — 52 mutants, 51 killed, 1 survivor

| Mutant id | Function | Diff | Blind spot demonstrated |
|---|---|---|---|
| 40 | `floatify_input` | trailing `continue` → `break` after the 0.0-masking fallback | No legacy test feeds ≥2 junk items, so early loop exit (silent truncation of the rest of the frame) is invisible. All other floatify/convert mutants were killed pre-patch — the masking bug itself (`0.0`) is *encoded* by the legacy tests, hence unkillable there. |

### 2b. `plugins/kalman/kalman_plugin.py` — 55 mutants, 42 killed, 13 survivors

| Mutant id | Function | Diff | Classification |
|---|---|---|---|
| 22 | `update()` | None-warning message text `XX…XX` | **in scope** — survived pre-patch (legacy test only checked `call_count`) |
| 23 | `update()` | `["FAIL"]` → `["XXFAILXX"]` | **in scope** — survived pre-patch (clrs never asserted) |
| 3, 4 | `__init__` | `window_size < 3` → `<= 3` / `< 4` | out of scope (catalogued) |
| 5 | `__init__` | error-message string | out of scope (catalogued) |
| 15–18 | `__init__` | `process_noise` diag / `observation_model` literals | out of scope; values feed only the conftest-mocked `simdkalman.KalmanFilter` |
| 28 | `render_reasoning` | `>` → `>=` threshold | out of scope (catalogued) |
| 33 | `render_reasoning` | `"TIME"` → `"XXTIMEXX"` | out of scope (catalogued) |
| 36, 38 | `_generate_residuals` | `frames[0]`→`frames[1]`; `>2`→`>3` | **documented-excluded** (`_predict`/`_generate_residuals` run against a conftest `MagicMock` simdkalman; behavior cannot be genuinely asserted) |

## 3. Pre-patch property falsifying examples (Step 2)

Files: `test/onair/data_handling/test_parser_util_properties.py`,
`test/plugins/kalman/test_kalman_properties.py`. Both were written **before** the
production patch and executed against the unpatched code
(`python -m pytest … -v --tb=long`): **11 failed, 3 passed**.

| Property | Result pre-patch | Falsifying example (verbatim from hypothesis) |
|---|---|---|
| P1 never-raises (json scalars) | FAIL | `items=[None]` → `TypeError: float() argument must be a string or a real number, not 'NoneType'` (leaks past `except ValueError`) |
| P2 length preserved (`remove_str=False`) | FAIL | `items=[None]` → same TypeError leak |
| P3 identity on finite floats | **PASS** (already held pre-patch, as predicted) | — |
| P4 `"-"` never 0.0 | FAIL | `items=['-']` → result `[0.0]`, `assert 0.0 != 0.0` (the 0.0-masking bug itself) |
| P5 non-finite spellings (strings) | FAIL | `s='inf'` → `math.isnan(inf)` is False (`inf` passed straight through) |
| P5 non-finite floats | FAIL | `v=inf` → passthrough |
| P6 valid `MM:SS` exact epoch | **PASS** (held pre-patch) | — |
| P6 valid `%Y-%j-%H:%M:%S.%f` exact epoch | **PASS** (held pre-patch) | — |
| P6 malformed → ValueError/nan | FAIL | `s='12:34:56'` → returned `946671154.0` (`:56` silently truncated) instead of ValueError/nan |
| P6 non-string input | FAIL | `v=None` → TypeError leak |
| Kalman short-frame rejection (@given) | FAIL | `num_headers=1, window_size=3, short_by=1` → frame stored / desync (no guard) |
| Kalman long-frame rejection (@given) | FAIL | `num_headers=1, extra=1` → `IndexError` at `kalman_plugin.py:114` (`self.frames[i].append`) |
| Kalman None-frame (@given) | **PASS** (None handling existed pre-patch) | — |
| Stateful machine (`runTest`) | FAIL | (a) `junk_frame(vals=[None,…])` → TypeError at `parser_util.py:55`; (b) `nan_spike(spike=nan)` → I4 violated (non-finite stored in window); (c) `long_frame(extra=1)` → frames mutated |

## 4. The hardening patches (Step 3)

Only the three target functions changed; no signatures changed; stdlib `math` only.

### 4a. `floatify_input` — policy ladder

```
try: x = float(i); if math.isfinite(x): append; continue
except (ValueError, TypeError): pass        # D1 dead: TypeError can no longer leak
# non-finite float() results (nan/inf strings) fall through to the timestamp
# attempt, which rejects them -> junk policy below
try: x = convert_str_to_timestamp(i); append; continue
except Exception: pass
if remove_str: continue                      # drop entirely
print_msg(f"floatify_input: could not parse {i!r} to a finite float; substituting math.nan", ["WARNING"])
floatified.append(math.nan)                  # no 0.0 anywhere (D2 dead)
```

`import math` and `from onair.src.util.print_io import print_msg` were added (no
import cycle: `print_io` imports nothing). Timestamp results are appended without an
isfinite gate (strptime epochs are inherently finite; keeps the legacy mock test at
former lines 478–503 green). Full policy is documented in the docstring.

### 4b. `convert_str_to_timestamp` — guarded `MM:SS` fallback

`except Exception` kept around `strptime` (legacy `Fake_Datetime` test raises plain
`Exception`). In the handler: `isinstance(time_str, str)` else `raise ValueError`;
`split(":")` must yield exactly 2 parts; both parts `int()`-parseable; minute and
second each in 0..59; else `ValueError` with a clear, distinct message per failure
mode. `datetime(2000, 1, 1, 1, minute, second, 0).timestamp()` kept (`12:34:56`,
`99:99`, `12:60`, `None` now raise — D4 dead; no IndexError/AttributeError leaks).

### 4c. `Plugin.update` — length guard + finite-only windows

`import math` added. New branch:
`elif len(low_level_data) != len(self.headers):` →
`print_msg(f"Kalman plugin update: frame length {len(low_level_data)} does not match
header length {len(self.headers)}; frame rejected.", ["WARNING"]); return`
(no IndexError, no frame/window desync — D5 dead). After `floatify_input`, a
per-attribute `if not math.isfinite(value): continue` keeps only finite floats in
the windows; skipped samples advance no window.

## 5. Property inventory (locked post-patch)

`test_parser_util_properties.py` — `json_scalar =
st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True,
allow_infinity=True), st.text(max_size=8))`; sentinels `-`, `nan`, `inf`, `-inf`,
`Infinity`, `""`, `12`, `12:34`, `12:34:56`, `2020-001-12:34:56.789`;
settings constant = `settings(max_examples=1000, deadline=None)` normally,
`settings(max_examples=25, deadline=None, derandomize=True)` when
`MUTMUT_HYPOTHESIS_FAST` is set; autouse fixture records `parser_util.print_msg`
(`raising=False`, pre-patch-safe).

- **P1** `floatify_input` never raises for any list of json scalars (both `remove_str` modes).
- **P2** `len(out) == len(in)` for `remove_str=False`, all inputs.
- **P3** identity on finite floats.
- **P4** `"-"` never yields `0.0`; post-patch `math.isnan` + `["WARNING"]` asserted + `remove_str=True` drops.
- **P5** `nan`/`inf`/`NaN`/`Infinity` strings and NaN/Inf floats → `math.isnan(out[0])`, never 0.0/crash.
- **P6** valid `MM:SS` (m,s ∈ 0..59) and `%Y-%j-%H:%M:%S.%f` → exact epoch equality vs
  `datetime` computed in the test; malformed (`12:34:56`, `99:99`, `-`, `None`, …) →
  `ValueError` or `nan`, never IndexError/AttributeError.

`test_kalman_properties.py` — real `Plugin` instances (conftest-mocked `simdkalman`;
`update()` never touches `kf`; `ai_plugin.ServiceManager` stubbed exactly like the
existing csv_output plugin tests); autouse fixture records `kalman_plugin.print_msg`
(`raising=False`).

- Stateful `RuleBasedStateMachine` (`@initialize` header count 1..6, `window_size` 3..8):
  rules `ValidFrame`, `JunkFrame` (json scalars incl. `None`/`-`/`nan`), `ShortFrame`,
  `LongFrame`, `NanSpike` (finite frame + injected NaN/Inf/`nan`/`-`).
- **I1** no rule ever raises; **I2** `len(frames) == len(headers)`; **I3** every window
  `<= window_size`; **I4** every stored value finite; **I5** all window lengths equal
  while windows are in lockstep and only valid frames arrive (the machine tracks
  factual equality per rule: skipping rules desync by design — the spiked attribute's
  window advances not — and saturation re-equalizes, at which point the strict
  equality assertion re-arms); **growth** — after `>= window_size` consecutive valid
  frames every window is exactly `window_size`.
- Extra `@given` properties: short/long frames leave the `frames` snapshot unchanged
  and the rejection warning contains both lengths; `None` frame warns exact text with
  `["FAIL"]` and stores nothing.

## 6. Legacy-test change log (Step 3d — only the tabulated edits)

| Test (original lines) | Change | Rationale |
|---|---|---|
| `test_parser_util.py` 435–452 | none (verified green) | plain-`Exception` float mock still propagates through the new `except (ValueError, TypeError)` gap → `Exception` leaks as before |
| `test_parser_util.py` 455–475 | float mock return `MagicMock()` → `7.5` | new `math.isfinite` gate would treat a mock as unparseable (mock-only variant comment) |
| `test_parser_util.py 478–503` | none (verified green) | timestamp results appended without isfinite gate |
| `test_parser_util.py` 506–531 | `result == [0.0]` → `len==1 and math.isnan(result[0])` | encodes removed 0.0-masking behavior |
| `test_parser_util.py` 534–550 | same → isnan | same |
| `test_parser_util.py` 553–576 | none (verified green) | `remove_str=True` still drops |
| `test_parser_util.py` 579–598 | float mock return → `3.25` | isfinite gate (mock-only variant) |
| `test_parser_util.py` 601–653 | mock float/convert returns → `pytest.gen.uniform(-10,10)` | isfinite gate (mock-only variant) |
| `test_parser_util.py` 656–709 | same + `expected_result.append(0.0)` → `math.nan` | encodes removed 0.0-masking behavior |
| `test_kalman_plugin.py` 152–165 | none (verified green) | None-warning path unchanged |
| `test_kalman_plugin.py` 168–190, 193–222, 225–253 | added `cut.headers = [...]` | new length guard reads `self.headers` (mock-only variant) |

No assertion was weakened anywhere. Added (additions only) deterministic kill-tests in
both legacy files: exact epochs (`12:34`, `00:00`, `59:59`, `2020-001-12:34:56.789`),
`^`-anchored exact ValueError message asserts, exact `print_msg` text +
`["WARNING"]`/`["FAIL"]` clrs, short/long rejection with unchanged frames, junk frame
stores nothing (3 floatify warnings), `remove_str=True` drops (incl. junk-then-valid
ordering), NaN input stores nothing, nan-spiked frame skips only the spiked
attribute, window saturates to exactly `window_size` with oldest-value eviction.

## 7. Mutant kill table (Step 4)

| Scoped pass | Mutants | Killed | Survivors | Survivors in target functions |
|---|---|---|---|---|
| parser_util pre-patch | 52 | 51 | 1 (#40) | 1 — `floatify_input` continue→break |
| kalman pre-patch | 55 | 42 | 13 | 2 — `update()` #22 (message), #23 (clrs) |
| parser_util post **Pass 1** (exact Step-1 runner) ×2 runs | 71 | **71** | **0** | **0** |
| parser_util post **Pass 2** (4-file runner, `MUTMUT_HYPOTHESIS_FAST=1`) ×2 runs | 71 | **71** | **0** | **0** |
| kalman post **Pass 1** (×2 runs) | 60 | 48–50 | 10–12 | **0** — every `update()` mutant dead in every run |
| kalman post **Pass 2** (×2 runs) | 60 | 50–51 | 8–10 (+1 "suspicious") | **0** |

Post-patch kalman survivors, itemized (all outside `update()`):
`__init__` #5 (error-message string); #15–#18 (kf config literals — conftest-mocked
simdkalman domain); `render_reasoning` #33 (`>`→`>=`), #38 (`"TIME"` literal);
`_generate_residuals` #41–#43 (documented-excluded, conftest MagicMock). Pass-1 also
survives #3/#4 (`__init__` window boundary — the two-file runner never constructs a
real `Plugin`; the Pass-2 property machine does construct real Plugins and kills both),
and #42/#43 fluctuate with the `--conftest-seed` (legacy `_generate_residuals` tests
draw random frame lengths). These fluctuations are confined to excluded/out-of-scope
functions; the in-scope survivor set was empty in **all four** post-patch kalman runs
and both post-patch parser runs (reproducible where the DoD demands it).

Mutant #24 (`update()` guard `!=` → `==`), flagged "suspicious" (slow-kill) in one
Pass-2 run, was proven deterministically killed out-of-band: `mutmut apply 24` →
legacy runner exit code 1 (`test_kill_update_valid_frame_stored_in_lockstep_windows`
fails) → source restored. In the repeat Pass-2 run it was killed outright.

Equivalent-mutant check: none of the zero-survivor target functions required an
"equivalent mutant" exemption; every in-scope mutant is killed by a deterministic
example test (no kill depends on hypothesis sampling — the Pass-1 runner, which
excludes the property files, achieves zero on its own).

## 8. Full verification (Step 5)

1. `python -m pytest test/ -q` → **401 passed** in 160.53s (371 baseline + 30 new:
   10 parser properties + 4 kalman properties/machine + 8 + 8 deterministic
   kill-tests). Zero failures, zero errors, zero skips. Full stdout read.
2. `python -m pytest test/onair/data_handling/test_parser_util_properties.py
   test/plugins/kalman/test_kalman_properties.py -v` (no env var → 1000 examples) →
   **14 passed** in 147.53s. Full stdout read.
3. `mutmut results` for every scoped pass recorded in §7.

## 9. Honest limits

- **mutmut operator set**: mutmut 2.x mutates strings, numbers, comparison operators,
  `continue`/`break`, and indices; it does not remove statements wholesale or mutate
  every logical fault (e.g. swapped independent arguments of the same type). Zero
  survivors means "zero survivors under this operator set", not absolute correctness.
- **Excluded domain**: `_predict` / `_generate_residuals` execute against a
  conftest-injected `MagicMock` simdkalman, so their mutants (#41–#43 here) are
  documented-excluded rather than force-killed; likewise the kf construction kwargs
  (#15–#18) only ever reach a mock.
- **Seed-dependent out-of-scope kills**: legacy `_generate_residuals` tests draw
  `pytest.gen` random frame lengths (seeded from wall clock), so #42/#43 flip between
  killed/survived across runs; #3/#4 are killed only by Pass 2 (property machine
  constructs real `Plugin`s). All fluctuation is outside the three target functions.
- **Hypothesis sampling**: properties are randomized (derandomized only in the
  25-example mutmut fast mode); 1000-example runs are strong evidence, not proof, of
  property validity. No in-scope mutant kill depends on hypothesis — the deterministic
  kill-tests alone reach zero, and Pass 2 verifies the properties are themselves
  killers of every reachable in-scope mutant.
- **I5 semantics**: unconditional window-length equality is false by design after a
  partial-skip frame (that is the finite-only-windows policy); I5 therefore asserts
  equality while windows are factually in lockstep (re-armed after every rule from
  actual state), and the growth invariant pins the re-synchronization point.

## 10. Out-of-scope guarantee

`telemetry_test_suite.py`, `redis_adapter.py`, `sbn_adapter.py`, `tlm_json_parser.py`,
`agent.py`, `vehicle_rep.py`, planners/complex-reasoning interfaces, and root
`conftest.py` were not touched. Changes were confined to
`onair/data_handling/parser_util.py`, `plugins/kalman/kalman_plugin.py`, and the two
legacy test files (tabulated updates + additive kill-tests), plus the two new property
files.

## 11. Post-hardening refinement (external review round)

An external review of the patched `convert_str_to_timestamp` raised three
refinements. Disposition of each:

### 11.1 ACCEPTED — type contract hoisted to the function boundary

The `isinstance(time_str, str)` guard moved out of the `except` handler to the
top of the function, so the input contract ("must be a `str`") is enforced
before any parsing path runs, rather than relying on `strptime` failing first.
Observable behavior for all tested inputs is unchanged (non-strings still get
the same `ValueError`, now raised earlier).

### 11.2 ACCEPTED — `strptime` handler narrowed from `except Exception` to `except ValueError`

Only the expected format-mismatch error falls through to the `MM:SS` fallback;
any other unexpected error (the review's hypothetical `.timestamp()` platform
failure) now propagates loudly instead of being masked as a format retry.
This required one legacy mock-fidelity fix: the `Fake_Datetime.strptime` in
`test_parser_util.py` raised plain `Exception` where the real
`datetime.strptime` raises `ValueError`; the fake now matches the real
dependency's contract (annotated as a mock-only update). The "both raise"
legacy test was left untouched — under the narrowed handler its plain
`Exception` propagates, which is precisely the new fail-loud semantics and
satisfies its `pytest.raises(Exception)` assertion.
`floatify_input`'s broad `except Exception` around its
`convert_str_to_timestamp` call was intentionally KEPT — that call site is the
pipeline's input-normalization boundary and must convert *any* parser failure
into the nan/drop policy (P1 never-raises).

### 11.3 REJECTED (documented instead) — timezone semantics of the `MM:SS` epoch

The review correctly notes `datetime(2000, 1, 1, 1, M, S).timestamp()` on a
naive datetime interprets LOCAL time, so the epoch varies by host timezone.
This is **pre-existing upstream behavior** (the original implementation called
`.timestamp()` identically); changing it (e.g. to UTC) would silently shift
every produced epoch, altering residuals pipeline-wide — outside this
hardening's no-redesign mandate. Documented in the function docstring
instead. Values remain self-consistent within any single machine, so
residuals and relative comparisons are unaffected. Flagged as a candidate
decision for upstream maintainers.

### 11.4 Review-driven test additions

- `"1:2"` pinned as ACCEPTED (== epoch of 1m 2s), matching `int()` semantics
  per the mandate ("both parse as int via try, minute/second in 0..59") — no
  two-digit requirement exists in the spec; decision now documented and
  regression-locked.
- `"12:-1"`, `"-1:30"` (negative components → range check) with anchored
  message asserts; `"12:ab"` (non-integer seconds → int-parse assert);
  `"1:2:3:4"` (4-part → part-count guard).
- `"2024-032-14:30:45.123456"` exact-epoch assert (post-2020 full format,
  leap-year day-of-year).
- Cases already covered from the original round: `"12:60"`, `"60:00"`,
  `"99:99"`, `"aa:bb"`, `"-"`, `""`, `"12:"`, `":34"`, non-strings
  (`None`, `5`, `5.0`, `True`, `[]`, `{}`).

### 11.5 Re-verification after refinement

| Check | Result |
|---|---|
| Full suite `pytest test/ -q` | **401 passed**, zero failures/errors/skips |
| Property files (dedicated run, `hypothesis profile 'default'` = 1000 examples) | **14/14 passed** (P1–P6, machine, guard properties) |
| mutmut Pass 1 (exact Step-1 runner, 2 legacy files) | **71/71 killed, 0 survived, 0 suspicious** |
| mutmut Pass 2 (4-file runner, `MUTMUT_HYPOTHESIS_FAST=1` derandomized cap) | **0 survivors, 0 suspicious** (`mutmut results` empty) |

`plugins/kalman/kalman_plugin.py` was not modified in this round; its prior
zero-survivor result for `Plugin.update()` stands.
