# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in
# the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""Property-based tests for the Kalman Plugin.update telemetry-ingestion path.

Real Plugin instances are used (the root conftest.py injects a MagicMock
simdkalman, and update() never touches kf), so every rule below exercises the
production update() code path: length guard, floatify_input, finite-only
window maintenance.

The RuleBasedStateMachine checks invariants I1-I5 plus the growth invariant;
the standalone @given properties pin the rejection behaviour of short / long /
None frames.  Pre-patch, these tests document the blind spots (TypeError leaks,
desynced frames, non-finite values stored in windows); post-patch they must
all pass.
"""

import copy
import math
import os
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

import onair.src.ai_components.ai_plugin_abstract.ai_plugin as ai_plugin_abstract
from plugins.kalman import kalman_plugin
from plugins.kalman.kalman_plugin import Plugin


# ---------------------------------------------------------------------------
# Hypothesis settings: full 1000-example runs normally; a derandomized
# 25-example cap when MUTMUT_HYPOTHESIS_FAST is set (mutation-testing runner).
# ---------------------------------------------------------------------------
if "MUTMUT_HYPOTHESIS_FAST" in os.environ:
    HYPO_SETTINGS = settings(max_examples=25, deadline=None, derandomize=True)
else:
    HYPO_SETTINGS = settings(max_examples=1000, deadline=None)

json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=8),
    st.sampled_from(["-", "nan", "inf", "-inf", "Infinity", ""]),
)

finite_float = st.floats(allow_nan=False, allow_infinity=False)

MAX_HEADERS = 6
MAX_WINDOW = 8

# Recorded kalman_plugin.print_msg traffic: list of (msg, clrs) tuples.
print_msg_calls = []


def _record_print_msg(msg, clrs=None):
    print_msg_calls.append((msg, clrs))


@pytest.fixture(autouse=True)
def _silence_and_record_print_msg(monkeypatch):
    """Silence AND record kalman_plugin.print_msg (raising=False so the file
    is also runnable against module states lacking the attribute).

    Also stubs AIPlugin's ServiceManager the same way the csv_output plugin
    tests do, so real Plugin instances can be constructed (the singleton
    ServiceManager would otherwise demand a service_dict)."""
    print_msg_calls.clear()
    monkeypatch.setattr(
        ai_plugin_abstract, "ServiceManager", MagicMock(), raising=False
    )
    monkeypatch.setattr(
        kalman_plugin, "print_msg", _record_print_msg, raising=False
    )
    yield
    print_msg_calls.clear()


def _make_plugin(num_headers, window_size):
    headers = [f"ATTR_{i}" for i in range(num_headers)]
    return Plugin("property_probe", headers, window_size)


# ---------------------------------------------------------------------------
# Standalone @given properties (rejection behaviour of the length guard).
# ---------------------------------------------------------------------------
@HYPO_SETTINGS
@given(
    num_headers=st.integers(min_value=1, max_value=MAX_HEADERS),
    window_size=st.integers(min_value=3, max_value=MAX_WINDOW),
    short_by=st.integers(min_value=1, max_value=MAX_HEADERS),
    vals=st.lists(finite_float, min_size=MAX_HEADERS, max_size=MAX_HEADERS),
)
def test_short_frames_are_rejected_and_leave_frames_unchanged(
    num_headers, window_size, short_by, vals
):
    plugin = _make_plugin(num_headers, window_size)
    n = max(num_headers - short_by, 0)
    frame = vals[:n]
    before = copy.deepcopy(plugin.frames)
    print_msg_calls.clear()
    plugin.update(frame)
    assert plugin.frames == before  # rejected: nothing stored
    assert any(
        "does not match" in msg and str(n) in msg and str(num_headers) in msg
        for msg, _ in print_msg_calls
    )


@HYPO_SETTINGS
@given(
    num_headers=st.integers(min_value=1, max_value=MAX_HEADERS),
    window_size=st.integers(min_value=3, max_value=MAX_WINDOW),
    extra=st.integers(min_value=1, max_value=3),
    vals=st.lists(finite_float, min_size=MAX_HEADERS + 3, max_size=MAX_HEADERS + 3),
)
def test_long_frames_are_rejected_and_leave_frames_unchanged(
    num_headers, window_size, extra, vals
):
    plugin = _make_plugin(num_headers, window_size)
    frame = vals[: num_headers + extra]
    before = copy.deepcopy(plugin.frames)
    print_msg_calls.clear()
    plugin.update(frame)
    assert plugin.frames == before  # rejected: nothing stored
    assert any(
        "does not match" in msg
        and str(num_headers + extra) in msg
        and str(num_headers) in msg
        for msg, _ in print_msg_calls
    )


@HYPO_SETTINGS
@given(
    num_headers=st.integers(min_value=1, max_value=MAX_HEADERS),
    window_size=st.integers(min_value=3, max_value=MAX_WINDOW),
)
def test_none_frame_warns_FAIL_and_leaves_frames_unchanged(num_headers, window_size):
    plugin = _make_plugin(num_headers, window_size)
    before = copy.deepcopy(plugin.frames)
    print_msg_calls.clear()
    plugin.update(None)
    assert plugin.frames == before
    assert any(
        msg == "Kalman plugin requires low_level_data but received None."
        and clrs == ["FAIL"]
        for msg, clrs in print_msg_calls
    )


# ---------------------------------------------------------------------------
# Stateful machine: random interleavings of valid / junk / short / long /
# nan-spike frames against one Plugin instance.
# ---------------------------------------------------------------------------
class KalmanUpdateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.plugin = None
        self.headers = []
        self.window_size = 3
        self.valid_streak = 0  # consecutive ValidFrame rules since any other rule
        self.windows_equal = True  # windows advance in lockstep while True

    @initialize(
        num_headers=st.integers(min_value=1, max_value=MAX_HEADERS),
        window_size=st.integers(min_value=3, max_value=MAX_WINDOW),
    )
    def create_plugin(self, num_headers, window_size):
        self.window_size = window_size
        self.valid_streak = 0
        self.windows_equal = True
        self.plugin = _make_plugin(num_headers, window_size)
        self.headers = self.plugin.headers

    @rule(
        vals=st.lists(finite_float, min_size=MAX_HEADERS, max_size=MAX_HEADERS)
    )
    def valid_frame(self, vals):
        frame = vals[: len(self.headers)]
        before_lengths = [len(f) for f in self.plugin.frames]
        print_msg_calls.clear()
        self.plugin.update(frame)  # I1: must not raise
        # every window advances by exactly one (finite values are never skipped)
        assert all(
            len(f) == min(n + 1, self.window_size)
            for n, f in zip(before_lengths, self.plugin.frames)
        )
        self.valid_streak += 1
        # re-arm lockstep tracking once saturation re-equalizes the windows
        self.windows_equal = len({len(f) for f in self.plugin.frames}) == 1

    @rule(
        vals=st.lists(json_scalar, min_size=MAX_HEADERS, max_size=MAX_HEADERS)
    )
    def junk_frame(self, vals):
        frame = vals[: len(self.headers)]
        print_msg_calls.clear()
        self.plugin.update(frame)  # I1: must not raise (junk becomes nan and is skipped)
        self.valid_streak = 0
        # partial skips may desynchronize window lengths; lockstep is only
        # guaranteed to continue while the windows are currently equal
        self.windows_equal = len({len(f) for f in self.plugin.frames}) == 1

    @rule(
        short_by=st.integers(min_value=1, max_value=MAX_HEADERS),
        vals=st.lists(finite_float, min_size=MAX_HEADERS, max_size=MAX_HEADERS),
    )
    def short_frame(self, short_by, vals):
        n = max(len(self.headers) - short_by, 0)
        frame = vals[:n]
        before = copy.deepcopy(self.plugin.frames)
        print_msg_calls.clear()
        self.plugin.update(frame)
        assert self.plugin.frames == before  # rejected: frames untouched
        self.valid_streak = 0

    @rule(
        extra=st.integers(min_value=1, max_value=3),
        vals=st.lists(
            finite_float, min_size=MAX_HEADERS + 3, max_size=MAX_HEADERS + 3
        ),
    )
    def long_frame(self, extra, vals):
        frame = vals[: len(self.headers) + extra]
        before = copy.deepcopy(self.plugin.frames)
        print_msg_calls.clear()
        self.plugin.update(frame)
        assert self.plugin.frames == before  # rejected: frames untouched
        self.valid_streak = 0

    @rule(
        vals=st.lists(finite_float, min_size=MAX_HEADERS, max_size=MAX_HEADERS),
        spike=st.sampled_from(
            [
                float("nan"),
                float("inf"),
                float("-inf"),
                "nan",
                "inf",
                "-inf",
                "Infinity",
                "-",
            ]
        ),
        position=st.integers(min_value=0, max_value=MAX_HEADERS - 1),
    )
    def nan_spike(self, vals, spike, position):
        frame = vals[: len(self.headers)]
        frame[position % len(frame)] = spike
        print_msg_calls.clear()
        self.plugin.update(frame)  # I1: must not raise; spliced attr skipped
        self.valid_streak = 0
        # the spiked attribute's window did not advance: lockstep may be lost
        self.windows_equal = len({len(f) for f in self.plugin.frames}) == 1

    @invariant()
    def I2_frame_count_matches_headers(self):
        assert len(self.plugin.frames) == len(self.headers)

    @invariant()
    def I3_windows_never_exceed_window_size(self):
        for frame in self.plugin.frames:
            assert len(frame) <= self.window_size

    @invariant()
    def I4_stored_values_are_all_finite(self):
        for frame in self.plugin.frames:
            for value in frame:
                assert math.isfinite(value)

    @invariant()
    def I5_windows_lockstep_while_only_valid_frames(self):
        # While only valid frames have arrived since the last junk / rejection
        # rule (valid_streak > 0) AND the windows are factually in lockstep
        # (windows_equal, recomputed after every rule -- skipping rules desync
        # by design, saturation re-equalizes), every window must keep the same
        # length: valid frames advance all windows together, never skipping.
        if self.valid_streak > 0 and self.windows_equal:
            lengths = {len(f) for f in self.plugin.frames}
            assert len(lengths) == 1

    @invariant()
    def growth_invariant_windows_saturate_after_full_streak(self):
        if self.valid_streak >= self.window_size:
            for frame in self.plugin.frames:
                assert len(frame) == self.window_size


class TestKalmanMachine(KalmanUpdateMachine.TestCase):
    settings = HYPO_SETTINGS
