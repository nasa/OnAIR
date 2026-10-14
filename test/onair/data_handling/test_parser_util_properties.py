# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in
# the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""Property-based tests for parser_util.floatify_input / convert_str_to_timestamp.

Properties P1-P6 lock the hardened parsing policy:
  finite-parseable -> float, timestamps -> epoch float,
  "-" / None / junk / non-finite -> math.nan + WARNING (or dropped when remove_str=True).

These tests are ALSO runnable pre-patch (the print_msg recorder is installed with
raising=False), in which case the falsifying examples document the pre-patch
blind spots (0.0-masking, TypeError leaks, silent ":56" truncation, inf passthrough).
"""

import datetime
import math
import os

import pytest
from hypothesis import given, settings, strategies as st

import onair.data_handling.parser_util as parser_util


# ---------------------------------------------------------------------------
# Hypothesis settings: full 1000-example runs normally; a derandomized 25-example
# cap when MUTMUT_HYPOTHESIS_FAST is set (used by the mutation-testing runner).
# ---------------------------------------------------------------------------
if "MUTMUT_HYPOTHESIS_FAST" in os.environ:
    HYPO_SETTINGS = settings(max_examples=25, deadline=None, derandomize=True)
else:
    HYPO_SETTINGS = settings(max_examples=1000, deadline=None)

# Any scalar a JSON telemetry field could carry.
json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=8),
)

# Sentinel strings the telemetry pipeline is known to emit.
SENTINEL_STRINGS = [
    "-",
    "nan",
    "inf",
    "-inf",
    "Infinity",
    "",
    "12",
    "12:34",
    "12:34:56",
    "2020-001-12:34:56.789",
]

finite_float = st.floats(allow_nan=False, allow_infinity=False)

# Recorded print_msg traffic: list of (msg, clrs) tuples.
print_msg_calls = []


def _record_print_msg(msg, clrs=None):
    print_msg_calls.append((msg, clrs))


@pytest.fixture(autouse=True)
def _silence_and_record_print_msg(monkeypatch):
    """Silence AND record parser_util.print_msg.

    raising=False keeps this file runnable pre-patch (the module gains a
    print_msg attribute only after the hardening patch) and lets P4 assert
    the WARNING post-patch.
    """
    print_msg_calls.clear()
    monkeypatch.setattr(parser_util, "print_msg", _record_print_msg, raising=False)
    yield
    print_msg_calls.clear()


# ---------------------------------------------------------------------------
# P1: floatify_input never raises, whatever the list of JSON scalars contains.
# ---------------------------------------------------------------------------
@HYPO_SETTINGS
@given(st.lists(json_scalar, max_size=20))
def test_P1_floatify_input_never_raises_on_json_scalars(items):
    parser_util.floatify_input(items)  # must not raise
    parser_util.floatify_input(items, remove_str=True)  # must not raise either


# ---------------------------------------------------------------------------
# P2: with remove_str=False the output length is preserved for ALL inputs.
# ---------------------------------------------------------------------------
@HYPO_SETTINGS
@given(st.lists(json_scalar, max_size=20))
def test_P2_len_preserved_when_remove_str_is_False(items):
    out = parser_util.floatify_input(items, remove_str=False)
    assert len(out) == len(items)


# ---------------------------------------------------------------------------
# P3: identity on finite floats (holds pre-patch as well).
# ---------------------------------------------------------------------------
@HYPO_SETTINGS
@given(st.lists(finite_float, max_size=20))
def test_P3_identity_on_finite_floats(items):
    out = parser_util.floatify_input(items)
    assert out == items


# ---------------------------------------------------------------------------
# P4: "-" must never degrade to 0.0. Post-patch: math.nan + WARNING, and
# remove_str=True drops it entirely.
# ---------------------------------------------------------------------------
@HYPO_SETTINGS
@given(st.lists(st.sampled_from(["-"]), min_size=1, max_size=5))
def test_P4_dash_is_never_zero_but_nan_with_warning(items):
    print_msg_calls.clear()
    out = parser_util.floatify_input(items, remove_str=False)
    assert len(out) == len(items)
    for value in out:
        assert value != 0.0  # the 0.0-masking failure mode is forbidden
        assert math.isnan(value)
    dropped = parser_util.floatify_input(items, remove_str=True)
    assert dropped == []
    assert any(clrs == ["WARNING"] for _, clrs in print_msg_calls)


# ---------------------------------------------------------------------------
# P5: non-finite spellings (strings and floats) become math.nan, never 0.0,
# never a crash.
# ---------------------------------------------------------------------------
@HYPO_SETTINGS
@given(
    st.sampled_from(
        ["nan", "inf", "-inf", "NaN", "Infinity", "-Infinity", "INF", "-INF"]
    )
)
def test_P5_nonfinite_strings_become_nan(s):
    out = parser_util.floatify_input([s])
    assert len(out) == 1
    assert math.isnan(out[0])
    assert out[0] != 0.0


@HYPO_SETTINGS
@given(st.sampled_from([float("nan"), float("inf"), float("-inf")]))
def test_P5_nonfinite_floats_become_nan(v):
    out = parser_util.floatify_input([v])
    assert len(out) == 1
    assert math.isnan(out[0])


# ---------------------------------------------------------------------------
# P6: valid MM:SS and %Y-%j-%H:%M:%S.%f timestamps convert to the EXACT epoch
# float computed independently via datetime; malformed inputs raise ValueError
# or become nan -- never IndexError / AttributeError.
# ---------------------------------------------------------------------------
@HYPO_SETTINGS
@given(st.integers(min_value=0, max_value=59), st.integers(min_value=0, max_value=59))
def test_P6_valid_mm_ss_converts_to_exact_epoch(minute, second):
    out = parser_util.floatify_input([f"{minute:02d}:{second:02d}"])
    expected = datetime.datetime(2000, 1, 1, 1, minute, second, 0).timestamp()
    assert out == [expected]


@HYPO_SETTINGS
@given(
    st.integers(min_value=1900, max_value=2100),
    st.integers(min_value=1, max_value=365),
    st.integers(min_value=0, max_value=23),
    st.integers(min_value=0, max_value=59),
    st.integers(min_value=0, max_value=59),
    st.integers(min_value=0, max_value=999999),
)
def test_P6_valid_doy_timestamp_converts_to_exact_epoch(
    year, doy, hour, minute, second, micro
):
    time_str = (
        f"{year:04d}-{doy:03d}-{hour:02d}:"
        f"{minute:02d}:{second:02d}.{micro:06d}"
    )
    out = parser_util.floatify_input([time_str])
    expected = (
        datetime.datetime(year, 1, 1, hour, minute, second, micro)
        + datetime.timedelta(days=doy - 1)
    ).timestamp()
    assert out == [expected]


@HYPO_SETTINGS
@given(
    st.sampled_from(
        [
            "12:34:56",
            "99:99",
            "-",
            "",
            "12:",
            ":34",
            "1:2:3",
            "aa:bb",
            "12:34:56.789",
        ]
    )
)
def test_P6_malformed_timestamps_raise_ValueError_or_become_nan(s):
    try:
        out = parser_util.floatify_input([s], remove_str=False)
    except ValueError:
        return  # acceptable outcome for malformed input
    assert len(out) == 1
    assert math.isnan(out[0])  # never a silently-truncated finite value


@HYPO_SETTINGS
@given(st.sampled_from([None, 5, 5.0, True]))
def test_P6_non_string_to_convert_str_to_timestamp_raises_ValueError_only(v):
    try:
        out = parser_util.floatify_input([v], remove_str=False)
    except ValueError:
        return
    assert len(out) == 1
    assert math.isnan(out[0]) or math.isfinite(out[0])
