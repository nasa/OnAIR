# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

from .tlm_json_parser import parseTlmConfJson, str2lst
from onair.src.util.print_io import print_msg
import datetime
import math


def extract_meta_data_handle_ss_breakdown(meta_data_file, ss_breakdown):
    parsed_meta_data = extract_meta_data(meta_data_file)
    if ss_breakdown == False:
        num_elements = len(parsed_meta_data["subsystem_assignments"])
        parsed_meta_data["subsystem_assignments"] = [
            ["MISSION"] for elem in range(num_elements)
        ]
    return parsed_meta_data


## Method to extract configuration data and return 3 dictionaries
def extract_meta_data(meta_data_file):
    assert meta_data_file != ""

    configs = parseTlmConfJson(meta_data_file)

    configs_len = len(configs["subsystem_assignments"])

    for i in range(configs_len):
        if configs["subsystem_assignments"][i] != "NONE":
            configs["subsystem_assignments"][i] = [configs["subsystem_assignments"][i]]
        else:
            configs["subsystem_assignments"][i] = []

        test_assign = configs["test_assignments"][i]

        for j in range(len(test_assign)):
            if len(test_assign[j]) > 1:
                test = [test_assign[j][0]]
                limits = str2lst(test_assign[j][1])
                test_assign[j] = test + limits

    return configs


def floatify_input(_input, remove_str=False):
    """Convert each item of ``_input`` to a finite float, preserving length.

    Parsing policy (hardened):

    - Values that ``float()`` parses to a FINITE float are appended as-is
      (finite floats pass through unchanged).
    - Values that ``convert_str_to_timestamp`` parses (a ``%Y-%j-%H:%M:%S.%f``
      telemetry timestamp, or a strict ``MM:SS`` pair with minute and second
      each in 0..59) are appended as their epoch float.
    - Everything else -- ``"-"``, ``None``, junk text, and non-finite
      spellings like ``"nan"``/``"inf"``/``"Infinity"`` (whose ``float()``
      result is not finite) -- is either dropped entirely when ``remove_str``
      is True, or substituted with ``math.nan`` plus a WARNING via
      ``print_msg`` when ``remove_str`` is False.  A ``0.0`` fallback is
      never used: masking junk as a real measurement value hides telemetry
      faults.
    """
    floatified = []
    for i in _input:
        try:
            x = float(i)
            if math.isfinite(x):
                floatified.append(x)
                continue
        except (ValueError, TypeError):
            pass
        # non-finite float() results (nan/inf strings and floats) fall through
        # to the timestamp attempt, which rejects them -> junk policy below
        try:
            x = convert_str_to_timestamp(i)
            floatified.append(x)
            continue
        except Exception:
            pass
        if remove_str:
            continue
        print_msg(
            f"floatify_input: could not parse {i!r} to a finite float; substituting math.nan",
            ["WARNING"],
        )
        floatified.append(math.nan)
    return floatified


def convert_str_to_timestamp(time_str):
    """Convert a telemetry time string to an epoch float.

    Contract: ``time_str`` must be a ``str``; any other type raises
    ``ValueError`` immediately at the function boundary.

    Accepted formats:

    - ``%Y-%j-%H:%M:%S.%f`` (e.g. ``"2020-001-12:34:56.789"``), parsed with
      ``datetime.strptime``; or
    - a strict ``MM:SS`` pair (e.g. ``"12:34"``) with exactly two colon
      separated parts, both integer-parseable via ``int()``, minute and
      second each in 0..59 (single-digit components such as ``"1:2"`` are
      accepted, matching ``int()`` semantics).  The date is fixed to 1am
      on Jan 1st, 2000.

    Anything else -- non-strings (``None``, numbers, objects), ``"12:34:56"``,
    ``"99:99"``, ``"-"``, empty strings -- raises ``ValueError`` instead of
    silently truncating extra parts or leaking ``IndexError`` /
    ``AttributeError``.

    Only ``ValueError`` from ``strptime`` falls through to the ``MM:SS``
    fallback; any other unexpected error propagates loudly instead of
    being masked as a format retry.

    Note: the ``MM:SS`` branch builds a naive ``datetime`` and calls
    ``.timestamp()``, which interprets it in the host's LOCAL timezone.
    This is the pre-existing upstream semantics (identical to the original
    implementation) and is intentionally preserved here: values are
    consistent within any single machine, so residuals and relative
    comparisons are unaffected.
    """
    if not isinstance(time_str, str):
        raise ValueError(
            f"convert_str_to_timestamp: expected a time string, got {type(time_str).__name__}: {time_str!r}"
        )
    try:
        t = datetime.datetime.strptime(time_str, "%Y-%j-%H:%M:%S.%f")
        return t.timestamp()
    except ValueError:
        pass
    min_sec = time_str.split(":")
    if len(min_sec) != 2:
        raise ValueError(
            f"convert_str_to_timestamp: expected MM:SS, got {time_str!r}"
        )
    try:
        minute = int(min_sec[0])
        second = int(min_sec[1])
    except ValueError:
        raise ValueError(
            f"convert_str_to_timestamp: expected MM:SS with integer minute and second, got {time_str!r}"
        )
    if not (0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(
            f"convert_str_to_timestamp: minute and second must be in 0..59, got {time_str!r}"
        )
    # Use 1 am on Jan 1st, 2000 as the date if only minutes and seconds are specified
    t = datetime.datetime(2000, 1, 1, 1, minute, second, 0)
    return t.timestamp()
