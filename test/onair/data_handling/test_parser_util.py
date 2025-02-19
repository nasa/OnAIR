# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""Test Parser Util Functionality"""
import pytest
from unittest.mock import MagicMock

import onair.data_handling.parser_util as parser_util


# extract_meta_data_handle_ss_breakdown
def test_parser_util_extract_meta_data_handle_ss_breakdown_returns_call_to_extract_meta_data_file_given_metadata_file_and_csv_set_to_True_when_given_ss_breakdown_does_not_resolve_to_False(
    mocker,
):
    # Arrange
    arg_configFile = MagicMock()
    arg_ss_breakdown = True if pytest.gen.randint(0, 1) else MagicMock()

    expected_result = MagicMock()

    mocker.patch(
        parser_util.__name__ + ".extract_meta_data", return_value=expected_result
    )
    mocker.patch(parser_util.__name__ + ".len")

    # Act
    result = parser_util.extract_meta_data_handle_ss_breakdown(
        arg_configFile, arg_ss_breakdown
    )

    # Assert
    assert parser_util.extract_meta_data.call_count == 1
    assert parser_util.extract_meta_data.call_args_list[0].args == (arg_configFile,)
    assert parser_util.len.call_count == 0
    assert result == expected_result


def test_parser_util_extract_meta_data_handle_ss_breakdown_returns_call_to_extract_meta_data_file_given_metadata_file_and_csv_set_to_True_with_dict_def_of_subsystem_assigments_def_of_call_to_process_filepath_given_configFile_and_kwarg_csv_set_to_True_set_to_empty_list_when_len_of_call_value_dict_def_of_subsystem_assigments_def_of_call_to_process_filepath_given_configFile_and_kwarg_csv_set_to_True_is_0_when_given_ss_breakdown_evaluates_to_False(
    mocker,
):
    # Arrange
    arg_configFile = MagicMock()
    arg_ss_breakdown = False if pytest.gen.randint(0, 1) else 0

    forced_return_extract_meta_data = {}
    forced_return_len = 0
    fake_empty_processed_filepath = MagicMock()
    forced_return_extract_meta_data["subsystem_assignments"] = (
        fake_empty_processed_filepath
    )

    expected_result = []

    mocker.patch(
        parser_util.__name__ + ".extract_meta_data",
        return_value=forced_return_extract_meta_data,
    )
    mocker.patch(parser_util.__name__ + ".len", return_value=forced_return_len)

    # Act
    result = parser_util.extract_meta_data_handle_ss_breakdown(
        arg_configFile, arg_ss_breakdown
    )

    # Assert
    assert parser_util.extract_meta_data.call_count == 1
    assert parser_util.extract_meta_data.call_args_list[0].args == (arg_configFile,)
    assert parser_util.len.call_count == 1
    assert parser_util.len.call_args_list[0].args == (fake_empty_processed_filepath,)
    assert result["subsystem_assignments"] == expected_result


def test_parser_util_extract_meta_data_handle_ss_breakdown_returns_call_to_extract_meta_data_given_metadata_file_and_csv_set_to_True_with_dict_def_subsystem_assignments_def_of_call_to_process_filepath_given_configFile_and_kwarg_csv_set_to_True_set_to_single_item_list_str_MISSION_for_each_item_when_given_ss_breakdown_evaluates_to_False(
    mocker,
):
    # Arrange
    arg_configFile = MagicMock()
    arg_ss_breakdown = False if pytest.gen.randint(0, 1) else 0

    forced_return_extract_meta_data = {}
    forced_return_process_filepath = MagicMock()
    fake_processed_filepath = []
    num_fake_processed_filepaths = pytest.gen.randint(
        1, 10
    )  # arbitrary, from 1 to 10 (0 has own test)
    for i in range(num_fake_processed_filepaths):
        fake_processed_filepath.append(i)
    forced_return_extract_meta_data["subsystem_assignments"] = fake_processed_filepath
    forced_return_len = num_fake_processed_filepaths

    expected_result = []
    for i in range(num_fake_processed_filepaths):
        expected_result.append(["MISSION"])

    mocker.patch(
        parser_util.__name__ + ".extract_meta_data",
        return_value=forced_return_extract_meta_data,
    )
    mocker.patch(parser_util.__name__ + ".len", return_value=forced_return_len)

    # Act
    result = parser_util.extract_meta_data_handle_ss_breakdown(
        arg_configFile, arg_ss_breakdown
    )

    # Assert
    assert parser_util.extract_meta_data.call_count == 1
    assert parser_util.extract_meta_data.call_args_list[0].args == (arg_configFile,)
    assert parser_util.len.call_count == 1
    assert parser_util.len.call_args_list[0].args == (fake_processed_filepath,)
    assert result["subsystem_assignments"] == expected_result


# extract_meta_data tests
def test_parser_util_extract_meta_data_raises_error_when_given_blank_meta_data_file():
    # Arrange
    arg_meta_data_file = ""

    # Act
    with pytest.raises(AssertionError) as e_info:
        result = parser_util.extract_meta_data(arg_meta_data_file)

    # Assert
    assert e_info.match("")


def test_parser_util_extract_meta_data_returns_expected_dicts_dict_when_configs_len_equal_to_zero(
    mocker,
):
    # Arrange
    arg_meta_data_file = MagicMock()

    fake_subsystem_assignments = MagicMock()
    fake_tests = MagicMock()
    fake_descs = MagicMock()

    forced_return_parse_tlm = {
        "subsystem_assignments": fake_subsystem_assignments,
        "test_assignments": fake_tests,
        "description_assignments": fake_descs,
    }
    forced_return_len = 0

    mocker.patch(
        parser_util.__name__ + ".parseTlmConfJson", return_value=forced_return_parse_tlm
    )
    mocker.patch(parser_util.__name__ + ".len", return_value=forced_return_len)
    mocker.patch(parser_util.__name__ + ".str2lst")

    # Act
    result = parser_util.extract_meta_data(arg_meta_data_file)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_meta_data_file,)
    assert parser_util.len.call_count == 1
    assert parser_util.len.call_args_list[0].args == (fake_subsystem_assignments,)
    assert parser_util.str2lst.call_count == 0
    assert result == forced_return_parse_tlm


def test_parser_util_extract_meta_data_returns_expected_dicts_dict_when_configs_len_equal_to_one(
    mocker,
):
    # Arrange
    arg_meta_data_file = MagicMock()

    fake_subsystem_assignments = [MagicMock()]
    fake_test_assign = MagicMock()
    fake_tests = [[[fake_test_assign]]]
    fake_descs = [MagicMock()]

    forced_return_parse_tlm = {
        "subsystem_assignments": fake_subsystem_assignments,
        "test_assignments": fake_tests,
        "description_assignments": fake_descs,
    }

    mocker.patch(
        parser_util.__name__ + ".parseTlmConfJson", return_value=forced_return_parse_tlm
    )
    mocker.patch(parser_util.__name__ + ".str2lst")

    expected_ss_assigns = [
        [fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments
    ]
    expected_result = {}
    expected_result["subsystem_assignments"] = expected_ss_assigns
    expected_result["test_assignments"] = [[[fake_test_assign]]]
    expected_result["description_assignments"] = fake_descs.copy()

    # Act
    result = parser_util.extract_meta_data(arg_meta_data_file)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_meta_data_file,)
    assert parser_util.str2lst.call_count == 0
    assert result == expected_result


def test_parser_util_extract_meta_data_returns_expected_dicts_dict_when_len_configs_greater_than_one(
    mocker,
):
    # Arrange
    arg_meta_data_file = MagicMock()

    len_configs = pytest.gen.randint(
        2, 10
    )  # arbitrary, from 2 to 10 (0 and 1 have own tests)
    fake_subsystem_assignments = [MagicMock()] * len_configs
    fake_test_assign = MagicMock()
    fake_tests = [[[fake_test_assign]]] * len_configs
    fake_descs = [MagicMock()] * len_configs

    forced_return_parse_tlm = {
        "subsystem_assignments": fake_subsystem_assignments,
        "test_assignments": fake_tests,
        "description_assignments": fake_descs,
    }

    mocker.patch(
        parser_util.__name__ + ".parseTlmConfJson", return_value=forced_return_parse_tlm
    )
    mocker.patch(parser_util.__name__ + ".str2lst")

    expected_ss_assigns = [
        [fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments
    ]
    expected_result = {}
    expected_result["subsystem_assignments"] = expected_ss_assigns
    expected_result["test_assignments"] = [[[fake_test_assign]]] * len_configs
    expected_result["description_assignments"] = fake_descs.copy()

    # Act
    result = parser_util.extract_meta_data(arg_meta_data_file)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_meta_data_file,)
    assert parser_util.str2lst.call_count == 0
    assert result == expected_result


def test_parser_util_extract_meta_data_returns_expected_dicts_dict_when_len_configs_greater_than_one_and_NOOPs_contained_in_test_assigns(
    mocker,
):
    # Arrange
    arg_meta_data_file = MagicMock()

    len_configs = pytest.gen.randint(
        2, 10
    )  # arbitrary, from 2 to 10 (0 and 1 have own tests)
    num_noops = pytest.gen.randint(2, 10)
    len_configs = len_configs + num_noops
    fake_subsystem_assignments = [MagicMock()] * len_configs
    fake_test_assign = MagicMock()
    noop_test_assign = "NOOP"
    fake_tests = [[[fake_test_assign]]] * (len_configs - num_noops) + [
        [[noop_test_assign]]
    ] * num_noops
    fake_descs = [MagicMock()] * len_configs

    forced_return_parse_tlm = {
        "subsystem_assignments": fake_subsystem_assignments,
        "test_assignments": fake_tests,
        "description_assignments": fake_descs,
    }

    mocker.patch(
        parser_util.__name__ + ".parseTlmConfJson", return_value=forced_return_parse_tlm
    )
    mocker.patch(parser_util.__name__ + ".str2lst")

    expected_ss_assigns = [
        [fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments
    ]
    expected_result = {}
    expected_result["subsystem_assignments"] = expected_ss_assigns
    expected_result["test_assignments"] = [[[fake_test_assign]]] * (
        len_configs - num_noops
    ) + [[[noop_test_assign]]] * num_noops
    expected_result["description_assignments"] = fake_descs.copy()

    # Act
    result = parser_util.extract_meta_data(arg_meta_data_file)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_meta_data_file,)
    assert parser_util.str2lst.call_count == 0
    assert result == expected_result


def test_parser_util_extract_meta_data_returns_expected_dicts_dict_when_len_configs_greater_than_one_and_len_test_assigns_greater_than_one(
    mocker,
):
    # Arrange
    arg_meta_data_file = MagicMock()

    len_configs = pytest.gen.randint(
        2, 10
    )  # arbitrary, from 2 to 10 (0 and 1 have own tests)
    fake_subsystem_assignments = [MagicMock()] * len_configs
    fake_tests = []
    fake_descs = [MagicMock()] * len_configs
    for i in range(len_configs):
        len_test_assigns = pytest.gen.randint(1, 10)  # arbitrary, from 1 to 10
        fake_test_assigns = [[MagicMock(), MagicMock()]] * len_test_assigns
        fake_tests.append(fake_test_assigns)

    forced_return_parse_tlm = {
        "subsystem_assignments": fake_subsystem_assignments,
        "test_assignments": fake_tests,
        "description_assignments": fake_descs,
    }
    forced_return_str2lst = [MagicMock()]

    mocker.patch(
        parser_util.__name__ + ".parseTlmConfJson", return_value=forced_return_parse_tlm
    )
    mocker.patch(parser_util.__name__ + ".str2lst", return_value=forced_return_str2lst)

    expected_ss_assigns = [
        [fake_ss_assign] for fake_ss_assign in fake_subsystem_assignments
    ]
    expected_result = {}
    expected_result["subsystem_assignments"] = expected_ss_assigns
    expected_result["test_assignments"] = []
    expected_result["description_assignments"] = fake_descs.copy()

    expected_str2lst_args = []
    expected_str2lst_call_count = 0
    for test_assigns in fake_tests:
        expected_test_assign = []
        for j in range(len(test_assigns)):
            expected_test_assign.append([test_assigns[j][0]] + forced_return_str2lst)
            expected_str2lst_args.append(test_assigns[j][1])
            expected_str2lst_call_count += 1
        expected_result["test_assignments"].append(expected_test_assign)

    # Act
    result = parser_util.extract_meta_data(arg_meta_data_file)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_meta_data_file,)
    assert parser_util.str2lst.call_count == expected_str2lst_call_count
    for i in range(expected_str2lst_call_count):
        assert parser_util.str2lst.call_args_list[i].args == (expected_str2lst_args[i],)
    assert result == expected_result


def test_parser_util_extract_meta_data_returns_expected_dicts_dict_when_configFiles_len_configs_greater_than_one_and_subsystem_NONE_exists(
    mocker,
):
    # Arrange
    arg_meta_data_file = MagicMock()

    len_configs = pytest.gen.randint(
        2, 10
    )  # arbitrary, from 2 to 10 (0 and 1 have own tests)
    fake_subsystem_assignments = []
    fake_tests = []
    fake_tests_copy = []
    fake_descs = []

    expected_subsystem_assignments = []
    for i in range(len_configs):
        if pytest.gen.randint(0, 1) == 1:
            fake_ss_assign = MagicMock()
            fake_subsystem_assignments.append(fake_ss_assign)
            expected_subsystem_assignments.append([fake_ss_assign])
        else:
            fake_subsystem_assignments.append("NONE")
            expected_subsystem_assignments.append([])
        fake_test_assign = MagicMock()
        fake_tests.append([[fake_test_assign]])
        fake_tests_copy.append([[fake_test_assign]])
        fake_descs.append(MagicMock())

    rand_index = pytest.gen.randint(
        0, len_configs - 1
    )  # arbitrary index in fake_subsystem_assignments
    fake_subsystem_assignments[rand_index] = "NONE"
    expected_subsystem_assignments[rand_index] = []

    forced_return_parse_tlm = {
        "subsystem_assignments": fake_subsystem_assignments,
        "test_assignments": fake_tests,
        "description_assignments": fake_descs,
    }

    mocker.patch(
        parser_util.__name__ + ".parseTlmConfJson", return_value=forced_return_parse_tlm
    )
    mocker.patch(parser_util.__name__ + ".str2lst")

    expected_result = {}
    expected_result["subsystem_assignments"] = expected_subsystem_assignments
    expected_result["test_assignments"] = fake_tests_copy
    expected_result["description_assignments"] = fake_descs.copy()

    # Act
    result = parser_util.extract_meta_data(arg_meta_data_file)

    # Assert
    assert parser_util.parseTlmConfJson.call_count == 1
    assert parser_util.parseTlmConfJson.call_args_list[0].args == (arg_meta_data_file,)
    assert result == expected_result


# floatify_input tests
def test_parser_util_flotify_input_returns_empty_list_when_given__input_is_vacant(
    mocker,
):
    # Arrange
    arg__input = []  # empty list, no iterations
    arg_remove_str = False

    # Act
    result = parser_util.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert result == []


def test_parser_util_flotify_input_raises_exception_when_float_returns_non_ValueError_exception(
    mocker,
):
    # Arrange
    arg__input = [str(MagicMock())]  # list of single str list, 1 iteration
    arg_remove_str = False

    exception_message = str(MagicMock())
    fake_exception = Exception(exception_message)

    mocker.patch("builtins.float", side_effect=[fake_exception])

    # Act
    with pytest.raises(Exception) as e_info:
        parser_util.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert e_info.match(exception_message)


def test_parser_util_flotify_input_returns_list_of_size_one_that_contains_the_call_to_float_when_no_Exception_is_thrown_and_given__input_is_str(
    mocker,
):
    # Arrange
    arg__input = []
    arg_remove_str = False

    fake_item = str(MagicMock())
    arg__input.append(fake_item)  # list of single str, one iteration

    expected_result = MagicMock()

    mocker.patch("builtins.float", return_value=expected_result)

    # Act
    result = parser_util.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert float.call_count == 1
    assert float.call_args_list[0].args == (arg__input[0],)
    assert result == [expected_result]


def test_parser_util_flotify_input_returns_list_of_size_one_that_contains_the_second_call_to_float_after_replace_call_when_single_Exception_is_thrown(
    mocker,
):
    # Arrange
    arg__input = []
    arg_remove_str = False

    fake_item = MagicMock()
    arg__input.append(fake_item)  # list of one item, one iteration

    expected_result = MagicMock()

    mocker.patch(parser_util.__name__ + ".float", side_effect=[ValueError])
    mocker.patch(
        parser_util.__name__ + ".convert_str_to_timestamp", return_value=expected_result
    )

    # Act
    result = parser_util.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert parser_util.float.call_count == 1
    assert parser_util.float.call_args_list[0].args == (arg__input[0],)
    assert parser_util.convert_str_to_timestamp.call_count == 1
    assert parser_util.convert_str_to_timestamp.call_args_list[0].args == (fake_item,)
    assert result == [expected_result]


def test_parser_util_flotify_input_returns_list_of_size_one_that_contains_0_dot_0_when_two_Exceptions_are_thrown_and_remove_str_is_False(
    mocker,
):
    # Arrange
    arg__input = []
    arg_remove_str = False

    fake_item = MagicMock()
    arg__input.append(fake_item)  # list of one item, one iteration

    mocker.patch(parser_util.__name__ + ".float", side_effect=[ValueError])
    mocker.patch(
        parser_util.__name__ + ".convert_str_to_timestamp", side_effect=[Exception]
    )

    # Act
    result = parser_util.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert parser_util.float.call_count == 1
    assert parser_util.float.call_args_list[0].args == (arg__input[0],)
    assert parser_util.convert_str_to_timestamp.call_count == 1
    assert parser_util.convert_str_to_timestamp.call_args_list[0].args == (
        arg__input[0],
    )
    assert result == [0.0]


def test_parser_util_flotify_input_default_arg_remove_str_is_False(mocker):
    # Arrange
    arg__input = []

    fake_item = MagicMock()
    arg__input.append(fake_item)  # list of one item, one iteration

    mocker.patch(parser_util.__name__ + ".float", side_effect=[ValueError])
    mocker.patch(
        parser_util.__name__ + ".convert_str_to_timestamp", side_effect=[Exception]
    )

    # Act
    result = parser_util.floatify_input(arg__input)

    # Assert
    assert result == [0.0]  # shows flow was correct for remove_str being False


def test_parser_util_flotify_input_returns_empty_list_when_two_Exceptions_are_thrown_and_remove_str_is_True(
    mocker,
):
    # Arrange
    arg__input = []
    arg_remove_str = True

    fake_item = MagicMock()
    arg__input.append(fake_item)  # list of one item, one iteration

    mocker.patch(parser_util.__name__ + ".float", side_effect=[ValueError])
    mocker.patch(
        parser_util.__name__ + ".convert_str_to_timestamp", side_effect=[Exception]
    )

    # Act
    result = parser_util.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert parser_util.float.call_count == 1
    assert parser_util.float.call_args_list[0].args == (arg__input[0],)
    assert parser_util.convert_str_to_timestamp.call_count == 1
    assert parser_util.convert_str_to_timestamp.call_args_list[0].args == (fake_item,)
    assert result == []


def test_parser_util_flotify_input_returns_call_to_float_that_was_given___input_item_when_type_of_item_is_not_str_and_there_is_single_item(
    mocker,
):
    # Arrange
    arg__input = []

    fake_item = MagicMock()
    arg__input.append(fake_item)  # list of one item, one iteration

    expected_result = MagicMock()

    mocker.patch(parser_util.__name__ + ".float", return_value=expected_result)

    # Act
    result = parser_util.floatify_input(arg__input)

    # Assert
    assert result == [
        expected_result
    ]  # shows flow was correct for remove_str being False


def test_parser_util_flotify_input_returns_expected_values_for_given__input_that_is_multi_typed_when_remove_str_is_True(
    mocker,
):
    # Arrange
    arg__input = []
    arg_remove_str = True

    side_effects_for_float = []
    side_effects_for_convert_str = []
    expected_result = []

    num_fakes = pytest.gen.randint(0, 10)  # arbitrary, from 0 to 10

    for i in range(num_fakes):
        rand_type_of_item = pytest.gen.sample(
            ["str", "str_need_replace", "str_fail_replace", "other"], 1
        )[0]

        if rand_type_of_item == "str":
            arg__input.append(MagicMock())
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == "str_need_replace":
            fake_input = MagicMock()
            arg__input.append(fake_input)
            resultant_float = MagicMock()
            side_effects_for_float.append(ValueError)
            side_effects_for_convert_str.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == "str_fail_replace":
            fake_input = MagicMock()
            arg__input.append(fake_input)
            resultant_float = MagicMock()
            side_effects_for_float.append(ValueError)
            side_effects_for_convert_str.append(Exception)
        else:
            arg__input.append(MagicMock())
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)

    mocker.patch(parser_util.__name__ + ".float", side_effect=side_effects_for_float)
    mocker.patch(
        parser_util.__name__ + ".convert_str_to_timestamp",
        side_effect=side_effects_for_convert_str,
    )

    # Act
    result = parser_util.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert result == expected_result


def test_parser_util_flotify_input_returns_expected_values_for_given__input_that_is_multi_typed_when_remove_str_is_False(
    mocker,
):
    # Arrange
    arg__input = []
    arg_remove_str = False

    side_effects_for_float = []
    side_effects_for_convert_str = []
    expected_result = []

    num_fakes = pytest.gen.randint(0, 10)  # arbitrary, from 0 to 10

    for i in range(num_fakes):
        rand_type_of_item = pytest.gen.sample(
            ["str", "str_need_replace", "str_fail_replace", "other"], 1
        )[0]

        if rand_type_of_item == "str":
            arg__input.append(MagicMock())
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == "str_need_replace":
            fake_input = MagicMock()
            arg__input.append(fake_input)
            resultant_float = MagicMock()
            side_effects_for_float.append(ValueError)
            side_effects_for_convert_str.append(resultant_float)
            expected_result.append(resultant_float)
        elif rand_type_of_item == "str_fail_replace":
            fake_input = MagicMock()
            arg__input.append(fake_input)
            resultant_float = MagicMock()
            side_effects_for_float.append(ValueError)
            side_effects_for_convert_str.append(Exception)
            expected_result.append(0.0)
        else:  # other
            arg__input.append(MagicMock())
            resultant_float = MagicMock()
            side_effects_for_float.append(resultant_float)
            expected_result.append(resultant_float)

    mocker.patch(parser_util.__name__ + ".float", side_effect=side_effects_for_float)
    mocker.patch(
        parser_util.__name__ + ".convert_str_to_timestamp",
        side_effect=side_effects_for_convert_str,
    )

    # Act
    result = parser_util.floatify_input(arg__input, arg_remove_str)

    # Assert
    assert result == expected_result


# convert_str_to_timestamp
def test_parser_util_convert_str_to_timestamp_returns_datetime_strptime_timestamp_on_success(
    mocker,
):
    # Arrange
    arg_time_str = str(MagicMock())

    fake_datetime = MagicMock()
    fake_timestamp = MagicMock()
    fake_dt_module = MagicMock()
    fake_dt_dt = MagicMock()

    mocker.patch(parser_util.__name__ + ".datetime", fake_dt_module)
    mocker.patch.object(fake_dt_module, "datetime", fake_dt_dt)
    mocker.patch.object(fake_dt_dt, "strptime", return_value=fake_datetime)
    mocker.patch.object(fake_datetime, "timestamp", return_value=fake_timestamp)
    # Act
    result = parser_util.convert_str_to_timestamp(arg_time_str)

    # Assert
    assert fake_dt_module.datetime.strptime.call_count == 1
    assert fake_dt_module.datetime.strptime.call_args_list[0].args == (
        arg_time_str,
        "%Y-%j-%H:%M:%S.%f",
    )
    assert fake_datetime.timestamp.call_count == 1
    assert result == fake_timestamp


def test_parser_util_convert_str_to_timestamp_returns_datetime_timestamp_when_strptime_raises_error(
    mocker,
):
    # Arrange
    arg_time_str = "59:20"

    fake_timestamp = MagicMock()
    fake_dt_module = MagicMock()

    class Fake_Datetime:
        timestamp_call_count = 0

        def __init__(self, year, month, day, hour, minute, second, subsecond):
            assert year == 2000
            assert month == 1
            assert day == 1
            assert hour == 1
            assert minute == 59
            assert second == 20
            assert subsecond == 0

        def strptime(arg1, arg2):
            raise Exception

        def timestamp(self):
            Fake_Datetime.timestamp_call_count = self.timestamp_call_count + 1
            return fake_timestamp

    mocker.patch(parser_util.__name__ + ".datetime", fake_dt_module)
    mocker.patch.object(fake_dt_module, "datetime", Fake_Datetime)

    # Act
    result = parser_util.convert_str_to_timestamp(arg_time_str)

    # Assert
    assert Fake_Datetime.timestamp_call_count == 1
    assert result == fake_timestamp


def test_parser_util_convert_str_to_timestamp_raises_error_when_both_strptime_and_datetime_raise_errors(
    mocker,
):
    # Arrange
    arg_time_str = str(MagicMock())

    fake_datetime = MagicMock()
    fake_timestamp = MagicMock()
    fake_dt_module = MagicMock()
    fake_dt_dt = MagicMock()

    class Fake_Datetime:
        def __init__(self, year, month, day, hour, minute, second, subsecond):
            raise Exception

        def strptime(arg1, arg2):
            raise Exception

        def timestamp():
            assert False

    mocker.patch(parser_util.__name__ + ".datetime", fake_dt_module)
    mocker.patch.object(fake_dt_module, "datetime", Fake_Datetime)

    # Act
    with pytest.raises(Exception) as e_info:
        parser_util.convert_str_to_timestamp(arg_time_str)

    # Assert
    assert e_info.match("")
