# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test TLM Json Parser Functionality """
import pytest
from unittest.mock import MagicMock

import onair.data_handling.tlm_json_parser as tlm_json_parser


# parseTlmConfJson tests
def test_tlm_json_parser_parseTlmConfJson_returns_configs_with_empty_dicts_when_reorg_dict_is_empty(
    mocker,
):
    # Arrange
    arg_file_path = MagicMock()

    fake_data = MagicMock()
    fake_organized_data = {}

    mocker.patch(tlm_json_parser.__name__ + ".parseJson", return_value=fake_data)
    mocker.patch(
        tlm_json_parser.__name__ + ".reorganizeTlmDict",
        return_value=fake_organized_data,
    )

    expected_result = {}
    expected_result["subsystem_assignments"] = []
    expected_result["test_assignments"] = []
    expected_result["description_assignments"] = []
    expected_result["data_labels"] = []

    # Act
    result = tlm_json_parser.parseTlmConfJson(arg_file_path)

    # Assert
    assert tlm_json_parser.parseJson.call_count == 1
    assert tlm_json_parser.parseJson.call_args_list[0].args == (arg_file_path,)
    assert tlm_json_parser.reorganizeTlmDict.call_count == 1
    assert tlm_json_parser.reorganizeTlmDict.call_args_list[0].args == (fake_data,)
    assert result == expected_result


def test_tlm_json_parser_parseTlmConfJson_returns_expected_configs_dict_when_reorg_dict_contains_only_one_label_and_order_key_does_not_exist(
    mocker,
):
    # Arrange
    arg_file_path = MagicMock()

    fake_data = MagicMock()
    fake_label = MagicMock()
    fake_subsystem = MagicMock()
    fake_limits = MagicMock()
    fake_mnemonics = MagicMock()
    fake_description = MagicMock()
    fake_organized_data = {}
    fake_organized_data[fake_label] = {
        "subsystem": fake_subsystem,
        "tests": {fake_mnemonics: fake_limits},
        "description": fake_description,
    }

    mocker.patch(tlm_json_parser.__name__ + ".parseJson", return_value=fake_data)
    mocker.patch(
        tlm_json_parser.__name__ + ".reorganizeTlmDict",
        return_value=fake_organized_data,
    )

    expected_result = {}
    expected_result["subsystem_assignments"] = [fake_subsystem]
    expected_result["test_assignments"] = [[[fake_mnemonics, fake_limits]]]
    expected_result["description_assignments"] = [fake_description]
    expected_result["data_labels"] = [fake_label]

    # Act
    result = tlm_json_parser.parseTlmConfJson(arg_file_path)

    # Assert
    assert tlm_json_parser.parseJson.call_count == 1
    assert tlm_json_parser.parseJson.call_args_list[0].args == (arg_file_path,)
    assert tlm_json_parser.reorganizeTlmDict.call_count == 1
    assert tlm_json_parser.reorganizeTlmDict.call_args_list[0].args == (fake_data,)
    assert result == expected_result


def test_tlm_json_parser_parseTlmConfJson_returns_expected_configs_dict_when_reorg_dict_contains_only_one_label_and_limits_test_and_description_keys_do_not_exist(
    mocker,
):
    # Arrange
    arg_file_path = MagicMock()

    fake_data = MagicMock()
    fake_label = MagicMock()
    fake_subsystem = MagicMock()
    fake_organized_data = {}
    fake_organized_data[fake_label] = {"subsystem": fake_subsystem}

    mocker.patch(tlm_json_parser.__name__ + ".parseJson", return_value=fake_data)
    mocker.patch(
        tlm_json_parser.__name__ + ".reorganizeTlmDict",
        return_value=fake_organized_data,
    )

    expected_result = {}
    expected_result["subsystem_assignments"] = [fake_subsystem]
    expected_result["test_assignments"] = [[["NOOP"]]]
    expected_result["description_assignments"] = [["No description"]]
    expected_result["data_labels"] = [fake_label]

    # Act
    result = tlm_json_parser.parseTlmConfJson(arg_file_path)

    # Assert
    assert tlm_json_parser.parseJson.call_count == 1
    assert tlm_json_parser.parseJson.call_args_list[0].args == (arg_file_path,)
    assert tlm_json_parser.reorganizeTlmDict.call_count == 1
    assert tlm_json_parser.reorganizeTlmDict.call_args_list[0].args == (fake_data,)
    assert result == expected_result


def test_tlm_json_parser_parseTlmConfJson_returns_expected_configs_dict_when_reorg_dict_contains_multiple_labels_and_limits_test_and_description_keys_do_not_exist(
    mocker,
):
    # Arrange
    arg_file_path = MagicMock()

    fake_data = MagicMock()
    fake_organized_data = {}
    fake_subsystems = []
    fake_labels = []
    num_labels = pytest.gen.randint(2, 10)  # arbitrary, from 2 to 10
    for i in range(num_labels):
        fake_label = MagicMock()
        fake_subsystem = MagicMock()
        fake_subsystems.append(fake_subsystem)
        fake_labels.append(fake_label)
        fake_organized_data[fake_label] = {"subsystem": fake_subsystem}

    mocker.patch(tlm_json_parser.__name__ + ".parseJson", return_value=fake_data)
    mocker.patch(
        tlm_json_parser.__name__ + ".reorganizeTlmDict",
        return_value=fake_organized_data,
    )

    expected_result = {}
    expected_result["subsystem_assignments"] = fake_subsystems
    expected_result["test_assignments"] = [[["NOOP"]]] * num_labels
    expected_result["description_assignments"] = [["No description"]] * num_labels
    expected_result["data_labels"] = fake_labels

    # Act
    result = tlm_json_parser.parseTlmConfJson(arg_file_path)

    # Assert
    assert tlm_json_parser.parseJson.call_count == 1
    assert tlm_json_parser.parseJson.call_args_list[0].args == (arg_file_path,)
    assert tlm_json_parser.reorganizeTlmDict.call_count == 1
    assert tlm_json_parser.reorganizeTlmDict.call_args_list[0].args == (fake_data,)
    assert result == expected_result


def test_tlm_json_parser_parseTlmConfJson_returns_expected_configs_dict_when_reorg_dict_contains_only_one_label_and_order_key_does_exist(
    mocker,
):
    # Arrange
    arg_file_path = MagicMock()

    fake_label = MagicMock()
    fake_data = {"order": [fake_label]}
    fake_subsystem = MagicMock()
    fake_limits = MagicMock()
    fake_mnemonics = MagicMock()
    fake_description = MagicMock()
    fake_organized_data = {}
    fake_organized_data[fake_label] = {
        "subsystem": fake_subsystem,
        "tests": {fake_mnemonics: fake_limits},
        "description": fake_description,
    }

    mocker.patch(tlm_json_parser.__name__ + ".parseJson", return_value=fake_data)
    mocker.patch(
        tlm_json_parser.__name__ + ".reorganizeTlmDict",
        return_value=fake_organized_data,
    )

    expected_result = {}
    expected_result["subsystem_assignments"] = [fake_subsystem]
    expected_result["test_assignments"] = [[[fake_mnemonics, fake_limits]]]
    expected_result["description_assignments"] = [fake_description]
    expected_result["data_labels"] = [fake_label]

    # Act
    result = tlm_json_parser.parseTlmConfJson(arg_file_path)

    # Assert
    assert tlm_json_parser.parseJson.call_count == 1
    assert tlm_json_parser.parseJson.call_args_list[0].args == (arg_file_path,)
    assert tlm_json_parser.reorganizeTlmDict.call_count == 1
    assert tlm_json_parser.reorganizeTlmDict.call_args_list[0].args == (fake_data,)
    assert result == expected_result


def test_tlm_json_parser_parseTlmConfJson_returns_expected_configs_dict_when_reorg_dict_contains_more_than_one_label_and_order_key_does_not_exist(
    mocker,
):
    # Arrange
    arg_file_path = MagicMock()

    fake_data = MagicMock()
    num_elems = pytest.gen.randint(2, 10)  # arbitrary, from 2 to 10
    fake_labels = [MagicMock() for i in range(num_elems)]
    fake_subsystem = MagicMock()
    fake_limits = MagicMock()
    fake_mnemonics = MagicMock()
    fake_description = MagicMock()
    fake_organized_data = {}
    for label in fake_labels:
        fake_organized_data[label] = {
            "subsystem": fake_subsystem,
            "tests": {fake_mnemonics: fake_limits},
            "description": fake_description,
        }

    mocker.patch(tlm_json_parser.__name__ + ".parseJson", return_value=fake_data)
    mocker.patch(
        tlm_json_parser.__name__ + ".reorganizeTlmDict",
        return_value=fake_organized_data,
    )

    expected_result = {}
    expected_result["subsystem_assignments"] = [fake_subsystem] * num_elems
    expected_result["test_assignments"] = [[[fake_mnemonics, fake_limits]]] * num_elems
    expected_result["description_assignments"] = [fake_description] * num_elems
    expected_result["data_labels"] = fake_labels

    # Act
    result = tlm_json_parser.parseTlmConfJson(arg_file_path)

    # Assert
    assert tlm_json_parser.parseJson.call_count == 1
    assert tlm_json_parser.parseJson.call_args_list[0].args == (arg_file_path,)
    assert tlm_json_parser.reorganizeTlmDict.call_count == 1
    assert tlm_json_parser.reorganizeTlmDict.call_args_list[0].args == (fake_data,)
    assert result == expected_result


def test_tlm_json_parser_parseTlmConfJson_returns_expected_configs_dict_when_reorg_dict_contains_more_than_one_label_and_order_key_does_exist(
    mocker,
):
    # Arrange
    arg_file_path = MagicMock()

    num_elems = pytest.gen.randint(2, 10)  # arbitrary, from 2 to 10
    fake_label = []
    fake_subsystem = []
    fake_limits = []
    fake_mnemonics = []
    fake_description = []
    for i in range(num_elems):
        fake_label.append(MagicMock())
        fake_subsystem.append(MagicMock())
        fake_limits.append(MagicMock())
        fake_mnemonics.append(MagicMock())
        fake_description.append(MagicMock())
    fake_order = fake_label.copy()
    pytest.gen.shuffle(fake_order)
    fake_data = {"order": fake_order}

    desired_order = {}
    for i in range(num_elems):
        desired_order[fake_order[i]] = i

    ordering_list = []
    for label in fake_label:
        ordering_list.append(desired_order[label])

    ordered_subsys = [y for x, y in sorted(zip(ordering_list, fake_subsystem))]
    ordered_mnemonics = [y for x, y in sorted(zip(ordering_list, fake_mnemonics))]
    ordered_limits = [y for x, y in sorted(zip(ordering_list, fake_limits))]
    ordered_descs = [y for x, y in sorted(zip(ordering_list, fake_description))]
    ordered_labels = [y for x, y in sorted(zip(ordering_list, fake_label))]

    fake_organized_data = {}
    for i in range(num_elems):
        fake_organized_data[fake_label[i]] = {
            "subsystem": fake_subsystem[i],
            "tests": {fake_mnemonics[i]: fake_limits[i]},
            "description": fake_description[i],
        }

    mocker.patch(tlm_json_parser.__name__ + ".parseJson", return_value=fake_data)
    mocker.patch(
        tlm_json_parser.__name__ + ".reorganizeTlmDict",
        return_value=fake_organized_data,
    )

    expected_result = {}
    expected_result["subsystem_assignments"] = []
    expected_result["test_assignments"] = []
    expected_result["description_assignments"] = []
    expected_result["data_labels"] = ordered_labels
    for i in range(num_elems):
        expected_result["subsystem_assignments"].append(ordered_subsys[i])
        expected_result["test_assignments"].append(
            [[ordered_mnemonics[i], ordered_limits[i]]]
        )
        expected_result["description_assignments"].append(ordered_descs[i])

    # Act
    result = tlm_json_parser.parseTlmConfJson(arg_file_path)

    # Assert
    assert tlm_json_parser.parseJson.call_count == 1
    assert tlm_json_parser.parseJson.call_args_list[0].args == (arg_file_path,)
    assert tlm_json_parser.reorganizeTlmDict.call_count == 1
    assert tlm_json_parser.reorganizeTlmDict.call_args_list[0].args == (fake_data,)
    assert result == expected_result


def test_tlm_json_parser_parseTlmConfJson_returns_expected_configs_dict_when_reorg_dict_contains_more_than_one_label_and_limits_are_interpreted_as_empty_lists(
    mocker,
):
    # Arrange
    arg_file_path = MagicMock()

    num_elems = pytest.gen.randint(2, 10)  # arbitrary, from 2 to 10
    fake_label = []
    fake_subsystem = []
    fake_limits = []
    fake_mnemonics = []
    fake_description = []
    for i in range(num_elems):
        fake_label.append(MagicMock())
        fake_subsystem.append(MagicMock())
        fake_limits.append(MagicMock())
        fake_mnemonics.append(MagicMock())
        fake_description.append(MagicMock())
    fake_order = fake_label.copy()
    pytest.gen.shuffle(fake_order)
    fake_data = {"order": fake_order}

    desired_order = {}
    for i in range(num_elems):
        desired_order[fake_order[i]] = i

    ordering_list = []
    for label in fake_label:
        ordering_list.append(desired_order[label])

    ordered_subsys = [y for x, y in sorted(zip(ordering_list, fake_subsystem))]
    ordered_mnemonics = [y for x, y in sorted(zip(ordering_list, fake_mnemonics))]
    ordered_limits = [y for x, y in sorted(zip(ordering_list, fake_limits))]
    ordered_descs = [y for x, y in sorted(zip(ordering_list, fake_description))]
    ordered_labels = [y for x, y in sorted(zip(ordering_list, fake_label))]

    fake_organized_data = {}
    for i in range(num_elems):
        fake_organized_data[fake_label[i]] = {
            "subsystem": fake_subsystem[i],
            "tests": {fake_mnemonics[i]: fake_limits[i]},
            "description": fake_description[i],
        }

    mocker.patch(tlm_json_parser.__name__ + ".parseJson", return_value=fake_data)
    mocker.patch(
        tlm_json_parser.__name__ + ".reorganizeTlmDict",
        return_value=fake_organized_data,
    )

    expected_result = {}
    expected_result["subsystem_assignments"] = []
    expected_result["test_assignments"] = []
    expected_result["description_assignments"] = []
    expected_result["data_labels"] = ordered_labels
    for i in range(num_elems):
        expected_result["subsystem_assignments"].append(ordered_subsys[i])
        expected_result["test_assignments"].append(
            [[ordered_mnemonics[i], ordered_limits[i]]]
        )
        expected_result["description_assignments"].append(ordered_descs[i])

    # Act
    result = tlm_json_parser.parseTlmConfJson(arg_file_path)

    # Assert
    assert tlm_json_parser.parseJson.call_count == 1
    assert tlm_json_parser.parseJson.call_args_list[0].args == (arg_file_path,)
    assert tlm_json_parser.reorganizeTlmDict.call_count == 1
    assert tlm_json_parser.reorganizeTlmDict.call_args_list[0].args == (fake_data,)
    assert result == expected_result


# reorganizeTlmDict tests
def test_tlm_json_parser_reorganizeTlmDict_raises_error_when_arg_data_does_not_contain_subsystems_key():
    # Arrange
    arg_data_len = pytest.gen.randint(0, 10)  # arbitrary, from 0 to 10
    arg_data = {}
    [arg_data.update({MagicMock(): MagicMock()}) for i in range(arg_data_len)]

    # Assert
    with pytest.raises(KeyError) as e_info:
        result = tlm_json_parser.reorganizeTlmDict(arg_data)

    # Act
    assert e_info.match("subsystems")


def test_tlm_json_parser_reorganizeTlmDict_returns_empty_dict_when_arg_data_subsystems_exists_and_is_empty():
    # Arrange
    arg_data = {"subsystems": {}}

    # Assert
    result = tlm_json_parser.reorganizeTlmDict(arg_data)

    # Act
    assert result == {}


def test_tlm_json_parser_reorganizeTlmDict_returns_empty_dict_when_arg_data_subsystems_exists_and_all_keys_map_to_empty():
    # Arrange
    arg_data = {"subsystems": {}}

    num_fake_subsystems = pytest.gen.randint(1, 10)  # arbitrary, from 1 to 10
    fake_subsystems = [MagicMock() for i in range(num_fake_subsystems)]

    for fs in fake_subsystems:
        arg_data["subsystems"][fs] = {}

    # Assert
    result = tlm_json_parser.reorganizeTlmDict(arg_data)

    # Act
    assert result == {}


def test_tlm_json_parser_reorganizeTlmDict_returns_expected_dict_when_arg_data_subsystems_exists_and_is_not_empty():
    # Arrange
    arg_data = {"subsystems": {}}

    num_fake_subsystems = pytest.gen.randint(
        1, 10
    )  # arbitrary, from 1 to 10, 0 has own test
    fake_subsystems = [MagicMock() for i in range(num_fake_subsystems)]

    expected_result = {}

    for fs in fake_subsystems:
        arg_data["subsystems"][fs] = {}
        num_fake_labels = pytest.gen.randint(
            1, 10
        )  # arbitrary, from 1 to 10, 0 has own test
        for i in range(num_fake_labels):
            fake_label = i
            fake_label_value = MagicMock()
            arg_data["subsystems"][fs][fake_label] = fake_label_value
            expected_result[fake_label] = fake_label_value
            expected_result[fake_label]["subsystem"] = fs

    # Assert
    result = tlm_json_parser.reorganizeTlmDict(arg_data)

    # Act
    assert result == expected_result


# str2lst tests
def test_tlm_json_parser_str2lst_returns_call_to_ast_literal_eval_which_receive_given_string(
    mocker,
):
    # Arrange
    arg_string = str(MagicMock())

    expected_result = MagicMock()

    mocker.patch(
        tlm_json_parser.__name__ + ".ast.literal_eval", return_value=expected_result
    )

    # Act
    result = tlm_json_parser.str2lst(arg_string)

    # Assert
    assert tlm_json_parser.ast.literal_eval.call_count == 1
    assert tlm_json_parser.ast.literal_eval.call_args_list[0].args == (arg_string,)
    assert result == expected_result


def test_tlm_json_parser_str2lst_prints_message_when_ast_literal_eval_receives_given_string_but_raises_exception(
    mocker,
):
    # Arrange
    arg_string = str(MagicMock())

    mocker.patch(tlm_json_parser.__name__ + ".ast.literal_eval", side_effect=Exception)
    mocker.patch(tlm_json_parser.__name__ + ".print")

    # Act
    result = tlm_json_parser.str2lst(arg_string)

    # Assert
    assert tlm_json_parser.ast.literal_eval.call_count == 1
    assert tlm_json_parser.ast.literal_eval.call_args_list[0].args == (arg_string,)
    assert tlm_json_parser.print.call_count == 1
    assert tlm_json_parser.print.call_args_list[0].args == (
        "Unable to process string representation of list",
    )
    assert result == None


# parseJson tests
def test_tlm_json_parser_parseJson_opens_given_path_and_returns_data_returned_by_json(
    mocker,
):
    # Arrange
    arg_path = MagicMock()

    fake_file = MagicMock()
    fake_file_str = MagicMock()
    fake_file_data = MagicMock()

    mocker.patch(tlm_json_parser.__name__ + ".open", return_value=fake_file)
    mocker.patch.object(fake_file, "read", return_value=fake_file_str)
    mocker.patch(tlm_json_parser.__name__ + ".json.loads", return_value=fake_file_data)
    mocker.patch.object(fake_file, "close")

    # Act
    result = tlm_json_parser.parseJson(arg_path)

    # Assert
    assert tlm_json_parser.open.call_count == 1
    assert tlm_json_parser.open.call_args_list[0].args == (arg_path, "rb")
    assert fake_file.read.call_count == 1
    assert tlm_json_parser.json.loads.call_count == 1
    assert tlm_json_parser.json.loads.call_args_list[0].args == (fake_file_str,)
    assert fake_file.close.call_count == 1
    assert result == fake_file_data
