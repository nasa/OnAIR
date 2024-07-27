# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"
import pytest
from unittest.mock import MagicMock

import onair.src.util.sim_io as sim_io


def test_sim_io_render_reasoning_writes_txt_and_csv_files_even_when_list_is_empty(
    mocker,
):
    # Arrange
    SAVE_PATH = "ONAIR_DIAGNOSIS_SAVE_PATH"
    diag1 = MagicMock()
    arg_diagnosis_list = []
    fake_system_filename = MagicMock()
    fake_full_path = MagicMock()
    fake_file_iterator = MagicMock()
    fake_file = MagicMock()
    fake_file.configure_mock(**{"__enter__.return_value": fake_file_iterator})

    mocker.patch(sim_io.__name__ + ".os.environ.get", return_value=fake_system_filename)
    mocker.patch(sim_io.__name__ + ".os.path.join", return_value=fake_full_path)
    mocker.patch("builtins.open", return_value=fake_file)

    # Act
    sim_io.render_reasoning(arg_diagnosis_list)

    # Assert
    assert open.call_count == 2
    assert fake_file_iterator.write.call_count == 4
    assert sim_io.os.environ.get.call_args_list[0].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[0].args == (
        fake_system_filename,
        "diagnosis.txt",
    )
    assert open.call_args_list[0].args == (fake_full_path,)
    assert open.call_args_list[0].kwargs == {"mode": "a"}
    assert fake_file_iterator.write.call_args_list[0].args == (
        "==========================================================\n",
    )
    assert fake_file_iterator.write.call_args_list[1].args == (
        "                        DIAGNOSIS                         \n",
    )
    assert fake_file_iterator.write.call_args_list[2].args == (
        "==========================================================\n",
    )
    assert sim_io.os.environ.get.call_args_list[1].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[1].args == (
        fake_system_filename,
        "diagnosis.csv",
    )
    assert open.call_args_list[1].args == (fake_full_path,)
    assert open.call_args_list[1].kwargs == {"mode": "a"}
    assert fake_file_iterator.write.call_args_list[3].args == (
        "time_step, cohens_kappa, faults, subgraph\n",
    )


def test_sim_io_render_reasoning_writes_txt_and_csv_files_with_entry_for_each_given_diagnosis_in_list(
    mocker,
):
    # Arrange
    SAVE_PATH = "ONAIR_DIAGNOSIS_SAVE_PATH"
    diag1 = MagicMock()
    arg_diagnosis_list = []
    fake_system_filename = MagicMock()
    fake_full_path = MagicMock()
    fake_file_iterator = MagicMock()
    fake_file = MagicMock()
    fake_file.configure_mock(**{"__enter__.return_value": fake_file_iterator})
    fake_timestep = "my fake time step"
    fake_str = MagicMock()
    fake_results_csv = MagicMock

    mocker.patch(sim_io.__name__ + ".os.environ.get", return_value=fake_system_filename)
    mocker.patch(sim_io.__name__ + ".os.path.join", return_value=fake_full_path)
    mocker.patch("builtins.open", return_value=fake_file)

    for i in range(5):
        fake_diag = MagicMock()
        fake_diag.configure_mock(
            **{
                "get_time_step.return_value": fake_timestep,
                "__str__.return_value": fake_str,
                "results_csv.return_value": fake_results_csv,
            }
        )
        arg_diagnosis_list.append(fake_diag)

    # Act
    sim_io.render_reasoning(arg_diagnosis_list)

    # Assert
    assert open.call_count == 2
    assert fake_file_iterator.write.call_count == 4 + 5 * 5
    assert sim_io.os.environ.get.call_args_list[0].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[0].args == (
        fake_system_filename,
        "diagnosis.txt",
    )
    assert open.call_args_list[0].args == (fake_full_path,)
    assert open.call_args_list[0].kwargs == {"mode": "a"}
    assert fake_file_iterator.write.call_args_list[0].args == (
        "==========================================================\n",
    )
    assert fake_file_iterator.write.call_args_list[1].args == (
        "                        DIAGNOSIS                         \n",
    )
    assert fake_file_iterator.write.call_args_list[2].args == (
        "==========================================================\n",
    )

    for i in range(5):
        assert fake_file_iterator.write.call_args_list[i * 4 + 3].args == (
            "\n----------------------------------------------------------\n",
        )
        assert fake_file_iterator.write.call_args_list[i * 4 + 4].args == (
            "***                DIAGNOSIS AT FRAME "
            + fake_timestep
            + "               ***\n",
        )
        assert fake_file_iterator.write.call_args_list[i * 4 + 5].args == (fake_str,)
        assert fake_file_iterator.write.call_args_list[i * 4 + 6].args == (
            "----------------------------------------------------------\n",
        )

    assert sim_io.os.environ.get.call_args_list[1].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[1].args == (
        fake_system_filename,
        "diagnosis.csv",
    )
    assert open.call_args_list[1].args == (fake_full_path,)
    assert open.call_args_list[1].kwargs == {"mode": "a"}
    assert fake_file_iterator.write.call_args_list[i * 4 + 7].args == (
        "time_step, cohens_kappa, faults, subgraph\n",
    )

    for j in range(5):
        assert fake_file_iterator.write.call_args_list[j + i * 4 + 8].args == (
            fake_results_csv,
        )


def test_sim_io_render_viz_does_only_stattest_render_viz_does_status_sensor_and_diagnosis_reports_when_diagnosis_is_givenus_and_sensor_reports_when_diagnosis_is_not_given(
    mocker,
):
    # Arrange
    SAVE_PATH = "ONAIR_VIZ_SAVE_PATH"
    arg_status_data = MagicMock()
    arg_sensor_data = MagicMock()
    arg_sim_name = MagicMock()

    fake_system_filename = MagicMock()
    fake_full_path = MagicMock()
    fake_iterator = MagicMock()
    fake_file = MagicMock()
    fake_file.configure_mock(**{"__enter__.return_value": fake_iterator})

    expected_status_report = {}
    expected_status_report["filename"] = arg_sim_name
    expected_status_report["data"] = arg_status_data
    expected_sensor_status_report = {}
    expected_sensor_status_report["name"] = "MISSION"
    expected_sensor_status_report["children"] = arg_sensor_data

    mocker.patch(sim_io.__name__ + ".os.environ.get", return_value=fake_system_filename)
    mocker.patch(sim_io.__name__ + ".os.path.join", return_value=fake_full_path)
    mocker.patch("builtins.open", return_value=fake_file)
    mocker.patch(sim_io.__name__ + ".json.dump")

    # Act
    sim_io.render_viz(arg_status_data, arg_sensor_data, arg_sim_name)

    # Assert
    assert open.call_count == 2
    assert sim_io.os.environ.get.call_args_list[0].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[0].args == (
        fake_system_filename,
        "system.json",
    )
    assert open.call_args_list[0].args == (fake_full_path, "w")
    assert sim_io.json.dump.call_args_list[0].args == (
        expected_status_report,
        fake_iterator,
    )
    assert sim_io.os.environ.get.call_args_list[1].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[1].args == (
        fake_system_filename,
        "faults.json",
    )
    assert open.call_args_list[1].args == (fake_full_path, "w")
    assert sim_io.json.dump.call_args_list[1].args == (
        expected_sensor_status_report,
        fake_iterator,
    )


def test_sim_io_render_viz_does_only_status_and_sensor_reports_when_diagnosis_is_given_as_None(
    mocker,
):
    # Arrange
    SAVE_PATH = "ONAIR_VIZ_SAVE_PATH"
    arg_status_data = MagicMock()
    arg_sensor_data = MagicMock()
    arg_sim_name = MagicMock()
    arg_diagnosis = None

    fake_system_filename = MagicMock()
    fake_full_path = MagicMock()
    fake_iterator = MagicMock()
    fake_file = MagicMock()
    fake_file.configure_mock(**{"__enter__.return_value": fake_iterator})

    expected_status_report = {}
    expected_status_report["filename"] = arg_sim_name
    expected_status_report["data"] = arg_status_data
    expected_sensor_status_report = {}
    expected_sensor_status_report["name"] = "MISSION"
    expected_sensor_status_report["children"] = arg_sensor_data

    mocker.patch(sim_io.__name__ + ".os.environ.get", return_value=fake_system_filename)
    mocker.patch(sim_io.__name__ + ".os.path.join", return_value=fake_full_path)
    mocker.patch("builtins.open", return_value=fake_file)
    mocker.patch(sim_io.__name__ + ".json.dump")

    # Act
    sim_io.render_viz(arg_status_data, arg_sensor_data, arg_sim_name, arg_diagnosis)

    # Assert
    assert open.call_count == 2
    assert sim_io.os.environ.get.call_args_list[0].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[0].args == (
        fake_system_filename,
        "system.json",
    )
    assert open.call_args_list[0].args == (fake_full_path, "w")
    assert sim_io.json.dump.call_args_list[0].args == (
        expected_status_report,
        fake_iterator,
    )
    assert sim_io.os.environ.get.call_args_list[1].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[1].args == (
        fake_system_filename,
        "faults.json",
    )
    assert open.call_args_list[1].args == (fake_full_path, "w")
    assert sim_io.json.dump.call_args_list[1].args == (
        expected_sensor_status_report,
        fake_iterator,
    )


def test_sim_io_render_viz_does_status_sensor_and_diagnosis_reports_when_diagnosis_is_given(
    mocker,
):
    # Arrange
    SAVE_PATH = "ONAIR_VIZ_SAVE_PATH"
    arg_status_data = MagicMock()
    arg_sensor_data = MagicMock()
    arg_sim_name = MagicMock()
    arg_diagnosis = MagicMock()

    fake_system_filename = MagicMock()
    fake_full_path = MagicMock()
    fake_iterator = MagicMock()
    fake_file = MagicMock()
    fake_file.configure_mock(**{"__enter__.return_value": fake_iterator})
    fake_results = MagicMock()

    expected_status_report = {}
    expected_status_report["filename"] = arg_sim_name
    expected_status_report["data"] = arg_status_data
    expected_sensor_status_report = {}
    expected_sensor_status_report["name"] = "MISSION"
    expected_sensor_status_report["children"] = arg_sensor_data

    mocker.patch(sim_io.__name__ + ".os.environ.get", return_value=fake_system_filename)
    mocker.patch(sim_io.__name__ + ".os.path.join", return_value=fake_full_path)
    mocker.patch("builtins.open", return_value=fake_file)
    mocker.patch(sim_io.__name__ + ".json.dump")
    arg_diagnosis.configure_mock(
        **{"get_diagnosis_viz_json.return_value": fake_results}
    )

    # Act
    sim_io.render_viz(arg_status_data, arg_sensor_data, arg_sim_name, arg_diagnosis)

    # Assert
    assert open.call_count == 3
    assert sim_io.os.environ.get.call_args_list[0].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[0].args == (
        fake_system_filename,
        "system.json",
    )
    assert open.call_args_list[0].args == (fake_full_path, "w")
    assert sim_io.json.dump.call_args_list[0].args == (
        expected_status_report,
        fake_iterator,
    )
    assert sim_io.os.environ.get.call_args_list[1].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[1].args == (
        fake_system_filename,
        "faults.json",
    )
    assert open.call_args_list[1].args == (fake_full_path, "w")
    assert sim_io.json.dump.call_args_list[1].args == (
        expected_sensor_status_report,
        fake_iterator,
    )
    arg_diagnosis.get_diagnosis_viz_json.assert_called_once()
    assert sim_io.os.environ.get.call_args_list[2].args == (SAVE_PATH,)
    assert sim_io.os.path.join.call_args_list[2].args == (
        fake_system_filename,
        "results.json",
    )
    assert open.call_args_list[2].args == (fake_full_path, "w")
    assert sim_io.json.dump.call_args_list[2].args == (fake_results, fake_iterator)


def test_sim_io_print_dots_uses_mod_10_plus_one_dots_when_ts_mod_20_is_less_than_10(
    mocker,
):
    # Arrange
    arg_ts = 20  # really want 0-9 + 20 * (arbitrary random 0 to some number)
    expected_num_dots = (arg_ts % 10) + 1
    dots_string = ""

    for i in range(expected_num_dots):
        dots_string = dots_string + "."

    mocker.patch("builtins.print")

    # Act
    sim_io.print_dots(arg_ts)

    # Assert
    print.assert_called_with("\033[95m" + dots_string + "\033[0m")


def test_sim_io_print_dots_uses_10_minus_mod_10_plus_one_dots_when_ts_mod_20_is_10(
    mocker,
):
    # Arrange
    arg_ts = (
        10  # 10 is a static value by design but should still add 20 * 0 to some number
    )
    expected_num_dots = 10 - (arg_ts % 10) + 1
    dots_string = ""

    for i in range(expected_num_dots):
        dots_string = dots_string + "."

    mocker.patch("builtins.print")

    # Act
    sim_io.print_dots(arg_ts)

    # Assert
    print.assert_called_with("\033[95m" + dots_string + "\033[0m")


def test_sim_io_print_dots_uses_10_minus_mod_10_plus_one_dots_when_ts_mod_20_is_greater_than_10(
    mocker,
):
    # Arrange
    arg_ts = 19  # really should be 11 to 19 + 20 * 0 to some random number
    expected_num_dots = 10 - (arg_ts % 10) + 1
    dots_string = ""

    for i in range(expected_num_dots):
        dots_string = dots_string + "."

    mocker.patch("builtins.print")

    # Act
    sim_io.print_dots(arg_ts)

    # Assert
    print.assert_called_with("\033[95m" + dots_string + "\033[0m")
