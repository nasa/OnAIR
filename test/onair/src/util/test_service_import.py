# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

import pytest
import os
from unittest.mock import MagicMock

import onair.src.util.service_import as service_import


def test_service_import_returns_empty_dict_when_given_service_dict_is_empty():
    # Arrange
    arg_service_dict = {}

    # Act
    result = service_import.import_services(arg_service_dict)

    # Assert
    assert result == {}


def test_service_import_returns_service_dict_containing_single_entry_with_return_key_as_arg_dict_key_and_return_value_as_Service_object_initialized_with_arg_value_after_popping_path_key_when_not_already_in_sys_modules(
    mocker,
):

    # Arrange
    fake_service_name = MagicMock()
    fake_service_kwarg = "i_am_fake"
    fake_service_kwarg_value = MagicMock()
    fake_service_info = MagicMock()
    fake_mod_name = MagicMock()
    fake_full_path = MagicMock()
    fake_true_path = MagicMock()

    fake_service_info = {
        "path": fake_true_path,
        fake_service_kwarg: fake_service_kwarg_value,
    }
    arg_module_dict = {fake_service_name: fake_service_info}

    fake_spec = MagicMock()
    fake_module = MagicMock()
    fake_service = MagicMock()
    fake_Service_instance = MagicMock()

    mocker.patch(
        service_import.__name__ + ".os.path.basename", return_value=fake_mod_name
    )
    mocker.patch(service_import.__name__ + ".os.path.join", return_value=fake_full_path)
    mocker.patch(
        service_import.__name__ + ".importlib.util.spec_from_file_location",
        return_value=fake_spec,
    )
    mocker.patch(
        service_import.__name__ + ".importlib.util.module_from_spec",
        return_value=fake_module,
    )

    mocker.patch.object(fake_spec, "loader.exec_module")

    mocker.patch.dict(service_import.sys.modules)
    import_mock = mocker.patch("builtins.__import__", return_value=fake_service)
    mocker.patch.object(fake_service, "Service", return_value=fake_Service_instance)

    # Act
    result = service_import.import_services(arg_module_dict)

    # Assert
    # If import checks fail, test fails with INTERNALERROR due to test output using patched code
    # Therefore import_mock is checked first then stopped, so other items failures output correctly
    # When this test fails because of INTERNALERROR the problem is with import_mock
    assert import_mock.call_count == 1
    assert import_mock.call_args_list[0].args == (
        f"{fake_mod_name}.{fake_mod_name}_service",
    )
    assert import_mock.call_args_list[0].kwargs == (
        {"fromlist": [f"{fake_mod_name}_service"]}
    )
    # # Without the stop of import_mock any other fails will also cause INTERNALERROR
    mocker.stop(import_mock)

    assert service_import.os.path.basename.call_count == 1
    assert service_import.os.path.basename.call_args_list[0].args == (fake_true_path,)
    assert service_import.os.path.join.call_count == 1
    assert service_import.os.path.join.call_args_list[0].args == (
        fake_true_path,
        "__init__.py",
    )
    assert service_import.importlib.util.spec_from_file_location.call_count == 1
    assert service_import.importlib.util.spec_from_file_location.call_args_list[
        0
    ].args == (fake_mod_name, fake_full_path)
    assert service_import.importlib.util.module_from_spec.call_count == 1
    assert service_import.importlib.util.module_from_spec.call_args_list[0].args == (
        fake_spec,
    )

    assert fake_spec.loader.exec_module.call_count == 1
    assert fake_spec.loader.exec_module.call_args_list[0].args == (fake_module,)
    assert fake_mod_name in service_import.sys.modules
    assert service_import.sys.modules[fake_mod_name] == fake_module

    assert fake_service.Service.call_count == 1
    assert result == {fake_mod_name: fake_Service_instance}
    assert fake_service.Service.call_args_list[0].kwargs == {
        fake_service_kwarg: fake_service_kwarg_value
    }


def test_service_import_returns_service_dict_containing_multiple_entries_with_return_keys_as_arg_dict_keys_and_return_values_as_Service_objects_initialized_with_arg_value_after_popping_path_key_when_not_already_in_sys_modules(
    mocker,
):

    # Arrange
    num_services = pytest.gen.randint(1, 5)
    num_service_kwargs = pytest.gen.randint(1, 5)

    fake_full_path = MagicMock()
    fake_mod_names = []
    fake_Service_instances = []

    arg_module_dict = {}

    for service_number in range(num_services):  # 1-5 arbitrary length
        temp_service_info = {"path": MagicMock()}
        for arg_number in range(num_service_kwargs):
            temp_service_info.update({f"service_arg_{arg_number}": MagicMock()})
        arg_module_dict.update({MagicMock(): temp_service_info})
        fake_Service_instances.append(MagicMock())
        fake_mod_names.append(MagicMock())

    fake_spec = MagicMock()
    fake_module = MagicMock()
    fake_service = MagicMock()

    mocker.patch(
        service_import.__name__ + ".os.path.basename", side_effect=fake_mod_names
    )
    mocker.patch(service_import.__name__ + ".os.path.join", return_value=fake_full_path)
    mocker.patch(
        service_import.__name__ + ".importlib.util.spec_from_file_location",
        return_value=fake_spec,
    )
    mocker.patch(
        service_import.__name__ + ".importlib.util.module_from_spec",
        return_value=fake_module,
    )

    mocker.patch.object(fake_spec, "loader.exec_module")

    mocker.patch.dict(service_import.sys.modules)
    import_mock = mocker.patch("builtins.__import__", return_value=fake_service)
    mocker.patch.object(fake_service, "Service", side_effect=fake_Service_instances)

    # Act
    result = service_import.import_services(arg_module_dict)

    # Assert
    mocker.stop(import_mock)

    for i, (service_name, service_instance) in enumerate(result.items()):
        assert service_name == fake_mod_names[i]
        assert service_instance == fake_Service_instances[i]
    e = arg_module_dict.values()
    print(e.__iter__().__next__())
    print(fake_service.Service.call_args_list[0].kwargs)
    assert fake_service.Service.call_count == num_services
    for i, service_info_dict in enumerate(arg_module_dict.values()):
        assert fake_service.Service.call_args_list[i].kwargs == service_info_dict


def test_service_import_returns_service_dict_containing_single_entry_with_return_key_as_arg_dict_key_and_return_value_as_Service_object_initialized_with_arg_value_after_popping_path_key_when_exists_in_sys_modules(
    mocker,
):

    # Arrange
    fake_service_name = MagicMock()
    fake_service_kwarg = "i_am_fake"
    fake_service_kwarg_value = MagicMock()
    fake_service_info = MagicMock()
    fake_mod_name = MagicMock()
    fake_true_path = MagicMock()

    fake_service_info = {
        "path": fake_true_path,
        fake_service_kwarg: fake_service_kwarg_value,
    }
    arg_module_dict = {fake_service_name: fake_service_info}

    fake_spec = MagicMock()
    fake_service = MagicMock()
    fake_Service_instance = MagicMock()

    mocker.patch(
        service_import.__name__ + ".os.path.basename", return_value=fake_mod_name
    )
    mocker.patch(service_import.__name__ + ".os.path.join")
    mocker.patch(service_import.__name__ + ".importlib.util.spec_from_file_location")
    mocker.patch(service_import.__name__ + ".importlib.util.module_from_spec")

    mocker.patch.object(fake_spec, "loader.exec_module")

    mocker.patch.dict(service_import.sys.modules, {fake_mod_name: None})
    import_mock = mocker.patch("builtins.__import__", return_value=fake_service)
    mocker.patch.object(fake_service, "Service", return_value=fake_Service_instance)

    # Act
    result = service_import.import_services(arg_module_dict)

    # Assert
    # If import checks fail, test fails with INTERNALERROR due to test output using patched code
    # Therefore import_mock is checked first then stopped, so other items failures output correctly
    # When this test fails because of INTERNALERROR the problem is with import_mock
    assert import_mock.call_count == 1
    assert import_mock.call_args_list[0].args == (
        f"{fake_mod_name}.{fake_mod_name}_service",
    )
    assert import_mock.call_args_list[0].kwargs == (
        {"fromlist": [f"{fake_mod_name}_service"]}
    )
    # # Without the stop of import_mock any other fails will also cause INTERNALERROR
    mocker.stop(import_mock)

    assert service_import.os.path.basename.call_count == 1
    assert service_import.os.path.join.call_count == 0
    assert service_import.importlib.util.spec_from_file_location.call_count == 0
    assert service_import.importlib.util.module_from_spec.call_count == 0
    assert fake_spec.loader.exec_module.call_count == 0
    assert fake_mod_name in service_import.sys.modules

    assert fake_service.Service.call_count == 1
    assert result == {fake_mod_name: fake_Service_instance}
    assert fake_service.Service.call_args_list[0].kwargs == {
        fake_service_kwarg: fake_service_kwarg_value
    }
