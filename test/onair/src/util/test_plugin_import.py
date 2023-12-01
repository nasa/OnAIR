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

import onair.src.util.plugin_import as plugin_import

def test_plugin_import_returns_empty_list_when_given_module_dict_is_empty():
    # Arrange
    arg_headers = MagicMock()
    arg_module_dict = {}

    # Act
    result = plugin_import.import_plugins(arg_headers, arg_module_dict)

    # Assert
    assert result == []

def test_plugin_import_returns_single_item_list_when_given_module_dict_contains_one_key_value_pair(mocker):
    # Arrange
    arg_headers = MagicMock()
    fake_module_name = MagicMock()
    fake_module_path = MagicMock()
    arg_module_dict = {fake_module_name:fake_module_path}

    fake_spec = MagicMock()
    fake_module = MagicMock()
    fake_plugin = MagicMock()
    fake_Plugin_instance = MagicMock()

    mocker.patch(plugin_import.__name__ + '.importlib.util.spec_from_file_location', return_value=fake_spec)
    mocker.patch(plugin_import.__name__ + '.importlib.util.module_from_spec', return_value=fake_module)
    mocker.patch.object(fake_spec, 'loader.exec_module')
    mocker.patch.dict(plugin_import.sys.modules)
    import_mock = mocker.patch('builtins.__import__', return_value=fake_plugin)
    mocker.patch.object(fake_plugin, 'Plugin', return_value=fake_Plugin_instance)

    # Act
    result = plugin_import.import_plugins(arg_headers, arg_module_dict)

    # Assert
    # If import checks fail, test fails with INTERNALERROR due to test output using patched code
    # Therefore import_mock is checked first then stopped, so other items failures output correctly
    # When this test fails because of INTERNALERROR the problem is with import_mock
    assert import_mock.call_count == 1
    assert import_mock.call_args_list[0].args == (f'{fake_module_name}.{fake_module_name}_plugin', )
    assert import_mock.call_args_list[0].kwargs == ({'fromlist': [f"{fake_module_name}_plugin"]})
    # Without the stop of import_mock any other fails will also cause INTERNALERROR
    mocker.stop(import_mock)

    assert plugin_import.importlib.util.spec_from_file_location.call_count == 1
    assert plugin_import.importlib.util.spec_from_file_location.call_args_list[0].args == (fake_module_name, fake_module_path)
    assert plugin_import.importlib.util.module_from_spec.call_count == 1
    assert plugin_import.importlib.util.module_from_spec.call_args_list[0].args == (fake_spec,)
    assert fake_spec.loader.exec_module.call_count == 1
    assert fake_spec.loader.exec_module.call_args_list[0].args == (fake_module, )
    assert fake_module_name in plugin_import.sys.modules
    assert plugin_import.sys.modules[fake_module_name] == fake_module
    assert result == [fake_Plugin_instance]
