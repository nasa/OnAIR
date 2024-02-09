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

import onair.src.util.plugin_import as plugin_import

def test_plugin_import_returns_empty_list_when_given_module_dict_is_empty():
    # Arrange
    arg_headers = MagicMock()
    arg_module_dict = {}

    # Act
    result = plugin_import.import_plugins(arg_headers, arg_module_dict)

    # Assert
    assert result == []

def test_plugin_import_returns_single_item_list_when_given_module_dict_contains_one_key_value_pair_no_init_and_not_already_in_sys(mocker):
    # Arrange
    arg_headers = MagicMock()
    fake_construct_name = MagicMock()
    fake_mod_name = MagicMock()
    fake_module_path = MagicMock()
    fake_full_path = MagicMock()
    arg_module_dict = {fake_construct_name:fake_module_path}

    fake_spec = MagicMock()
    fake_module = MagicMock()
    fake_plugin = MagicMock()
    fake_Plugin_instance = MagicMock()

    mocker.patch.object(fake_module_path, 'endswith',
                        return_value=False)
    mocker.patch(plugin_import.__name__ + '.os.path.basename',
                 return_value=fake_mod_name)
    mocker.patch(plugin_import.__name__ + '.os.path.join',
                 return_value=fake_full_path)
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
    assert import_mock.call_args_list[0].args == (f'{fake_mod_name}.{fake_mod_name}_plugin', )
    assert import_mock.call_args_list[0].kwargs == ({'fromlist': [f"{fake_mod_name}_plugin"]})
    # # Without the stop of import_mock any other fails will also cause INTERNALERROR
    mocker.stop(import_mock)

    assert plugin_import.os.path.basename.call_count == 1
    assert plugin_import.os.path.basename.call_args_list[0].args == (fake_module_path, )
    assert plugin_import.importlib.util.spec_from_file_location.call_count == 1
    assert plugin_import.importlib.util.spec_from_file_location.call_args_list[0].args == (fake_mod_name, fake_full_path)
    assert plugin_import.importlib.util.module_from_spec.call_count == 1
    assert plugin_import.importlib.util.module_from_spec.call_args_list[0].args == (fake_spec,)
    assert fake_spec.loader.exec_module.call_count == 1
    assert fake_spec.loader.exec_module.call_args_list[0].args == (fake_module, )
    assert fake_mod_name in plugin_import.sys.modules
    assert plugin_import.sys.modules[fake_mod_name] == fake_module
    assert result == [fake_Plugin_instance]

def test_plugin_import_returns_single_item_list_when_given_module_dict_contains_one_key_value_pair_has_init_and_not_already_in_sys(mocker):
    # Arrange
    arg_headers = MagicMock()
    fake_construct_name = MagicMock()
    fake_mod_name = MagicMock()
    fake_pathing = []
    for _ in range(pytest.gen.randint(1, 5)): # 1-5 arbitrary length
        fake_pathing.append(str(MagicMock()))
    expected_true_path = os.path.join(*fake_pathing)
    fake_pathing.append("__init__.py")
    fake_module_path = os.path.join(*fake_pathing)
    fake_full_path = MagicMock()
    arg_module_dict = {fake_construct_name:fake_module_path}

    fake_spec = MagicMock()
    fake_module = MagicMock()
    fake_plugin = MagicMock()
    fake_Plugin_instance = MagicMock()

    mocker.patch(plugin_import.__name__ + '.os.path.basename',
                 return_value=fake_mod_name)
    mocker.patch(plugin_import.__name__ + '.os.path.join',
                 return_value=fake_full_path)
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
    assert import_mock.call_args_list[0].args == (f'{fake_mod_name}.{fake_mod_name}_plugin', )
    assert import_mock.call_args_list[0].kwargs == ({'fromlist': [f"{fake_mod_name}_plugin"]})
    # # Without the stop of import_mock any other fails will also cause INTERNALERROR
    mocker.stop(import_mock)

    assert plugin_import.os.path.basename.call_count == 1
    assert plugin_import.os.path.basename.call_args_list[0].args == (expected_true_path, )
    assert plugin_import.importlib.util.spec_from_file_location.call_count == 1
    assert plugin_import.importlib.util.spec_from_file_location.call_args_list[0].args == (fake_mod_name, fake_full_path)
    assert plugin_import.importlib.util.module_from_spec.call_count == 1
    assert plugin_import.importlib.util.module_from_spec.call_args_list[0].args == (fake_spec,)
    assert fake_spec.loader.exec_module.call_count == 1
    assert fake_spec.loader.exec_module.call_args_list[0].args == (fake_module, )
    assert fake_mod_name in plugin_import.sys.modules
    assert plugin_import.sys.modules[fake_mod_name] == fake_module
    assert result == [fake_Plugin_instance]

def test_plugin_import_returns_single_item_list_when_given_module_dict_contains_one_key_value_pair_no_init_and_exists_in_sys(mocker):
    # Arrange
    arg_headers = MagicMock()
    fake_construct_name = MagicMock()
    fake_mod_name = MagicMock()
    fake_module_path = MagicMock()
    fake_full_path = MagicMock()
    arg_module_dict = {fake_construct_name:fake_module_path}

    fake_plugin = MagicMock()
    fake_Plugin_instance = MagicMock()

    mocker.patch.object(fake_module_path, 'endswith',
                        return_value=False)
    mocker.patch(plugin_import.__name__ + '.os.path.basename',
                 return_value=fake_mod_name)
    mocker.patch(plugin_import.__name__ + '.os.path.join',
                 return_value=fake_full_path)
    mocker.patch(plugin_import.__name__ + '.importlib.util.spec_from_file_location')
    mocker.patch.dict(plugin_import.sys.modules, {fake_mod_name:None})
    import_mock = mocker.patch('builtins.__import__', return_value=fake_plugin)
    mocker.patch.object(fake_plugin, 'Plugin', return_value=fake_Plugin_instance)

    # Act
    result = plugin_import.import_plugins(arg_headers, arg_module_dict)

    # Assert
    # If import checks fail, test fails with INTERNALERROR due to test output using patched code
    # Therefore import_mock is checked first then stopped, so other items failures output correctly
    # When this test fails because of INTERNALERROR the problem is with import_mock
    assert import_mock.call_count == 1
    assert import_mock.call_args_list[0].args == (f'{fake_mod_name}.{fake_mod_name}_plugin', )
    assert import_mock.call_args_list[0].kwargs == ({'fromlist': [f"{fake_mod_name}_plugin"]})
    # # Without the stop of import_mock any other fails will also cause INTERNALERROR
    mocker.stop(import_mock)

    assert plugin_import.os.path.basename.call_count == 1
    assert plugin_import.os.path.basename.call_args_list[0].args == (fake_module_path, )
    assert plugin_import.importlib.util.spec_from_file_location.call_count == 0
    assert fake_mod_name in plugin_import.sys.modules
    assert plugin_import.sys.modules[fake_mod_name] == None
    assert result == [fake_Plugin_instance]

def test_plugin_import_returns_single_item_list_when_given_module_dict_contains_one_key_value_pair_has_init_and_exists_in_sys(mocker):
    # Arrange
    arg_headers = MagicMock()
    fake_construct_name = MagicMock()
    fake_mod_name = MagicMock()
    fake_pathing = []
    for _ in range(pytest.gen.randint(1, 5)): # 1-5 arbitrary length
        fake_pathing.append(str(MagicMock()))
    expected_true_path = os.path.join(*fake_pathing)
    fake_pathing.append("__init__.py")
    fake_module_path = os.path.join(*fake_pathing)
    fake_full_path = MagicMock()
    arg_module_dict = {fake_construct_name:fake_module_path}

    fake_plugin = MagicMock()
    fake_Plugin_instance = MagicMock()

    mocker.patch(plugin_import.__name__ + '.os.path.basename',
                 return_value=fake_mod_name)
    mocker.patch(plugin_import.__name__ + '.os.path.join',
                 return_value=fake_full_path)
    mocker.patch(plugin_import.__name__ + '.importlib.util.spec_from_file_location')
    mocker.patch.dict(plugin_import.sys.modules, {fake_mod_name:None})
    import_mock = mocker.patch('builtins.__import__', return_value=fake_plugin)
    mocker.patch.object(fake_plugin, 'Plugin', return_value=fake_Plugin_instance)

    # Act
    result = plugin_import.import_plugins(arg_headers, arg_module_dict)

    # Assert
    # If import checks fail, test fails with INTERNALERROR due to test output using patched code
    # Therefore import_mock is checked first then stopped, so other items failures output correctly
    # When this test fails because of INTERNALERROR the problem is with import_mock
    assert import_mock.call_count == 1
    assert import_mock.call_args_list[0].args == (f'{fake_mod_name}.{fake_mod_name}_plugin', )
    assert import_mock.call_args_list[0].kwargs == ({'fromlist': [f"{fake_mod_name}_plugin"]})
    # # Without the stop of import_mock any other fails will also cause INTERNALERROR
    mocker.stop(import_mock)

    assert plugin_import.os.path.basename.call_count == 1
    assert plugin_import.os.path.basename.call_args_list[0].args == (expected_true_path, )
    assert plugin_import.importlib.util.spec_from_file_location.call_count == 0
    assert fake_mod_name in plugin_import.sys.modules
    assert plugin_import.sys.modules[fake_mod_name] == None
    assert result == [fake_Plugin_instance]

def test_plugin_import_returns_two_item_list_when_given_module_dict_contains_two_key_value_pairs_that_use_same_module(mocker):
    # Arrange
    arg_headers = MagicMock()
    fake_construct_name_1 = MagicMock()
    fake_construct_name_2 = MagicMock()
    fake_mod_name = MagicMock()
    fake_module_path = MagicMock()
    fake_full_path = MagicMock()
    arg_module_dict = {fake_construct_name_1:fake_module_path,
                       fake_construct_name_2:fake_module_path}

    fake_spec = MagicMock()
    fake_module = MagicMock()
    fake_plugin = MagicMock()
    fake_Plugin_instance_1 = MagicMock()
    fake_Plugin_instance_2 = MagicMock()

    mocker.patch.object(fake_module_path, 'endswith',
                        return_value=False)
    mocker.patch(plugin_import.__name__ + '.os.path.basename',
                 return_value=fake_mod_name)
    mocker.patch(plugin_import.__name__ + '.os.path.join',
                 return_value=fake_full_path)
    mocker.patch(plugin_import.__name__ + '.importlib.util.spec_from_file_location', return_value=fake_spec)
    mocker.patch(plugin_import.__name__ + '.importlib.util.module_from_spec', return_value=fake_module)
    mocker.patch.object(fake_spec, 'loader.exec_module')
    mocker.patch.dict(plugin_import.sys.modules)
    import_mock = mocker.patch('builtins.__import__', return_value=fake_plugin)
    mocker.patch.object(fake_plugin,
                        'Plugin',
                        side_effect=[fake_Plugin_instance_1,
                                     fake_Plugin_instance_2])

    # Act
    result = plugin_import.import_plugins(arg_headers, arg_module_dict)

    # Assert
    # If import checks fail, test fails with INTERNALERROR due to test output using patched code
    # Therefore import_mock is checked first then stopped, so other items failures output correctly
    # When this test fails because of INTERNALERROR the problem is with import_mock
    assert import_mock.call_count == 2
    assert import_mock.call_args_list[0].args == (f'{fake_mod_name}.{fake_mod_name}_plugin', )
    assert import_mock.call_args_list[0].kwargs == ({'fromlist': [f"{fake_mod_name}_plugin"]})
    assert import_mock.call_args_list[1].args == (f'{fake_mod_name}.{fake_mod_name}_plugin', )
    assert import_mock.call_args_list[1].kwargs == ({'fromlist': [f"{fake_mod_name}_plugin"]})
    # # Without the stop of import_mock any other fails will also cause INTERNALERROR
    mocker.stop(import_mock)

    assert plugin_import.os.path.basename.call_count == 2
    assert plugin_import.os.path.basename.call_args_list[0].args == (fake_module_path, )
    assert plugin_import.importlib.util.spec_from_file_location.call_count == 1
    assert plugin_import.importlib.util.spec_from_file_location.call_args_list[0].args == (fake_mod_name, fake_full_path)
    assert plugin_import.importlib.util.module_from_spec.call_count == 1
    assert plugin_import.importlib.util.module_from_spec.call_args_list[0].args == (fake_spec,)
    assert fake_spec.loader.exec_module.call_count == 1
    assert fake_spec.loader.exec_module.call_args_list[0].args == (fake_module, )
    assert fake_mod_name in plugin_import.sys.modules
    assert plugin_import.sys.modules[fake_mod_name] == fake_module
    assert result == [fake_Plugin_instance_1, fake_Plugin_instance_2]
