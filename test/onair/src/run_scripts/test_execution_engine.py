# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

""" Test Execution Engine Functionality """
import pytest
from unittest.mock import MagicMock

import onair.src.run_scripts.execution_engine as execution_engine
from onair.src.run_scripts.execution_engine import ExecutionEngine

# __init__ tests


def test_ExecutionEngine__init__sets_expected_values_but_does_no_calls_when_config_file_is_empty_string(mocker):
    # Arrange
    arg_config_file = ''
    arg_run_name = MagicMock()
    arg_save_flag = MagicMock()

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.object(cut, 'init_save_paths')
    mocker.patch.object(cut, 'parse_configs')
    mocker.patch.object(cut, 'parse_data')
    mocker.patch.object(cut, 'setup_sim')

    # Act
    cut.__init__(arg_config_file, arg_run_name, arg_save_flag)

    # Assert
    assert cut.run_name == arg_run_name
    assert cut.IO_Enabled == False
    assert cut.dataFilePath == ''
    assert cut.telemetryFile == ''
    assert cut.fullTelemetryFile == ''
    assert cut.metadataFilePath == ''
    assert cut.metaFile == ''
    assert cut.fullMetaFile == ''
    assert cut.data_source_file == ''
    assert cut.simDataSource == None
    assert cut.sim == None
    assert cut.save_flag == arg_save_flag
    assert cut.save_name == arg_run_name
    assert cut.init_save_paths.call_count == 0
    assert cut.parse_configs.call_count == 0
    assert cut.parse_data.call_count == 0
    assert cut.setup_sim.call_count == 0


def test_ExecutionEngine__init__does_calls_when_config_file_is_an_occupied_string(mocker):
    # Arrange
    arg_config_file = str(MagicMock())
    arg_run_name = MagicMock()
    arg_save_flag = MagicMock()

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.object(cut, 'init_save_paths')
    mocker.patch.object(cut, 'parse_configs')
    mocker.patch.object(cut, 'parse_data')
    mocker.patch.object(cut, 'setup_sim')

    # Act
    cut.__init__(arg_config_file, arg_run_name, arg_save_flag)

    # Assert
    assert cut.init_save_paths.call_count == 1
    assert cut.parse_configs.call_count == 1
    assert cut.parse_configs.call_args_list[0].args == (arg_config_file, )
    assert cut.parse_data.call_count == 1
    assert cut.parse_data.call_args_list[0].args == (
        cut.data_source_file, cut.dataFilePath, cut.metadataFilePath, )
    assert cut.setup_sim.call_count == 1


def test_ExecutionEngine__init__accepts_no_arguments_using_defaults_instead_with_config_file_default_as_empty_string(mocker):
    # Arrange
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.object(cut, 'init_save_paths')
    mocker.patch.object(cut, 'parse_configs')
    mocker.patch.object(cut, 'parse_data')
    mocker.patch.object(cut, 'setup_sim')

    # Act
    cut.__init__()

    # Assert
    assert cut.run_name == ''
    assert cut.save_flag == False
    assert cut.save_name == ''
    assert cut.init_save_paths.call_count == 0

# parse_configs tests


def test_ExecutionEngine_parse_configs_raises_FileNotFoundError_when_config_cannot_be_read(mocker):
    # Arrange
    arg_bad_config_filepath = MagicMock()

    fake_config = MagicMock()
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 0

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ +
                 '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read',
                        return_value=fake_config_read_result)

    # Act
    with pytest.raises(FileNotFoundError) as e_info:
        cut.parse_configs(arg_bad_config_filepath)

    # Assert
    assert e_info.match(f"Config file at '{
                        arg_bad_config_filepath}' could not be read.")


def test_ExecutionEngine_parse_configs_raises_KeyError_with_config_file_info_when_the_required_key_FILES_is_not_in_config(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    missing_key = 'FILES'
    fake_dict_for_Config = {
        "DATA_HANDLING": MagicMock(),
        "PLUGINS": MagicMock(),
        "OPTIONS": MagicMock()
    }
    fake_config = MagicMock()
    fake_config.__getitem__.side_effect = fake_dict_for_Config.__getitem__
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ +
                 '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read',
                        return_value=fake_config_read_result)
    mocker.patch.object(cut, 'parse_plugins_dict', return_value=None)

    # Act
    with pytest.raises(KeyError) as e_info:
        cut.parse_configs(arg_config_filepath)

    # Assert
    assert e_info.match(
        f"Config file: '{arg_config_filepath}', missing key: {missing_key}")


def test_ExecutionEngine_parse_configs_raises_KeyError_with_config_file_info_when_a_required_FILES_subkey_is_not_in_config(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    fake_files = {
        'TelemetryFilePath': MagicMock(),
        'TelemetryFile': MagicMock(),
        'MetaFilePath': MagicMock(),
        'MetaFile': MagicMock()
    }
    fake_data_handling = {
        'DataSourceFile': MagicMock()
    }
    fake_plugins = {
        'KnowledgeRepPluginDict': "{fake_name:fake_path}",
        'LearnersPluginDict': "{fake_name:fake_path}",
        'PlannersPluginDict': "{fake_name:fake_path}",
        'ComplexPluginDict': "{fake_name:fake_path}"
    }
    required_keys = [item for item in list(fake_files.keys())]
    missing_key = pytest.gen.choice(required_keys)
    del fake_files[missing_key]
    fake_options = MagicMock()
    fake_dict_for_Config = {
        "FILES": fake_files,
        "DATA_HANDLING": fake_data_handling,
        "PLUGINS": fake_plugins,
        "OPTIONS": fake_options
    }
    fake_config = MagicMock()
    fake_config.__getitem__.side_effect = fake_dict_for_Config.__getitem__
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ +
                 '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read',
                        return_value=fake_config_read_result)
    mocker.patch.object(cut, 'parse_plugins_dict', return_value=None)

    # Act
    with pytest.raises(KeyError) as e_info:
        cut.parse_configs(arg_config_filepath)

    # Assert
    assert e_info.match(
        f"Config file: '{arg_config_filepath}', missing key: {missing_key}")


def test_ExecutionEngine_parse_configs_skips_OPTIONS_when_the_required_section_OPTIONS_is_not_in_config(mocker):
    # Arrange
    arg_config_filepath = MagicMock()
    fake_files = {
        'TelemetryFilePath': MagicMock(),
        'TelemetryFile': MagicMock(),
        'MetaFilePath': MagicMock(),
        'MetaFile': MagicMock()
    }
    fake_data_handling = {
        'DataSourceFile': MagicMock()
    }
    fake_plugins = {
        'KnowledgeRepPluginDict': "{fake_name:fake_path}",
        'LearnersPluginDict': "{fake_name:fake_path}",
        'PlannersPluginDict': "{fake_name:fake_path}",
        'ComplexPluginDict': "{fake_name:fake_path}"
    }
    fake_plugin_dict = MagicMock()
    fake_plugin_dict.body = MagicMock()
    fake_plugin_dict.body.keys = MagicMock()
    fake_plugin_dict.body.keys.__len__.return_value = 1
    fake_dict_for_Config = {
        "FILES": fake_files,
        "DATA_HANDLING": fake_data_handling,
        "PLUGINS": fake_plugins
    }
    fake_config = MagicMock()
    fake_config.__getitem__.side_effect = fake_dict_for_Config.__getitem__
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1
    fake_knowledge_rep_plugin_list = MagicMock()
    fake_learners_plugin_list = MagicMock()
    fake_planners_plugin_list = MagicMock()
    fake_complex_plugin_list = MagicMock()
    fake_plugins = [fake_knowledge_rep_plugin_list,
                    fake_learners_plugin_list,
                    fake_planners_plugin_list,
                    fake_complex_plugin_list]
    fake_IO_enabled = MagicMock()
    fake_Dev_enabled = MagicMock()
    fake_Viz_enabled = MagicMock()
    fake_plugin_dict = MagicMock()
    fake_keys = MagicMock()
    fake_plugin = MagicMock()
    fake_path = MagicMock()

    fake_keys.__len__.return_value = 1
    fake_keys.__iter__.return_value = iter([str(fake_plugin)])

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ +
                 '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read',
                        return_value=fake_config_read_result)
    mocker.patch.object(fake_config, "has_section", return_value=False)
    mocker.patch.object(cut, 'parse_plugins_dict', side_effect=fake_plugins)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)
    mocker.patch(execution_engine.__name__ +
                 '.os.path.exists', return_value=True)
    mocker.patch.object(fake_plugin_dict, 'keys', return_value=fake_keys)
    mocker.patch.object(fake_plugin_dict, '__getitem__',
                        return_value=fake_path)

    # Act
    cut.parse_configs(arg_config_filepath)

    # Assert
    assert execution_engine.configparser.ConfigParser.call_count == 1
    assert fake_config.read.call_count == 1
    assert cut.dataFilePath == fake_files['TelemetryFilePath']
    assert cut.telemetryFile == fake_files['TelemetryFile']
    assert cut.metadataFilePath == fake_files['MetaFilePath']
    assert cut.metaFile == fake_files['MetaFile']
    assert cut.data_source_file == fake_data_handling['DataSourceFile']
    assert cut.parse_plugins_dict.call_count == 4
    assert cut.knowledge_rep_plugin_dict == fake_knowledge_rep_plugin_list
    assert cut.learners_plugin_dict == fake_learners_plugin_list
    assert cut.planners_plugin_dict == fake_planners_plugin_list
    assert cut.complex_plugin_dict == fake_complex_plugin_list
    assert cut.IO_Enabled == False


def test_ExecutionEngine_parse_configs_sets_all_items_without_error(mocker):
    # Arrange
    arg_config_filepath = MagicMock()
    fake_files = {
        'TelemetryFilePath': MagicMock(),
        'TelemetryFile': MagicMock(),
        'MetaFilePath': MagicMock(),
        'MetaFile': MagicMock()
    }
    fake_data_handling = {
        'DataSourceFile': MagicMock()
    }
    fake_plugins = {
        'KnowledgeRepPluginDict': "{fake_name:fake_path}",
        'LearnersPluginDict': "{fake_name:fake_path}",
        'PlannersPluginDict': "{fake_name:fake_path}",
        'ComplexPluginDict': "{fake_name:fake_path}"
    }
    fake_options = MagicMock()
    fake_plugin_dict = MagicMock()
    fake_plugin_dict.body = MagicMock()
    fake_plugin_dict.body.keys = MagicMock()
    fake_plugin_dict.body.keys.__len__.return_value = 1
    fake_dict_for_Config = {
        "FILES": fake_files,
        "DATA_HANDLING": fake_data_handling,
        "PLUGINS": fake_plugins,
        "OPTIONS": fake_options
    }
    fake_config = MagicMock()
    fake_config.__getitem__.side_effect = fake_dict_for_Config.__getitem__
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1
    fake_knowledge_rep_plugin_list = MagicMock()
    fake_learners_plugin_list = MagicMock()
    fake_planners_plugin_list = MagicMock()
    fake_complex_plugin_list = MagicMock()
    fake_plugins = [fake_knowledge_rep_plugin_list,
                    fake_learners_plugin_list,
                    fake_planners_plugin_list,
                    fake_complex_plugin_list]
    fake_IO_enabled = MagicMock()
    fake_Dev_enabled = MagicMock()
    fake_Viz_enabled = MagicMock()
    fake_plugin_dict = MagicMock()
    fake_keys = MagicMock()
    fake_plugin = MagicMock()
    fake_path = MagicMock()

    fake_keys.__len__.return_value = 1
    fake_keys.__iter__.return_value = iter([str(fake_plugin)])

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ +
                 '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read',
                        return_value=fake_config_read_result)
    mocker.patch.object(fake_config, "has_section", return_value=True)
    mocker.patch.object(cut, 'parse_plugins_dict', side_effect=fake_plugins)
    mocker.patch.object(fake_options, 'getboolean',
                        return_value=fake_IO_enabled)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)
    mocker.patch(execution_engine.__name__ +
                 '.os.path.exists', return_value=True)
    mocker.patch.object(fake_plugin_dict, 'keys', return_value=fake_keys)
    mocker.patch.object(fake_plugin_dict, '__getitem__',
                        return_value=fake_path)

    # Act
    cut.parse_configs(arg_config_filepath)

    # Assert
    assert execution_engine.configparser.ConfigParser.call_count == 1
    assert fake_config.read.call_count == 1
    assert cut.dataFilePath == fake_files['TelemetryFilePath']
    assert cut.telemetryFile == fake_files['TelemetryFile']
    assert cut.metadataFilePath == fake_files['MetaFilePath']
    assert cut.metaFile == fake_files['MetaFile']
    assert cut.data_source_file == fake_data_handling['DataSourceFile']
    assert cut.parse_plugins_dict.call_count == 4
    assert cut.knowledge_rep_plugin_dict == fake_knowledge_rep_plugin_list
    assert cut.learners_plugin_dict == fake_learners_plugin_list
    assert cut.planners_plugin_dict == fake_planners_plugin_list
    assert cut.complex_plugin_dict == fake_complex_plugin_list
    assert fake_options.getboolean.call_count == 1
    assert fake_options.getboolean.call_args_list[0].args == ('IO_Enabled', )
    assert cut.IO_Enabled == fake_IO_enabled

# parse_plugins_dict


def test_ExecutionEngine_parse_plugins_list_raises_ValueError_when_config_plugin_dict_is_not_dict(mocker):
    # Arrange
    arg_config_plugin_dict = MagicMock()

    fake_plugin_dict = MagicMock()
    fake_plugin_dict.body = MagicMock()
    fake_config_filepath = MagicMock()

    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.config_filepath = fake_config_filepath

    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_dict)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=False)

    # Act
    with pytest.raises(ValueError) as e_info:
        cut.parse_plugins_dict(arg_config_plugin_dict)

    # Assert
    assert e_info.match(f"Plugin dict {arg_config_plugin_dict} from {
                        fake_config_filepath} is invalid. It must be a dict.")
    assert cut.ast_parse_eval.call_count == 1
    assert cut.ast_parse_eval.call_args_list[0].args == (
        arg_config_plugin_dict,)
    assert execution_engine.isinstance.call_count == 1
    assert execution_engine.isinstance.call_args_list[0].args == (
        fake_plugin_dict.body, execution_engine.ast.Dict, )


def test_ExecutionEngine_parse_plugins_list_raises_FileNotFoundError_when_single_config_plugin_dict_key_maps_to_non_existing_file(mocker):
    # Arrange
    arg_config_plugin_dict = MagicMock()

    fake_plugin_dict = MagicMock()
    fake_plugin_dict.body = MagicMock()
    fake_config_filepath = MagicMock()
    fake_temp_plugin_dict = MagicMock()
    fake_values = MagicMock()
    fake_path = MagicMock()

    fake_values.__iter__.return_value = iter([fake_path])

    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.config_filepath = fake_config_filepath

    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_dict)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)
    mocker.patch(execution_engine.__name__ + '.ast.literal_eval',
                 return_value=fake_temp_plugin_dict)
    mocker.patch.object(fake_temp_plugin_dict, 'values',
                        return_value=fake_values)
    mocker.patch(execution_engine.__name__ +
                 '.os.path.exists', return_value=False)

    # Act
    with pytest.raises(FileNotFoundError) as e_info:
        cut.parse_plugins_dict(arg_config_plugin_dict)

    # Assert
    assert e_info.match(f"In config file '{fake_config_filepath}' Plugin path '{
                        fake_path}' does not exist.")
    assert cut.ast_parse_eval.call_count == 1
    assert cut.ast_parse_eval.call_args_list[0].args == (
        arg_config_plugin_dict,)
    assert execution_engine.isinstance.call_count == 1
    assert execution_engine.isinstance.call_args_list[0].args == (
        fake_plugin_dict.body, execution_engine.ast.Dict, )
    assert execution_engine.ast.literal_eval.call_count == 1
    assert execution_engine.ast.literal_eval.call_args_list[0].args == (
        arg_config_plugin_dict, )
    assert fake_temp_plugin_dict.values.call_count == 1
    assert fake_temp_plugin_dict.values.call_args_list[0].args == ()
    assert execution_engine.os.path.exists.call_count == 1
    assert execution_engine.os.path.exists.call_args_list[0].args == (
        fake_path, )


def test_ExecutionEngine_parse_plugins_list_raises_FileNotFoundError_when_any_config_plugin_dict_key_maps_to_non_existing_file(mocker):
    # Arrange
    arg_config_plugin_dict = MagicMock()

    fake_plugin_dict = MagicMock()
    fake_plugin_dict.body = MagicMock()
    fake_config_filepath = MagicMock()
    fake_temp_plugin_dict = MagicMock()
    fake_values = MagicMock()
    fake_path = MagicMock()
    # from 2 to 10 arbitrary, 1 has own test
    num_fake_items = pytest.gen.randint(2, 10)
    num_fake_existing_files = pytest.gen.randint(1, num_fake_items-1)
    exists_side_effects = [True] * num_fake_existing_files
    exists_side_effects.append(False)

    fake_values.__iter__.return_value = iter([fake_path] * num_fake_items)

    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.config_filepath = fake_config_filepath

    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_dict)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)
    mocker.patch(execution_engine.__name__ + '.ast.literal_eval',
                 return_value=fake_temp_plugin_dict)
    mocker.patch.object(fake_temp_plugin_dict, 'values',
                        return_value=fake_values)
    mocker.patch(execution_engine.__name__ + '.os.path.exists',
                 side_effect=exists_side_effects)

    # Act
    with pytest.raises(FileNotFoundError) as e_info:
        cut.parse_plugins_dict(arg_config_plugin_dict)

    # Assert
    assert e_info.match(f"In config file '{fake_config_filepath}' Plugin path '{
                        fake_path}' does not exist.")
    assert cut.ast_parse_eval.call_count == 1
    assert cut.ast_parse_eval.call_args_list[0].args == (
        arg_config_plugin_dict,)
    assert execution_engine.isinstance.call_count == 1
    assert execution_engine.isinstance.call_args_list[0].args == (
        fake_plugin_dict.body, execution_engine.ast.Dict, )
    assert execution_engine.ast.literal_eval.call_count == 1
    assert execution_engine.ast.literal_eval.call_args_list[0].args == (
        arg_config_plugin_dict, )
    assert fake_temp_plugin_dict.values.call_count == 1
    assert fake_temp_plugin_dict.values.call_args_list[0].args == ()
    assert execution_engine.os.path.exists.call_count == len(
        exists_side_effects)
    for i in range(len(exists_side_effects)):
        assert execution_engine.os.path.exists.call_args_list[i].args == (
            fake_path, )


def test_ExecutionEngine_returns_empty_dict_when_config_dict_is_empty(mocker):
    # Arrange
    arg_config_plugin_dict = MagicMock()

    fake_plugin_dict = MagicMock()
    fake_plugin_dict.body = MagicMock()
    fake_config_filepath = MagicMock()
    fake_temp_plugin_dict = {}

    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.config_filepath = fake_config_filepath

    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_dict)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)
    mocker.patch(execution_engine.__name__ + '.ast.literal_eval',
                 return_value=fake_temp_plugin_dict)

    # Act
    result = cut.parse_plugins_dict(arg_config_plugin_dict)

    # Assert
    assert result == {}
    assert cut.ast_parse_eval.call_count == 1
    assert cut.ast_parse_eval.call_args_list[0].args == (
        arg_config_plugin_dict,)
    assert execution_engine.isinstance.call_count == 1
    assert execution_engine.isinstance.call_args_list[0].args == (
        fake_plugin_dict.body, execution_engine.ast.Dict, )
    assert execution_engine.ast.literal_eval.call_count == 1
    assert execution_engine.ast.literal_eval.call_args_list[0].args == (
        arg_config_plugin_dict, )


def test_ExecutionEngine_returns_expected_dict_when_all_mapped_files_exist(mocker):
    # Arrange
    arg_config_plugin_dict = MagicMock()

    fake_plugin_dict = MagicMock()
    fake_plugin_dict.body = MagicMock()
    fake_config_filepath = MagicMock()
    fake_temp_plugin_dict = MagicMock()
    fake_values = MagicMock()
    fake_path = MagicMock()
    # from 2 to 10 arbitrary, 0 has own test
    num_fake_items = pytest.gen.randint(1, 10)
    exists_side_effects = [True] * num_fake_items

    fake_values.__iter__.return_value = iter([fake_path] * num_fake_items)

    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.config_filepath = fake_config_filepath

    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_dict)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)
    mocker.patch(execution_engine.__name__ + '.ast.literal_eval',
                 return_value=fake_temp_plugin_dict)
    mocker.patch.object(fake_temp_plugin_dict, 'values',
                        return_value=fake_values)
    mocker.patch(execution_engine.__name__ + '.os.path.exists',
                 side_effect=exists_side_effects)

    # Act
    result = cut.parse_plugins_dict(arg_config_plugin_dict)

    # Assert
    assert result == fake_temp_plugin_dict
    assert cut.ast_parse_eval.call_count == 1
    assert cut.ast_parse_eval.call_args_list[0].args == (
        arg_config_plugin_dict,)
    assert execution_engine.isinstance.call_count == 1
    assert execution_engine.isinstance.call_args_list[0].args == (
        fake_plugin_dict.body, execution_engine.ast.Dict, )
    assert execution_engine.ast.literal_eval.call_count == 1
    assert execution_engine.ast.literal_eval.call_args_list[0].args == (
        arg_config_plugin_dict, )
    assert fake_temp_plugin_dict.values.call_count == 1
    assert fake_temp_plugin_dict.values.call_args_list[0].args == ()
    assert execution_engine.os.path.exists.call_count == len(
        exists_side_effects)
    for i in range(len(exists_side_effects)):
        assert execution_engine.os.path.exists.call_args_list[i].args == (
            fake_path, )

# parse_data tests


def test_ExecutionEngine_parse_data_sets_the_simDataSource_to_a_new_data_source_module_DataSource(mocker):
    # Arrange
    arg_parser_file_name = MagicMock()
    arg_dataFile = str(MagicMock())
    arg_metadataFile = str(MagicMock())
    arg_subsystems_breakdown = MagicMock()

    fake_spec = MagicMock()
    fake_module = MagicMock()
    fake_parser_class_instance = MagicMock()

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ +
                 '.importlib.util.spec_from_file_location', return_value=fake_spec)
    mocker.patch(execution_engine.__name__ +
                 '.importlib.util.module_from_spec', return_value=fake_module)
    mocker.patch.object(fake_spec, 'loader.exec_module', return_value=None)
    mocker.patch.object(fake_module, 'DataSource',
                        return_value=fake_parser_class_instance)

    # Act
    cut.parse_data(arg_parser_file_name, arg_dataFile,
                   arg_metadataFile, arg_subsystems_breakdown)

    # Assert
    assert execution_engine.importlib.util.spec_from_file_location.call_count == 1
    assert execution_engine.importlib.util.spec_from_file_location.call_args_list[0].args == (
        'data_source', arg_parser_file_name, )
    assert execution_engine.importlib.util.module_from_spec.call_count == 1
    assert execution_engine.importlib.util.module_from_spec.call_args_list[0].args == (
        fake_spec, )
    assert fake_spec.loader.exec_module.call_count == 1
    assert fake_module.DataSource.call_count == 1
    assert fake_module.DataSource.call_args_list[0].args == (
        arg_dataFile, arg_metadataFile, arg_subsystems_breakdown, )
    assert cut.simDataSource == fake_parser_class_instance


def test_ExecutionEngine_parse_data_argument_subsystems_breakdown_optional_default_is_False(mocker):
    # Arrange
    arg_parser_file_name = MagicMock()
    arg_dataFile = MagicMock()
    arg_metadataFile = str(MagicMock())

    class FakeParser:
        init_data_file = None
        init_meta_data_file = None
        init_subsystems_breakdown = None

        def __init__(self, data_file, meta_file, subsystems_breakdown):
            FakeParser.init_data_file = data_file
            FakeParser.init_meta_data_file = meta_file
            FakeParser.init_subsystems_breakdown = subsystems_breakdown

    fake_spec = MagicMock()
    fake_module = MagicMock()
    fake_parser_class_instance = MagicMock()

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ +
                 '.importlib.util.spec_from_file_location', return_value=fake_spec)
    mocker.patch(execution_engine.__name__ +
                 '.importlib.util.module_from_spec', return_value=fake_module)
    mocker.patch.object(fake_spec, '.loader.exec_module', return_value=None)
    mocker.patch.object(fake_module, '.DataSource',
                        return_value=fake_parser_class_instance)

    # Act
    cut.parse_data(arg_parser_file_name, arg_dataFile, arg_metadataFile)

    # Assert
    assert fake_module.DataSource.call_count == 1
    assert fake_module.DataSource.call_args_list[0].args == (
        arg_dataFile, arg_metadataFile, False, )

# setup_sim tests


def test_ExecutionEngine_setup_sim_sets_self_sim_to_new_Simulator(mocker):
    # Arrange
    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.simDataSource = MagicMock()
    cut.knowledge_rep_plugin_dict = MagicMock()
    cut.learners_plugin_dict = MagicMock()
    cut.planners_plugin_dict = MagicMock()
    cut.complex_plugin_dict = MagicMock()

    fake_sim = MagicMock()

    mocker.patch(execution_engine.__name__ +
                 '.Simulator', return_value=fake_sim)

    # Act
    cut.setup_sim()

    # Assert
    assert execution_engine.Simulator.call_count == 1
    assert execution_engine.Simulator.call_args_list[0].args == (cut.simDataSource,
                                                                 cut.knowledge_rep_plugin_dict,
                                                                 cut.learners_plugin_dict,
                                                                 cut.planners_plugin_dict,
                                                                 cut.complex_plugin_dict)
    assert cut.sim == fake_sim

# run_sim tests


def test_ExecutionEngine_run_sim_runs_but_does_not_save_results_when_save_flag_is_False(mocker):
    # Arrange
    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.sim = MagicMock()
    cut.IO_Enabled = MagicMock()
    cut.save_flag = False

    mocker.patch.object(cut.sim, 'run_sim')
    mocker.patch.object(cut, 'save_results')

    # Act
    cut.run_sim()

    # Assert
    assert cut.sim.run_sim.call_count == 1
    assert cut.sim.run_sim.call_args_list[0].args == (cut.IO_Enabled, )
    assert cut.save_results.call_count == 0


def test_ExecutionEngine_run_sim_runs_and_saves_results_when_save_flag_is_True(mocker):
    # Arrange
    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.sim = MagicMock()
    cut.IO_Enabled = MagicMock()
    cut.save_flag = True
    cut.save_name = MagicMock()

    mocker.patch.object(cut.sim, 'run_sim')
    mocker.patch.object(cut, 'save_results')

    # Act
    cut.run_sim()

    # Assert
    assert cut.sim.run_sim.call_count == 1
    assert cut.sim.run_sim.call_args_list[0].args == (cut.IO_Enabled, )
    assert cut.save_results.call_count == 1
    assert cut.save_results.call_args_list[0].args == (cut.save_name, )

# init_save_paths tests


def test_ExecutionEngine_init_save_paths_makes_tmp_and_models_and_diagnosis_directories_and_adds_them_to_os_environ(mocker):
    # Arrange
    fake_save_path = str(MagicMock())
    fake_environ = {'RESULTS_PATH': fake_save_path}
    fake_tmp_save_path = str(MagicMock())
    fake_tmp_models_path = str(MagicMock())
    fake_tmp_diagnosis_path = str(MagicMock())

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ + '.os.path.join',
                 side_effect=[fake_tmp_save_path, fake_tmp_models_path, fake_tmp_diagnosis_path])
    mocker.patch.object(cut, 'delete_save_paths')
    mocker.patch(execution_engine.__name__ + '.os.mkdir')

    # Act
    cut.init_save_paths()

    # Assert
    # NOTE: assert execution_engine.os.path.join.call_count must assert correctly or there are odd errors? Is this due to using side_effect instead of return_value?
    assert execution_engine.os.path.join.call_count == 3
    # NOTE: similar problem with the args lists, bad expected values do not error nicely with good outputs, so beware but correct values pass
    assert execution_engine.os.path.join.call_args_list[0].args == (
        fake_save_path, 'tmp')
    assert execution_engine.os.path.join.call_args_list[1].args == (
        fake_tmp_save_path, 'models')
    assert execution_engine.os.path.join.call_args_list[2].args == (
        fake_tmp_save_path, 'diagnosis')
    # NOTE: apparently the problem persists to other failures because these asserts have the same problem, bad values error, but not correct outputs, good values pass
    assert execution_engine.os.environ['ONAIR_SAVE_PATH'] == fake_save_path
    assert execution_engine.os.environ['ONAIR_TMP_SAVE_PATH'] == fake_tmp_save_path
    assert execution_engine.os.environ['ONAIR_MODELS_SAVE_PATH'] == fake_tmp_models_path
    assert execution_engine.os.environ['ONAIR_DIAGNOSIS_SAVE_PATH'] == fake_tmp_diagnosis_path

# delete_save_path tests


def test_ExecutionEngine_delete_save_paths_does_nothing_when_save_path_has_no_tmp_dir(mocker):
    # Arrange
    fake_save_path = str(MagicMock())
    fake_environ = {'RESULTS_PATH': fake_save_path}
    fake_dirs = []

    for i in range(pytest.gen.randint(0, 5)):  # 0 to 5
        fake_dirs.append(str(MagicMock()))

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ +
                 '.os.listdir', return_value=fake_dirs)
    mocker.patch(execution_engine.__name__ + '.shutil.rmtree')

    # Act
    cut.delete_save_paths()

    # Assert
    assert execution_engine.os.listdir.call_count == 1
    assert execution_engine.os.listdir.call_args_list[0].args == (
        fake_save_path, )
    assert execution_engine.shutil.rmtree.call_count == 0


def test_ExecutionEngine_delete_save_paths_removes_tmp_tree_when_it_exists(mocker):
    # Arrange
    fake_save_path = str(MagicMock())
    fake_environ = {'RESULTS_PATH': fake_save_path}
    fake_dirs = []

    for i in range(pytest.gen.randint(0, 5)):  # 0 to 5
        fake_dirs.append(str(MagicMock()))
    fake_dirs.append('tmp')
    for i in range(pytest.gen.randint(0, 5)):  # 0 to 5
        fake_dirs.append(str(MagicMock()))

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ +
                 '.os.listdir', return_value=fake_dirs)
    mocker.patch(execution_engine.__name__ + '.shutil.rmtree')
    mocker.patch(execution_engine.__name__ + '.print')

    # Act
    cut.delete_save_paths()

    # Assert
    assert execution_engine.os.listdir.call_count == 1
    assert execution_engine.os.listdir.call_args_list[0].args == (
        fake_save_path, )
    assert execution_engine.shutil.rmtree.call_count == 1
    assert execution_engine.shutil.rmtree.call_args_list[0].args == (
        fake_save_path + '/tmp', )
    assert execution_engine.print.call_count == 0


def test_ExecutionEngine_delete_save_paths_prints_error_message_when_rmtree_raises_OSError(mocker):
    # Arrange
    fake_save_path = str(MagicMock())
    fake_environ = {'RESULTS_PATH': fake_save_path}
    fake_dirs = []
    fake_error_message = str(MagicMock())

    for i in range(pytest.gen.randint(0, 5)):  # 0 to 5
        fake_dirs.append(str(MagicMock()))
    fake_dirs.append('tmp')
    for i in range(pytest.gen.randint(0, 5)):  # 0 to 5
        fake_dirs.append(str(MagicMock()))

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ +
                 '.os.listdir', return_value=fake_dirs)
    mocker.patch(execution_engine.__name__ + '.shutil.rmtree', side_effect=OSError(
        # 0 to 10 arbitrary error value for errno
        pytest.gen.randint(0, 10), fake_error_message))
    mocker.patch(execution_engine.__name__ + '.print')

    # Act
    cut.delete_save_paths()

    # Assert
    assert execution_engine.os.listdir.call_count == 1
    assert execution_engine.os.listdir.call_args_list[0].args == (
        fake_save_path, )
    assert execution_engine.shutil.rmtree.call_count == 1
    assert execution_engine.shutil.rmtree.call_args_list[0].args == (
        fake_save_path + '/tmp', )
    assert execution_engine.print.call_count == 1
    assert execution_engine.print.call_args_list[0].args == (
        ("Error: " + fake_save_path + " : " + fake_error_message), )

# save_results tests


def test_ExecutionEngine_save_results_creates_expected_save_path_and_copies_proper_tree_to_it(mocker):
    # Arrange
    arg_save_name = str(MagicMock())

    fake_gmtime = str(MagicMock())
    fake_complete_time = str(MagicMock())
    fake_onair_save_path = str(MagicMock())
    fake_onair_tmp_save_path = str(MagicMock())
    fake_environ = {'ONAIR_SAVE_PATH': fake_onair_save_path,
                    'ONAIR_TMP_SAVE_PATH': fake_onair_tmp_save_path}
    fake_save_path = fake_onair_save_path + 'saved/' + \
        arg_save_name + '_' + fake_complete_time

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ +
                 '.gmtime', return_value=fake_gmtime)
    mocker.patch(execution_engine.__name__ + '.strftime',
                 return_value=fake_complete_time)
    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ + '.os.makedirs')
    mocker.patch(execution_engine.__name__ + '.copy_tree')

    # Act
    cut.save_results(arg_save_name)

    # Assert
    assert execution_engine.gmtime.call_count == 1
    assert execution_engine.gmtime.call_args_list[0].args == ()
    assert execution_engine.strftime.call_count == 1
    assert execution_engine.strftime.call_args_list[0].args == (
        "%H-%M-%S", fake_gmtime,)
    assert execution_engine.os.makedirs.call_count == 1
    assert execution_engine.os.makedirs.call_args_list[0].args == (
        fake_save_path, )
    assert execution_engine.os.makedirs.call_args_list[0].kwargs == {
        "exist_ok": True}
    assert execution_engine.copy_tree.call_count == 1
    assert execution_engine.copy_tree.call_args_list[0].args == (
        fake_onair_tmp_save_path, fake_save_path, )

# set_run_param tests


def test_ExecutionEngine_set_run_param_passes_given_arguments_to_setattr(mocker):
    # Arrange
    arg_name = MagicMock()
    arg_val = MagicMock()
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.setattr')

    # Act
    cut.set_run_param(arg_name, arg_val)

    # Assert
    assert execution_engine.setattr.call_count == 1
    assert execution_engine.setattr.call_args_list[0].args == (
        cut, arg_name, arg_val, )

# ast_parse_eval tests


def test_ExecutionEngine_ast_parse_eval_returns_call_to_ast_parse_with_mode_eval(mocker):
    # Arrange
    arg_config_list = MagicMock()
    expected_result = MagicMock()
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + ".ast.parse",
                 return_value=expected_result)

    # Act
    result = cut.ast_parse_eval(arg_config_list)

    # Assert
    assert result == expected_result
    assert execution_engine.ast.parse.call_count == 1
    assert execution_engine.ast.parse.call_args_list[0].args == (
        arg_config_list, )
    assert execution_engine.ast.parse.call_args_list[0].kwargs == {
        'mode': 'eval'}
