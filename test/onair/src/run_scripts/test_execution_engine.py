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
from mock import MagicMock

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
    assert cut.IO_Flag == False
    assert cut.Dev_Flag == False
    assert cut.Viz_Flag == False
    assert cut.dataFilePath == ''
    assert cut.telemetryFile == ''
    assert cut.fullTelemetryFileName == ''
    assert cut.metadataFilePath == ''
    assert cut.metaFile == ''
    assert cut.fullMetaDataFileName == ''
    assert cut.benchmarkFilePath == ''
    assert cut.benchmarkFiles == ''
    assert cut.benchmarkIndices == ''
    assert cut.parser_file_name == ''
    assert cut.parser_name == ''
    assert cut.sim_name == ''
    assert cut.simDataParser == None
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
    assert cut.parse_data.call_args_list[0].args == (cut.parser_name, cut.parser_file_name, cut.dataFilePath, cut.metadataFilePath, )
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

    mocker.patch(execution_engine.__name__ + '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read', return_value=fake_config_read_result)
   
    # Act
    with pytest.raises(FileNotFoundError) as e_info:
        cut.parse_configs(arg_bad_config_filepath)

    # Assert
    assert e_info.match(f"Config file at '{arg_bad_config_filepath}' could not be read.")

def test_ExecutionEngine_parse_configs_raises_KeyError_with_config_file_info_when_a_required_key_is_not_in_config(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    fake_default = {'TelemetryDataFilePath':MagicMock(),
                    'TelemetryFile':MagicMock(),
                    'TelemetryMetadataFilePath':MagicMock(),
                    'MetaFile':MagicMock(),
                    'BenchmarkFilePath':MagicMock(),
                    'BenchmarkFiles':MagicMock(),
                    'BenchmarkIndices':MagicMock(),
                    'ParserFileName':MagicMock(),
                    'ParserName':MagicMock(),
                    'SimName':MagicMock(),
                    'PluginList':MagicMock()
                    }
    required_keys = [item for item in list(fake_default.keys()) if 'Benchmark' not in item]
    missing_key = pytest.gen.choice(required_keys)
    del fake_default[missing_key]
    fake_run_flags = MagicMock()
    fake_dict_for_Config = {'DEFAULT':fake_default, 'RUN_FLAGS':fake_run_flags}
    fake_config = MagicMock()
    fake_config.__getitem__.side_effect = fake_dict_for_Config.__getitem__
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read', return_value=fake_config_read_result)

    # Act
    with pytest.raises(KeyError) as e_info:
        cut.parse_configs(arg_config_filepath)

    # Assert
    assert e_info.match(f"Config file: '{arg_config_filepath}', missing key: {missing_key}")

def test_ExecutionEngine_parse_configs_raises_ValueError_when_PluginList_from_config_is_not_dict(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    fake_config = MagicMock()
    fake_paths_and_filenames = str(MagicMock)
    fake_plugin_list = MagicMock()
    fake_plugin_list.body = MagicMock()
    fake_default_item = MagicMock()
    fake_config.__getitem__.return_value = fake_default_item
    fake_default_item.__getitem__.side_effect = [fake_paths_and_filenames] * 4 + [None] * 3 + [fake_plugin_list]
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read', return_value=fake_config_read_result)
    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_list)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=False)

    # Act
    with pytest.raises(ValueError) as e_info:
        cut.parse_configs(arg_config_filepath)

    # Assert
    assert e_info.match(f"{fake_plugin_list} is an invalid PluginList. It must be a dict of at least 1 key/value pair.")
    assert cut.ast_parse_eval.call_count == 1
    assert cut.ast_parse_eval.call_args_list[0].args == (fake_plugin_list,)
    assert execution_engine.isinstance.call_count == 1
    assert execution_engine.isinstance.call_args_list[0].args == (fake_plugin_list.body, execution_engine.ast.Dict, )

def test_ExecutionEngine_parse_configs_raises_ValueError_when_PluginList_from_config_is_empty_dict(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    fake_config = MagicMock()
    fake_paths_and_filenames = str(MagicMock)
    fake_plugin_list = MagicMock()
    fake_plugin_list.body = MagicMock()
    fake_plugin_list.body.keys = MagicMock()
    fake_plugin_list.body.keys.__len__.return_value = 0
    fake_default_item = MagicMock()
    fake_config.__getitem__.return_value = fake_default_item
    fake_default_item.__getitem__.side_effect = [fake_paths_and_filenames] * 4 + [None] * 3 + [fake_plugin_list]
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1


    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read', return_value=fake_config_read_result)
    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_list)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)

    # Act
    with pytest.raises(ValueError) as e_info:
        cut.parse_configs(arg_config_filepath)

    # Assert
    assert e_info.match(f"{fake_plugin_list} is an invalid PluginList. It must be a dict of at least 1 key/value pair.")
    assert cut.ast_parse_eval.call_count == 1
    assert cut.ast_parse_eval.call_args_list[0].args == (fake_plugin_list,)
    assert execution_engine.isinstance.call_count == 1
    assert execution_engine.isinstance.call_args_list[0].args == (fake_plugin_list.body, execution_engine.ast.Dict, )

def test_ExecutionEngine_parse_configs_raises_FileNotFoundError_when_given_plugin_path_is_not_valid(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    fake_config = MagicMock()
    fake_paths_and_filenames = str(MagicMock)
    fake_plugin_list = MagicMock()
    fake_plugin_list.body = MagicMock()
    fake_plugin_list.body.keys = MagicMock()
    fake_plugin_list.body.keys.__len__.return_value = 1
    fake_temp_plugin_list = MagicMock()
    fake_plugin_name = MagicMock()
    fake_temp_iter = iter([fake_plugin_name])
    fake_default_item = MagicMock()
    fake_config.__getitem__.return_value = fake_default_item
    fake_default_item.__getitem__.side_effect = [fake_paths_and_filenames] * 4 + [None] * 3 + [fake_plugin_list]
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1


    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read', return_value=fake_config_read_result)
    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_list)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)
    mocker.patch(execution_engine.__name__ + '.ast.literal_eval', return_value=fake_temp_plugin_list)
    mocker.patch.object(fake_temp_plugin_list, 'values', return_value=fake_temp_iter)
    mocker.patch(execution_engine.__name__ + '.os.path.exists', return_value=False)
    # Act
    with pytest.raises(FileNotFoundError) as e_info:
        cut.parse_configs(arg_config_filepath)

    # Assert
    assert e_info.match(f"In config file '{arg_config_filepath}', path '{fake_plugin_name}' does not exist or is formatted incorrectly.")
    assert cut.ast_parse_eval.call_count == 1
    assert cut.ast_parse_eval.call_args_list[0].args == (fake_plugin_list,)
    assert execution_engine.isinstance.call_count == 1
    assert execution_engine.isinstance.call_args_list[0].args == (fake_plugin_list.body, execution_engine.ast.Dict, )


def test_ExecutionEngine_parse_configs_sets_all_items_without_error(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    fake_default = {'TelemetryDataFilePath':MagicMock(),
                    'TelemetryFile':MagicMock(),
                    'TelemetryMetadataFilePath':MagicMock(),
                    'MetaFile':MagicMock(),
                    'BenchmarkFilePath':MagicMock(),
                    'BenchmarkFiles':MagicMock(),
                    'BenchmarkIndices':MagicMock(),
                    'ParserFileName':MagicMock(),
                    'ParserName':MagicMock(),
                    'SimName':MagicMock(),
                    'PluginList':"{fake_name:fake_path}"
                    }
    fake_run_flags = MagicMock()
    fake_plugin_list = MagicMock()
    fake_plugin_list.body = MagicMock()
    fake_plugin_list.body.keys = MagicMock()
    fake_plugin_list.body.keys.__len__.return_value = 1
    fake_temp_plugin_list = MagicMock()
    fake_plugin_name = MagicMock()
    fake_temp_iter = iter([fake_plugin_name])
    fake_dict_for_Config = {'DEFAULT':fake_default, 'RUN_FLAGS':fake_run_flags}
    fake_config = MagicMock()
    fake_config.__getitem__.side_effect = fake_dict_for_Config.__getitem__
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1
    fake_IO_flags = MagicMock()
    fake_Dev_flags = MagicMock()
    fake_Viz_flags = MagicMock()
    fake_plugin_dict= MagicMock()
    fake_keys = MagicMock()
    fake_plugin = MagicMock()
    fake_path = MagicMock()

    fake_keys.__len__.return_value = 1
    fake_keys.__iter__.return_value = iter([str(fake_plugin)])
    
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read', return_value=fake_config_read_result)
    mocker.patch.object(fake_run_flags, 'getboolean', side_effect=[fake_IO_flags, fake_Dev_flags, fake_Viz_flags])
    mocker.patch.object(cut, 'ast_parse_eval', return_value=fake_plugin_list)
    mocker.patch(execution_engine.__name__ + '.isinstance', return_value=True)
    mocker.patch(execution_engine.__name__ + '.ast.literal_eval', return_value=fake_temp_plugin_list)
    mocker.patch.object(fake_temp_plugin_list, 'values', return_value=fake_temp_iter)
    mocker.patch(execution_engine.__name__ + '.os.path.exists', return_value=True)
    mocker.patch.object(fake_plugin_dict, 'keys', return_value=fake_keys)
    mocker.patch.object(fake_plugin_dict, '__getitem__', return_value=fake_path)

    # Act
    cut.parse_configs(arg_config_filepath)

    # Assert
    assert execution_engine.configparser.ConfigParser.call_count == 1
    assert fake_config.read.call_count == 1
    assert cut.dataFilePath == fake_default['TelemetryDataFilePath']
    assert cut.telemetryFile == fake_default['TelemetryFile']
    assert cut.metadataFilePath == fake_default['TelemetryMetadataFilePath']
    assert cut.metaFile == fake_default['MetaFile']
    assert cut.benchmarkFilePath == fake_default['BenchmarkFilePath']
    assert cut.benchmarkFiles == fake_default['BenchmarkFiles']
    assert cut.benchmarkIndices == fake_default['BenchmarkIndices']
    assert cut.parser_file_name == fake_default['ParserFileName']
    assert cut.parser_name == fake_default['ParserName']
    assert cut.sim_name == fake_default['SimName']
    assert cut.plugin_list == fake_temp_plugin_list
    assert fake_run_flags.getboolean.call_count == 3
    assert fake_run_flags.getboolean.call_args_list[0].args == ('IO_Flag', )
    assert cut.IO_Flag == fake_IO_flags
    assert fake_run_flags.getboolean.call_args_list[1].args == ('Dev_Flag', )
    assert cut.Dev_Flag == fake_Dev_flags
    assert fake_run_flags.getboolean.call_args_list[2].args == ('Viz_Flag', )
    assert cut.Viz_Flag == fake_Viz_flags

def test_ExecutionEngine_parse_configs_bypasses_benchmarks_when_access_raises_error(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    # NOTE: not including the benchmark strings causes the exception
    fake_default = {'TelemetryDataFilePath':MagicMock(),
                    'TelemetryFile':MagicMock(),
                    'TelemetryMetadataFilePath':MagicMock(),
                    'MetaFile':MagicMock(),
                    'ParserFileName':MagicMock(),
                    'ParserName':MagicMock(),
                    'SimName':MagicMock(),
                    'PluginList':"{fake_name:fake_path}"
                    }
    fake_run_flags = MagicMock()
    fake_dict_for_Config = {'DEFAULT':fake_default, 'RUN_FLAGS':fake_run_flags}
    fake_config = MagicMock()
    fake_config.__getitem__.side_effect = fake_dict_for_Config.__getitem__
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1
    fake_IO_flags = MagicMock()
    fake_Dev_flags = MagicMock()
    fake_Viz_flags = MagicMock()
    fake_plugin_dict = MagicMock()
    fake_keys = MagicMock()
    fake_plugin = MagicMock()
    fake_path = MagicMock()

    fake_keys.__len__.return_value = 1
    fake_keys.__iter__.return_value = iter([str(fake_plugin)])
    
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read', return_value=fake_config_read_result)
    mocker.patch.object(fake_run_flags, 'getboolean', side_effect=[fake_IO_flags, fake_Dev_flags, fake_Viz_flags])
    mocker.patch('ast.literal_eval',return_value=fake_plugin_dict)
    mocker.patch.object(fake_plugin_dict, 'keys', return_value=fake_keys)
    mocker.patch.object(fake_plugin_dict, '__getitem__', return_value=fake_path)
    mocker.patch('os.path.exists', return_value=True)


    # Act
    cut.parse_configs(arg_config_filepath)

    # Assert
    assert hasattr(cut, 'benchmarkFilePath') == False
    assert hasattr(cut, 'benchmarkFiles') == False
    assert hasattr(cut, 'benchmarkIndices') == False

def test_ExecutionEngine_parse_configs_raises_KeyError_with_config_file_info_when_a_required_key_is_not_in_config(mocker):
    # Arrange
    arg_config_filepath = MagicMock()

    fake_default = {'TelemetryDataFilePath':MagicMock(),
                    'TelemetryFile':MagicMock(),
                    'TelemetryMetadataFilePath':MagicMock(),
                    'MetaFile':MagicMock(),
                    'BenchmarkFilePath':MagicMock(),
                    'BenchmarkFiles':MagicMock(),
                    'BenchmarkIndices':MagicMock(),
                    'ParserFileName':MagicMock(),
                    'ParserName':MagicMock(),
                    'SimName':MagicMock(),
                    }
    required_keys = [item for item in list(fake_default.keys()) if 'Benchmark' not in item]
    missing_key = pytest.gen.choice(required_keys)
    del fake_default[missing_key]
    fake_run_flags = MagicMock()
    fake_dict_for_Config = {'DEFAULT':fake_default, 'RUN_FLAGS':fake_run_flags}
    fake_config = MagicMock()
    fake_config.__getitem__.side_effect = fake_dict_for_Config.__getitem__
    fake_config_read_result = MagicMock()
    fake_config_read_result.__len__.return_value = 1
    
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.configparser.ConfigParser', return_value=fake_config)
    mocker.patch.object(fake_config, 'read', return_value=fake_config_read_result)

    # Act
    with pytest.raises(KeyError) as e_info:
        cut.parse_configs(arg_config_filepath)

    # Assert
    assert e_info.match(f"Config file: '{arg_config_filepath}', missing key: {missing_key}")

# parse_data tests
def test_ExecutionEngine_parse_data_sets_the_simDataParser_to_the_data_parser(mocker):
    # Arrange
    arg_parser_name = MagicMock()
    arg_parser_file_name = MagicMock()
    arg_dataFile = str(MagicMock())
    arg_metadataFile = str(MagicMock())
    arg_subsystems_breakdown = MagicMock()

    class FakeParser:
        def __init__(self, data_file, metadata_file, subsystems_breakdown):
            pass

    fake_parser = MagicMock()
    fake_parser_class = FakeParser
    fake_parser_class_instance = MagicMock()
    fake_parsed_data = MagicMock()

    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.telemetryFile = MagicMock()
    cut.metaFile = MagicMock()

    mocker.patch(execution_engine.__name__ + '.importlib.import_module', return_value=fake_parser)
    mocker.patch(execution_engine.__name__ + '.getattr', return_value=fake_parser_class)
    mocker.patch.object(fake_parser_class, '__new__', return_value=fake_parser_class_instance)

    # Act
    cut.parse_data(arg_parser_name, arg_parser_file_name, arg_dataFile, arg_metadataFile, arg_subsystems_breakdown)

    # Assert
    assert execution_engine.importlib.import_module.call_count == 1
    assert execution_engine.importlib.import_module.call_args_list[0].args == ('data_handling.parsers.' + arg_parser_file_name, )
    assert execution_engine.getattr.call_count == 1
    assert execution_engine.getattr.call_args_list[0].args == (fake_parser, arg_parser_name,)
    assert cut.simDataParser == fake_parser_class_instance
    assert fake_parser_class.__new__.call_count == 1
    assert fake_parser_class.__new__.call_args_list[0].args == (fake_parser_class, arg_dataFile, arg_metadataFile, arg_subsystems_breakdown, )

    # subsystems_breakdown

def test_ExecutionEngine_parse_data_argument_subsystems_breakdown_optional_default_is_False(mocker):
    # Arrange
    arg_parser_name = MagicMock()
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

    fake_parser = MagicMock()
    fake_parser_class = FakeParser
    fake_run_path = str(MagicMock())
    fake_environ = {'RUN_PATH':fake_run_path}
    fake_parsed_data = MagicMock()
    fake_processdSimData = MagicMock()

    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.telemetryFile = MagicMock()
    cut.metaFile = MagicMock()

    mocker.patch(execution_engine.__name__ + '.importlib.import_module', return_value=fake_parser)
    mocker.patch(execution_engine.__name__ + '.getattr', return_value=fake_parser_class)
    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)

    # Act
    cut.parse_data(arg_parser_name, arg_parser_file_name, arg_dataFile, arg_metadataFile)

    # Assert
    assert FakeParser.init_subsystems_breakdown == False

# setup_sim tests
def test_ExecutionEngine_setup_sim_sets_self_sim_to_new_Simulator_and_sets_benchmark_data_when_no_exceptions_are_encountered(mocker):
    # Arrange
    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.sim_name = MagicMock()
    cut.simDataParser = MagicMock()
    cut.benchmarkFiles = MagicMock()
    cut.benchmarkFilePath = MagicMock()
    cut.benchmarkIndices = MagicMock()
    cut.plugin_list = MagicMock()

    fake_sim = MagicMock()
    fake_fls = MagicMock()
    fake_fp = str(MagicMock())
    fake_bi = MagicMock()
    fake__file__ = str(MagicMock())

    print(fake__file__)

    mocker.patch(execution_engine.__name__ + '.Simulator', return_value=fake_sim)
    mocker.patch(execution_engine.__name__ + '.ast.literal_eval', side_effect=[fake_fls, fake_bi])
    mocker.patch(execution_engine.__name__ + '.__file__', fake__file__)
    mocker.patch(execution_engine.__name__ + '.os.path.realpath', return_value=fake_fp)
    mocker.patch(execution_engine.__name__ + '.os.path.dirname', return_value=fake_fp)
    mocker.patch.object(fake_sim, 'set_benchmark_data')

    # Act
    cut.setup_sim()

    # Assert
    assert execution_engine.Simulator.call_count == 1
    assert execution_engine.Simulator.call_args_list[0].args == (cut.sim_name, cut.simDataParser, cut.plugin_list)
    assert cut.sim == fake_sim
    assert execution_engine.ast.literal_eval.call_count == 2
    assert execution_engine.ast.literal_eval.call_args_list[0].args == (cut.benchmarkFiles, )
    assert execution_engine.ast.literal_eval.call_args_list[1].args == (cut.benchmarkIndices, )
    assert execution_engine.os.path.realpath.call_count == 1
    assert execution_engine.os.path.realpath.call_args_list[0].args == (fake__file__, )
    assert fake_sim.set_benchmark_data.call_count == 1
    assert fake_sim.set_benchmark_data.call_args_list[0].args == (fake_fp + '/../..' + cut.benchmarkFilePath, fake_fls, fake_bi, )

def test_ExecutionEngine_setup_sim_sets_self_sim_to_new_Simulator_but_does_not_set_bencmark_data_because_exception_is_encountered(mocker):
    # Arrange
    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.sim_name = MagicMock()
    cut.simDataParser = MagicMock()
    cut.benchmarkFiles = MagicMock()
    cut.benchmarkFilePath = MagicMock()
    cut.benchmarkIndices = MagicMock()
    cut.plugin_list = MagicMock()

    fake_sim = MagicMock()
    fake_fls = MagicMock()
    fake_fp = str(MagicMock())
    fake_bi = MagicMock()
    fake__file__ = str(MagicMock())

    print(fake__file__)

    mocker.patch(execution_engine.__name__ + '.Simulator', return_value=fake_sim)
    mocker.patch(execution_engine.__name__ + '.ast.literal_eval', side_effect=Exception)
    mocker.patch(execution_engine.__name__ + '.os.path.realpath')
    mocker.patch(execution_engine.__name__ + '.os.path.dirname')
    mocker.patch.object(fake_sim, 'set_benchmark_data')

    # Act
    cut.setup_sim()

    # Assert
    assert execution_engine.Simulator.call_count == 1
    assert execution_engine.Simulator.call_args_list[0].args == (cut.sim_name, cut.simDataParser,  cut.plugin_list)
    assert cut.sim == fake_sim
    assert execution_engine.ast.literal_eval.call_count == 1
    assert execution_engine.ast.literal_eval.call_args_list[0].args == (cut.benchmarkFiles, )
    assert execution_engine.os.path.realpath.call_count == 0
    assert fake_sim.set_benchmark_data.call_count == 0

# run_sim tests
def test_ExecutionEngine_run_sim_runs_but_does_not_save_results_when_save_flag_is_False(mocker):
    # Arrange
    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.sim = MagicMock()
    cut.IO_Flag = MagicMock()
    cut.Dev_Flag = MagicMock()
    cut.Viz_Flag = MagicMock()
    cut.save_flag = False

    mocker.patch.object(cut.sim, 'run_sim')
    mocker.patch.object(cut, 'save_results')

    # Act
    cut.run_sim()

    # Assert
    assert cut.sim.run_sim.call_count == 1
    assert cut.sim.run_sim.call_args_list[0].args == (cut.IO_Flag, cut.Dev_Flag, cut.Viz_Flag, )
    assert cut.save_results.call_count == 0

def test_ExecutionEngine_run_sim_runs_and_saves_results_when_save_flag_is_True(mocker):
    # Arrange
    cut = ExecutionEngine.__new__(ExecutionEngine)
    cut.sim = MagicMock()
    cut.IO_Flag = MagicMock()
    cut.Dev_Flag = MagicMock()
    cut.Viz_Flag = MagicMock()
    cut.save_flag = True
    cut.save_name = MagicMock()

    mocker.patch.object(cut.sim, 'run_sim')
    mocker.patch.object(cut, 'save_results')

    # Act
    cut.run_sim()

    # Assert
    assert cut.sim.run_sim.call_count == 1
    assert cut.sim.run_sim.call_args_list[0].args == (cut.IO_Flag, cut.Dev_Flag, cut.Viz_Flag, )
    assert cut.save_results.call_count == 1
    assert cut.save_results.call_args_list[0].args == (cut.save_name, )

# init_save_paths tests
def test_ExecutionEngine_init_save_paths_makes_tmp_and_models_and_diagnosis_directories_and_adds_them_to_os_environ(mocker):
    # Arrange
    fake_save_path = str(MagicMock())
    fake_environ = {'RESULTS_PATH':fake_save_path}
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
    assert execution_engine.os.path.join.call_args_list[0].args == (fake_save_path, 'tmp')
    assert execution_engine.os.path.join.call_args_list[1].args == (fake_tmp_save_path, 'models')
    assert execution_engine.os.path.join.call_args_list[2].args == (fake_tmp_save_path, 'diagnosis')
    # NOTE: apparently the problem persists to other failures because these asserts have the same problem, bad values error, but not correct outputs, good values pass
    assert execution_engine.os.environ['ONAIR_SAVE_PATH'] == fake_save_path
    assert execution_engine.os.environ['ONAIR_TMP_SAVE_PATH'] == fake_tmp_save_path
    assert execution_engine.os.environ['ONAIR_MODELS_SAVE_PATH'] == fake_tmp_models_path
    assert execution_engine.os.environ['ONAIR_DIAGNOSIS_SAVE_PATH'] == fake_tmp_diagnosis_path

# delete_save_path tests
def test_ExecutionEngine_delete_save_paths_does_nothing_when_save_path_has_no_tmp_dir(mocker):
    # Arrange
    fake_save_path = str(MagicMock())
    fake_environ = {'RESULTS_PATH':fake_save_path}
    fake_dirs = []

    for i in range(pytest.gen.randint(0, 5)): # 0 to 5
        fake_dirs.append(str(MagicMock()))

    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ + '.os.listdir', return_value=fake_dirs)
    mocker.patch(execution_engine.__name__ + '.shutil.rmtree')

    # Act
    cut.delete_save_paths()

    # Assert
    assert execution_engine.os.listdir.call_count == 1
    assert execution_engine.os.listdir.call_args_list[0].args == (fake_save_path, )
    assert execution_engine.shutil.rmtree.call_count == 0

def test_ExecutionEngine_delete_save_paths_removes_tmp_tree_when_it_exists(mocker):
    # Arrange
    fake_save_path = str(MagicMock())
    fake_environ = {'RESULTS_PATH':fake_save_path}
    fake_dirs = []

    for i in range(pytest.gen.randint(0, 5)): # 0 to 5
        fake_dirs.append(str(MagicMock()))
    fake_dirs.append('tmp')
    for i in range(pytest.gen.randint(0, 5)): # 0 to 5
        fake_dirs.append(str(MagicMock()))
        
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ + '.os.listdir', return_value=fake_dirs)
    mocker.patch(execution_engine.__name__ + '.shutil.rmtree')
    mocker.patch(execution_engine.__name__ + '.print')

    # Act
    cut.delete_save_paths()

    # Assert
    assert execution_engine.os.listdir.call_count == 1
    assert execution_engine.os.listdir.call_args_list[0].args == (fake_save_path, )
    assert execution_engine.shutil.rmtree.call_count == 1
    assert execution_engine.shutil.rmtree.call_args_list[0].args == (fake_save_path + '/tmp', )
    assert execution_engine.print.call_count == 0

def test_ExecutionEngine_delete_save_paths_prints_error_message_when_rmtree_raises_OSError(mocker):
    # Arrange
    fake_save_path = str(MagicMock())
    fake_environ = {'RESULTS_PATH':fake_save_path}
    fake_dirs = []
    fake_error_message = str(MagicMock())

    for i in range(pytest.gen.randint(0, 5)): # 0 to 5
        fake_dirs.append(str(MagicMock()))
    fake_dirs.append('tmp')
    for i in range(pytest.gen.randint(0, 5)): # 0 to 5
        fake_dirs.append(str(MagicMock()))
        
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ + '.os.listdir', return_value=fake_dirs)
    mocker.patch(execution_engine.__name__ + '.shutil.rmtree', side_effect=OSError(pytest.gen.randint(0,10),fake_error_message)) # 0 to 10 arbitrary error value for errno
    mocker.patch(execution_engine.__name__ + '.print')

    # Act
    cut.delete_save_paths()

    # Assert
    assert execution_engine.os.listdir.call_count == 1
    assert execution_engine.os.listdir.call_args_list[0].args == (fake_save_path, )
    assert execution_engine.shutil.rmtree.call_count == 1
    assert execution_engine.shutil.rmtree.call_args_list[0].args == (fake_save_path + '/tmp', )
    assert execution_engine.print.call_count == 1
    assert execution_engine.print.call_args_list[0].args == (("Error: " + fake_save_path + " : " + fake_error_message), )

# save_results tests
def test_ExecutionEngine_save_results_creates_expected_save_path_and_copies_proper_tree_to_it(mocker):
    # Arrange
    arg_save_name = str(MagicMock())

    fake_gmtime = str(MagicMock())
    fake_complete_time = str(MagicMock())
    fake_onair_save_path = str(MagicMock())
    fake_onair_tmp_save_path = str(MagicMock())
    fake_environ = {'ONAIR_SAVE_PATH':fake_onair_save_path, 'ONAIR_TMP_SAVE_PATH':fake_onair_tmp_save_path}
    fake_save_path = fake_onair_save_path + '/saved/' + arg_save_name + '_' + fake_complete_time
    
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + '.gmtime', return_value=fake_gmtime)
    mocker.patch(execution_engine.__name__ + '.strftime', return_value=fake_complete_time)
    mocker.patch.dict(execution_engine.__name__ + '.os.environ', fake_environ)
    mocker.patch(execution_engine.__name__ + '.os.mkdir')
    mocker.patch(execution_engine.__name__ + '.copy_tree')

    # Act
    cut.save_results(arg_save_name)

    # Assert
    assert execution_engine.gmtime.call_count == 1
    assert execution_engine.gmtime.call_args_list[0].args == ()
    assert execution_engine.strftime.call_count == 1
    assert execution_engine.strftime.call_args_list[0].args == ("%H-%M-%S", fake_gmtime,)
    assert execution_engine.os.mkdir.call_count == 1
    assert execution_engine.os.mkdir.call_args_list[0].args == (fake_save_path, )
    assert execution_engine.copy_tree.call_count == 1
    assert execution_engine.copy_tree.call_args_list[0].args == (fake_onair_tmp_save_path, fake_save_path, )
    
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
    assert execution_engine.setattr.call_args_list[0].args == (cut, arg_name, arg_val, )

# ast_parse_eval tests
def test_ExecutionEngine_ast_parse_eval_returns_call_to_ast_parse_with_mode_eval(mocker):
    # Arrange
    arg_config_list = MagicMock()
    expected_result = MagicMock()
    cut = ExecutionEngine.__new__(ExecutionEngine)

    mocker.patch(execution_engine.__name__ + ".ast.parse", return_value=expected_result)

    # Act
    result = cut.ast_parse_eval(arg_config_list)

    # Assert
    assert result == expected_result
    assert execution_engine.ast.parse.call_count == 1
    assert execution_engine.ast.parse.call_args_list[0].args == (arg_config_list, )
    assert execution_engine.ast.parse.call_args_list[0].kwargs == {'mode':'eval'}

