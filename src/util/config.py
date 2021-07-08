import configparser
import os


default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../test/config/default_config.ini')
CONFIG_PATH = os.environ.get('CONFIG_PATH', default_config_path) 

def load_config():
    """
    Loads config from env variable 'CONFIG_PATH'; if not defined, loads default_config.ini
    :return: (config object) parsed config
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return config

CONFIG = load_config()

def get_config():
    """
    :return: (config object) returns configparser object parsed from command line path
    """
    return CONFIG