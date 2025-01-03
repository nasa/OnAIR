# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"
#
# NOTE: For testing singleton-like classes, a teardown procedure must be implemented
# to delete the instance after every test. Otherwise, proceeding tests will have
# access to the last test's instance. This happens due to the nature of singletons,
# which have a single instance per global scope (which the tests are running in).
#

import pytest
from unittest.mock import MagicMock
import types
import sys
import string

fake_singleton = types.ModuleType("singleton")
fake_singleton.Singleton = object


@pytest.fixture
def ServiceManager():
    # Setup
    # create a fake Singleton for ServiceManager to inherit
    sys.modules["onair.src.util.singleton"] = fake_singleton
    if "onair.services.service_manager" in sys.modules:
        del sys.modules["onair.services.service_manager"]
    from onair.services.service_manager import ServiceManager

    yield ServiceManager
    # Teardown
    # Remove service manager and singleton for test isolation
    del sys.modules["onair.services.service_manager"]
    del sys.modules["onair.src.util.singleton"]


# special test
def test_ServiceManager_inherits_Singleton(ServiceManager):
    # Proves that ServiceManager not only inherits Singleton but also test replaced it
    assert ServiceManager.__base__ is fake_singleton.Singleton


# __init__ tests
def test_ServiceManager__init__raises_ValueError_when_service_dict_is_not_provided_on_first_instantiation(
    mocker, ServiceManager
):
    # Arrange / Act
    with pytest.raises(ValueError) as e_info:
        ServiceManager()

    # Assert
    assert (
        str(e_info.value) == "'service_dict' parameter required on first instantiation"
    )


def test_ServiceManager__init__raises_ValueError_when_service_dict_is_None_on_first_instantiation(
    mocker, ServiceManager
):
    # Arrange / Act
    with pytest.raises(ValueError) as e_info:
        ServiceManager(None)

    # Assert
    assert (
        str(e_info.value) == "'service_dict' parameter required on first instantiation"
    )


def test_ServiceManager__init__sets_services_to_empty_dict_when_import_services_returns_empty_dict(
    mocker, ServiceManager
):
    # Arrange
    arg_service_dict = MagicMock()
    fake_imported_services = {}
    mock_import_services = mocker.patch(
        ServiceManager.__module__ + ".import_services",
        return_value=fake_imported_services,
    )

    # Act
    result = ServiceManager(arg_service_dict)

    # Assert
    assert result.services == dict()
    assert mock_import_services.call_count == 1
    assert mock_import_services.call_args_list[0].args == (arg_service_dict,)


def test_ServiceManager__init__when_single_service_has_only_a_noncallable_function_sets_its_services_to_empty_set(
    mocker, ServiceManager
):
    # Arrange
    arg_service_dict = MagicMock()
    forced_return_import_services = {}

    fake_service_name = generate_random_string()
    fake_path_value = generate_random_string()
    fake_function_name = generate_random_string()
    fake_functions = [fake_function_name]
    fake_attr = MagicMock()

    forced_return_import_services[fake_service_name] = fake_path_value

    mock_import_services = mocker.patch(
        ServiceManager.__module__ + ".import_services",
        return_value=forced_return_import_services,
    )
    mock_setattr = mocker.patch(ServiceManager.__module__ + ".setattr")
    mock_dir = mocker.patch(
        ServiceManager.__module__ + ".dir", return_value=fake_functions
    )
    mock_getattr = mocker.patch(
        ServiceManager.__module__ + ".getattr", return_value=fake_attr
    )
    mock_callable = mocker.patch(
        ServiceManager.__module__ + ".callable", return_value=False
    )

    # Act
    result = ServiceManager(arg_service_dict)

    # Assert
    # assert result.services == forced_return_import_services
    assert mock_import_services.call_count == 1
    assert mock_import_services.call_args_list[0].args == (arg_service_dict,)
    assert mock_setattr.call_count == 1
    assert mock_setattr.call_args_list[0].args == (
        result,
        fake_service_name,
        fake_path_value,
    )
    assert mock_dir.call_count == 1
    assert mock_dir.call_args_list[0].args == (fake_path_value,)
    assert mock_getattr.call_count == 1
    assert mock_getattr.call_args_list[0].args == (fake_path_value, fake_function_name)
    assert mock_callable.call_count == 1
    assert mock_callable.call_args_list[0].args == (fake_attr,)
    assert result.services[fake_service_name] == set([])


def test_ServiceManager__init__when_single_service_has_only_a_callable_function_starting_with_underscore_sets_its_services_to_empty_set(
    mocker, ServiceManager
):
    # Arrange
    arg_service_dict = MagicMock()
    forced_return_import_services = {}

    fake_service_name = generate_random_string()
    fake_path_value = generate_random_string()
    fake_function_name = "_" + generate_random_string()
    fake_functions = [fake_function_name]
    fake_attr = MagicMock()

    forced_return_import_services[fake_service_name] = fake_path_value

    mock_import_services = mocker.patch(
        ServiceManager.__module__ + ".import_services",
        return_value=forced_return_import_services,
    )
    mock_setattr = mocker.patch(ServiceManager.__module__ + ".setattr")
    mock_dir = mocker.patch(
        ServiceManager.__module__ + ".dir", return_value=fake_functions
    )
    mock_getattr = mocker.patch(
        ServiceManager.__module__ + ".getattr", return_value=fake_attr
    )
    mock_callable = mocker.patch(
        ServiceManager.__module__ + ".callable", return_value=True
    )

    # Act
    result = ServiceManager(arg_service_dict)

    # Assert
    # assert result.services == forced_return_import_services
    assert mock_import_services.call_count == 1
    assert mock_import_services.call_args_list[0].args == (arg_service_dict,)
    assert mock_setattr.call_count == 1
    assert mock_setattr.call_args_list[0].args == (
        result,
        fake_service_name,
        fake_path_value,
    )
    assert mock_dir.call_count == 1
    assert mock_dir.call_args_list[0].args == (fake_path_value,)
    assert mock_getattr.call_count == 1
    assert mock_getattr.call_args_list[0].args == (fake_path_value, fake_function_name)
    assert mock_callable.call_count == 1
    assert mock_callable.call_args_list[0].args == (fake_attr,)
    assert result.services[fake_service_name] == set([])


def test_ServiceManager__init__when_single_service_has_only_a_callable_function_not_starting_in_underscore_sets_its_services_to_set_with_that_function(
    mocker, ServiceManager
):
    # Arrange
    arg_service_dict = MagicMock()
    forced_return_import_services = {}

    fake_service_name = generate_random_string()
    fake_path_value = generate_random_string()
    fake_function_name = generate_random_string()
    fake_functions = [fake_function_name]
    fake_attr = MagicMock()

    forced_return_import_services[fake_service_name] = fake_path_value

    mock_import_services = mocker.patch(
        ServiceManager.__module__ + ".import_services",
        return_value=forced_return_import_services,
    )
    mock_setattr = mocker.patch(ServiceManager.__module__ + ".setattr")
    mock_dir = mocker.patch(
        ServiceManager.__module__ + ".dir", return_value=fake_functions
    )
    mock_getattr = mocker.patch(
        ServiceManager.__module__ + ".getattr", return_value=fake_attr
    )
    mock_callable = mocker.patch(
        ServiceManager.__module__ + ".callable", return_value=True
    )

    # Act
    result = ServiceManager(arg_service_dict)

    # Assert
    # assert result.services == forced_return_import_services
    assert mock_import_services.call_count == 1
    assert mock_import_services.call_args_list[0].args == (arg_service_dict,)
    assert mock_setattr.call_count == 1
    assert mock_setattr.call_args_list[0].args == (
        result,
        fake_service_name,
        fake_path_value,
    )
    assert mock_dir.call_count == 1
    assert mock_dir.call_args_list[0].args == (fake_path_value,)
    assert mock_getattr.call_count == 1
    assert mock_getattr.call_args_list[0].args == (fake_path_value, fake_function_name)
    assert mock_callable.call_count == 1
    assert mock_callable.call_args_list[0].args == (fake_attr,)
    assert result.services[fake_service_name] == set([fake_function_name])


def generate_random_string(length=10):
    return "".join(pytest.gen.choice(string.ascii_lowercase) for _ in range(length))


def generate_random_functions():
    fake_non_callable_functions = [
        generate_random_string() for _ in range(pytest.gen.randint(1, 5))
    ]

    fake_callable_start_underscore_functions = [
        "_" + generate_random_string() for _ in range(pytest.gen.randint(1, 5))
    ]

    fake_callable_non_underscore_start_functions = [
        generate_random_string() for _ in range(pytest.gen.randint(1, 5))
    ]

    fake_callable_functions = (
        fake_callable_start_underscore_functions
        + fake_callable_non_underscore_start_functions
    )
    fake_functions = fake_non_callable_functions + fake_callable_functions
    pytest.gen.shuffle(fake_functions)

    return {
        "all_functions": fake_functions,
        "callable_non_underscore": fake_callable_non_underscore_start_functions,
    }


def test_ServiceManager__init__when_single_service_has_functions_its_services_are_set_to_only_callable_not_starting_with_underscore_functions(
    mocker, ServiceManager
):
    # Arrange
    arg_service_dict = MagicMock()
    forced_return_import_services = {}

    fake_service_name = generate_random_string()
    fake_path_value = generate_random_string()

    random_functions = generate_random_functions()
    fake_functions = random_functions["all_functions"]
    fake_callable_non_underscore_start_functions = random_functions[
        "callable_non_underscore"
    ]

    forced_return_import_services[fake_service_name] = fake_path_value

    def determine_attr_callability(_service, f):
        return f in fake_callable_non_underscore_start_functions

    def determine_callable(fake_attr_determination):
        return fake_attr_determination

    mock_import_services = mocker.patch(
        ServiceManager.__module__ + ".import_services",
        return_value=forced_return_import_services,
    )
    mock_setattr = mocker.patch(ServiceManager.__module__ + ".setattr")
    mock_dir = mocker.patch(
        ServiceManager.__module__ + ".dir", return_value=fake_functions
    )
    mock_getattr = mocker.patch(
        ServiceManager.__module__ + ".getattr", side_effect=determine_attr_callability
    )
    mock_callable = mocker.patch(
        ServiceManager.__module__ + ".callable", side_effect=determine_callable
    )

    # Act
    result = ServiceManager(arg_service_dict)

    # Assert
    assert mock_import_services.call_count == 1
    assert mock_import_services.call_args_list[0].args == (arg_service_dict,)
    assert mock_setattr.call_count == 1
    assert mock_setattr.call_args_list[0].args == (
        result,
        fake_service_name,
        fake_path_value,
    )
    assert mock_dir.call_count == 1
    assert mock_dir.call_args_list[0].args == (fake_path_value,)
    assert mock_getattr.call_count == len(fake_functions)
    for i in range(len(fake_functions)):
        assert mock_getattr.call_args_list[i].args == (
            fake_path_value,
            fake_functions[i],
        )
    assert mock_callable.call_count == len(fake_functions)
    for i in range(len(fake_functions)):
        assert mock_callable.call_args_list[i].args == (
            fake_functions[i] in fake_callable_non_underscore_start_functions,
        )
    assert result.services[fake_service_name] == set(
        fake_callable_non_underscore_start_functions
    )


def test_ServiceManager__init__when_multiple_services_all_services_set_to_only_callable_not_starting_with_underscore_functions(
    mocker, ServiceManager
):
    # Arrange
    arg_service_dict = MagicMock()
    forced_return_import_services = {}

    num_services = pytest.gen.randint(1, 10)  # arbitrary 1 to 10
    services_data = {}

    for _ in range(num_services):
        fake_service_name = generate_random_string()
        fake_path_value = generate_random_string()
        random_functions = generate_random_functions()
        fake_functions = random_functions["all_functions"]
        fake_callable_non_underscore_start_functions = random_functions[
            "callable_non_underscore"
        ]

        services_data[fake_service_name] = {
            "path": fake_path_value,
            "all_functions": fake_functions,
            "callable_non_underscore": fake_callable_non_underscore_start_functions,
        }

        forced_return_import_services[fake_service_name] = fake_path_value

    def determine_attr_callability(service, f):
        for service_name, data in services_data.items():
            if data["path"] == service:
                return f in data["callable_non_underscore"]
        return False

    def determine_callable(fake_attr_determination):
        return fake_attr_determination

    mock_import_services = mocker.patch(
        ServiceManager.__module__ + ".import_services",
        return_value=forced_return_import_services,
    )
    mock_setattr = mocker.patch(ServiceManager.__module__ + ".setattr")
    mock_dir = mocker.patch(
        ServiceManager.__module__ + ".dir",
        side_effect=lambda x: next(
            data["all_functions"]
            for data in services_data.values()
            if data["path"] == x
        ),
    )
    mock_getattr = mocker.patch(
        ServiceManager.__module__ + ".getattr", side_effect=determine_attr_callability
    )
    mock_callable = mocker.patch(
        ServiceManager.__module__ + ".callable", side_effect=determine_callable
    )

    # Act
    result = ServiceManager(arg_service_dict)

    # Assert
    assert mock_import_services.call_count == 1
    assert mock_import_services.call_args_list[0].args == (arg_service_dict,)

    # Check that setattr was called for each service in result.services
    assert mock_setattr.call_count == len(result.services)
    setattr_calls = {
        mock_setattr.call_args_list[i].args[1:] for i in range(mock_setattr.call_count)
    }
    for service_name, service_path in forced_return_import_services.items():
        if service_name in result.services:
            assert (service_name, service_path) in setattr_calls

    assert mock_dir.call_count == len(result.services)
    for i in range(mock_dir.call_count):
        assert mock_dir.call_args_list[i].args == (
            mock_setattr.call_args_list[i].args[2],
        )

    getattr_calls = set(
        mock_getattr.call_args_list[i].args for i in range(mock_getattr.call_count)
    )

    # Check that getattr was called for all functions
    for service_name, service_data in services_data.items():
        if service_name in result.services:
            for func in service_data["all_functions"]:
                assert (service_data["path"], func) in getattr_calls

    assert mock_callable.call_count == mock_getattr.call_count

    # Verify the final result
    for service_name, service_data in services_data.items():
        if service_name in result.services:
            assert result.services[service_name] == set(
                service_data["callable_non_underscore"]
            )


def test_ServiceManager__init__does_not_import_services_when_services_already_exist(
    mocker, ServiceManager
):
    # Arrange
    mock_hasattr = mocker.patch(
        ServiceManager.__module__ + ".hasattr", return_value=True
    )
    mock_import_services = mocker.patch(ServiceManager.__module__ + ".import_services")

    # Act
    result = ServiceManager()

    # Assert
    assert mock_hasattr.call_count == 1
    assert mock_hasattr.call_args_list[0].args == (result, "services")
    assert mock_import_services.call_count == 0


# get_services tests
def test_ServiceManager_get_services_returns_services_attribute(mocker, ServiceManager):
    # Arrange
    cut = ServiceManager.__new__(ServiceManager)
    fake_services = MagicMock()
    cut.services = fake_services

    # Act
    result = cut.get_services()

    # Assert
    assert result is fake_services
