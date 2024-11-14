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
from unittest.mock import MagicMock, patch
from onair.services.service_manager import ServiceManager
import onair.services.service_manager as service_manager_import


def test_ServiceManager__init__raises_ValueError_when_service_dict_is_None_on_first_instantiation(
    mocker,
):
    # Arrange / Act
    with pytest.raises(ValueError) as e_info:
        ServiceManager()

    # Assert
    assert (
        str(e_info.value) == "'service_dict' parameter required on first instantiation"
    )


def test_ServiceManager__init__imports_services_and_sets_attributes(mocker):
    # Arrange
    fake_service_dict = {
        "service1": {"path": "path/to/service1"},
        "service2": {"path": "path/to/service2"},
    }
    fake_imported_services = {"service1": MagicMock(), "service2": MagicMock()}
    mocker.patch(
        "onair.services.service_manager.import_services",
        return_value=fake_imported_services,
    )

    # Act
    service_manager = ServiceManager(fake_service_dict)

    # Assert
    assert service_manager.service1 == fake_imported_services["service1"]
    assert service_manager.service2 == fake_imported_services["service2"]
    assert service_manager._initialized == True

    # Teardown
    del ServiceManager.instance


def test_ServiceManager__init__does_not_reinitialize_if_already_initialized(mocker):
    # Arrange
    fake_service_dict = {"service1": {"path": "path/to/service1"}}
    mocker.patch.object(ServiceManager, "_initialized", True, create=True)
    mock_import_services = mocker.patch(
        "onair.src.util.service_import.import_services"
    )  # called in __init__

    # Act
    ServiceManager(fake_service_dict)

    # Assert
    assert mock_import_services.call_count == 0

    # Teardown
    del ServiceManager.instance


def test_ServiceManager_get_services_returns_dict_of_services_and_their_functions(
    mocker,
):
    # Arrange
    class FakeService1:
        def func1(self):
            pass

        def _private_func(self):
            pass

    class FakeService2:
        def func2(self):
            pass

        def func3(self):
            pass

    service_manager = ServiceManager.__new__(ServiceManager)
    service_manager.service1 = FakeService1()
    service_manager.service2 = FakeService2()

    # Act
    result = service_manager.get_services()

    # Assert
    assert result == {
        "service1": {"func1"},  # correctly avoids _private_func
        "service2": {"func2", "func3"},
    }

    # Teardown
    del ServiceManager.instance


def test_ServiceManager_get_services_returns_empty_dict_when_no_services(mocker):
    # Arrange
    service_manager = ServiceManager.__new__(ServiceManager)

    # Act
    result = service_manager.get_services()

    # Assert
    assert result == {}

    # Teardown
    del ServiceManager.instance


def test_ServiceManager_get_services_returns_empty_dict_and_does_reach_second_for_loop_when_own_items_returns_only_internal_or_private_attributes(
    mocker,
):
    # Arrange
    service_manager = ServiceManager.__new__(ServiceManager)
    fake_vars_return = MagicMock()
    fake_internal_variable = MagicMock()

    mocker.patch(
        service_manager_import.__name__ + ".vars", return_value=fake_vars_return
    )
    mocker.patch(service_manager_import.__name__ + ".dir")
    fake_vars_return.items.return_value = iter([(fake_internal_variable, MagicMock())])
    fake_internal_variable.startswith.return_value = True

    # Act
    result = service_manager.get_services()

    # Assert
    assert result == {}
    assert fake_internal_variable.startswith.call_count == 1
    assert fake_internal_variable.startswith.call_args_list[0].args == ("_",)
    assert service_manager_import.dir.call_count == 0


def test_ServiceManager_behaves_as_singleton(mocker):
    # Arrange
    fake_service_dict1 = {"service1": "path1"}
    fake_imported_service = {"service1": MagicMock()}
    mocker.patch(
        "onair.services.service_manager.import_services",
        return_value=fake_imported_service,
    )

    # Act
    service_manager1 = ServiceManager(fake_service_dict1)
    service_manager2 = ServiceManager()

    # Assert
    assert service_manager1 is service_manager2
    assert hasattr(service_manager2, "service1")

    # Teardown
    del ServiceManager.instance
