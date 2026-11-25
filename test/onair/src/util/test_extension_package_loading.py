# GSC-19165-1, "The On-Board Artificial Intelligence Research (OnAIR) Platform"
#
# Copyright © 2023 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in the
# United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# Licensed under the NASA Open Source Agreement version 1.3
# See "NOSA GSC-19165-1 OnAIR.pdf"

"""Exercise extension initialization using real packages outside sys.path."""

import importlib
import sys
from uuid import uuid4

import pytest

from onair.src.util.plugin_import import import_plugins
from onair.src.util.service_import import import_services


@pytest.fixture(name="extension_package")
def external_package_fixture(tmp_path):
    """Create an external package and remove its imported modules afterward."""
    name = "onair_test_extension_" + uuid4().hex
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    (package / "constants.py").write_text("VALUE = 42\n", encoding="utf-8")
    (package / (name + "_plugin.py")).write_text(
        "from . import VALUE\n"
        "class Plugin:\n"
        "    def __init__(self, name, headers):\n"
        "        self.name = name\n"
        "        self.headers = headers\n"
        "        self.value = VALUE\n",
        encoding="utf-8",
    )
    (package / (name + "_service.py")).write_text(
        "from . import VALUE\n"
        "class Service:\n"
        "    def __init__(self, option):\n"
        "        self.option = option\n"
        "        self.value = VALUE\n",
        encoding="utf-8",
    )
    try:
        yield package
    finally:
        for module_name in list(sys.modules):
            if module_name == name or module_name.startswith(name + "."):
                sys.modules.pop(module_name, None)
        importlib.invalidate_caches()


def load_extension(kind, package):
    """Load a plugin or service through its public import function."""
    if kind == "service":
        return import_services(
            {"configured": {"path": str(package), "option": "kept"}}
        )[package.name]
    path = package / "__init__.py" if kind == "legacy_plugin" else package
    return import_plugins(["telemetry"], {"configured": str(path)})[0]


@pytest.mark.parametrize("kind", ["plugin", "legacy_plugin", "service"])
def test_initializer_can_import_relative_submodules(extension_package, kind):
    """A package need not be on sys.path to import its own children."""
    (extension_package / "__init__.py").write_text(
        "from .constants import VALUE\n", encoding="utf-8"
    )
    loaded = load_extension(kind, extension_package)
    assert loaded.value == 42
    if kind == "service":
        assert loaded.option == "kept"
    else:
        assert loaded.name == "configured"
        assert loaded.headers == ["telemetry"]


@pytest.mark.parametrize("kind", ["plugin", "service"])
def test_initializer_sees_its_registered_module(extension_package, kind):
    """Imports during initialization resolve to the module being executed."""
    (extension_package / "__init__.py").write_text(
        "import importlib\n"
        "VALUE = 42\n"
        "SAME_MODULE = importlib.import_module(__name__)\n",
        encoding="utf-8",
    )
    load_extension(kind, extension_package)
    module = sys.modules[extension_package.name]
    assert module.SAME_MODULE is module


@pytest.mark.parametrize("kind", ["plugin", "service"])
@pytest.mark.parametrize("error_type", [RuntimeError, SystemExit, KeyboardInterrupt])
def test_failed_initializer_is_removed_and_can_retry(
    extension_package, kind, error_type
):
    """Failed initialization must not cache a partially initialized package."""
    observed = extension_package / "registered.txt"
    (extension_package / "__init__.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        f"Path({str(observed)!r}).write_text(str(__name__ in sys.modules))\n"
        f"raise {error_type.__name__}('initialization failed')\n",
        encoding="utf-8",
    )
    with pytest.raises(error_type, match="initialization failed"):
        load_extension(kind, extension_package)
    assert observed.read_text() == "True"
    assert extension_package.name not in sys.modules

    (extension_package / "__init__.py").write_text(
        "from .constants import VALUE\n", encoding="utf-8"
    )
    assert load_extension(kind, extension_package).value == 42


@pytest.mark.parametrize("kind", ["plugin", "service"])
def test_cached_package_is_not_initialized_twice(extension_package, kind):
    """Repeated loads instantiate new objects without rerunning the package."""
    first = load_extension(kind, extension_package)
    original_module = sys.modules[extension_package.name]
    (extension_package / "__init__.py").write_text(
        "raise RuntimeError('initializer should not run again')\n", encoding="utf-8"
    )
    second = load_extension(kind, extension_package)
    assert first is not second
    assert first.value == second.value == 42
    assert sys.modules[extension_package.name] is original_module


@pytest.mark.parametrize("kind", ["plugin", "service"])
def test_missing_initializer_does_not_leave_module_cached(extension_package, kind):
    """File errors still propagate and leave no placeholder in sys.modules."""
    (extension_package / "__init__.py").unlink()
    with pytest.raises(FileNotFoundError):
        load_extension(kind, extension_package)
    assert extension_package.name not in sys.modules
