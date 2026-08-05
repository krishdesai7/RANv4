import logging

import pytest
from rich.logging import RichHandler


@pytest.fixture(autouse=True)
def restore_root_logging():
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    yield
    root.handlers[:] = old_handlers
    root.setLevel(old_level)


def test_configure_logging_installs_one_rich_handler_and_level() -> None:
    from ran.logging_config import configure_logging

    configure_logging("debug")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], RichHandler)


def test_configure_logging_is_deterministic_when_called_twice() -> None:
    from ran.logging_config import configure_logging

    configure_logging("INFO")
    configure_logging("WARNING")

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], RichHandler)


def test_configure_logging_rejects_unknown_level() -> None:
    from ran.logging_config import configure_logging

    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("verbose")
