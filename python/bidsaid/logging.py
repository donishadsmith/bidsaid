"""Module for logging."""

import logging

from rich.logging import RichHandler


def setup_logger(
    logger_name: str = None, level: int | str | None = "INFO"
) -> logging.Logger:
    """
    Sets up the logger.

    .. note::
       Defaults to ``RichHandler`` if a module or root handler is
       not detected.

    Parameters
    ----------
    logger_name : :obj:`str`
        Name of the logger to return, if None, the root logger is returned.

    level : :obj:`int`, :obj:`str`, or :obj:`None`, default="INFO"
        The logging level. If None, the logging level is not set

    Returns
    -------
    Logger
        A ``Logger`` object.
    """
    logger = logging.getLogger(logger_name)
    if not _has_handler(logger):
        logger = _add_default_handler(logger)

    if level:
        logger.setLevel(level)

    return logger


def _has_handler(logger):
    """
    Check if a handler is present.

    Checks the root logger and module logger.

    Parameters
    ----------
    logger : :obj:`Logger`
        A logging object.

    Returns
    -------
    bool
        True if a handler is present and False if no handler is present
    """
    has_root_handler = logging.getLogger().hasHandlers()
    has_module_handler = bool(logger.handlers)

    return True if (has_root_handler or has_module_handler) else False


def _add_default_handler(logger: logging.Logger, format: str | None = None):
    """
    Add a default and format handler. Uses ``RichHandler`` as the default logger.

    Parameters
    ----------
    logger : :obj:`Logger`
        A logging object.

    format : :obj:`str`
        String specifying the format of the logged message.

    Returns
    -------
    Logger
        A logger object.
    """

    format = format if format else "%(asctime)s %(name)s [%(levelname)s] %(message)s"

    handler = RichHandler()
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)

    return logger


__all__ = ["setup_logger"]
