# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Wetiang Li <liwt31@163.com>

import logging
from logging import ERROR, WARN, INFO, DEBUG

import numpy as np

package_logger = logging.getLogger("renormalizer")
default_stream_handler = logging.StreamHandler()
default_formatter = logging.Formatter("%(asctime)s[%(levelname)s] %(message)s")
PROFILING = 5


def _profiling(self, message, *args, **kwargs):
    if self.isEnabledFor(PROFILING):
        self._log(PROFILING, message, args, **kwargs)


logging.addLevelName(PROFILING, "PROFILING")
if not hasattr(logging, "PROFILING"):
    logging.PROFILING = PROFILING
if not hasattr(logging.Logger, "profiling"):
    logging.Logger.profiling = _profiling


def parse_log_level(level):
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        text = level.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        name_to_level = {
            "PROFILING": PROFILING,
            "DEBUG": DEBUG,
            "INFO": INFO,
            "WARN": WARN,
            "WARNING": WARN,
            "ERROR": ERROR,
        }
        upper = text.upper()
        if upper in name_to_level:
            return name_to_level[upper]
    raise ValueError(f"Invalid log level: {level!r}")


def getLogger(*args):
    return package_logger

# alias for the Python convention
get_logger = getLogger


def init_log(level=logging.DEBUG):
    level = parse_log_level(level)
    package_logger.setLevel(level)

    default_stream_handler.setLevel(min(logging.DEBUG, level))

    default_stream_handler.setFormatter(default_formatter)

    package_logger.addHandler(default_stream_handler)


def set_stream_level(level):
    default_stream_handler.setLevel(level)


def disable_stream_output():
    if default_stream_handler in package_logger.handlers:
        package_logger.removeHandler(default_stream_handler)


def register_file_output(file_path, mode="w", level=DEBUG):
    level = parse_log_level(level)
    file_handler = logging.FileHandler(file_path, mode=mode)
    file_handler.setLevel(min(level, package_logger.level))
    file_handler.setFormatter(default_formatter)
    file_handler.addFilter(logging.Filter("renormalizer"))
    package_logger.addHandler(file_handler)
    return file_handler


NP_ERRCONFIG = {"divide": "raise", "over": "raise", "under": "ignore", "invalid": "raise"}

DEFAULT_NP_ERRCONFIG = np.seterr(**NP_ERRCONFIG)
