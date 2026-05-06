# -*- coding: utf-8 -*-

import logging
import random
import subprocess

import numpy as np

from renormalizer.backend.proxy import BackendManager, BackendProxy

logger = logging.getLogger(__name__)


def get_git_commit_hash():
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE)
        return commit_hash.strip().decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unknown"


_manager = BackendManager()
backend = BackendProxy(_manager)
xp = backend


def set_backend(name):
    selected = _manager.set_backend(name, explicit=True)
    selected.random.seed(2019)
    return selected


def get_backend():
    return _manager.get_backend()


def runtime_backend():
    return _manager.get_backend()


backend.random.seed(2019)
np.random.seed(9012)
random.seed(1092)

logger.info("Use %s as backend", backend.name)
logger.info("numpy random seed is 9012")
logger.info("backend random seed is 2019")
logger.info("random seed is 1092")
logger.info("Git Commit Hash: %s", get_git_commit_hash())
