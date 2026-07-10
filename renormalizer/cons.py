# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Process-wide backend compatibility facade."""

import random
import subprocess

import numpy as np

from renormalizer.backend.config import BackendConfig
from renormalizer.backend.factory import SUPPORTED_BACKENDS, available_backends, is_backend_available
from renormalizer.backend.proxy import BackendManager, BackendProxy


def get_git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.PIPE
        ).strip().decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unknown"


_manager = BackendManager()
backend = BackendProxy(_manager)
xp = backend


def set_backend(name: str, config=None, **options):
    return _manager.set_backend(name, config=config, **options)


def get_backend():
    return _manager.get_backend()


def runtime_backend():
    return get_backend()


np.random.seed(9012)
random.seed(1092)
