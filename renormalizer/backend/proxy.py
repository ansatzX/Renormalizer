# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Stable proxy object that follows the selected backend."""

from renormalizer.backend.factory import create_backend
from renormalizer.backend.protocol import BackendProtocol


class BackendManager:
    def __init__(self, initial_backend=None, *, config=None, **options):
        selected = create_backend(initial_backend, config=config, **options)
        try:
            selected.activate()
            self._seed_backend(selected)
        except Exception:
            selected.deactivate()
            raise
        self.current: BackendProtocol = selected

    @staticmethod
    def _seed_backend(selected):
        seed = 2019 if selected.config.seed is None else selected.config.seed
        selected.random.seed(seed)

    def set_backend(self, name: str, *, config=None, **options):
        previous = self.current
        previous.deactivate()
        selected = None
        try:
            selected = create_backend(name, config=config, **options)
            selected.activate()
            self._seed_backend(selected)
        except Exception:
            try:
                if selected is not None:
                    selected.deactivate()
            finally:
                previous.activate()
            raise
        self.current = selected
        return selected

    def get_backend(self):
        return self.current


class BackendProxy:
    def __init__(self, manager: BackendManager):
        object.__setattr__(self, "_manager", manager)

    @property
    def current(self):
        return self._manager.current

    def __getattr__(self, name):
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        if name == "_manager":
            object.__setattr__(self, name, value)
        else:
            setattr(self.current, name, value)

    def __array_namespace__(self, *args, **kwargs):
        return self.current.array_namespace
