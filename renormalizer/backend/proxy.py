# -*- coding: utf-8 -*-

from renormalizer.backend.factory import create_backend


class BackendManager:
    def __init__(self, initial_backend=None):
        self.current = create_backend(initial_backend, explicit=False)

    def set_backend(self, name, *, explicit=True):
        self.current = create_backend(name, explicit=explicit)
        return self.current

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
            return
        setattr(self.current, name, value)

    def __array_namespace__(self):
        return self.current.array_namespace
