# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Autodiff placeholders for backends without transform support."""


class UnavailableTransforms:
    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def _raise(self, name: str):
        raise NotImplementedError(
            "Backend {!r} does not provide {} transforms".format(self.backend_name, name)
        )

    def grad(self, *args, **kwargs):
        self._raise("grad")

    def value_and_grad(self, *args, **kwargs):
        self._raise("value_and_grad")

    def jit(self, *args, **kwargs):
        self._raise("jit")

    def vmap(self, *args, **kwargs):
        self._raise("vmap")

    def stop_gradient(self, value):
        return value
