# -*- coding: utf-8 -*-

from renormalizer.backend.abstract import AbstractBackend
from renormalizer.backend.boundary import eye_like, flatten_backend, scalar_to_python
from renormalizer.backend.config import BackendConfig
from renormalizer.backend.execution import (
    ArrayInfo,
    BackendCapabilities,
    BackendCopyError,
    BackendFeatureError,
    BlockContractionSpec,
    BlockTensor,
    CopyPolicy,
    DenseBlock,
    DeviceSpec,
    EinsumSpec,
    FallbackPolicy,
    LayoutSpec,
    LayoutTransform,
    MatmulDesc,
    MatmulPlan,
    PairContractionSpec,
    TensorOperand,
    parse_device_spec,
)
from renormalizer.backend.factory import SUPPORTED_BACKENDS, available_backends, is_backend_available
from renormalizer.backend.protocol import BackendProtocol
from renormalizer.backend.transforms import UnavailableTransforms

__all__ = [
    "AbstractBackend",
    "BackendConfig",
    "ArrayInfo",
    "BackendCapabilities",
    "BackendCopyError",
    "BackendFeatureError",
    "BackendProtocol",
    "BlockContractionSpec",
    "BlockTensor",
    "CopyPolicy",
    "DenseBlock",
    "DeviceSpec",
    "EinsumSpec",
    "FallbackPolicy",
    "LayoutSpec",
    "LayoutTransform",
    "MatmulDesc",
    "MatmulPlan",
    "PairContractionSpec",
    "SUPPORTED_BACKENDS",
    "TensorOperand",
    "UnavailableTransforms",
    "available_backends",
    "eye_like",
    "flatten_backend",
    "is_backend_available",
    "parse_device_spec",
    "scalar_to_python",
]
