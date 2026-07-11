# -*- coding: utf-8 -*-
# Author: Cunxi Gong <ansatzMe@outlook.com>

"""Public facade for generic sharded contraction execution."""

from renormalizer.backend._distributed.local_operator import (
    DistributedLocalOperator,
    run_root_fallback,
)
from renormalizer.backend._distributed.planner import (
    DistributedBlockPlan,
    DistributedMemoryEstimate,
    DistributedPlan,
    plan_distributed_execution,
)
from renormalizer.backend._distributed.providers import (
    DeviceResidentProvider,
    OperandLease,
    OperandProvider,
    OperandRequest,
)
from renormalizer.backend._distributed.sharding import (
    DistributedTensor,
    ShardingSpec,
    shard_axis,
)
from renormalizer.backend._distributed.solvers import (
    distributed_norm,
    distributed_vdot,
    run_sharded_davidson,
    run_sharded_krylov,
)
from renormalizer.backend.config import DistributedExecutionConfig


__all__ = [
    "DeviceResidentProvider",
    "DistributedBlockPlan",
    "DistributedExecutionConfig",
    "DistributedLocalOperator",
    "DistributedMemoryEstimate",
    "DistributedPlan",
    "DistributedTensor",
    "OperandLease",
    "OperandProvider",
    "OperandRequest",
    "ShardingSpec",
    "distributed_norm",
    "distributed_vdot",
    "plan_distributed_execution",
    "run_root_fallback",
    "run_sharded_davidson",
    "run_sharded_krylov",
    "shard_axis",
]
