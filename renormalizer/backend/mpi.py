# -*- coding: utf-8 -*-

import os


class SingleProcessDistributedMixin:
    rank = 0
    size = 1
    is_distributed = False

    def barrier(self):
        return None

    def allreduce(self, x, op="sum"):
        return x

    def broadcast(self, x, root=0):
        return x

    def gather(self, x, root=0):
        return [x]

    def allgather(self, x):
        return [x]

    def reduce_scatter(self, x, op="sum", axis=0):
        return x

    def alltoall(self, x, split_axis=0, concat_axis=0):
        return x


class TorchDistributedMixin:
    """torch.distributed-backed collectives for multi-process GPU execution."""

    _distributed = None
    _distributed_backend = None

    def _torch_module(self):
        module = getattr(self, "array_namespace", None)
        if module is not None and hasattr(module, "Tensor"):
            return module
        try:
            import torch as module
        except (ImportError, OSError):
            return None
        return module

    def _init_distributed_runtime(self):
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
        if world_size <= 1:
            self._distributed = None
            self._distributed_backend = None
            return
        torch_module = self._torch_module()
        dist = getattr(torch_module, "distributed", None) if torch_module is not None else None
        if dist is None or not dist.is_available():
            self._distributed = None
            self._distributed_backend = None
            return
        backend = os.environ.get("RENO_DISTRIBUTED_BACKEND")
        if backend is None:
            backend = "nccl" if getattr(self, "device", None) == "gpu" else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=os.environ.get("RENO_DISTRIBUTED_INIT_METHOD", "env://"),
            )
        self._distributed = dist
        self._distributed_backend = backend

    def _distributed_ready(self):
        dist = getattr(self, "_distributed", None)
        if dist is None:
            torch_module = self._torch_module()
            dist = getattr(torch_module, "distributed", None) if torch_module is not None else None
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return False
        if dist.get_world_size() <= 1:
            return False
        self._distributed = dist
        return True

    @property
    def rank(self):
        if not self._distributed_ready():
            return 0
        return int(self._distributed.get_rank())

    @property
    def size(self):
        if not self._distributed_ready():
            return 1
        return int(self._distributed.get_world_size())

    @property
    def is_distributed(self):
        return self.size > 1

    def _reduce_op(self, op):
        reduce_op = getattr(self._distributed, "ReduceOp", None)
        if reduce_op is None:
            return None
        name = str(op or "sum").lower()
        mapping = {
            "sum": "SUM",
            "add": "SUM",
            "prod": "PRODUCT",
            "product": "PRODUCT",
            "max": "MAX",
            "min": "MIN",
        }
        attr = mapping.get(name)
        if attr is None or not hasattr(reduce_op, attr):
            raise ValueError("unsupported distributed reduction op {0!r}".format(op))
        return getattr(reduce_op, attr)

    @staticmethod
    def _normalize_axis(axis, ndim):
        axis = int(axis)
        if axis < 0:
            axis += int(ndim)
        if axis < 0 or axis >= int(ndim):
            raise ValueError("axis {0} is out of range for rank {1}".format(axis, ndim))
        return axis

    @staticmethod
    def _rank_slice(dim, parts, rank):
        base = int(dim) // int(parts)
        remainder = int(dim) % int(parts)
        start = int(rank) * base + min(int(rank), remainder)
        stop = start + base + (1 if int(rank) < remainder else 0)
        return slice(start, stop)

    def barrier(self):
        if self._distributed_ready():
            self._distributed.barrier()
        return None

    def allreduce(self, x, op="sum"):
        if not self._distributed_ready():
            return x
        self._distributed.all_reduce(x, op=self._reduce_op(op))
        return x

    def broadcast(self, x, root=0):
        if not self._distributed_ready():
            return x
        torch_module = self._torch_module()
        if torch_module is not None and isinstance(x, torch_module.Tensor):
            self._distributed.broadcast(x, src=int(root))
            return x
        values = [x if self.rank == int(root) else None]
        self._distributed.broadcast_object_list(values, src=int(root))
        return values[0]

    def _allgather_tensor(self, x):
        torch_module = self._torch_module()
        shape = torch_module.tensor(tuple(x.shape), dtype=torch_module.long, device=x.device)
        shapes = [torch_module.empty_like(shape) for _ in range(self.size)]
        self._distributed.all_gather(shapes, shape)
        shapes = [tuple(int(dim.item()) for dim in item) for item in shapes]
        max_shape = tuple(max(shape[axis] for shape in shapes) for axis in range(len(shapes[0])))
        slices = tuple(slice(0, dim) for dim in x.shape)
        padded = torch_module.zeros(max_shape, dtype=x.dtype, device=x.device)
        padded[slices] = x
        gathered = [torch_module.empty_like(padded) for _ in range(self.size)]
        self._distributed.all_gather(gathered, padded)
        return [
            value[tuple(slice(0, dim) for dim in value_shape)].clone()
            for value, value_shape in zip(gathered, shapes)
        ]

    def allgather(self, x):
        if not self._distributed_ready():
            return [x]
        torch_module = self._torch_module()
        if torch_module is not None and isinstance(x, torch_module.Tensor):
            return self._allgather_tensor(x)
        values = [None for _ in range(self.size)]
        self._distributed.all_gather_object(values, x)
        return values

    def gather(self, x, root=0):
        if not self._distributed_ready():
            return [x]
        values = self.allgather(x)
        return values if self.rank == int(root) else None

    def reduce_scatter(self, x, op="sum", axis=0):
        if not self._distributed_ready():
            return x
        torch_module = self._torch_module()
        if torch_module is None or not isinstance(x, torch_module.Tensor):
            reduced = self.allreduce(x, op=op)
            return reduced
        axis = self._normalize_axis(axis, x.ndim)
        if x.shape[axis] % self.size == 0 and hasattr(self._distributed, "reduce_scatter_tensor"):
            moved = torch_module.movedim(x, axis, 0).contiguous()
            local_dim = moved.shape[0] // self.size
            output = torch_module.empty((local_dim,) + tuple(moved.shape[1:]), dtype=x.dtype, device=x.device)
            self._distributed.reduce_scatter_tensor(output, moved, op=self._reduce_op(op))
            return torch_module.movedim(output, 0, axis)
        reduced = self.allreduce(x, op=op)
        local_slice = self._rank_slice(reduced.shape[axis], self.size, self.rank)
        slices = [slice(None)] * reduced.ndim
        slices[axis] = local_slice
        return reduced[tuple(slices)]

    def alltoall(self, x, split_axis=0, concat_axis=0):
        if not self._distributed_ready():
            return x
        torch_module = self._torch_module()
        if torch_module is None or not isinstance(x, torch_module.Tensor):
            values = self.allgather(x)
            return values[self.rank]
        split_axis = self._normalize_axis(split_axis, x.ndim)
        concat_axis = self._normalize_axis(concat_axis, x.ndim)
        if x.shape[split_axis] % self.size != 0 or not hasattr(self._distributed, "all_to_all_single"):
            gathered = self.allgather(x)
            chunks = [torch_module.chunk(item, self.size, dim=split_axis)[self.rank] for item in gathered]
            return torch_module.cat(chunks, dim=concat_axis)
        moved = torch_module.movedim(x, split_axis, 0).contiguous()
        output = torch_module.empty_like(moved)
        self._distributed.all_to_all_single(output, moved)
        return torch_module.movedim(output, 0, concat_axis)
