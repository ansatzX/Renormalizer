# -*- coding: utf-8 -*-

import os


class SingleProcessDistributedMixin:
    rank = 0
    size = 1
    is_distributed = False

    _single_process_reduce_ops = frozenset(("sum", "add", "prod", "product", "max", "min"))

    @staticmethod
    def _single_process_collective_axis(x, axis, label):
        ndim = getattr(x, "ndim", None)
        if ndim is None:
            shape = getattr(x, "shape", None)
            if shape is None:
                raise ValueError("{0} requires array-like input when axis is provided".format(label))
            ndim = len(shape)
        ndim = int(ndim)
        axis = int(axis)
        normalized = axis + ndim if axis < 0 else axis
        if normalized < 0 or normalized >= ndim:
            raise ValueError("{0} {1} is out of range for rank {2}".format(label, axis, ndim))
        return normalized

    @staticmethod
    def _single_process_collective_root(root, label):
        root = int(root)
        if root != 0:
            raise ValueError("{0} root {1} is out of range for world size 1".format(label, root))
        return root

    @staticmethod
    def _single_process_collective_peer(peer, label):
        peer = int(peer)
        if peer != 0:
            raise ValueError("{0} {1} is out of range for world size 1".format(label, peer))
        return peer

    @classmethod
    def _single_process_reduce_op(cls, op):
        name = str(op or "sum").lower()
        if name not in cls._single_process_reduce_ops:
            raise ValueError("unsupported distributed reduction op {0!r}".format(op))
        return name

    def barrier(self):
        return None

    def allreduce(self, x, op="sum"):
        self._single_process_reduce_op(op)
        return x

    def broadcast(self, x, root=0):
        self._single_process_collective_root(root, "broadcast")
        return x

    def gather(self, x, root=0):
        self._single_process_collective_root(root, "gather")
        return [x]

    def allgather(self, x, axis=None):
        if axis is None:
            return [x]
        self._single_process_collective_axis(x, axis, "allgather axis")
        return x

    def reduce_scatter(self, x, op="sum", axis=0):
        self._single_process_reduce_op(op)
        self._single_process_collective_axis(x, axis, "reduce_scatter axis")
        return x

    def alltoall(self, x, split_axis=0, concat_axis=0):
        self._single_process_collective_axis(x, split_axis, "alltoall split_axis")
        self._single_process_collective_axis(x, concat_axis, "alltoall concat_axis")
        return x

    def send(self, x, *, dst, tag=0):
        self._single_process_collective_peer(dst, "send dst")
        int(tag)
        return None

    def recv(self, *, src, tag=0, like=None):
        self._single_process_collective_peer(src, "recv src")
        int(tag)
        if like is None:
            raise ValueError("recv requires a like buffer in single-process mode")
        return like


class TorchDistributedMixin:
    """torch.distributed-backed collectives for multi-process GPU execution."""

    _distributed = None
    _distributed_backend = None

    def _distributed_peer(self, peer, label):
        peer = int(peer)
        size = int(self.size)
        if peer < 0 or peer >= size:
            raise ValueError("{0} {1} is out of range for world size {2}".format(label, peer, size))
        return peer

    def _distributed_root(self, root, label):
        return self._distributed_peer(root, label)

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

    @staticmethod
    def _reduce_op_attr(op):
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
        if attr is None:
            raise ValueError("unsupported distributed reduction op {0!r}".format(op))
        return attr

    def _reduce_op(self, op):
        attr = self._reduce_op_attr(op)
        reduce_op = getattr(self._distributed, "ReduceOp", None)
        if reduce_op is None:
            return None
        if not hasattr(reduce_op, attr):
            raise ValueError("unsupported distributed reduction op {0!r}".format(op))
        return getattr(reduce_op, attr)

    @staticmethod
    def _normalize_axis(axis, ndim, label=None):
        raw_axis = int(axis)
        axis = int(axis)
        if axis < 0:
            axis += int(ndim)
        if axis < 0 or axis >= int(ndim):
            label = "axis" if label is None else str(label)
            raise ValueError("{0} {1} is out of range for rank {2}".format(label, raw_axis, ndim))
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
        self._reduce_op_attr(op)
        if not self._distributed_ready():
            return x
        self._distributed.all_reduce(x, op=self._reduce_op(op))
        return x

    def broadcast(self, x, root=0):
        root = self._distributed_root(root, "broadcast root")
        if not self._distributed_ready():
            return x
        torch_module = self._torch_module()
        if torch_module is not None and isinstance(x, torch_module.Tensor):
            self._distributed.broadcast(x, src=root)
            return x
        values = [x if self.rank == root else None]
        self._distributed.broadcast_object_list(values, src=root)
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

    def _gather_tensor(self, x, root):
        torch_module = self._torch_module()
        shape = torch_module.tensor(tuple(x.shape), dtype=torch_module.long, device=x.device)
        shape_list = None
        if self.rank == root:
            shape_list = [torch_module.empty_like(shape) for _ in range(self.size)]
        self._distributed.gather(shape, gather_list=shape_list, dst=root)

        shapes = None
        if self.rank == root:
            shapes = [tuple(int(dim.item()) for dim in item) for item in shape_list]
            max_shape = tuple(max(shape[axis] for shape in shapes) for axis in range(len(shapes[0])))
            max_shape_tensor = torch_module.tensor(max_shape, dtype=torch_module.long, device=x.device)
        else:
            max_shape_tensor = torch_module.empty_like(shape)
        self._distributed.broadcast(max_shape_tensor, src=root)
        max_shape = tuple(int(dim.item()) for dim in max_shape_tensor)

        slices = tuple(slice(0, dim) for dim in x.shape)
        padded = torch_module.zeros(max_shape, dtype=x.dtype, device=x.device)
        padded[slices] = x
        gather_list = None
        if self.rank == root:
            gather_list = [torch_module.empty_like(padded) for _ in range(self.size)]
        self._distributed.gather(padded, gather_list=gather_list, dst=root)
        if self.rank != root:
            return None
        return [
            value[tuple(slice(0, dim) for dim in value_shape)].clone()
            for value, value_shape in zip(gather_list, shapes)
        ]

    def allgather(self, x, axis=None):
        if axis is not None:
            axis = self._normalize_axis(axis, x.ndim, "allgather axis")
        if not self._distributed_ready():
            return [x] if axis is None else x
        torch_module = self._torch_module()
        if torch_module is not None and isinstance(x, torch_module.Tensor):
            values = self._allgather_tensor(x)
            if axis is None:
                return values
            return torch_module.cat(values, dim=axis)
        values = [None for _ in range(self.size)]
        self._distributed.all_gather_object(values, x)
        return values

    def gather(self, x, root=0):
        root = self._distributed_root(root, "gather root")
        if not self._distributed_ready():
            return [x]
        torch_module = self._torch_module()
        if (
            torch_module is not None
            and isinstance(x, torch_module.Tensor)
            and hasattr(self._distributed, "gather")
        ):
            return self._gather_tensor(x, root)
        values = self.allgather(x)
        return values if self.rank == root else None

    def send(self, x, *, dst, tag=0):
        dst = self._distributed_peer(dst, "send dst")
        tag = int(tag)
        if not self._distributed_ready():
            if int(self.size) == 1:
                return None
            raise RuntimeError("torch point-to-point send requires initialized torch.distributed")
        torch_module = self._torch_module()
        if torch_module is None or not isinstance(x, torch_module.Tensor):
            raise TypeError("torch point-to-point send requires a torch.Tensor")
        self._distributed.send(x, dst=dst, tag=tag)
        return None

    def recv(self, *, src, tag=0, like=None):
        src = self._distributed_peer(src, "recv src")
        tag = int(tag)
        if like is None:
            raise ValueError("recv requires a like buffer")
        if not self._distributed_ready():
            if int(self.size) == 1:
                return like
            raise RuntimeError("torch point-to-point recv requires initialized torch.distributed")
        torch_module = self._torch_module()
        if torch_module is None or not isinstance(like, torch_module.Tensor):
            raise TypeError("torch point-to-point recv requires a torch.Tensor like buffer")
        output = torch_module.empty_like(like)
        self._distributed.recv(output, src=src, tag=tag)
        return output

    def reduce_scatter(self, x, op="sum", axis=0):
        self._reduce_op_attr(op)
        axis = self._normalize_axis(axis, x.ndim, "reduce_scatter axis")
        if not self._distributed_ready():
            return x
        torch_module = self._torch_module()
        if torch_module is None or not isinstance(x, torch_module.Tensor):
            reduced = self.allreduce(x, op=op)
            return reduced
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
        split_axis = self._normalize_axis(split_axis, x.ndim, "alltoall split_axis")
        concat_axis = self._normalize_axis(concat_axis, x.ndim, "alltoall concat_axis")
        if not self._distributed_ready():
            return x
        torch_module = self._torch_module()
        if torch_module is None or not isinstance(x, torch_module.Tensor):
            values = self.allgather(x)
            return values[self.rank]
        if x.shape[split_axis] % self.size != 0 or not hasattr(self._distributed, "all_to_all_single"):
            gathered = self.allgather(x)
            chunks = []
            for item in gathered:
                local_slice = self._rank_slice(item.shape[split_axis], self.size, self.rank)
                slices = [slice(None)] * item.ndim
                slices[split_axis] = local_slice
                chunks.append(item[tuple(slices)])
            return torch_module.cat(chunks, dim=concat_axis)
        moved = torch_module.movedim(x, split_axis, 0).contiguous()
        output = torch_module.empty_like(moved)
        self._distributed.all_to_all_single(output, moved)
        if split_axis == concat_axis:
            return torch_module.movedim(output, 0, concat_axis)
        local_split = moved.shape[0] // self.size
        # all_to_all_single concatenates received chunks in source-rank order.
        staged = output.reshape((self.size, local_split) + tuple(moved.shape[1:]))
        remaining_axes = [axis for axis in range(x.ndim) if axis != split_axis]
        axis_to_staged_dim = {
            axis: 2 + index
            for index, axis in enumerate(remaining_axes)
        }
        perm = []
        final_shape = []
        for axis, dim in enumerate(x.shape):
            if axis == split_axis:
                perm.append(1)
                final_shape.append(local_split)
            elif axis == concat_axis:
                perm.extend((0, axis_to_staged_dim[axis]))
                final_shape.append(int(dim) * int(self.size))
            else:
                perm.append(axis_to_staged_dim[axis])
                final_shape.append(int(dim))
        return staged.permute(tuple(perm)).contiguous().reshape(tuple(final_shape))
