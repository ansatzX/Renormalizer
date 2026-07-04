# -*- coding: utf-8 -*-

from renormalizer.backend.factory import create_backend
from renormalizer.backend.protocol import BackendProtocol


def _delegate_backend_property(name):
    def getter(self):
        return getattr(self.current, name)

    return property(getter)


def _delegate_backend_method(name):
    def method(self, *args, **kwargs):
        return getattr(self.current, name)(*args, **kwargs)

    method.__name__ = name
    method.__qualname__ = "BackendProxy.{0}".format(name)
    return method


class BackendManager:
    def __init__(self, initial_backend=None, config=None, **options):
        self.current: BackendProtocol = create_backend(initial_backend, explicit=False, config=config, **options)

    def set_backend(self, name, *, explicit=True, config=None, **options) -> BackendProtocol:
        self.current = create_backend(name, explicit=explicit, config=config, **options)
        return self.current

    def get_backend(self) -> BackendProtocol:
        return self.current


class BackendProxy:
    def __init__(self, manager: BackendManager):
        object.__setattr__(self, "_manager", manager)

    @property
    def current(self):
        return self._manager.current

    name = _delegate_backend_property("name")
    config = _delegate_backend_property("config")
    device = _delegate_backend_property("device")
    device_spec = _delegate_backend_property("device_spec")
    capabilities = _delegate_backend_property("capabilities")
    fallback_policy = _delegate_backend_property("fallback_policy")
    supported_device_kinds = _delegate_backend_property("supported_device_kinds")
    available_device_kinds = _delegate_backend_property("available_device_kinds")
    supports_cpu = _delegate_backend_property("supports_cpu")
    array_namespace = _delegate_backend_property("array_namespace")
    opt_einsum_name = _delegate_backend_property("opt_einsum_name")
    supports_gpu = _delegate_backend_property("supports_gpu")
    supports_autodiff = _delegate_backend_property("supports_autodiff")
    supports_jit = _delegate_backend_property("supports_jit")
    supports_sparse = _delegate_backend_property("supports_sparse")
    supports_functional_update = _delegate_backend_property("supports_functional_update")
    supports_complex64 = _delegate_backend_property("supports_complex64")
    supports_complex128 = _delegate_backend_property("supports_complex128")
    supports_fp32 = _delegate_backend_property("supports_fp32")
    supports_fp64 = _delegate_backend_property("supports_fp64")
    supports_mixed_precision = _delegate_backend_property("supports_mixed_precision")
    supports_device_index = _delegate_backend_property("supports_device_index")
    supports_matmul = _delegate_backend_property("supports_matmul")
    supports_batched_matmul = _delegate_backend_property("supports_batched_matmul")
    supports_grouped_gemm = _delegate_backend_property("supports_grouped_gemm")
    supports_strided_batched_gemm = _delegate_backend_property("supports_strided_batched_gemm")
    supports_einsum = _delegate_backend_property("supports_einsum")
    supports_contract_expression = _delegate_backend_property("supports_contract_expression")
    supports_contraction_path = _delegate_backend_property("supports_contraction_path")
    supports_custom_contraction_plan = _delegate_backend_property("supports_custom_contraction_plan")
    supports_streams = _delegate_backend_property("supports_streams")
    supports_events = _delegate_backend_property("supports_events")
    supports_memory_pool = _delegate_backend_property("supports_memory_pool")
    supports_block_sparse = _delegate_backend_property("supports_block_sparse")
    supports_packed_blocks = _delegate_backend_property("supports_packed_blocks")
    supports_scatter_add = _delegate_backend_property("supports_scatter_add")
    supports_distributed = _delegate_backend_property("supports_distributed")
    supports_distributed_array = _delegate_backend_property("supports_distributed_array")
    supports_allreduce = _delegate_backend_property("supports_allreduce")
    supports_broadcast = _delegate_backend_property("supports_broadcast")
    supports_allgather = _delegate_backend_property("supports_allgather")
    supports_reduce_scatter = _delegate_backend_property("supports_reduce_scatter")
    supports_alltoall = _delegate_backend_property("supports_alltoall")
    supports_point_to_point = _delegate_backend_property("supports_point_to_point")
    host_array_types = _delegate_backend_property("host_array_types")
    device_array_types = _delegate_backend_property("device_array_types")
    ndarray = _delegate_backend_property("ndarray")
    memory_errors = _delegate_backend_property("memory_errors")
    transforms = _delegate_backend_property("transforms")
    random = _delegate_backend_property("random")
    linalg = _delegate_backend_property("linalg")
    rank = _delegate_backend_property("rank")
    size = _delegate_backend_property("size")
    is_distributed = _delegate_backend_property("is_distributed")
    is_32bits = _delegate_backend_property("is_32bits")
    real_dtype = _delegate_backend_property("real_dtype")
    complex_dtype = _delegate_backend_property("complex_dtype")
    dtypes = _delegate_backend_property("dtypes")
    canonical_atol = _delegate_backend_property("canonical_atol")
    canonical_rtol = _delegate_backend_property("canonical_rtol")

    use_32bits = _delegate_backend_method("use_32bits")
    use_64bits = _delegate_backend_method("use_64bits")
    array = _delegate_backend_method("array")
    asarray = _delegate_backend_method("asarray")
    from_numpy = _delegate_backend_method("from_numpy")
    to_numpy = _delegate_backend_method("to_numpy")
    numpy = _delegate_backend_method("numpy")
    to_host = _delegate_backend_method("to_host")
    to_backend = _delegate_backend_method("to_backend")
    tensordot = _delegate_backend_method("tensordot")
    einsum = _delegate_backend_method("einsum")
    is_array = _delegate_backend_method("is_array")
    is_host_array = _delegate_backend_method("is_host_array")
    is_device_array = _delegate_backend_method("is_device_array")
    is_distributed_array = _delegate_backend_method("is_distributed_array")
    array_info = _delegate_backend_method("array_info")
    layout = _delegate_backend_method("layout")
    permute = _delegate_backend_method("permute")
    reshape_view = _delegate_backend_method("reshape_view")
    can_reshape_view = _delegate_backend_method("can_reshape_view")
    make_contiguous = _delegate_backend_method("make_contiguous")
    parse_einsum = _delegate_backend_method("parse_einsum")
    contract = _delegate_backend_method("contract")
    contract_expression = _delegate_backend_method("contract_expression")
    contract_path = _delegate_backend_method("contract_path")
    plan_contraction = _delegate_backend_method("plan_contraction")
    plan_distributed_contraction_path = _delegate_backend_method("plan_distributed_contraction_path")
    estimate_matmul = _delegate_backend_method("estimate_matmul")
    estimate_contraction = _delegate_backend_method("estimate_contraction")
    estimate_redistribute = _delegate_backend_method("estimate_redistribute")
    default_stream = _delegate_backend_method("default_stream")
    new_stream = _delegate_backend_method("new_stream")
    record_event = _delegate_backend_method("record_event")
    wait_event = _delegate_backend_method("wait_event")
    allocate_workspace = _delegate_backend_method("allocate_workspace")
    release_workspace = _delegate_backend_method("release_workspace")
    execute = _delegate_backend_method("execute")
    last_execution_profile = _delegate_backend_method("last_execution_profile")
    unpack_masked_vectors = _delegate_backend_method("unpack_masked_vectors")
    pack_masked_vectors = _delegate_backend_method("pack_masked_vectors")
    shard_tensor = _delegate_backend_method("shard_tensor")
    gather_tensor = _delegate_backend_method("gather_tensor")
    redistribute = _delegate_backend_method("redistribute")
    replicate_tensor = _delegate_backend_method("replicate_tensor")
    distributed_contract = _delegate_backend_method("distributed_contract")
    astype = _delegate_backend_method("astype")
    ascontiguousarray = _delegate_backend_method("ascontiguousarray")
    current_device = _delegate_backend_method("current_device")
    set_device = _delegate_backend_method("set_device")
    device_count = _delegate_backend_method("device_count")
    lower_pair_contraction_to_matmul = _delegate_backend_method("lower_pair_contraction_to_matmul")
    lower_block_contraction = _delegate_backend_method("lower_block_contraction")
    execute_matmul_plan = _delegate_backend_method("execute_matmul_plan")
    execute_grouped_gemm_plan = _delegate_backend_method("execute_grouped_gemm_plan")
    matmul = _delegate_backend_method("matmul")
    batched_matmul = _delegate_backend_method("batched_matmul")
    grouped_gemm = _delegate_backend_method("grouped_gemm")
    gemv_batch = _delegate_backend_method("gemv_batch")
    prepack_grouped_gemm = _delegate_backend_method("prepack_grouped_gemm")
    execute_prepacked_grouped_gemm = _delegate_backend_method("execute_prepacked_grouped_gemm")
    synchronize = _delegate_backend_method("synchronize")
    sync = _delegate_backend_method("sync")
    free_all_blocks = _delegate_backend_method("free_all_blocks")
    log_memory_usage = _delegate_backend_method("log_memory_usage")
    at_set = _delegate_backend_method("at_set")
    at_add = _delegate_backend_method("at_add")
    at_sub = _delegate_backend_method("at_sub")
    at_mul = _delegate_backend_method("at_mul")
    barrier = _delegate_backend_method("barrier")
    allreduce = _delegate_backend_method("allreduce")
    broadcast = _delegate_backend_method("broadcast")
    gather = _delegate_backend_method("gather")
    allgather = _delegate_backend_method("allgather")
    reduce_scatter = _delegate_backend_method("reduce_scatter")
    alltoall = _delegate_backend_method("alltoall")
    send = _delegate_backend_method("send")
    recv = _delegate_backend_method("recv")

    def __getattr__(self, name):
        return getattr(self.current, name)

    def __setattr__(self, name, value):
        if name == "_manager":
            object.__setattr__(self, name, value)
            return
        setattr(self.current, name, value)

    def __array_namespace__(self):
        return self.current.array_namespace
