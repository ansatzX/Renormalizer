import numpy as np
import opt_einsum as oe
import pytest
from renormalizer.backend.context import make_context
from renormalizer.backend.execution import bind_backend, record_execution, contract


def test_bound_contraction_is_witnessed_and_context_restored():
    from renormalizer.backend.context import current_backend
    ctx = make_context(host_policy="explicit")
    before = current_backend()
    @bind_backend
    def f():
        return contract('ij,jk->ik', np.eye(2), np.eye(2))
    with record_execution(ctx) as ledger:
        result = f(backend_context=ctx)
    np.testing.assert_array_equal(result, np.eye(2))
    assert current_backend() is before
    assert ledger.operations[0]['adapter_id'] == id(ctx.adapter)
    assert ledger.operations[0]['operation'] == 'contraction'


def test_explicit_hybrid_policy_rejected_before_algorithm():
    ctx = make_context()
    @bind_backend
    def f():
        raise AssertionError('must reject before algorithm')
    with pytest.raises(Exception, match='explicit'):
        f(backend_context=ctx)


def test_cpu_foreign_array_records_host_boundary_without_size_method():
    from renormalizer.backend.execution import _record_transfer
    class Foreign:
        device = 'cpu'
        dtype = 'float64'
        shape = (2, 3)
        def size(self):
            raise AssertionError('not a NumPy size attribute')
    with record_execution(make_context()) as ledger:
        _record_transfer(np.zeros((2, 3)), Foreign(), 'test')
    assert ledger.transfers[0]['direction'] == 'host_to_host'
    assert ledger.transfers[0]['logical_bytes'] == 48


def test_expression_constants_captured_and_cross_instance_refused(monkeypatch):
    from renormalizer.backend.context import capture_backend
    from renormalizer.backend.execution import contract_expression
    ctx = make_context(host_policy='explicit')
    constant = np.eye(2)
    converted = []
    original = oe.contract_expression
    def traced(*args, **kwargs):
        converted.extend(args)
        return original(*args, **kwargs)
    # NumPy placement is the identity, so the captured constant itself must
    # reach opt_einsum, without a copy or a process-default conversion.
    monkeypatch.setattr(oe, 'contract_expression', traced)
    with capture_backend(ctx.adapter):
        expr = contract_expression('ij,jk->ik',constant,(2,2),constants=[0])
        np.testing.assert_array_equal(expr(np.eye(2)),constant)
    assert any(value is constant for value in converted)
    with capture_backend(make_context().adapter):
        with pytest.raises(Exception,match='another backend'):
            expr(np.eye(2))
