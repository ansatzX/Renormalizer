import importlib.util
import json
import os
import subprocess
import sys
import weakref

import numpy as np
import pytest

pytestmark=pytest.mark.skipif(importlib.util.find_spec('jax') is None,reason='JAX environment required')


def test_x64_requirement_does_not_mutate_process_configuration():
    code='''
import jax,json
from renormalizer.backend.context import make_context
before=bool(jax.config.x64_enabled)
make_context('jax',device='cpu',real_dtype='float32')
try:
    make_context('jax',device='cpu',real_dtype='float64')
except ValueError as error:
    message=str(error)
else:
    raise AssertionError('float64 silently accepted without x64')
print(json.dumps([before,bool(jax.config.x64_enabled),message]))
'''
    env=os.environ.copy()
    env['JAX_ENABLE_X64']='0'
    env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    result=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=60,check=True)
    before,after,message=json.loads(result.stdout.strip().splitlines()[-1])
    assert before is False and after is False
    assert 'JAX_ENABLE_X64=1' in message


def test_instance_rng_precision_and_sync_does_not_retain_arrays():
    # JAX import itself may consume host RNG state; measure the adapter after
    # that third-party initialization, not package-import side effects.
    import jax
    from renormalizer.backend.context import make_context
    state=np.random.get_state()
    ctx=make_context('jax',device='cpu',real_dtype='float64')
    random=ctx.adapter.random.normal(size=3)
    assert ctx.adapter.owns(random) and random.dtype==np.float64
    values=ctx.adapter.random.randint(5,size=3)
    assert ctx.adapter.owns(values)
    assert bool((values>=0).all()) and bool((values<5).all())
    x=ctx.ops.ones((3,))
    reference=weakref.ref(x)
    ctx.ops.sync()
    del x
    assert reference() is None
    after=np.random.get_state()
    np.testing.assert_array_equal(state[1],after[1])
    assert state[2:]==after[2:]


def test_sync_waits_for_live_random_result(monkeypatch):
    from renormalizer.backend.context import make_context
    ctx=make_context('jax',device='cpu',real_dtype='float64')
    value=ctx.adapter.random.normal(size=3)
    waited=[]
    original=type(value).block_until_ready
    def wait(array):
        waited.append(id(array))
        return original(array)
    monkeypatch.setattr(type(value),'block_until_ready',wait)
    ctx.ops.sync()
    assert id(value) in waited


def test_sync_waits_for_live_legacy_linalg_result(monkeypatch):
    from renormalizer.backend.context import make_context
    ctx=make_context('jax',device='cpu',real_dtype='float64')
    q,r=ctx.adapter.linalg.qr(ctx.ops.ones((3,2)))
    waited=[]
    original=type(q).block_until_ready
    def wait(array):
        waited.append(id(array))
        return original(array)
    monkeypatch.setattr(type(q),'block_until_ready',wait)
    ctx.ops.sync()
    assert id(q) in waited and id(r) in waited
