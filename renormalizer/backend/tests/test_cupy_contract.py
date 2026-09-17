"""CPU-only control for selected-device synchronization; not GPU evidence."""
import importlib
from types import SimpleNamespace


def test_sync_uses_device_captured_at_construction(monkeypatch):
    module = importlib.import_module('renormalizer.backend.cupy_backend')
    synchronized = []
    active = [3]
    class Device:
        def __init__(self, device_id=None):
            self.id = active[0] if device_id is None else device_id
        def synchronize(self):
            synchronized.append(self.id)
    fake = SimpleNamespace(ndarray=type('FakeArray', (), {}), linalg=object(), random=object(),
            cuda=SimpleNamespace(Device=Device,
                memory=SimpleNamespace(OutOfMemoryError=MemoryError)))
    monkeypatch.setattr(module, '_cupy', fake)
    monkeypatch.setattr(module, '_cupy_available', True)
    adapter = module.CupyBackend()
    active[0] = 5
    adapter.sync()
    assert synchronized == [3]
