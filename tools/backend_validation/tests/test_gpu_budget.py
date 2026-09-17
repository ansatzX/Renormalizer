import numpy as np
import pytest
from tools.backend_validation.gpu import run_reviewed_gpu


def test_output_work_budget_checked_before_launch(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a,**k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='budget'):
        run_reviewed_gpu(np.empty((257,0)),np.empty((0,257)),np.empty((0,0)),tolerances={})
