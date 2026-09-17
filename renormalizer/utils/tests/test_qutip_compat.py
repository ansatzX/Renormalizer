"""Independent small-state controls for the supported QuTiP public API."""
import numpy as np
import qutip

from renormalizer.utils.qutip_utils import get_gs


def test_ground_state_dimensions_and_values():
    state = get_gs(2, 3)
    expected = np.zeros((36, 1), dtype=complex)
    expected[0] = 1
    np.testing.assert_array_equal(state.full(), expected)
    assert abs(state.norm() - 1) < 1e-14


def test_public_solver_against_analytic_spin():
    initial = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    result = qutip.sesolve(0.5 * qutip.sigmaz(), initial, [0, 0.02],
                          options={'atol': 1e-12, 'rtol': 1e-12})
    exact = np.exp(-1j * np.array([0.5, -0.5]) * 0.02) / np.sqrt(2)
    np.testing.assert_allclose(result.states[-1].full().ravel(), exact,
                               atol=1e-10, rtol=1e-10)
