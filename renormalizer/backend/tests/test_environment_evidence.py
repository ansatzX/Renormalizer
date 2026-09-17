from tools.backend_validation.environment import validate_identity


def test_actual_numpy_major_and_backend_are_required():
    assert validate_identity('numpy2', '1.26.4', 'numpy')['reason'] == 'numpy_major_mismatch'
    assert validate_identity('jax', '1.26.4', 'numpy')['reason'] == 'backend_mismatch'
    assert validate_identity('numpy2', '2.0.0', 'numpy')['ok']
    assert not validate_identity('invented', '2.0.0', 'numpy')['ok']
