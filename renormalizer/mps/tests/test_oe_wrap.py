from unittest.mock import patch

import pytest

from renormalizer.mps.oe_contract_wrap import oe_contract, oe_contract_expression
from renormalizer.mps.backend import MEMORY_ERRORS, np, xp
from renormalizer.mps.matrix import asnumpy


def test_oe_contract_expression_matches_numpy():
    rng = np.random.default_rng(11)
    left = rng.normal(size=(2, 3))
    right = rng.normal(size=(3, 4))
    expr = oe_contract_expression(
        "ab,bc->ac", xp.asarray(left), right.shape, constants=[0]
    )

    actual = asnumpy(expr(xp.asarray(right)))
    expected = np.einsum("ab,bc->ac", left, right)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_oe_contract():
    with patch("logging.Logger.fatal") as mock_logger_fatal:
        with pytest.raises(MEMORY_ERRORS):
            a = np.random.rand(2<<20)
            oe_args = []
            for i in range(5):
                oe_args.extend([a, [i]])
            oe_args.append(list(range(5)))
            oe_contract(*oe_args)

        # Verify that logger.fatal was called multiple times
        assert mock_logger_fatal.call_count > 1, "logger.fatal was not called multiple times"
        # Verify that one of the calls contains the specific message
        messages = [call[0][0] for call in mock_logger_fatal.call_args_list]
        assert "Out of memory error calling oe.contract" in messages, (
            "Expected message not found in logger.fatal calls"
        )


def test_oe_contract_expression():
    with patch("logging.Logger.fatal") as mock_logger_fatal:
        with pytest.raises(MEMORY_ERRORS):
            a = np.random.rand(2 << 20)
            expr = oe_contract_expression(
                "a, b, c, d, e -> abcde",
                a, a, a, a, (2 << 20, ),
                constants=[0, 1, 2, 3])
            expr(a)

        # Verify that logger.fatal was called multiple times
        assert mock_logger_fatal.call_count > 1, "logger.fatal was not called multiple times"
        # Verify that one of the calls contains the specific message
        messages = [call[0][0] for call in mock_logger_fatal.call_args_list]
        assert "Out of memory error calling oe contract expression" in messages, (
            "Expected message not found in logger.fatal calls"
        )
