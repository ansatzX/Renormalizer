import json
import logging
from unittest.mock import patch

import pytest

from renormalizer.mps.oe_contract_wrap import oe_contract, oe_contract_expression
from renormalizer.mps.backend import backend, np


def test_oe_contract():
    with patch("logging.Logger.fatal") as mock_logger_fatal:
        with pytest.raises(backend.memory_errors):
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
        with pytest.raises(backend.memory_errors):
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


def test_oe_contract_wrap_uses_active_backend_metadata():
    import renormalizer as r
    from renormalizer.mps import oe_contract_wrap

    assert oe_contract_wrap.active_memory_errors() == r.backend.memory_errors
    assert oe_contract_wrap.active_array_types() == r.backend.ndarray


def test_oe_contract_wrap_metadata_helpers_follow_runtime_backend(monkeypatch):
    from renormalizer.mps import oe_contract_wrap

    class SentinelMemoryError(MemoryError):
        pass

    class SentinelArray:
        pass

    class FakeBackend:
        memory_errors = (SentinelMemoryError,)
        ndarray = (SentinelArray,)

    monkeypatch.setattr(oe_contract_wrap, "backend", FakeBackend())

    assert oe_contract_wrap.active_memory_errors() == (SentinelMemoryError,)
    assert oe_contract_wrap.active_array_types() == (SentinelArray,)


def _jsonl_payloads(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_oe_contract_profiling_records_operand_array_backends(caplog, monkeypatch, tmp_path):
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING
    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    a = np.ones((2, 2))
    b = np.ones((2, 2))
    try:
        oe_contract("ij,jk->ik", a, b)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    messages = [
        record.getMessage() for record in caplog.records
        if record.levelno == PROFILING and record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    payloads = [json.loads(message[len(profiling.LOG_PREFIX):]) for message in messages]
    assert not [payload for payload in payloads if payload["event"] == "oe_contract"]
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "oe_contract")
    assert event["operand_array_types"] == ["numpy.ndarray", "numpy.ndarray"]
    assert event["operand_array_backends"] == ["numpy", "numpy"]


def test_oe_contract_expression_profiling_records_operand_array_backends(caplog, monkeypatch, tmp_path):
    from renormalizer.utils import profiling
    from renormalizer.utils.log import PROFILING
    caplog.set_level(PROFILING, logger="renormalizer")
    event_path = tmp_path / "profile-events.jsonl"
    profiling.register_event_output(event_path)

    a = np.ones((2, 2))
    try:
        expr = oe_contract_expression("ij,jk->ik", a, (2, 2), constants=[0])
        expr(a)
        profiling.flush_event_output()
    finally:
        profiling.close_event_output()

    messages = [
        record.getMessage() for record in caplog.records
        if record.levelno == PROFILING and record.getMessage().startswith(profiling.LOG_PREFIX)
    ]
    payloads = [json.loads(message[len(profiling.LOG_PREFIX):]) for message in messages]
    assert not [payload for payload in payloads if payload["event"] == "oe_contract_expression"]
    event = next(payload for payload in _jsonl_payloads(event_path) if payload["event"] == "oe_contract_expression")
    assert event["operand_array_types"] == ["numpy.ndarray", "numpy.ndarray"]
    assert event["operand_array_backends"] == ["numpy", "numpy"]
