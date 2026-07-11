import logging

from renormalizer.utils.log import (
    INFO,
    PROFILING,
    default_stream_handler,
    init_log,
    parse_log_level,
    register_file_output,
)


def test_profiling_level_is_registered_and_enabled_by_init_log():
    init_log(PROFILING)

    assert PROFILING == 5
    assert logging.getLevelName(PROFILING) == "PROFILING"
    assert parse_log_level("PROFILING") == PROFILING
    assert parse_log_level("5") == PROFILING
    assert logging.getLogger("renormalizer").isEnabledFor(PROFILING)
    assert default_stream_handler.level <= PROFILING
    assert hasattr(logging.getLogger("renormalizer"), "profiling")


def test_file_output_respects_explicit_info_threshold(tmp_path):
    logger = logging.getLogger("renormalizer")
    path = tmp_path / "renormalizer.log"
    init_log(PROFILING)
    handler = register_file_output(path, level=INFO)

    try:
        logger.profiling("profiling record must be filtered")
        logger.info("info record must be retained")
        handler.flush()
    finally:
        logger.removeHandler(handler)
        handler.close()
        init_log(logging.DEBUG)

    output = path.read_text()
    assert "profiling record must be filtered" not in output
    assert "info record must be retained" in output
