"""Central logger factory used by every pipeline stage."""

import logging
from pathlib import Path

def get_logger(name: str = "gdelt_pipeline", log_to_file: bool = True) -> logging.Logger:
    """
    Return a named logger with console (and optional file) handlers.

    Handlers are attached only once per logger name: repeated calls with the
    same `name` (e.g. across module re-imports) return the same configured
    logger instead of duplicating handlers and doubling log output.

    Parameters
    ----------
    name : str
        Logger name, typically `__name__` of the calling module.
    log_to_file : bool
        If True, also attach a FileHandler writing to logs/pipeline.log
        (directory created if missing).

    Returns
    -------
    logging.Logger
        Configured at INFO level with a timestamped formatter.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # Prevent duplicate handlers during imports
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        # ---- console handler ----
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ---- optional file handler ----
        if log_to_file:
            log_path = Path("logs")
            log_path.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_path / "pipeline.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger