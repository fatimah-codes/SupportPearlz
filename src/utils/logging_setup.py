"""
Centralized logging configuration for SupportPearlz.
Supports both console streaming and persistent file logging with structured formatting.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from src.config import settings


def setup_logger(name: str = "supportpearlz") -> logging.Logger:
    """Configure and return the central application logger."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    log_level_str = getattr(settings, "LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Standard formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        log_file = Path(settings.LOG_FILE_PATH)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to attach file logger handler: {e}")

    logger.propagate = False
    return logger


logger = setup_logger()


def log_turn(
    session_id: str,
    original_query: str,
    rewritten_query: str,
    retrieved_chunk_ids: List[str],
    retrieved_scores: List[float],
    latency_ms: float,
    refusal_flag: bool,
    error: Optional[str] = None
) -> None:
    """Log a complete conversation turn with structured telemetry."""
    score_summary = [f"{cid} (score: {s:.3f})" for cid, s in zip(retrieved_chunk_ids, retrieved_scores)]
    log_msg = (
        f"TURN COMPLETED | Session: {session_id} | "
        f"Original: '{original_query}' | "
        f"Rewritten: '{rewritten_query}' | "
        f"Retrieved Chunks: {score_summary} | "
        f"Latency: {latency_ms:.1f}ms | "
        f"Refusal: {refusal_flag}"
    )
    if error:
        log_msg += f" | Error: {error}"
        logger.error(log_msg)
    else:
        logger.info(log_msg)
