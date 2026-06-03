"""
Structured Logging System
=========================
Centralized logging configuration for all modules.
Supports both console and rotating file output with structured formatting.
"""

import logging
import os
from datetime import datetime

import yaml


def setup_logger(name: str, config_path: str = "configs/config.yaml") -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name: Logger name (typically module name like 'ingestion.api_client')
        config_path: Path to the YAML config file

    Returns:
        Configured logging.Logger instance with console + file handlers
    """
    # ----- Load logging config from YAML -----
    log_level = "INFO"
    log_format = "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s"
    file_enabled = True
    log_dir = "logs"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        log_cfg = config.get("logging", {})
        log_level = log_cfg.get("level", log_level)
        log_format = log_cfg.get("format", log_format)
        file_enabled = log_cfg.get("file_enabled", file_enabled)
        log_dir = config.get("paths", {}).get("logs", log_dir)
    except FileNotFoundError:
        pass  # Use defaults if config not found

    level = getattr(logging, log_level, logging.INFO)

    # ----- Create logger -----
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(log_format)

    # ----- Console handler -----
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ----- File handler (one log file per module per day) -----
    if file_enabled:
        os.makedirs(log_dir, exist_ok=True)
        safe_name = name.replace(".", "_")
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{safe_name}_{date_str}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
