"""
Utility Helpers
===============
Common utility functions shared across all modules.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load YAML config and inject secrets from .env file.

    The .env file in the project root provides all sensitive values
    (API keys, SMTP credentials). These override the "ENV" placeholders
    in config.yaml at runtime.

    This is the SINGLE integration point — all other modules just call
    load_config() and receive a complete config dict with real values.
    """
    # Step 1: Load .env from project root
    root = Path(__file__).resolve().parent.parent.parent
    env_path = root / ".env"
    load_dotenv(env_path)

    # Step 2: Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Step 3: Inject secrets from environment variables into config
    config["api"]["aqicn"]["api_key"] = os.getenv("AQICN_API_KEY", config["api"]["aqicn"]["api_key"])
    config["alerting"]["sender_email"] = os.getenv("SENDER_EMAIL", config["alerting"]["sender_email"])
    config["alerting"]["sender_password"] = os.getenv("SENDER_PASSWORD", config["alerting"]["sender_password"])

    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create all required data directories defined in config['paths'].

    This is called at startup to guarantee the folder structure exists
    before any module tries to write data.
    """
    paths = config.get("paths", {})
    for _key, path in paths.items():
        os.makedirs(path, exist_ok=True)


def save_json(data: Dict, filepath: str) -> None:
    """Save a dictionary to a JSON file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict:
    """Load a dictionary from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_timestamp() -> str:
    """Return the current timestamp as 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    Assumes this file lives at <root>/src/utils/helpers.py
    """
    return Path(__file__).resolve().parent.parent.parent
