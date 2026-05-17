"""
Base API Ingestion Client Module
=================================
Implements abstract HTTP client patterns, local caching, 
exponential backoffs, and the Circuit Breaker resilience pattern.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import requests

from src.utils.logger import setup_logger

logger = setup_logger("ingestion.base_client")


# ============================================================
# Circuit Breaker Pattern
# ============================================================

@dataclass
class CircuitBreaker:
    """
    Circuit Breaker for API resilience.

    CLOSED    → requests pass through
    OPEN      → too many failures, block for cooldown
    HALF_OPEN → cooldown elapsed, allow one test request
    """
    failure_threshold: int = 5
    cooldown_seconds: int = 300
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure: Optional[datetime] = field(default=None, init=False, repr=False)
    _state: str = field(default="CLOSED", init=False, repr=False)

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure = datetime.now()
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(
                f"Circuit breaker OPENED after {self._failure_count} failures. "
                f"Cooldown: {self.cooldown_seconds}s"
            )

    def can_proceed(self) -> bool:
        if self._state == "CLOSED":
            return True
        if self._state == "OPEN" and self._last_failure:
            elapsed = (datetime.now() - self._last_failure).total_seconds()
            if elapsed >= self.cooldown_seconds:
                self._state = "HALF_OPEN"
                logger.info("Circuit breaker → HALF_OPEN (testing)")
                return True
            return False
        return True  # HALF_OPEN allows one probe


# ============================================================
# Base API Client
# ============================================================

class BaseAPIClient:
    """Base class: resilient HTTP GET with retry + circuit breaker + cache."""

    def __init__(self, config: Dict[str, Any], api_name: str):
        self.config = config
        self.api_name = api_name
        res = config.get("resilience", {})
        self.max_retries = res.get("max_retries", 3)
        self.base_backoff = res.get("base_backoff_seconds", 2)
        self.timeout = res.get("request_timeout_seconds", 30)

        self.cache_dir = os.path.join(config["paths"]["cache"], api_name)
        os.makedirs(self.cache_dir, exist_ok=True)

        cb = res.get("circuit_breaker", {})
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=cb.get("failure_threshold", 5),
            cooldown_seconds=cb.get("cooldown_seconds", 300),
        )

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """HTTP GET with retry loop, circuit breaker, and caching."""
        if not self.circuit_breaker.can_proceed():
            logger.warning(f"[{self.api_name}] Circuit OPEN — skipping request")
            return None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"[{self.api_name}] Attempt {attempt}/{self.max_retries}")
                resp = requests.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                self.circuit_breaker.record_success()
                self._cache_response(url, data)
                return data
            except requests.exceptions.Timeout:
                logger.warning(f"[{self.api_name}] Timeout (attempt {attempt})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"[{self.api_name}] Connection error (attempt {attempt})")
            except requests.exceptions.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else 0
                if 400 <= code < 500 and code != 429:
                    logger.error(f"[{self.api_name}] Client error {code}: {exc}")
                    self.circuit_breaker.record_failure()
                    return None
                logger.warning(f"[{self.api_name}] HTTP {code} (attempt {attempt})")
            except requests.exceptions.RequestException as exc:
                logger.warning(f"[{self.api_name}] Error (attempt {attempt}): {exc}")

            if attempt < self.max_retries:
                wait = self.base_backoff ** attempt
                logger.info(f"[{self.api_name}] Retrying in {wait}s …")
                time.sleep(wait)

        self.circuit_breaker.record_failure()
        logger.error(f"[{self.api_name}] All {self.max_retries} retries exhausted")
        return None

    def _cache_key(self, url: str) -> str:
        return url.replace("/", "_").replace(":", "").replace("?", "_").replace("&", "_")[:120]

    def _cache_response(self, url: str, data: Dict) -> None:
        try:
            path = os.path.join(self.cache_dir, f"{self._cache_key(url)}.json")
            with open(path, "w") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f)
        except Exception as exc:
            logger.warning(f"[{self.api_name}] Cache write failed: {exc}")

    def _get_cached_response(self, url: str) -> Optional[Dict]:
        try:
            path = os.path.join(self.cache_dir, f"{self._cache_key(url)}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    cached = json.load(f)
                age = (datetime.now() - datetime.fromisoformat(cached["timestamp"])).total_seconds()
                logger.warning(f"[{self.api_name}] Serving CACHED data (age {age:.0f}s)")
                return cached["data"]
        except Exception as exc:
            logger.warning(f"[{self.api_name}] Cache read failed: {exc}")
        return None
