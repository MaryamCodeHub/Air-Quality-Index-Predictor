"""
Error Handling & Resilience Module

Provides production-grade error handling for API calls and data operations:
- Exponential backoff retry decorator
- Connection error recovery
- Request timeout handling
- Rate limit detection and backoff
- Structured error logging
- Circuit breaker pattern (optional)

Ensures pipeline resilience against transient API failures.
"""

import logging
import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
from datetime import datetime, timedelta

import requests
from requests.exceptions import (
    RequestException,
    ConnectionError,
    Timeout,
    HTTPError
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        backoff_factor: float = 1.0
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exponential_base: Factor for exponential growth (e.g., 2.0 = double each time)
            jitter: Add randomness to delay to prevent thundering herd
            backoff_factor: Multiplier for base_delay
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.backoff_factor = backoff_factor
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number (0-indexed).
        
        Formula: min(base * factor * (exponential_base ^ attempt), max_delay)
        With optional jitter added.
        
        Args:
            attempt: Attempt number (0, 1, 2, ...)
        
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * self.backoff_factor * (
            self.exponential_base ** attempt
        )
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            # Add ±20% jitter
            jitter_amount = delay * 0.2 * (random.random() - 0.5)
            delay = max(0.1, delay + jitter_amount)
        
        return delay


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    retriable_exceptions: Tuple[Type[Exception], ...] = (
        RequestException,
        ConnectionError,
        Timeout
    )
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Retries on:
    - Network errors (ConnectionError)
    - Timeouts
    - HTTP errors (unless explicitly non-retriable)
    - Other RequestExceptions
    
    Does NOT retry on:
    - Authentication errors (HTTP 401)
    - Permission errors (HTTP 403)
    - Not found errors (HTTP 404)
    - Validation errors (HTTP 400)
    
    Example:
        @retry_with_backoff(max_retries=3, base_delay=2)
        def fetch_api_data(url):
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
    
    Args:
        max_retries: Number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 60.0)
        exponential_base: Exponential growth factor (default: 2.0)
        jitter: Add randomness to prevent thundering herd (default: True)
        on_retry: Optional callback(attempt_num, exception) for logging
        retriable_exceptions: Tuple of exceptions to retry on
    
    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                
                except HTTPError as e:
                    # Check for non-retriable HTTP errors
                    status_code = e.response.status_code
                    
                    if status_code in [400, 401, 403, 404]:
                        # Client errors - don't retry
                        logger.error(
                            f"Non-retriable HTTP error: {status_code} {e.response.reason}. "
                            f"Func: {func.__name__}"
                        )
                        raise
                    
                    # Server errors (5xx) - retriable
                    if attempt < max_retries - 1:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"HTTP {status_code} error in {func.__name__}. "
                            f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                        )
                        if on_retry:
                            on_retry(attempt + 1, e)
                        time.sleep(delay)
                        last_exception = e
                        continue
                    else:
                        raise
                
                except retriable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = config.get_delay(attempt)
                        error_type = type(e).__name__
                        
                        logger.warning(
                            f"{error_type} in {func.__name__}: {str(e)[:100]}... "
                            f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                        )
                        
                        if on_retry:
                            on_retry(attempt + 1, e)
                        
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                            f"Final error: {error_type}: {str(e)[:100]}"
                        )
                        raise
                
                except Exception as e:
                    # Unexpected exceptions - fail immediately
                    logger.error(
                        f"Unexpected error in {func.__name__}: {type(e).__name__}: {e}"
                    )
                    raise
            
            # Should not reach here, but safety net
            if last_exception:
                raise last_exception
        
        return wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests rejected
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    
    Prevents cascading failures by failing fast when service is down.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = RequestException
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        logger.info(
            f"CircuitBreaker initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s"
        )
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: Original exception or CircuitBreakerOpen
        """
        if self.state == "OPEN":
            if self._should_attempt_recovery():
                self.state = "HALF_OPEN"
                logger.info("CircuitBreaker: Attempting recovery (HALF_OPEN)")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker OPEN. Recovery in "
                    f"{self._time_until_recovery():.0f}s"
                )
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self._on_success()
            
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_failure(self) -> None:
        """Record failure and update state."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(
            f"CircuitBreaker: Failure #{self.failure_count}. "
            f"State: {self.state}"
        )
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"CircuitBreaker: OPENED after {self.failure_count} failures"
            )
    
    def _on_success(self) -> None:
        """Record success and reset if recovering."""
        self.failure_count = 0
        self.state = "CLOSED"
        logger.info("CircuitBreaker: CLOSED. Service recovered.")
    
    def _should_attempt_recovery(self) -> bool:
        """Check if recovery timeout has elapsed."""
        if not self.last_failure_time:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _time_until_recovery(self) -> float:
        """Time remaining until recovery attempt."""
        if not self.last_failure_time:
            return 0
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        remaining = max(0, self.recovery_timeout - elapsed)
        return remaining


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


def make_resilient_request(
    url: str,
    method: str = 'GET',
    timeout: int = 10,
    max_retries: int = 3,
    backoff_base: float = 2,
    **kwargs
) -> requests.Response:
    """
    Make HTTP request with automatic retry and exponential backoff.
    
    Handles:
    - Connection errors
    - Timeouts
    - Server errors (5xx)
    - Rate limiting (429)
    
    Args:
        url: URL to request
        method: HTTP method (GET, POST, etc.)
        timeout: Request timeout in seconds
        max_retries: Number of retry attempts
        backoff_base: Exponential backoff multiplier
        **kwargs: Additional requests.request() arguments
    
    Returns:
        requests.Response object
    
    Raises:
        requests.RequestException: On final failure
    """
    @retry_with_backoff(
        max_retries=max_retries,
        base_delay=1.0,
        exponential_base=backoff_base
    )
    def _request():
        response = requests.request(
            method,
            url,
            timeout=timeout,
            **kwargs
        )
        response.raise_for_status()
        return response
    
    return _request()


def make_request_safe(
    url: str,
    method: str = 'GET',
    timeout: int = 10,
    max_retries: int = 3
) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Convenience function: make request, return structured response.
    
    Args:
        url: URL to request
        method: HTTP method
        timeout: Request timeout
        max_retries: Retry attempts
    
    Returns:
        (success, data_dict, error_message)
    """
    try:
        response = make_resilient_request(
            url,
            method=method,
            timeout=timeout,
            max_retries=max_retries
        )
        return True, response.json(), None
    
    except requests.Timeout:
        return False, None, f"Request timeout after {timeout}s"
    
    except requests.ConnectionError as e:
        return False, None, f"Connection error: {str(e)[:100]}"
    
    except requests.HTTPError as e:
        return False, None, f"HTTP {e.response.status_code}: {e.response.reason}"
    
    except Exception as e:
        return False, None, f"Unexpected error: {type(e).__name__}: {str(e)[:100]}"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Demo: Retry decorator
    @retry_with_backoff(max_retries=3, base_delay=0.5)
    def example_api_call(fail_times: int = 0):
        """Example function that fails N times then succeeds."""
        global attempt_count
        attempt_count = getattr(example_api_call, 'attempt', 0) + 1
        example_api_call.attempt = attempt_count
        
        if attempt_count <= fail_times:
            raise ConnectionError(f"Simulated failure #{attempt_count}")
        
        return {"status": "success", "attempt": attempt_count}
    
    print("\n" + "="*80)
    print("RETRY DECORATOR DEMO")
    print("="*80)
    
    # Test 1: Immediate success
    print("\nTest 1: Immediate success (no retries needed)")
    example_api_call.attempt = 0
    result = example_api_call(fail_times=0)
    print(f"Result: {result}")
    
    # Test 2: Success after retries
    print("\nTest 2: Success after 2 retries")
    example_api_call.attempt = 0
    result = example_api_call(fail_times=2)
    print(f"Result: {result}")
    
    # Test 3: Circuit breaker demo
    print("\n" + "="*80)
    print("CIRCUIT BREAKER DEMO")
    print("="*80)
    
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=5)
    
    def failing_service():
        raise ConnectionError("Service down")
    
    for i in range(5):
        try:
            print(f"\nAttempt {i+1}: Circuit state = {breaker.state}")
            breaker.call(failing_service)
        except (ConnectionError, CircuitBreakerOpen) as e:
            print(f"  Error: {e}")
