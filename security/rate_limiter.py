"""
Sliding window rate limiter for API protection.
"""

import time
from collections import defaultdict
from typing import Dict, List

from config import settings


class RateLimiter:
    """
    In-memory sliding window rate limiter.
    Tracks per-user request timestamps and enforces limits.
    """

    def __init__(self, max_requests: int = None, window_seconds: int = None):
        self.max_requests = max_requests or settings.RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or settings.RATE_LIMIT_WINDOW_SECONDS
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, user_id: str) -> bool:
        """
        Check if a request from user_id is allowed.

        Returns True if under rate limit, False otherwise.
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove expired timestamps
        self._requests[user_id] = [
            ts for ts in self._requests[user_id] if ts > window_start
        ]

        # Check limit
        if len(self._requests[user_id]) >= self.max_requests:
            return False

        # Record this request
        self._requests[user_id].append(now)
        return True

    def get_remaining(self, user_id: str) -> int:
        """Get the number of remaining requests for a user."""
        now = time.time()
        window_start = now - self.window_seconds

        self._requests[user_id] = [
            ts for ts in self._requests[user_id] if ts > window_start
        ]

        return max(0, self.max_requests - len(self._requests[user_id]))

    def get_reset_time(self, user_id: str) -> float:
        """Get seconds until the rate limit window resets for a user."""
        if not self._requests[user_id]:
            return 0

        oldest = min(self._requests[user_id])
        reset_at = oldest + self.window_seconds
        return max(0, reset_at - time.time())
