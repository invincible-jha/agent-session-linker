"""Cross-platform file locking for concurrent USF file access.

This module provides a simple, cross-platform advisory file lock based on
exclusive file creation.  It works on POSIX and Windows without any
third-party dependencies.

Classes
-------
FileLock
    Acquires an exclusive lock on a sentinel ``.lock`` file.  Supports both
    explicit ``acquire``/``release`` calls and the context-manager protocol.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import IO

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

_POLL_INTERVAL_SECONDS: float = 0.05  # 50 ms between lock-acquisition retries


# ---------------------------------------------------------------------------
# FileLock
# ---------------------------------------------------------------------------


class FileLock:
    """Cross-platform advisory file lock using exclusive file creation.

    The lock is implemented by atomically creating a sentinel lock file.
    If the file already exists, the caller retries until the timeout expires.

    Parameters
    ----------
    lock_path:
        Path to the sentinel lock file.  The file is created on acquisition
        and deleted on release.
    timeout:
        Maximum number of seconds to wait before raising :class:`TimeoutError`.
        Defaults to 10 seconds.

    Raises
    ------
    TimeoutError
        If the lock cannot be acquired within *timeout* seconds.
    """

    def __init__(self, lock_path: str | Path, timeout: float = 10.0) -> None:
        self._lock_path: Path = Path(lock_path)
        self._timeout: float = timeout
        self._lock_file: IO[str] | None = None

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    def acquire(self) -> None:
        """Acquire the file lock, blocking until success or timeout.

        Uses ``open(..., "x")`` (exclusive creation) as the atomic primitive.
        On Windows and POSIX alike, this either succeeds immediately or raises
        :class:`FileExistsError` when another process holds the lock.

        Raises
        ------
        TimeoutError
            If the lock cannot be acquired within :attr:`_timeout` seconds.
        """
        start = time.monotonic()
        while True:
            try:
                self._lock_file = open(self._lock_path, "x")  # noqa: WPS515
                return
            except FileExistsError:
                elapsed = time.monotonic() - start
                if elapsed >= self._timeout:
                    raise TimeoutError(
                        f"Could not acquire lock {self._lock_path} within {self._timeout}s"
                    )
                time.sleep(_POLL_INTERVAL_SECONDS)

    def release(self) -> None:
        """Release the file lock and delete the sentinel file.

        Idempotent: calling ``release()`` when the lock is not held is safe.
        """
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> FileLock:
        """Acquire the lock on context entry."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Release the lock on context exit, even if an exception occurred."""
        self.release()
