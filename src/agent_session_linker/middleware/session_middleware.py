"""Session auto-save / auto-load middleware.

Intercepts the start and end of each request cycle to transparently
load the appropriate session state before the request is processed and
persist any context changes afterwards.

Classes
-------
- SessionMiddleware  — before/after request hooks for session management
"""
from __future__ import annotations

import logging

from agent_session_linker.session.manager import SessionManager, SessionNotFoundError
from agent_session_linker.session.state import ContextSegment, SessionState

logger = logging.getLogger(__name__)


class SessionMiddleware:
    """Auto-save and auto-load session state around each request cycle.

    This middleware wraps a ``SessionManager`` and exposes two hook
    methods: ``before_request`` (loads the session) and ``after_request``
    (appends new context and persists the session).

    It is intentionally framework-agnostic: callers are responsible for
    calling the hooks at the right points in their own request pipeline.

    Parameters
    ----------
    manager:
        The session manager to delegate to.
    auto_create:
        When True (default), if no session exists for ``session_id`` a
        new empty session is created rather than raising an error.
    """

    def __init__(
        self,
        manager: SessionManager,
        auto_create: bool = True,
    ) -> None:
        self._manager = manager
        self.auto_create = auto_create
        self._active_sessions: dict[str, SessionState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def before_request(self, session_id: str) -> SessionState:
        """Load (or create) the session for ``session_id``.

        The loaded session is cached in-memory for the duration of the
        request and returned for immediate use.

        Parameters
        ----------
        session_id:
            The session identifier for the incoming request.

        Returns
        -------
        SessionState
            The existing or newly created session.

        Raises
        ------
        SessionNotFoundError
            If ``auto_create`` is False and no session exists for
            ``session_id``.
        """
        if self._manager.session_exists(session_id):
            session = self._manager.load_session(session_id)
            logger.debug("SessionMiddleware: loaded session %r", session_id)
        elif self.auto_create:
            session = self._manager.create_session()
            # Force the session ID to match the requested one.
            session.session_id = session_id
            logger.debug("SessionMiddleware: created new session %r", session_id)
        else:
            raise SessionNotFoundError(session_id)

        self._active_sessions[session_id] = session
        return session

    def after_request(
        self,
        session_id: str,
        new_context: list[ContextSegment] | str | None = None,
    ) -> str:
        """Persist the session and optionally append new context.

        If ``new_context`` is provided the segments (or single text string)
        are appended to the session before it is saved.

        Parameters
        ----------
        session_id:
            The session identifier used in the preceding ``before_request``
            call.
        new_context:
            Context to append before saving.  May be:
            - A list of ``ContextSegment`` objects appended directly.
            - A plain string added as a single ``"assistant"`` segment.
            - None to save without any new context.

        Returns
        -------
        str
            The ``session_id`` under which the session was saved.

        Raises
        ------
        KeyError
            If ``before_request`` was not called for ``session_id`` in this
            request cycle (session not in the active cache).
        """
        session = self._active_sessions.get(session_id)
        if session is None:
            raise KeyError(
                f"No active session for {session_id!r}. "
                "Call before_request() before after_request()."
            )

        if isinstance(new_context, str):
            session.add_segment(role="assistant", content=new_context)
        elif isinstance(new_context, list):
            for segment in new_context:
                session.segments.append(segment)

        saved_id = self._manager.save_session(session)
        del self._active_sessions[session_id]
        logger.debug("SessionMiddleware: saved session %r", session_id)
        return saved_id

    def get_active(self, session_id: str) -> SessionState | None:
        """Return the currently active (in-flight) session, if any.

        Parameters
        ----------
        session_id:
            Session identifier to look up.

        Returns
        -------
        SessionState | None
            The cached session state, or None if not currently active.
        """
        return self._active_sessions.get(session_id)

    def clear_active(self, session_id: str) -> None:
        """Discard the cached active session without saving.

        Useful for rolling back a request cycle that encountered an error.

        Parameters
        ----------
        session_id:
            Session identifier to discard.
        """
        self._active_sessions.pop(session_id, None)
        logger.debug("SessionMiddleware: cleared active session %r", session_id)
