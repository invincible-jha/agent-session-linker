/**
 * HTTP client for the agent-session-linker session management API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
 *
 * @example
 * ```ts
 * import { createAgentSessionLinkerClient } from "@aumos/agent-session-linker";
 *
 * const client = createAgentSessionLinkerClient({ baseUrl: "http://localhost:8091" });
 *
 * const session = await client.createSession({ agent_id: "my-agent" });
 * if (session.ok) {
 *   console.log("Session created:", session.data.session_id);
 * }
 * ```
 */

import type {
  ApiError,
  ApiResult,
  CreateSessionRequest,
  ResumptionToken,
  ResumeSessionRequest,
  SaveContextRequest,
  SessionState,
  SessionStats,
  SessionSummary,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the AgentSessionLinkerClient. */
export interface AgentSessionLinkerClientConfig {
  /** Base URL of the agent-session-linker server (e.g. "http://localhost:8091"). */
  readonly baseUrl: string;
  /** Optional request timeout in milliseconds (default: 30000). */
  readonly timeoutMs?: number;
  /** Optional extra HTTP headers sent with every request. */
  readonly headers?: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    const body = await response.json() as unknown;

    if (!response.ok) {
      const errorBody = body as Partial<ApiError>;
      return {
        ok: false,
        error: {
          error: errorBody.error ?? "Unknown error",
          detail: errorBody.detail ?? "",
        },
        status: response.status,
      };
    }

    return { ok: true, data: body as T };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-session-linker server. */
export interface AgentSessionLinkerClient {
  /**
   * Create a new agent session.
   *
   * @param request - Optional session creation parameters including agent_id and preferences.
   * @returns The newly created SessionState record.
   */
  createSession(request?: CreateSessionRequest): Promise<ApiResult<SessionState>>;

  /**
   * Retrieve a session by its unique identifier.
   *
   * @param sessionId - The session to retrieve.
   * @returns The full SessionState for the requested session.
   */
  getSession(sessionId: string): Promise<ApiResult<SessionState>>;

  /**
   * Resume a session using a previously issued resumption token.
   *
   * @param request - Contains the opaque resumption token.
   * @returns The deserialised SessionState linked to the token.
   */
  resumeSession(request: ResumeSessionRequest): Promise<ApiResult<SessionState>>;

  /**
   * Save or update the context payload for an existing session.
   *
   * @param request - Contains the session_id and partial context fields to update.
   * @returns The updated SessionState after the save operation.
   */
  saveContext(request: SaveContextRequest): Promise<ApiResult<SessionState>>;

  /**
   * List all sessions, optionally filtered by agent identifier.
   *
   * @param options - Optional filter parameters.
   * @returns Array of SessionSummary records ordered by creation time descending.
   */
  listSessions(options?: {
    readonly agentId?: string;
    readonly limit?: number;
  }): Promise<ApiResult<readonly SessionSummary[]>>;

  /**
   * Delete a session from the session store.
   *
   * @param sessionId - The session to delete.
   * @returns An empty object on successful deletion.
   */
  deleteSession(sessionId: string): Promise<ApiResult<Readonly<Record<string, never>>>>;

  /**
   * Generate a resumption token for a session to enable cross-boundary transfer.
   *
   * @param sessionId - The session for which to generate a token.
   * @param options - Optional token lifetime in seconds (default: 3600).
   * @returns A ResumptionToken that can be serialised and passed to resumeSession().
   */
  getResumptionToken(
    sessionId: string,
    options?: { readonly ttlSeconds?: number },
  ): Promise<ApiResult<ResumptionToken>>;

  /**
   * Retrieve aggregate statistics across all stored sessions.
   *
   * @returns A SessionStats record with totals and agent breakdown.
   */
  getStats(): Promise<ApiResult<SessionStats>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-session-linker server.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentSessionLinkerClient instance.
 */
export function createAgentSessionLinkerClient(
  config: AgentSessionLinkerClientConfig,
): AgentSessionLinkerClient {
  const { baseUrl, timeoutMs = 30_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async createSession(
      request?: CreateSessionRequest,
    ): Promise<ApiResult<SessionState>> {
      return fetchJson<SessionState>(
        `${baseUrl}/sessions`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request ?? {}),
        },
        timeoutMs,
      );
    },

    async getSession(sessionId: string): Promise<ApiResult<SessionState>> {
      return fetchJson<SessionState>(
        `${baseUrl}/sessions/${encodeURIComponent(sessionId)}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async resumeSession(
      request: ResumeSessionRequest,
    ): Promise<ApiResult<SessionState>> {
      return fetchJson<SessionState>(
        `${baseUrl}/sessions/resume`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async saveContext(
      request: SaveContextRequest,
    ): Promise<ApiResult<SessionState>> {
      return fetchJson<SessionState>(
        `${baseUrl}/sessions/${encodeURIComponent(request.session_id)}/context`,
        {
          method: "PATCH",
          headers: baseHeaders,
          body: JSON.stringify(request.context),
        },
        timeoutMs,
      );
    },

    async listSessions(options?: {
      readonly agentId?: string;
      readonly limit?: number;
    }): Promise<ApiResult<readonly SessionSummary[]>> {
      const params = new URLSearchParams();
      if (options?.agentId !== undefined) {
        params.set("agent_id", options.agentId);
      }
      if (options?.limit !== undefined) {
        params.set("limit", String(options.limit));
      }
      const queryString = params.toString();
      const url = queryString
        ? `${baseUrl}/sessions?${queryString}`
        : `${baseUrl}/sessions`;
      return fetchJson<readonly SessionSummary[]>(
        url,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async deleteSession(
      sessionId: string,
    ): Promise<ApiResult<Readonly<Record<string, never>>>> {
      return fetchJson<Readonly<Record<string, never>>>(
        `${baseUrl}/sessions/${encodeURIComponent(sessionId)}`,
        { method: "DELETE", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getResumptionToken(
      sessionId: string,
      options?: { readonly ttlSeconds?: number },
    ): Promise<ApiResult<ResumptionToken>> {
      const params = new URLSearchParams();
      if (options?.ttlSeconds !== undefined) {
        params.set("ttl_seconds", String(options.ttlSeconds));
      }
      const queryString = params.toString();
      const url = queryString
        ? `${baseUrl}/sessions/${encodeURIComponent(sessionId)}/token?${queryString}`
        : `${baseUrl}/sessions/${encodeURIComponent(sessionId)}/token`;
      return fetchJson<ResumptionToken>(
        url,
        { method: "POST", headers: baseHeaders, body: JSON.stringify({}) },
        timeoutMs,
      );
    },

    async getStats(): Promise<ApiResult<SessionStats>> {
      return fetchJson<SessionStats>(
        `${baseUrl}/sessions/stats`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },
  };
}

/** Re-export config type for convenience. */
export type {
  CreateSessionRequest,
  ResumeSessionRequest,
  SaveContextRequest,
  SessionState,
  SessionSummary,
  SessionStats,
  ResumptionToken,
};
