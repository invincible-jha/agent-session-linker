/**
 * HTTP client for the agent-session-linker session management API.
 *
 * Delegates all HTTP transport to `@aumos/sdk-core` which provides
 * automatic retry with exponential back-off, timeout management via
 * `AbortSignal.timeout`, interceptor support, and a typed error hierarchy.
 *
 * The public-facing `ApiResult<T>` envelope is preserved for full
 * backward compatibility with existing callers.
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

import {
  createHttpClient,
  HttpError,
  NetworkError,
  TimeoutError,
  AumosError,
  type HttpClient,
} from "@aumos/sdk-core";

import type {
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
// Internal adapter
// ---------------------------------------------------------------------------

async function callApi<T>(
  operation: () => Promise<{ readonly data: T; readonly status: number }>,
): Promise<ApiResult<T>> {
  try {
    const response = await operation();
    return { ok: true, data: response.data };
  } catch (error: unknown) {
    if (error instanceof HttpError) {
      return {
        ok: false,
        error: { error: error.message, detail: String(error.body ?? "") },
        status: error.statusCode,
      };
    }
    if (error instanceof TimeoutError) {
      return {
        ok: false,
        error: { error: "Request timed out", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof NetworkError) {
      return {
        ok: false,
        error: { error: "Network error", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof AumosError) {
      return {
        ok: false,
        error: { error: error.code, detail: error.message },
        status: error.statusCode ?? 0,
      };
    }
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: { error: "Unexpected error", detail: message },
      status: 0,
    };
  }
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
  const http: HttpClient = createHttpClient({
    baseUrl: config.baseUrl,
    timeout: config.timeoutMs ?? 30_000,
    defaultHeaders: config.headers,
  });

  return {
    createSession(request?: CreateSessionRequest): Promise<ApiResult<SessionState>> {
      return callApi(() => http.post<SessionState>("/sessions", request ?? {}));
    },

    getSession(sessionId: string): Promise<ApiResult<SessionState>> {
      return callApi(() =>
        http.get<SessionState>(`/sessions/${encodeURIComponent(sessionId)}`),
      );
    },

    resumeSession(request: ResumeSessionRequest): Promise<ApiResult<SessionState>> {
      return callApi(() => http.post<SessionState>("/sessions/resume", request));
    },

    saveContext(request: SaveContextRequest): Promise<ApiResult<SessionState>> {
      return callApi(() =>
        http.patch<SessionState>(
          `/sessions/${encodeURIComponent(request.session_id)}/context`,
          request.context,
        ),
      );
    },

    listSessions(options?: {
      readonly agentId?: string;
      readonly limit?: number;
    }): Promise<ApiResult<readonly SessionSummary[]>> {
      const queryParams: Record<string, string> = {};
      if (options?.agentId !== undefined) queryParams["agent_id"] = options.agentId;
      if (options?.limit !== undefined) queryParams["limit"] = String(options.limit);
      return callApi(() =>
        http.get<readonly SessionSummary[]>("/sessions", { queryParams }),
      );
    },

    deleteSession(
      sessionId: string,
    ): Promise<ApiResult<Readonly<Record<string, never>>>> {
      return callApi(() =>
        http.delete<Readonly<Record<string, never>>>(
          `/sessions/${encodeURIComponent(sessionId)}`,
        ),
      );
    },

    getResumptionToken(
      sessionId: string,
      options?: { readonly ttlSeconds?: number },
    ): Promise<ApiResult<ResumptionToken>> {
      const queryParams: Record<string, string> = {};
      if (options?.ttlSeconds !== undefined) {
        queryParams["ttl_seconds"] = String(options.ttlSeconds);
      }
      return callApi(() =>
        http.post<ResumptionToken>(
          `/sessions/${encodeURIComponent(sessionId)}/token`,
          {},
          { queryParams },
        ),
      );
    },

    getStats(): Promise<ApiResult<SessionStats>> {
      return callApi(() => http.get<SessionStats>("/sessions/stats"));
    },
  };
}
