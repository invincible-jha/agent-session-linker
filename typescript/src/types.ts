/**
 * TypeScript interfaces for the agent-session-linker session management library.
 *
 * Mirrors the Pydantic models defined in:
 *   agent_session_linker.session.state
 *   agent_session_linker.session.manager
 *   agent_session_linker.session.serializer
 *
 * All interfaces use readonly fields to match Python's frozen Pydantic models.
 */

// ---------------------------------------------------------------------------
// Task lifecycle
// ---------------------------------------------------------------------------

/**
 * Lifecycle state of a tracked task within a session.
 * Maps to TaskStatus enum in Python.
 */
export type TaskStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed"
  | "cancelled";

// ---------------------------------------------------------------------------
// Context segment
// ---------------------------------------------------------------------------

/**
 * A discrete unit of context captured from a single conversation turn.
 * Maps to ContextSegment in Python.
 */
export interface ContextSegment {
  /** Unique identifier for this segment. */
  readonly segment_id: string;
  /** Message role: "user", "assistant", "system", or "tool". */
  readonly role: string;
  /** Raw text content of the segment. */
  readonly content: string;
  /** Estimated token count for this segment. */
  readonly token_count: number;
  /** Categorical label such as "conversation", "reasoning", "code", "plan", "output", or "metadata". */
  readonly segment_type: string;
  /** ISO-8601 UTC timestamp when this segment was captured. */
  readonly timestamp: string;
  /** Zero-based index of the conversation turn this segment belongs to. */
  readonly turn_index: number;
  /** Arbitrary additional key-value data attached to this segment. */
  readonly metadata: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Entity reference
// ---------------------------------------------------------------------------

/**
 * A cross-session pointer to a tracked domain entity.
 * Maps to EntityReference in Python.
 */
export interface EntityReference {
  /** Unique identifier for this entity record. */
  readonly entity_id: string;
  /** Primary normalised name used for matching. */
  readonly canonical_name: string;
  /** Categorical label such as "person", "project", "file", "concept", "tool", or "organisation". */
  readonly entity_type: string;
  /** Alternative names or spellings for this entity. */
  readonly aliases: readonly string[];
  /** Key-value attributes describing the entity. */
  readonly attributes: Readonly<Record<string, string>>;
  /** Session ID where this entity was first observed. */
  readonly first_seen_session: string;
  /** Session ID where this entity was most recently observed. */
  readonly last_seen_session: string;
  /** Match/extraction confidence in the range [0.0, 1.0]. */
  readonly confidence: number;
}

// ---------------------------------------------------------------------------
// Task state
// ---------------------------------------------------------------------------

/**
 * A tracked task with its current lifecycle status.
 * Maps to TaskState in Python.
 */
export interface TaskState {
  /** Unique identifier for this task. */
  readonly task_id: string;
  /** Short human-readable title. */
  readonly title: string;
  /** Detailed description of what the task requires. */
  readonly description: string;
  /** Current lifecycle status. */
  readonly status: TaskStatus;
  /** Integer priority where 1 is highest. Range [1, 10]. */
  readonly priority: number;
  /** ISO-8601 UTC timestamp when the task was first recorded. */
  readonly created_at: string;
  /** ISO-8601 UTC timestamp when the task was last modified. */
  readonly updated_at: string;
  /** Optional reference to a parent task for sub-task hierarchies. */
  readonly parent_task_id: string | null;
  /** Free-form labels for categorisation and filtering. */
  readonly tags: readonly string[];
  /** Additional free-text notes about the task. */
  readonly notes: string;
}

// ---------------------------------------------------------------------------
// Tool context
// ---------------------------------------------------------------------------

/**
 * A record of a single tool invocation within a session.
 * Maps to ToolContext in Python.
 */
export interface ToolContext {
  /** Unique identifier for this invocation. */
  readonly invocation_id: string;
  /** Name of the tool that was called. */
  readonly tool_name: string;
  /** Brief summary of the input arguments (not the full payload). */
  readonly input_summary: string;
  /** Brief summary of the output returned. */
  readonly output_summary: string;
  /** Wall-clock execution time in milliseconds. */
  readonly duration_ms: number;
  /** Whether the invocation completed without error. */
  readonly success: boolean;
  /** Error description when success is false. */
  readonly error_message: string;
  /** ISO-8601 UTC timestamp when the invocation started. */
  readonly timestamp: string;
  /** Tokens consumed by the tool call, if measurable. */
  readonly token_cost: number;
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

/**
 * Complete snapshot of an agent session.
 * Maps to SessionState in Python.
 */
export interface SessionState {
  /** Globally unique session identifier. */
  readonly session_id: string;
  /** Identifier for the agent or agent-type that owns this session. */
  readonly agent_id: string;
  /** Schema version string used for forward/backward compatibility. */
  readonly schema_version: string;
  /** Ordered list of conversation context segments. */
  readonly segments: readonly ContextSegment[];
  /** Named entities tracked within this session. */
  readonly entities: readonly EntityReference[];
  /** Tasks created or referenced during this session. */
  readonly tasks: readonly TaskState[];
  /** Chronological log of tool invocations. */
  readonly tools_used: readonly ToolContext[];
  /** Agent or user preferences captured during the session. */
  readonly preferences: Readonly<Record<string, string>>;
  /** Optional compressed summary of the session's key content. */
  readonly summary: string;
  /** If this is a continuation, the ID of the preceding session. */
  readonly parent_session_id: string | null;
  /** Accumulated LLM API cost in USD for this session. */
  readonly total_cost_usd: number;
  /** ISO-8601 UTC timestamp of session creation. */
  readonly created_at: string;
  /** ISO-8601 UTC timestamp of last modification. */
  readonly updated_at: string;
  /** SHA-256 of the session's canonical JSON (excluding this field). */
  readonly checksum: string;
}

// ---------------------------------------------------------------------------
// API request types
// ---------------------------------------------------------------------------

/** Configuration for creating a new session. */
export interface CreateSessionRequest {
  /** Agent identifier that owns this session. */
  readonly agent_id?: string;
  /** Optional parent session to create a continuation from. */
  readonly parent_session_id?: string;
  /** Initial preference key-value pairs. */
  readonly preferences?: Readonly<Record<string, string>>;
}

/** Request to resume a session using a resumption token. */
export interface ResumeSessionRequest {
  /** The opaque resumption token previously issued by the server. */
  readonly resumption_token: string;
}

/** Request to save a serialised context payload. */
export interface SaveContextRequest {
  /** The session ID to update. */
  readonly session_id: string;
  /** Partial or full updated session state fields. */
  readonly context: Partial<
    Pick<SessionState, "segments" | "entities" | "tasks" | "tools_used" | "preferences" | "summary" | "total_cost_usd">
  >;
}

// ---------------------------------------------------------------------------
// Resumption token
// ---------------------------------------------------------------------------

/**
 * An opaque token that enables resuming a serialised session across boundaries.
 * Maps to the resumption token concept in agent_session_linker.portable.
 */
export interface ResumptionToken {
  /** The token value to pass to resumeSession(). */
  readonly token: string;
  /** Session ID this token references. */
  readonly session_id: string;
  /** ISO-8601 UTC timestamp after which this token expires. */
  readonly expires_at: string;
  /** Schema version of the serialised payload. */
  readonly schema_version: string;
}

// ---------------------------------------------------------------------------
// Session list entry
// ---------------------------------------------------------------------------

/** Summary information returned when listing sessions. */
export interface SessionSummary {
  /** Unique session identifier. */
  readonly session_id: string;
  /** Agent that owns this session. */
  readonly agent_id: string;
  /** ISO-8601 UTC creation timestamp. */
  readonly created_at: string;
  /** ISO-8601 UTC last-update timestamp. */
  readonly updated_at: string;
  /** Total number of context segments. */
  readonly segment_count: number;
  /** Total number of tracked tasks. */
  readonly task_count: number;
  /** Whether this session has a parent (is a continuation). */
  readonly is_continuation: boolean;
}

// ---------------------------------------------------------------------------
// Session statistics
// ---------------------------------------------------------------------------

/** Aggregate statistics across all stored sessions. */
export interface SessionStats {
  /** Total number of stored sessions. */
  readonly total_sessions: number;
  /** Sum of all segment counts across sessions. */
  readonly total_segments: number;
  /** Sum of all token counts across sessions. */
  readonly total_tokens: number;
  /** Sum of all task counts across sessions. */
  readonly total_tasks: number;
  /** Sum of all entity counts across sessions. */
  readonly total_entities: number;
  /** Summed cost across all sessions in USD. */
  readonly total_cost_usd: number;
  /** List of unique agent IDs that have sessions. */
  readonly agents: readonly string[];
}

// ---------------------------------------------------------------------------
// API result wrapper
// ---------------------------------------------------------------------------

/** Standard error payload returned by the agent-session-linker API. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for all client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
