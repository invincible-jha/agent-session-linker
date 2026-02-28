/**
 * @aumos/agent-session-linker
 *
 * TypeScript client for the AumOS agent-session-linker library.
 * Provides HTTP client and session type definitions for cross-session
 * context persistence, resumption tokens, and session lifecycle management.
 */

// Client and configuration
export type { AgentSessionLinkerClient, AgentSessionLinkerClientConfig } from "./client.js";
export { createAgentSessionLinkerClient } from "./client.js";

// Core types
export type {
  TaskStatus,
  ContextSegment,
  EntityReference,
  TaskState,
  ToolContext,
  SessionState,
  CreateSessionRequest,
  ResumeSessionRequest,
  SaveContextRequest,
  ResumptionToken,
  SessionSummary,
  SessionStats,
  ApiError,
  ApiResult,
} from "./types.js";
