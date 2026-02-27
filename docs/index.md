# agent-session-linker

Cross-Session Context Persistence — session store, context serialization, cross-framework portability.

[![CI](https://github.com/invincible-jha/agent-session-linker/actions/workflows/ci.yaml/badge.svg)](https://github.com/invincible-jha/agent-session-linker/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-session-linker.svg)](https://pypi.org/project/agent-session-linker/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-session-linker.svg)](https://pypi.org/project/agent-session-linker/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/invincible-jha/agent-session-linker/blob/main/LICENSE)

---

## Installation

```bash
pip install agent-session-linker
```

Verify the installation:

```bash
agent-session-linker version
```

---

## Quick Start

```python
import agent_session_linker

# See examples/01_quickstart.py for a complete working example
```

---

## Key Features

- **`SessionManager`** provides create/save/load/list/delete lifecycle management over a pluggable `StorageBackend` ABC with checksum validation on every serialization round-trip
- **Five storage backends** out of the box — in-memory, filesystem (JSON files), SQLite, Redis, and S3 — swap backends without changing application code
- **`ContextInjector`** selects and injects prior session context into a new conversation based on relevance scoring and configurable freshness thresholds
- **Entity extraction and cross-session linking** tracks named entities (people, places, topics) across sessions so the agent can resume threads without being explicitly told what it discussed
- **`ContextSummarizer`** compresses long session histories to fit within token budgets while preserving the entities and decisions that matter
- **Session middleware** with automatic checkpoint creation and context-window management that prevents unbounded context growth
- **`SessionLinker`** chains related sessions by shared entities, enabling multi-session reasoning across conversations that span days or weeks

---

## Links

- [GitHub Repository](https://github.com/invincible-jha/agent-session-linker)
- [PyPI Package](https://pypi.org/project/agent-session-linker/)
- [Architecture](architecture.md)
- [Contributing](https://github.com/invincible-jha/agent-session-linker/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/invincible-jha/agent-session-linker/blob/main/CHANGELOG.md)

---

## License

Apache 2.0 — see [LICENSE](https://github.com/invincible-jha/agent-session-linker/blob/main/LICENSE) for full terms.

---

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure.
