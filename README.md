# agent-session-linker

Cross-session context persistence and resumption

[![CI](https://github.com/aumos-ai/agent-session-linker/actions/workflows/ci.yaml/badge.svg)](https://github.com/aumos-ai/agent-session-linker/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/agent-session-linker.svg)](https://pypi.org/project/agent-session-linker/)
[![Python versions](https://img.shields.io/pypi/pyversions/agent-session-linker.svg)](https://pypi.org/project/agent-session-linker/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [AumOS](https://github.com/aumos-ai) open-source agent infrastructure portfolio.

---

## Features

- `SessionManager` provides create/save/load/list/delete lifecycle management over a pluggable `StorageBackend` ABC with checksum validation on every serialization round-trip
- Five storage backends out of the box: in-memory, filesystem (JSON files), SQLite, Redis, and S3 — swap backends without changing application code
- `ContextInjector` selects and injects prior session context into a new conversation based on relevance scoring and configurable freshness thresholds
- Entity extraction and cross-session linking tracks named entities (people, places, topics) across sessions so the agent can resume threads without being explicitly told what it discussed
- `ContextSummarizer` compresses long session histories to fit within token budgets while preserving the entities and decisions that matter
- Session middleware with automatic checkpoint creation and context-window management that prevents unbounded context growth
- `SessionLinker` chains related sessions by shared entities, enabling multi-session reasoning across conversations that span days or weeks

## Current Limitations

> **Transparency note**: We list known limitations to help you evaluate fit.

- **Security**: No encryption at rest for exported sessions. No file locking for concurrent access.
- **Async**: Synchronous API only.
- **Format**: USF 1.0 — no schema migration tooling yet.

## Quick Start

Install from PyPI:

```bash
pip install agent-session-linker
```

Verify the installation:

```bash
agent-session-linker version
```

Basic usage:

```python
import agent_session_linker

# See examples/01_quickstart.py for a working example
```

## Documentation

- [Architecture](docs/architecture.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Examples](examples/README.md)

## Enterprise Upgrade

For production deployments requiring SLA-backed support and advanced
integrations, contact the maintainers or see the commercial extensions documentation.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request.

## License

Apache 2.0 — see [LICENSE](LICENSE) for full terms.

---

Part of [AumOS](https://github.com/aumos-ai) — open-source agent infrastructure.
