# Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | [Quickstart](01_quickstart.py) | Create a session, add context, save, and resume |
| 02 | [Storage Backends](02_storage_backends.py) | In-memory, filesystem, and SQLite storage backends |
| 03 | [Context Injection](03_context_injection.py) | Inject context into a prompt window with freshness decay |
| 04 | [Entity Tracking](04_entity_tracking.py) | Extract, track, and link named entities across turns |
| 05 | [Session Linking](05_session_linking.py) | Link sessions into chains and create restore checkpoints |
| 06 | [Portable Sessions](06_portable_sessions.py) | Export/import via USF and encrypt session payloads |
| 07 | [LangChain Sessions](07_langchain_sessions.py) | Export sessions to LangChain memory format |

## Running the examples

```bash
pip install agent-session-linker
python examples/01_quickstart.py
```

For framework integrations:

```bash
pip install langchain   # for example 07
```
