"""CLI entry point for agent-session-linker.

Invoked as::

    agent-session-linker [OPTIONS] COMMAND [ARGS]...

or, during development::

    python -m agent_session_linker.cli.main

Commands
--------
- version      — Show detailed version information
- plugins      — List all registered plugins
- session      — Session management command group

Session sub-commands
---------------------
- session save       — Persist the current session
- session load       — Load and display a session
- session list       — List all stored sessions
- session link       — Create a link between two sessions
- session context    — Show the injected context for a session
- session checkpoint — Create or restore a session checkpoint
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Storage backend factory
# ---------------------------------------------------------------------------


def _make_backend(
    storage: str,
    db_path: str | None,
    storage_dir: str | None,
) -> object:
    """Instantiate the requested storage backend.

    Parameters
    ----------
    storage:
        Backend name: ``"memory"``, ``"filesystem"``, or ``"sqlite"``.
    db_path:
        Path to the SQLite database (used when ``storage="sqlite"``).
    storage_dir:
        Directory for the filesystem backend (used when ``storage="filesystem"``).

    Returns
    -------
    StorageBackend
        A configured storage backend instance.
    """
    from agent_session_linker.storage.memory import InMemoryBackend
    from agent_session_linker.storage.filesystem import FilesystemBackend
    from agent_session_linker.storage.sqlite import SQLiteBackend

    if storage == "memory":
        return InMemoryBackend()
    if storage == "filesystem":
        directory = Path(storage_dir) if storage_dir else Path.home() / ".agent-sessions"
        return FilesystemBackend(storage_dir=directory)
    if storage == "sqlite":
        db = Path(db_path) if db_path else Path.home() / ".agent-sessions" / "sessions.db"
        return SQLiteBackend(db_path=db)
    console.print(f"[red]Unknown storage backend: {storage!r}[/red]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
def cli() -> None:
    """Cross-session context persistence and resumption"""


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@cli.command(name="version")
def version_command() -> None:
    """Show detailed version information."""
    from agent_session_linker import __version__

    console.print(f"[bold]agent-session-linker[/bold] v{__version__}")


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------


@cli.command(name="plugins")
def plugins_command() -> None:
    """List all registered plugins loaded from entry-points."""
    console.print("[bold]Registered plugins:[/bold]")
    console.print("  (No plugins registered. Install a plugin package to see entries here.)")


# ---------------------------------------------------------------------------
# session command group
# ---------------------------------------------------------------------------


@cli.group(name="session")
@click.option(
    "--storage",
    default="filesystem",
    show_default=True,
    type=click.Choice(["memory", "filesystem", "sqlite"], case_sensitive=False),
    help="Storage backend to use.",
)
@click.option("--db-path", default=None, help="Path to SQLite database (sqlite backend).")
@click.option(
    "--storage-dir",
    default=None,
    help="Directory for filesystem backend.",
)
@click.pass_context
def session_group(
    ctx: click.Context,
    storage: str,
    db_path: str | None,
    storage_dir: str | None,
) -> None:
    """Session management commands."""
    ctx.ensure_object(dict)
    ctx.obj["backend"] = _make_backend(storage, db_path, storage_dir)


# ---------------------------------------------------------------------------
# session save
# ---------------------------------------------------------------------------


@session_group.command(name="save")
@click.option("--agent-id", default="default", show_default=True, help="Agent identifier.")
@click.option(
    "--parent",
    default=None,
    help="Parent session ID (for continuations).",
)
@click.option(
    "--content",
    default=None,
    help="Initial segment content to add (role=user).",
)
@click.pass_context
def session_save(
    ctx: click.Context,
    agent_id: str,
    parent: str | None,
    content: str | None,
) -> None:
    """Create and persist a new session.

    Prints the new session ID on success.
    """
    from agent_session_linker.session.manager import SessionManager

    backend = ctx.obj["backend"]
    manager = SessionManager(backend=backend, default_agent_id=agent_id)
    session = manager.create_session(agent_id=agent_id, parent_session_id=parent)

    if content:
        session.add_segment(role="user", content=content)

    session_id = manager.save_session(session)
    console.print(f"[green]Session saved:[/green] {session_id}")


# ---------------------------------------------------------------------------
# session load
# ---------------------------------------------------------------------------


@session_group.command(name="load")
@click.argument("session_id")
@click.option("--json-output", is_flag=True, help="Output raw JSON instead of formatted view.")
@click.pass_context
def session_load(
    ctx: click.Context,
    session_id: str,
    json_output: bool,
) -> None:
    """Load and display a session by SESSION_ID."""
    from agent_session_linker.session.manager import SessionManager, SessionNotFoundError

    backend = ctx.obj["backend"]
    manager = SessionManager(backend=backend)

    try:
        session = manager.load_session(session_id)
    except SessionNotFoundError:
        console.print(f"[red]Session not found:[/red] {session_id}")
        sys.exit(1)

    if json_output:
        console.print_json(session.model_dump_json(indent=2))
        return

    table = Table(title=f"Session {session.session_id[:8]}", show_lines=True)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")

    table.add_row("session_id", session.session_id)
    table.add_row("agent_id", session.agent_id)
    table.add_row("schema_version", session.schema_version)
    table.add_row("segments", str(len(session.segments)))
    table.add_row("tasks", str(len(session.tasks)))
    table.add_row("entities", str(len(session.entities)))
    table.add_row("total_tokens", str(session.total_tokens()))
    table.add_row("created_at", session.created_at.isoformat())
    table.add_row("updated_at", session.updated_at.isoformat())
    if session.parent_session_id:
        table.add_row("parent_session_id", session.parent_session_id)
    if session.summary:
        table.add_row("summary", session.summary[:200])

    console.print(table)

    if session.segments:
        console.print("\n[bold]Context Segments:[/bold]")
        for segment in session.segments:
            role_style = {
                "user": "green",
                "assistant": "blue",
                "system": "yellow",
                "tool": "magenta",
            }.get(segment.role, "white")
            header = f"[{role_style}]{segment.role.upper()}[/{role_style}] | turn={segment.turn_index} | type={segment.segment_type}"
            console.print(Panel(segment.content, title=header, expand=False))


# ---------------------------------------------------------------------------
# session list
# ---------------------------------------------------------------------------


@session_group.command(name="list")
@click.option("--agent-id", default=None, help="Filter by agent ID.")
@click.option("--limit", default=50, show_default=True, help="Maximum sessions to show.")
@click.pass_context
def session_list(
    ctx: click.Context,
    agent_id: str | None,
    limit: int,
) -> None:
    """List all sessions in the storage backend."""
    from agent_session_linker.session.manager import SessionManager

    backend = ctx.obj["backend"]
    manager = SessionManager(backend=backend)

    if agent_id:
        session_ids = manager.list_sessions_for_agent(agent_id)
    else:
        session_ids = manager.list_sessions()

    if not session_ids:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    session_ids = session_ids[:limit]

    table = Table(title="Sessions", show_lines=False)
    table.add_column("Session ID", style="cyan")
    table.add_column("Agent", style="green")
    table.add_column("Segments", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Updated")

    for sid in session_ids:
        try:
            session = manager.load_session(sid)
            table.add_row(
                session.session_id[:16] + "...",
                session.agent_id,
                str(len(session.segments)),
                str(session.total_tokens()),
                session.updated_at.strftime("%Y-%m-%d %H:%M"),
            )
        except Exception:  # noqa: BLE001
            table.add_row(sid[:16] + "...", "[red]<unreadable>[/red]", "-", "-", "-")

    console.print(table)
    console.print(f"\n[dim]Showing {len(session_ids)} of {len(manager.list_sessions())} sessions.[/dim]")


# ---------------------------------------------------------------------------
# session link
# ---------------------------------------------------------------------------


@session_group.command(name="link")
@click.argument("source_session_id")
@click.argument("target_session_id")
@click.option(
    "--relationship",
    default="references",
    show_default=True,
    help="Relationship label (e.g. continues, references, spawned).",
)
@click.option(
    "--links-file",
    default=None,
    help="JSON file to persist links. Defaults to ~/.agent-sessions/links.json.",
)
@click.pass_context
def session_link(
    ctx: click.Context,
    source_session_id: str,
    target_session_id: str,
    relationship: str,
    links_file: str | None,
) -> None:
    """Create a directed relationship between two sessions.

    SOURCE_SESSION_ID is the originating session.
    TARGET_SESSION_ID is the related session.
    """
    from agent_session_linker.linking.session_linker import SessionLinker

    links_path = Path(links_file) if links_file else Path.home() / ".agent-sessions" / "links.json"

    linker = SessionLinker()

    if links_path.exists():
        try:
            existing_data: list[dict[str, object]] = json.loads(links_path.read_text())
            linker.import_links(existing_data)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning: could not load existing links: {exc}[/yellow]")

    try:
        linked = linker.link(source_session_id, target_session_id, relationship)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    links_path.parent.mkdir(parents=True, exist_ok=True)
    links_path.write_text(json.dumps(linker.export_links(), indent=2, default=str))

    console.print(
        f"[green]Linked[/green] {source_session_id[:8]}... "
        f"--[{relationship}]--> {target_session_id[:8]}..."
    )


# ---------------------------------------------------------------------------
# session context
# ---------------------------------------------------------------------------


@session_group.command(name="context")
@click.argument("session_id")
@click.option("--query", default="", help="Query string to guide relevance scoring.")
@click.option("--token-budget", default=2000, show_default=True, help="Token budget for injection.")
@click.option(
    "--include-linked",
    is_flag=True,
    help="Also include context from linked sessions.",
)
@click.option(
    "--links-file",
    default=None,
    help="JSON file to load links from.",
)
@click.pass_context
def session_context(
    ctx: click.Context,
    session_id: str,
    query: str,
    token_budget: int,
    include_linked: bool,
    links_file: str | None,
) -> None:
    """Show the injected context block for a session."""
    from agent_session_linker.session.manager import SessionManager, SessionNotFoundError
    from agent_session_linker.context.injector import ContextInjector, InjectionConfig
    from agent_session_linker.linking.session_linker import SessionLinker

    backend = ctx.obj["backend"]
    manager = SessionManager(backend=backend)

    try:
        session = manager.load_session(session_id)
    except SessionNotFoundError:
        console.print(f"[red]Session not found:[/red] {session_id}")
        sys.exit(1)

    sessions_to_inject = [session]

    if include_linked:
        links_path = (
            Path(links_file) if links_file else Path.home() / ".agent-sessions" / "links.json"
        )
        linker = SessionLinker()
        if links_path.exists():
            try:
                existing_data: list[dict[str, object]] = json.loads(links_path.read_text())
                linker.import_links(existing_data)
            except Exception:  # noqa: BLE001
                pass

        for related_id in linker.get_related_session_ids(session_id):
            try:
                related_session = manager.load_session(related_id)
                sessions_to_inject.append(related_session)
            except Exception:  # noqa: BLE001
                pass

    config = InjectionConfig(token_budget=token_budget)
    injector = ContextInjector(config=config)
    context_block = injector.inject(sessions_to_inject, query or "")

    console.print(Panel(context_block, title="Injected Context", expand=True))


# ---------------------------------------------------------------------------
# session checkpoint
# ---------------------------------------------------------------------------


@session_group.command(name="checkpoint")
@click.argument("action", type=click.Choice(["create", "restore", "list"], case_sensitive=False))
@click.argument("session_id")
@click.option("--label", default="", help="Human-readable label for the checkpoint (create only).")
@click.option(
    "--checkpoint-id",
    default=None,
    help="Checkpoint ID to restore (restore only).",
)
@click.pass_context
def session_checkpoint(
    ctx: click.Context,
    action: str,
    session_id: str,
    label: str,
    checkpoint_id: str | None,
) -> None:
    """Create, restore, or list checkpoints for a session.

    ACTION is one of: create, restore, list.
    SESSION_ID is the target session.

    Examples::

      agent-session-linker session checkpoint create <session_id> --label before-refactor

      agent-session-linker session checkpoint list <session_id>

      agent-session-linker session checkpoint restore <session_id> --checkpoint-id __checkpoint__<...>
    """
    from agent_session_linker.session.manager import SessionManager, SessionNotFoundError
    from agent_session_linker.middleware.checkpoint import CheckpointManager

    backend = ctx.obj["backend"]
    manager = SessionManager(backend=backend)
    checkpoint_manager = CheckpointManager(backend=backend, manager=manager)

    if action == "create":
        try:
            session = manager.load_session(session_id)
        except SessionNotFoundError:
            console.print(f"[red]Session not found:[/red] {session_id}")
            sys.exit(1)

        record = checkpoint_manager.create_checkpoint(session, label=label)
        console.print(f"[green]Checkpoint created:[/green] {record.checkpoint_id}")
        console.print(f"  label:     {record.label}")
        console.print(f"  segments:  {record.segment_count}")
        console.print(f"  tokens:    {record.token_count}")
        console.print(f"  created:   {record.created_at.isoformat()}")

    elif action == "restore":
        if not checkpoint_id:
            console.print("[red]--checkpoint-id is required for the restore action.[/red]")
            sys.exit(1)
        try:
            restored = checkpoint_manager.restore_checkpoint(checkpoint_id)
        except KeyError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)

        manager.save_session(restored)
        console.print(f"[green]Checkpoint restored and saved:[/green] {restored.session_id}")

    elif action == "list":
        records = checkpoint_manager.list_checkpoints(session_id)
        if not records:
            console.print(f"[yellow]No checkpoints found for session:[/yellow] {session_id}")
            return

        table = Table(title=f"Checkpoints for {session_id[:16]}...", show_lines=False)
        table.add_column("Checkpoint ID", style="cyan")
        table.add_column("Label")
        table.add_column("Segments", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Created")

        for record in records:
            table.add_row(
                record.checkpoint_id,
                record.label,
                str(record.segment_count),
                str(record.token_count),
                record.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)


# ---------------------------------------------------------------------------
# portable command group
# ---------------------------------------------------------------------------


@cli.group(name="portable")
def portable_group() -> None:
    """Cross-framework session portability commands (USF).

    Export, import, or convert sessions between LangChain, CrewAI, and OpenAI
    formats using the Universal Session Format as the interchange layer.
    """


@portable_group.command(name="export")
@click.option(
    "--format",
    "fmt",
    required=True,
    type=click.Choice(["langchain", "crewai", "openai"], case_sensitive=False),
    help="Target framework format.",
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a USF session JSON file.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to write the exported JSON file.",
)
def portable_export(fmt: str, input_file: str, output_file: str) -> None:
    """Export a USF session file to a framework-native JSON format.

    Reads a UniversalSession from INPUT and writes the framework-specific
    representation to OUTPUT.

    Examples::

        agent-session-linker portable export --format langchain \\
            --input session.usf.json --output lc_session.json

        agent-session-linker portable export --format openai \\
            --input session.usf.json --output openai_thread.json
    """
    import json
    from pathlib import Path
    from agent_session_linker.portable.usf import UniversalSession
    from agent_session_linker.portable.exporters import (
        LangChainExporter,
        CrewAIExporter,
        OpenAIExporter,
    )

    try:
        json_str = Path(input_file).read_text(encoding="utf-8")
        session = UniversalSession.from_json(json_str)
    except (ValueError, OSError) as exc:
        console.print(f"[red]Failed to load session:[/red] {exc}")
        sys.exit(1)

    exporter_map = {
        "langchain": LangChainExporter(),
        "crewai": CrewAIExporter(),
        "openai": OpenAIExporter(),
    }
    exporter = exporter_map[fmt.lower()]
    result = exporter.export(session)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]Exported ({fmt}):[/green] {output_file}")


@portable_group.command(name="import")
@click.option(
    "--format",
    "fmt",
    required=True,
    type=click.Choice(["langchain", "crewai", "openai"], case_sensitive=False),
    help="Source framework format.",
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the framework-native JSON file.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to write the USF session JSON file.",
)
def portable_import(fmt: str, input_file: str, output_file: str) -> None:
    """Import a framework-native JSON file as a USF session.

    Reads the framework-specific representation from INPUT and writes a
    UniversalSession to OUTPUT.

    Examples::

        agent-session-linker portable import --format langchain \\
            --input lc_memory.json --output session.usf.json

        agent-session-linker portable import --format crewai \\
            --input crewai_ctx.json --output session.usf.json
    """
    import json as _json
    from pathlib import Path
    from agent_session_linker.portable.importers import (
        LangChainImporter,
        CrewAIImporter,
        OpenAIImporter,
    )

    try:
        raw_text = Path(input_file).read_text(encoding="utf-8")
        data: dict[str, object] = _json.loads(raw_text)
    except (ValueError, OSError) as exc:
        console.print(f"[red]Failed to read input:[/red] {exc}")
        sys.exit(1)

    importer_map = {
        "langchain": LangChainImporter(),
        "crewai": CrewAIImporter(),
        "openai": OpenAIImporter(),
    }
    importer = importer_map[fmt.lower()]

    try:
        session = importer.import_session(data)  # type: ignore[arg-type]
    except (ValueError, KeyError) as exc:
        console.print(f"[red]Import failed:[/red] {exc}")
        sys.exit(1)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(session.to_json(), encoding="utf-8")
    console.print(f"[green]Imported ({fmt}):[/green] {output_file}")


@portable_group.command(name="convert")
@click.option(
    "--from",
    "from_fmt",
    required=True,
    type=click.Choice(["langchain", "crewai", "openai"], case_sensitive=False),
    help="Source framework format.",
)
@click.option(
    "--to",
    "to_fmt",
    required=True,
    type=click.Choice(["langchain", "crewai", "openai"], case_sensitive=False),
    help="Target framework format.",
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the source framework JSON file.",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to write the target framework JSON file.",
)
def portable_convert(from_fmt: str, to_fmt: str, input_file: str, output_file: str) -> None:
    """Convert a session between two framework-native formats via USF.

    The conversion pipeline is: source -> USF -> target.

    Examples::

        agent-session-linker portable convert --from langchain --to openai \\
            --input lc_memory.json --output openai_thread.json

        agent-session-linker portable convert --from crewai --to langchain \\
            --input crewai_ctx.json --output lc_memory.json
    """
    import json as _json
    from pathlib import Path
    from agent_session_linker.portable.importers import (
        LangChainImporter,
        CrewAIImporter,
        OpenAIImporter,
    )
    from agent_session_linker.portable.exporters import (
        LangChainExporter,
        CrewAIExporter,
        OpenAIExporter,
    )

    importer_map = {
        "langchain": LangChainImporter(),
        "crewai": CrewAIImporter(),
        "openai": OpenAIImporter(),
    }
    exporter_map = {
        "langchain": LangChainExporter(),
        "crewai": CrewAIExporter(),
        "openai": OpenAIExporter(),
    }

    try:
        raw_text = Path(input_file).read_text(encoding="utf-8")
        data: dict[str, object] = _json.loads(raw_text)
    except (ValueError, OSError) as exc:
        console.print(f"[red]Failed to read input:[/red] {exc}")
        sys.exit(1)

    importer = importer_map[from_fmt.lower()]
    exporter = exporter_map[to_fmt.lower()]

    try:
        session = importer.import_session(data)  # type: ignore[arg-type]
    except (ValueError, KeyError) as exc:
        console.print(f"[red]Import failed:[/red] {exc}")
        sys.exit(1)

    result = exporter.export(session)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json.dumps(result, indent=2, default=str), encoding="utf-8")
    console.print(f"[green]Converted ({from_fmt} -> {to_fmt}):[/green] {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
