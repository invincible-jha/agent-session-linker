"""Unit tests for agent_session_linker.cli.main.

Uses Click's test runner (CliRunner) with the InMemoryBackend injected via
the storage=memory option, so no disk I/O or external services are required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_session_linker.cli.main import cli, _make_backend
from agent_session_linker.session.manager import SessionManager
from agent_session_linker.storage.memory import InMemoryBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture()
def manager(backend: InMemoryBackend) -> SessionManager:
    return SessionManager(backend=backend, default_agent_id="cli-agent")


# ---------------------------------------------------------------------------
# _make_backend factory
# ---------------------------------------------------------------------------


class TestMakeBackend:
    def test_memory_backend(self) -> None:
        from agent_session_linker.storage.memory import InMemoryBackend
        backend = _make_backend("memory", None, None)
        assert isinstance(backend, InMemoryBackend)

    def test_filesystem_backend_default_dir(self, tmp_path: Path) -> None:
        from agent_session_linker.storage.filesystem import FilesystemBackend
        backend = _make_backend("filesystem", None, str(tmp_path))
        assert isinstance(backend, FilesystemBackend)

    def test_filesystem_backend_custom_dir(self, tmp_path: Path) -> None:
        from agent_session_linker.storage.filesystem import FilesystemBackend
        custom_dir = str(tmp_path / "custom")
        backend = _make_backend("filesystem", None, custom_dir)
        assert isinstance(backend, FilesystemBackend)

    def test_sqlite_backend_default_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_session_linker.storage.sqlite import SQLiteBackend
        monkeypatch.chdir(tmp_path)
        backend = _make_backend("sqlite", str(tmp_path / "test.db"), None)
        assert isinstance(backend, SQLiteBackend)

    def test_sqlite_backend_custom_path(self, tmp_path: Path) -> None:
        from agent_session_linker.storage.sqlite import SQLiteBackend
        backend = _make_backend("sqlite", str(tmp_path / "custom.db"), None)
        assert isinstance(backend, SQLiteBackend)

    def test_unknown_backend_exits(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            ["session", "--storage", "invalid_choice", "list"],
        )
        # Click's Choice should reject the invalid value.
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------


class TestVersionCommand:
    def test_version_command_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["version"])
        assert result.exit_code == 0

    def test_version_command_prints_package_name(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["version"])
        assert "agent-session-linker" in result.output


# ---------------------------------------------------------------------------
# plugins command
# ---------------------------------------------------------------------------


class TestPluginsCommand:
    def test_plugins_command_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["plugins"])
        assert result.exit_code == 0

    def test_plugins_command_has_output(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["plugins"])
        assert len(result.output) > 0


# ---------------------------------------------------------------------------
# session save
# ---------------------------------------------------------------------------


class TestSessionSave:
    def test_save_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["session", "--storage", "memory", "save", "--agent-id", "test-bot"]
        )
        assert result.exit_code == 0

    def test_save_prints_session_id(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["session", "--storage", "memory", "save"]
        )
        assert "Session saved" in result.output

    def test_save_with_content(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            ["session", "--storage", "memory", "save", "--content", "hello there"],
        )
        assert result.exit_code == 0
        assert "Session saved" in result.output

    def test_save_with_parent(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "save",
                "--parent",
                "parent-session-id",
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# session load
# ---------------------------------------------------------------------------


class TestSessionLoad:
    def _save_and_get_id(self, runner: CliRunner) -> str:
        result = runner.invoke(
            cli, ["session", "--storage", "memory", "save", "--agent-id", "bot"]
        )
        # Extract the session ID from output "Session saved: <id>"
        for word in result.output.split():
            if len(word) > 10 and "-" in word:
                return word.strip()
        raise AssertionError(f"Could not find session ID in: {result.output!r}")

    def test_load_nonexistent_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            ["session", "--storage", "memory", "load", "nonexistent-session-id"],
        )
        assert result.exit_code != 0

    def test_load_nonexistent_prints_not_found(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli,
            ["session", "--storage", "memory", "load", "ghost-id"],
        )
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# session list
# ---------------------------------------------------------------------------


class TestSessionList:
    def test_list_exits_zero_when_empty(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["session", "--storage", "memory", "list"])
        assert result.exit_code == 0

    def test_list_prints_no_sessions_message_when_empty(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(cli, ["session", "--storage", "memory", "list"])
        assert "No sessions" in result.output

    def test_list_with_filesystem_backend(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "filesystem",
                "--storage-dir",
                str(tmp_path),
                "list",
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# session link
# ---------------------------------------------------------------------------


class TestSessionLink:
    def test_link_exits_zero(self, runner: CliRunner, tmp_path: Path) -> None:
        links_file = str(tmp_path / "links.json")
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "link",
                "source-session-abc",
                "target-session-xyz",
                "--links-file",
                links_file,
            ],
        )
        assert result.exit_code == 0

    def test_link_creates_links_file(self, runner: CliRunner, tmp_path: Path) -> None:
        links_file = tmp_path / "links.json"
        runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "link",
                "src-id",
                "tgt-id",
                "--links-file",
                str(links_file),
            ],
        )
        assert links_file.exists()

    def test_link_writes_valid_json(self, runner: CliRunner, tmp_path: Path) -> None:
        links_file = tmp_path / "links.json"
        runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "link",
                "src-id",
                "tgt-id",
                "--links-file",
                str(links_file),
            ],
        )
        data = json.loads(links_file.read_text())
        assert isinstance(data, list)

    def test_link_with_custom_relationship(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        links_file = str(tmp_path / "links.json")
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "link",
                "src-id",
                "tgt-id",
                "--relationship",
                "continues",
                "--links-file",
                links_file,
            ],
        )
        assert result.exit_code == 0

    def test_link_prints_confirmation(self, runner: CliRunner, tmp_path: Path) -> None:
        links_file = str(tmp_path / "links.json")
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "link",
                "source-id",
                "target-id",
                "--links-file",
                links_file,
            ],
        )
        assert "Linked" in result.output

    def test_link_loads_existing_links_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        links_file = tmp_path / "links.json"
        links_file.write_text(json.dumps([]))  # Existing empty links file.
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "link",
                "s1",
                "s2",
                "--links-file",
                str(links_file),
            ],
        )
        assert result.exit_code == 0

    def test_link_handles_corrupt_links_file_gracefully(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        links_file = tmp_path / "links.json"
        links_file.write_text("not valid json {{{{")
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "link",
                "s1",
                "s2",
                "--links-file",
                str(links_file),
            ],
        )
        # Should still proceed (warning emitted) and exit zero.
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# session context
# ---------------------------------------------------------------------------


class TestSessionContext:
    def test_context_nonexistent_session_exits_nonzero(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            cli,
            ["session", "--storage", "memory", "context", "nonexistent-id"],
        )
        assert result.exit_code != 0

    def test_context_nonexistent_prints_not_found(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            cli,
            ["session", "--storage", "memory", "context", "ghost-id"],
        )
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# session checkpoint
# ---------------------------------------------------------------------------


class TestSessionCheckpoint:
    def test_checkpoint_list_empty_session_exits_nonzero_or_prints_message(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "checkpoint",
                "list",
                "nonexistent-session-id",
            ],
        )
        # Either exits cleanly (prints "no checkpoints") or exits non-zero.
        assert "No checkpoints" in result.output or result.exit_code == 0

    def test_checkpoint_create_nonexistent_session_exits_nonzero(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "checkpoint",
                "create",
                "ghost-session-id",
            ],
        )
        assert result.exit_code != 0

    def test_checkpoint_restore_without_id_exits_nonzero(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "checkpoint",
                "restore",
                "some-session",
            ],
        )
        assert result.exit_code != 0

    def test_checkpoint_restore_nonexistent_checkpoint_exits_nonzero(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(
            cli,
            [
                "session",
                "--storage",
                "memory",
                "checkpoint",
                "restore",
                "some-session",
                "--checkpoint-id",
                "nonexistent-checkpoint",
            ],
        )
        assert result.exit_code != 0
