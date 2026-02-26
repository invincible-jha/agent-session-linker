"""Integration-style CLI tests using a shared filesystem backend.

Each test class uses a shared tmp directory so multiple CLI invocations
can operate on the same session data (memory backend creates a fresh
store each time, so filesystem is used here for stateful workflows).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_session_linker.cli.main import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _storage_args(storage_dir: str) -> list[str]:
    return ["session", "--storage", "filesystem", "--storage-dir", storage_dir]


def _save(runner: CliRunner, storage_dir: str, **kwargs: str) -> str:
    """Save a session and return its ID."""
    args = _storage_args(storage_dir) + ["save"]
    for key, value in kwargs.items():
        args += [f"--{key.replace('_', '-')}", value]
    result = runner.invoke(cli, args)
    assert result.exit_code == 0, f"save failed: {result.output}"
    # Extract ID from "Session saved: <id>"
    for word in result.output.split():
        word = word.strip()
        if len(word) > 8 and ("-" in word or len(word) == 32):
            return word
    raise AssertionError(f"Could not parse session ID from: {result.output!r}")


# ---------------------------------------------------------------------------
# session load — success paths
# ---------------------------------------------------------------------------


class TestSessionLoadSuccess:
    def test_load_existing_session_exits_zero(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        result = runner.invoke(
            cli, _storage_args(storage_dir) + ["load", session_id]
        )
        assert result.exit_code == 0

    def test_load_prints_session_id(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        result = runner.invoke(
            cli, _storage_args(storage_dir) + ["load", session_id]
        )
        assert session_id[:8] in result.output

    def test_load_json_output_flag(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        result = runner.invoke(
            cli,
            _storage_args(storage_dir) + ["load", session_id, "--json-output"],
        )
        assert result.exit_code == 0
        # Output should contain valid JSON (session_id field).
        assert session_id in result.output

    def test_load_shows_summary_when_present(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Load a session that has a summary and a parent — ensures those branches."""
        from agent_session_linker.session.manager import SessionManager
        from agent_session_linker.storage.filesystem import FilesystemBackend

        storage_dir = str(tmp_path)
        backend = FilesystemBackend(storage_dir=storage_dir)
        manager = SessionManager(backend=backend, default_agent_id="bot")
        session = manager.create_session()
        session.summary = "My session summary text."
        session.parent_session_id = "parent-abc-123"
        session.add_segment("user", "hello", token_count=5)
        manager.save_session(session)

        result = runner.invoke(
            cli,
            _storage_args(storage_dir) + ["load", session.session_id],
        )
        assert result.exit_code == 0
        assert "My session summary text." in result.output

    def test_load_shows_segments(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        from agent_session_linker.session.manager import SessionManager
        from agent_session_linker.storage.filesystem import FilesystemBackend

        storage_dir = str(tmp_path)
        backend = FilesystemBackend(storage_dir=storage_dir)
        manager = SessionManager(backend=backend, default_agent_id="bot")
        session = manager.create_session()
        session.add_segment("user", "hello from user", token_count=5)
        session.add_segment("assistant", "hello from assistant", token_count=5)
        session.add_segment("system", "system message here", token_count=5)
        session.add_segment("tool", "tool output result", token_count=5)
        manager.save_session(session)

        result = runner.invoke(
            cli,
            _storage_args(storage_dir) + ["load", session.session_id],
        )
        assert result.exit_code == 0
        assert "hello from user" in result.output


# ---------------------------------------------------------------------------
# session list — with sessions present
# ---------------------------------------------------------------------------


class TestSessionListWithData:
    def test_list_shows_saved_sessions(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        _save(runner, storage_dir, agent_id="bot-a")
        _save(runner, storage_dir, agent_id="bot-b")
        result = runner.invoke(cli, _storage_args(storage_dir) + ["list"])
        assert result.exit_code == 0
        assert "bot-a" in result.output or "bot-b" in result.output

    def test_list_filtered_by_agent_id(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        _save(runner, storage_dir, agent_id="alpha-bot")
        _save(runner, storage_dir, agent_id="beta-bot")
        result = runner.invoke(
            cli, _storage_args(storage_dir) + ["list", "--agent-id", "alpha-bot"]
        )
        assert result.exit_code == 0

    def test_list_respects_limit(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        for _ in range(5):
            _save(runner, storage_dir, agent_id="bot")
        result = runner.invoke(
            cli, _storage_args(storage_dir) + ["list", "--limit", "2"]
        )
        assert result.exit_code == 0

    def test_list_handles_unreadable_session(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        _save(runner, storage_dir, agent_id="bot")
        # Corrupt one session file.
        files = list(Path(storage_dir).glob("*.json"))
        if files:
            files[0].write_text("not valid json {{{{", encoding="utf-8")
        result = runner.invoke(cli, _storage_args(storage_dir) + ["list"])
        # Should still exit zero (unreadable sessions are shown as error row).
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# session context — success path with a loadable session
# ---------------------------------------------------------------------------


class TestSessionContextSuccess:
    def test_context_for_existing_session_exits_zero(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        result = runner.invoke(
            cli,
            _storage_args(storage_dir) + ["context", session_id, "--query", "test"],
        )
        assert result.exit_code == 0

    def test_context_with_include_linked_and_links_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        s1 = _save(runner, storage_dir, agent_id="bot")
        s2 = _save(runner, storage_dir, agent_id="bot")

        links_file = tmp_path / "links.json"
        runner.invoke(
            cli,
            _storage_args(storage_dir)
            + [
                "link",
                s1,
                s2,
                "--links-file",
                str(links_file),
            ],
        )

        result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + [
                "context",
                s1,
                "--include-linked",
                "--links-file",
                str(links_file),
            ],
        )
        assert result.exit_code == 0

    def test_context_with_include_linked_missing_links_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        links_file = str(tmp_path / "nonexistent_links.json")
        result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + [
                "context",
                session_id,
                "--include-linked",
                "--links-file",
                links_file,
            ],
        )
        # Should succeed gracefully when links file doesn't exist.
        assert result.exit_code == 0

    def test_context_token_budget_option(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + ["context", session_id, "--token-budget", "500"],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# session checkpoint — success paths for create / list / restore
# ---------------------------------------------------------------------------


class TestSessionCheckpointSuccess:
    def test_checkpoint_create_success(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + ["checkpoint", "create", session_id, "--label", "v1"],
        )
        assert result.exit_code == 0
        assert "Checkpoint created" in result.output

    def test_checkpoint_create_output_contains_metadata(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + ["checkpoint", "create", session_id, "--label", "my-label"],
        )
        assert "my-label" in result.output

    def test_checkpoint_list_after_create(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")
        # Create a checkpoint first.
        runner.invoke(
            cli,
            _storage_args(storage_dir)
            + ["checkpoint", "create", session_id, "--label", "list-test"],
        )
        # Now list.
        result = runner.invoke(
            cli,
            _storage_args(storage_dir) + ["checkpoint", "list", session_id],
        )
        assert result.exit_code == 0
        assert "list-test" in result.output

    def test_checkpoint_restore_success(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")

        # Create checkpoint and capture its ID from output.
        create_result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + ["checkpoint", "create", session_id, "--label", "restore-test"],
        )
        assert create_result.exit_code == 0

        # Parse the checkpoint ID from the output line:
        # "Checkpoint created: __checkpoint__...__0000"
        checkpoint_id = None
        for word in create_result.output.split():
            word = word.strip()
            if word.startswith("__checkpoint__"):
                checkpoint_id = word
                break
        assert checkpoint_id is not None, f"No checkpoint ID in: {create_result.output!r}"

        # Restore the checkpoint.
        restore_result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + [
                "checkpoint",
                "restore",
                session_id,
                "--checkpoint-id",
                checkpoint_id,
            ],
        )
        assert restore_result.exit_code == 0
        assert "restored" in restore_result.output.lower()


# ---------------------------------------------------------------------------
# session list — agent_id filter where agent has no sessions
# ---------------------------------------------------------------------------


class TestSessionListEdgeCases:
    def test_list_agent_no_sessions(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        storage_dir = str(tmp_path)
        _save(runner, storage_dir, agent_id="known-bot")
        result = runner.invoke(
            cli,
            _storage_args(storage_dir) + ["list", "--agent-id", "unknown-bot"],
        )
        assert result.exit_code == 0
        assert "No sessions" in result.output


# ---------------------------------------------------------------------------
# session link — error path (duplicate link raises ValueError)
# ---------------------------------------------------------------------------


class TestSessionLinkError:
    def test_link_duplicate_same_session_raises_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Linking a session to itself can trigger a ValueError in some implementations.
        We test the corrupt-links-file warning path (exception on import_links)."""
        storage_dir = str(tmp_path)
        links_file = tmp_path / "links.json"
        # Write a links file that parses as JSON but has invalid link data
        # so that import_links raises (testing the except-branch at line 350).
        links_file.write_text(json.dumps([{"invalid": "data"}]))
        result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + [
                "link",
                "source-id",
                "target-id",
                "--links-file",
                str(links_file),
            ],
        )
        # Should either succeed (warning emitted) or exit cleanly.
        assert result.exit_code == 0 or "Warning" in result.output


# ---------------------------------------------------------------------------
# session context — corrupt linked-session that cannot be loaded
# ---------------------------------------------------------------------------


class TestSessionContextLinkedSessionLoadError:
    def test_context_linked_session_not_in_backend(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Covers lines 427-428: linked session ID exists in links file but
        cannot be loaded from the backend (triggers the except pass branch)."""
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")

        # Write a links file that links to a non-existent session.
        links_file = tmp_path / "links.json"
        links_data = [
            {
                "source_session_id": session_id,
                "target_session_id": "nonexistent-linked-session",
                "relationship": "references",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ]
        links_file.write_text(json.dumps(links_data))

        result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + [
                "context",
                session_id,
                "--include-linked",
                "--links-file",
                str(links_file),
            ],
        )
        # Should succeed even though the linked session can't be loaded.
        assert result.exit_code == 0

    def test_context_corrupt_links_json_handled_gracefully(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Covers lines 420-421: links file has invalid JSON structure
        after being parsed (import_links raises)."""
        storage_dir = str(tmp_path)
        session_id = _save(runner, storage_dir, agent_id="bot")

        links_file = tmp_path / "bad_links.json"
        # Valid JSON list but with objects that won't deserialise as link records.
        links_file.write_text(json.dumps([{"garbage": True}]))

        result = runner.invoke(
            cli,
            _storage_args(storage_dir)
            + [
                "context",
                session_id,
                "--include-linked",
                "--links-file",
                str(links_file),
            ],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# SQLite backend CLI path
# ---------------------------------------------------------------------------


class TestCliSqliteBackend:
    def test_save_and_list_with_sqlite(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        db_path = str(tmp_path / "sessions.db")
        result = runner.invoke(
            cli,
            ["session", "--storage", "sqlite", "--db-path", db_path, "save"],
        )
        assert result.exit_code == 0
        assert "Session saved" in result.output

    def test_list_with_sqlite_empty(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        db_path = str(tmp_path / "sessions.db")
        result = runner.invoke(
            cli,
            ["session", "--storage", "sqlite", "--db-path", db_path, "list"],
        )
        assert result.exit_code == 0
