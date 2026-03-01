"""Tests for USF encryption, file locking, and schema versioning.

Covers:
- SessionEncryptor: encrypt/decrypt round-trips, wrong-key failure,
  key generation, invalid key length
- EncryptedPayload: to_bytes/from_bytes round-trip
- FileLock: acquire/release, timeout on contention, context manager,
  cleanup on error
- SchemaVersion: is_supported, current version
- SchemaMigrator: register + migrate, no-path error, detect_version
- Integration: encrypt -> write -> read -> decrypt round-trip
- Backward compatibility: exporter works without encryptor
- Without cryptography: helpful ImportError
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_session_linker.portable.encryption import (
    _CRYPTO_AVAILABLE,
    EncryptedPayload,
    SessionEncryptor,
)
from agent_session_linker.portable.locking import FileLock
from agent_session_linker.portable.versioning import SchemaVersion, SchemaMigrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def key() -> bytes:
    """Return a fresh 32-byte AES-256 key."""
    return SessionEncryptor.generate_key()


@pytest.fixture()
def encryptor(key: bytes) -> SessionEncryptor:
    """Return a SessionEncryptor backed by a random key."""
    return SessionEncryptor(key)


@pytest.fixture()
def sample_payload() -> dict[str, object]:
    return {
        "session_id": "test-session-001",
        "version": "1.0",
        "messages": [{"role": "user", "content": "hello"}],
    }


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# SessionEncryptor — key generation
# ---------------------------------------------------------------------------


def test_generate_key_returns_32_bytes() -> None:
    """generate_key must return exactly 32 bytes."""
    generated = SessionEncryptor.generate_key()
    assert len(generated) == 32


def test_generate_key_is_random() -> None:
    """Two successive calls must not produce the same key."""
    key_a = SessionEncryptor.generate_key()
    key_b = SessionEncryptor.generate_key()
    assert key_a != key_b


# ---------------------------------------------------------------------------
# SessionEncryptor — invalid key length
# ---------------------------------------------------------------------------


def test_invalid_key_length_16_raises() -> None:
    """A 16-byte key (AES-128) must raise ValueError."""
    with pytest.raises(ValueError, match="32 bytes"):
        SessionEncryptor(b"\x00" * 16)


def test_invalid_key_length_0_raises() -> None:
    """An empty key must raise ValueError."""
    with pytest.raises(ValueError, match="32 bytes"):
        SessionEncryptor(b"")


# ---------------------------------------------------------------------------
# SessionEncryptor — encrypt / decrypt round-trips
# ---------------------------------------------------------------------------


def test_encrypt_decrypt_simple_payload(
    encryptor: SessionEncryptor,
    sample_payload: dict[str, object],
) -> None:
    """Encrypt then decrypt must recover the original dict."""
    encrypted = encryptor.encrypt(sample_payload)
    recovered = encryptor.decrypt(encrypted)
    assert recovered == sample_payload


def test_encrypt_decrypt_empty_payload(encryptor: SessionEncryptor) -> None:
    """An empty dict must survive the round-trip."""
    empty: dict[str, object] = {}
    encrypted = encryptor.encrypt(empty)
    recovered = encryptor.decrypt(encrypted)
    assert recovered == empty


def test_encrypt_decrypt_nested_payload(encryptor: SessionEncryptor) -> None:
    """Nested dicts and lists must survive the round-trip."""
    payload: dict[str, object] = {
        "level1": {"level2": [1, 2, 3]},
        "flag": True,
        "score": 0.99,
    }
    recovered = encryptor.decrypt(encryptor.encrypt(payload))
    assert recovered == payload


# ---------------------------------------------------------------------------
# SessionEncryptor — wrong key failure
# ---------------------------------------------------------------------------


def test_decrypt_with_wrong_key_raises(
    encryptor: SessionEncryptor,
    sample_payload: dict[str, object],
) -> None:
    """Decryption with a different key must raise an error (authentication tag mismatch)."""
    encrypted = encryptor.encrypt(sample_payload)
    wrong_key = bytes(b ^ 0xFF for b in encryptor._aesgcm._key) if hasattr(encryptor._aesgcm, "_key") else os.urandom(32)
    # Use a different key that is guaranteed to differ
    different_key = bytes((b + 1) % 256 for b in SessionEncryptor.generate_key())
    bad_encryptor = SessionEncryptor(different_key)
    with pytest.raises(Exception):  # cryptography.exceptions.InvalidTag
        bad_encryptor.decrypt(encrypted)


def test_decrypt_tampered_ciphertext_raises(
    encryptor: SessionEncryptor,
    sample_payload: dict[str, object],
) -> None:
    """Flipping a bit in the ciphertext must cause authentication failure."""
    encrypted = encryptor.encrypt(sample_payload)
    # Flip the first byte of ciphertext
    tampered_ciphertext = bytes([encrypted.ciphertext[0] ^ 0x01]) + encrypted.ciphertext[1:]
    tampered = EncryptedPayload(
        ciphertext=tampered_ciphertext,
        nonce=encrypted.nonce,
        version=encrypted.version,
    )
    with pytest.raises(Exception):
        encryptor.decrypt(tampered)


# ---------------------------------------------------------------------------
# EncryptedPayload — to_bytes / from_bytes round-trip
# ---------------------------------------------------------------------------


def test_encrypted_payload_to_bytes_from_bytes_round_trip(
    encryptor: SessionEncryptor,
    sample_payload: dict[str, object],
) -> None:
    """Serialising and deserialising EncryptedPayload must be lossless."""
    encrypted = encryptor.encrypt(sample_payload)
    wire = encrypted.to_bytes()
    restored = EncryptedPayload.from_bytes(wire)
    assert restored.ciphertext == encrypted.ciphertext
    assert restored.nonce == encrypted.nonce
    assert restored.version == encrypted.version


def test_encrypted_payload_from_bytes_then_decrypt(
    encryptor: SessionEncryptor,
    sample_payload: dict[str, object],
) -> None:
    """A round-tripped payload must decrypt back to the original dict."""
    encrypted = encryptor.encrypt(sample_payload)
    wire = encrypted.to_bytes()
    restored = EncryptedPayload.from_bytes(wire)
    recovered = encryptor.decrypt(restored)
    assert recovered == sample_payload


def test_encrypted_payload_from_bytes_truncated_raises() -> None:
    """from_bytes on data that is too short must raise ValueError."""
    with pytest.raises(ValueError):
        EncryptedPayload.from_bytes(b"")


def test_encrypted_payload_from_bytes_truncated_in_version_raises() -> None:
    """from_bytes when version field is truncated must raise ValueError."""
    # version_len = 5 but only 2 bytes follow
    bad = bytes([5, 0x61, 0x62])
    with pytest.raises(ValueError):
        EncryptedPayload.from_bytes(bad)


# ---------------------------------------------------------------------------
# FileLock — acquire / release
# ---------------------------------------------------------------------------


def test_file_lock_acquire_creates_file(tmp_dir: Path) -> None:
    """Acquiring a lock must create the sentinel lock file."""
    lock_path = tmp_dir / "session.lock"
    lock = FileLock(lock_path)
    lock.acquire()
    assert lock_path.exists()
    lock.release()


def test_file_lock_release_deletes_file(tmp_dir: Path) -> None:
    """Releasing a lock must delete the sentinel file."""
    lock_path = tmp_dir / "session.lock"
    lock = FileLock(lock_path)
    lock.acquire()
    lock.release()
    assert not lock_path.exists()


def test_file_lock_release_idempotent(tmp_dir: Path) -> None:
    """Calling release() twice must not raise."""
    lock_path = tmp_dir / "session.lock"
    lock = FileLock(lock_path)
    lock.acquire()
    lock.release()
    lock.release()  # second call must be a no-op


# ---------------------------------------------------------------------------
# FileLock — context manager
# ---------------------------------------------------------------------------


def test_file_lock_context_manager_acquires_and_releases(tmp_dir: Path) -> None:
    """The context manager must acquire on entry and release on exit."""
    lock_path = tmp_dir / "ctx.lock"
    with FileLock(lock_path) as lock:
        assert lock_path.exists()
        assert isinstance(lock, FileLock)
    assert not lock_path.exists()


def test_file_lock_context_manager_releases_on_exception(tmp_dir: Path) -> None:
    """The context manager must release the lock even when an exception is raised."""
    lock_path = tmp_dir / "err.lock"
    with pytest.raises(RuntimeError):
        with FileLock(lock_path):
            assert lock_path.exists()
            raise RuntimeError("deliberate error")
    assert not lock_path.exists()


# ---------------------------------------------------------------------------
# FileLock — timeout on contention
# ---------------------------------------------------------------------------


def test_file_lock_timeout_when_already_held(tmp_dir: Path) -> None:
    """A second FileLock on the same file must time out quickly."""
    lock_path = tmp_dir / "contended.lock"

    # Manually create the sentinel file to simulate a held lock
    lock_path.write_text("")

    short_timeout = FileLock(lock_path, timeout=0.15)
    with pytest.raises(TimeoutError, match="Could not acquire lock"):
        short_timeout.acquire()

    # Clean up the sentinel
    lock_path.unlink()


# ---------------------------------------------------------------------------
# SchemaVersion
# ---------------------------------------------------------------------------


def test_schema_version_current_is_one_zero() -> None:
    """CURRENT must be '1.0'."""
    assert SchemaVersion.CURRENT == "1.0"


def test_schema_version_is_supported_true() -> None:
    """'1.0' must be in SUPPORTED."""
    assert SchemaVersion.is_supported("1.0") is True


def test_schema_version_is_supported_false() -> None:
    """An unknown version string must not be supported."""
    assert SchemaVersion.is_supported("99.99") is False


# ---------------------------------------------------------------------------
# SchemaMigrator
# ---------------------------------------------------------------------------


def test_schema_migrator_no_migration_needed_returns_data() -> None:
    """migrate() with the same source and target version must return data unchanged."""
    migrator = SchemaMigrator()
    data: dict[str, object] = {"version": "1.0", "field": "value"}
    result = migrator.migrate(data, target_version="1.0")
    assert result is data


def test_schema_migrator_registered_migration_applied() -> None:
    """A registered migration function must be applied when versions differ."""

    def upgrade(d: dict[str, object]) -> dict[str, object]:
        return {**d, "version": "2.0", "new_field": "added"}

    migrator = SchemaMigrator()
    migrator.register_migration("1.0", "2.0", upgrade)
    data: dict[str, object] = {"version": "1.0", "existing": True}
    result = migrator.migrate(data, target_version="2.0")
    assert result["version"] == "2.0"
    assert result["new_field"] == "added"
    assert result["existing"] is True


def test_schema_migrator_no_path_registered_raises() -> None:
    """migrate() must raise ValueError when no migration path is registered."""
    migrator = SchemaMigrator()
    data: dict[str, object] = {"version": "1.0"}
    with pytest.raises(ValueError, match="No migration path"):
        migrator.migrate(data, target_version="3.0")


def test_schema_migrator_detect_version_present() -> None:
    """detect_version must return the version declared in the payload."""
    migrator = SchemaMigrator()
    assert migrator.detect_version({"version": "1.0"}) == "1.0"


def test_schema_migrator_detect_version_missing_defaults_to_current() -> None:
    """detect_version must fall back to SchemaVersion.CURRENT when absent."""
    migrator = SchemaMigrator()
    assert migrator.detect_version({}) == SchemaVersion.CURRENT


# ---------------------------------------------------------------------------
# Integration — encrypt -> write -> read -> decrypt
# ---------------------------------------------------------------------------


def test_integration_encrypt_write_read_decrypt_file(
    encryptor: SessionEncryptor,
    sample_payload: dict[str, object],
    tmp_dir: Path,
) -> None:
    """Encrypting, writing to disk, reading back, and decrypting must recover payload."""
    session_file = tmp_dir / "session.usf.enc"
    encrypted = encryptor.encrypt(sample_payload)
    wire = encrypted.to_bytes()
    session_file.write_bytes(wire)

    read_back = session_file.read_bytes()
    restored = EncryptedPayload.from_bytes(read_back)
    recovered = encryptor.decrypt(restored)
    assert recovered == sample_payload


def test_integration_encrypt_file_with_locking(
    encryptor: SessionEncryptor,
    sample_payload: dict[str, object],
    tmp_dir: Path,
) -> None:
    """Encrypt and write using FileLock to guard concurrent access."""
    session_file = tmp_dir / "session.usf.enc"
    lock_path = tmp_dir / "session.usf.enc.lock"

    with FileLock(lock_path):
        encrypted = encryptor.encrypt(sample_payload)
        session_file.write_bytes(encrypted.to_bytes())

    with FileLock(lock_path):
        wire = session_file.read_bytes()
        restored = EncryptedPayload.from_bytes(wire)
        recovered = encryptor.decrypt(restored)

    assert recovered == sample_payload


# ---------------------------------------------------------------------------
# Backward compatibility — exporter without encryptor still works
# ---------------------------------------------------------------------------


def test_langchain_exporter_without_encryptor() -> None:
    """LangChainExporter.export() without encryptor must return a plain dict."""
    from agent_session_linker.portable.exporters import LangChainExporter
    from agent_session_linker.portable.usf import UniversalSession

    session = UniversalSession(framework_source="langchain")
    result = LangChainExporter().export(session)
    assert "messages" in result
    assert "memory_variables" in result
    assert "_encrypted" not in result


def test_openai_exporter_without_encryptor() -> None:
    """OpenAIExporter.export() without encryptor must return a plain dict."""
    from agent_session_linker.portable.exporters import OpenAIExporter
    from agent_session_linker.portable.usf import UniversalSession

    session = UniversalSession(framework_source="openai")
    result = OpenAIExporter().export(session)
    assert "thread_id" in result
    assert "messages" in result
    assert "_encrypted" not in result


def test_crewai_exporter_without_encryptor() -> None:
    """CrewAIExporter.export() without encryptor must return a plain dict."""
    from agent_session_linker.portable.exporters import CrewAIExporter
    from agent_session_linker.portable.usf import UniversalSession

    session = UniversalSession(framework_source="crewai")
    result = CrewAIExporter().export(session)
    assert "context" in result
    assert "task_results" in result
    assert "_encrypted" not in result


# ---------------------------------------------------------------------------
# Exporter with encryptor — returns encrypted envelope
# ---------------------------------------------------------------------------


def test_langchain_exporter_with_encryptor_returns_envelope(
    encryptor: SessionEncryptor,
) -> None:
    """LangChainExporter.export(encryptor=...) must return an encrypted envelope."""
    from agent_session_linker.portable.exporters import LangChainExporter
    from agent_session_linker.portable.usf import UniversalSession

    session = UniversalSession(framework_source="langchain")
    result = LangChainExporter().export(session, encryptor=encryptor)
    assert "_encrypted" in result
    assert "_encryption_version" in result
    assert "messages" not in result


def test_openai_exporter_with_encryptor_decryptable(
    encryptor: SessionEncryptor,
) -> None:
    """The encrypted envelope produced by OpenAIExporter must be decryptable."""
    from agent_session_linker.portable.exporters import OpenAIExporter
    from agent_session_linker.portable.usf import UniversalSession

    session = UniversalSession(
        framework_source="openai",
        working_memory={"key": "val"},
    )
    result = OpenAIExporter().export(session, encryptor=encryptor)
    wire = base64.b64decode(str(result["_encrypted"]))
    restored_payload = EncryptedPayload.from_bytes(wire)
    decrypted = encryptor.decrypt(restored_payload)
    assert decrypted["session_id"] == session.session_id


# ---------------------------------------------------------------------------
# Without cryptography — helpful ImportError
# ---------------------------------------------------------------------------


def test_session_encryptor_raises_import_error_without_cryptography() -> None:
    """SessionEncryptor must raise ImportError with an actionable message
    when the cryptography package is not installed."""
    with patch(
        "agent_session_linker.portable.encryption._CRYPTO_AVAILABLE", False
    ):
        with pytest.raises(ImportError, match="cryptography"):
            # Re-instantiate with the patched flag
            import agent_session_linker.portable.encryption as enc_mod
            original = enc_mod._CRYPTO_AVAILABLE
            enc_mod._CRYPTO_AVAILABLE = False
            try:
                enc_mod.SessionEncryptor(b"\x00" * 32)
            finally:
                enc_mod._CRYPTO_AVAILABLE = original
