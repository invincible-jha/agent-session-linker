"""AES-256-GCM encryption for Universal Session Format payloads.

This module provides commodity AES-256-GCM encryption as an optional layer
over USF session data.  The ``cryptography`` package is a soft dependency;
a helpful ``ImportError`` is raised when it is absent so that callers who
do not need encryption pay no installation cost.

Classes
-------
EncryptedPayload
    Immutable value object holding ciphertext and nonce; serialises to/from
    a compact binary wire format.
SessionEncryptor
    Encrypts and decrypts ``dict`` payloads with a caller-supplied 32-byte
    AES-256 key.

Constants
---------
_CRYPTO_AVAILABLE
    ``True`` when the ``cryptography`` package is installed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover — only missing when cryptography absent
    _CRYPTO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Wire-format constants
# ---------------------------------------------------------------------------

_NONCE_LENGTH: int = 12  # 96-bit nonce per NIST SP 800-38D
_KEY_LENGTH: int = 32  # 256-bit key for AES-256


# ---------------------------------------------------------------------------
# EncryptedPayload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EncryptedPayload:
    """Immutable holder for AES-256-GCM ciphertext and its associated nonce.

    Parameters
    ----------
    ciphertext:
        The encrypted bytes, including the 16-byte GCM authentication tag
        appended by the AESGCM primitive.
    nonce:
        The 12-byte (96-bit) nonce used during encryption.  Must never be
        reused with the same key.
    version:
        Wire-format version string; defaults to ``"1.0"``.
    """

    ciphertext: bytes
    nonce: bytes
    version: str = "1.0"

    def to_bytes(self) -> bytes:
        """Serialise to a compact binary format.

        Wire layout (big-endian lengths)::

            version_len (1 byte)
            version     (version_len bytes, UTF-8)
            nonce_len   (1 byte)
            nonce       (nonce_len bytes)
            ciphertext  (remainder)

        Returns
        -------
        bytes
            The serialised payload.
        """
        version_bytes = self.version.encode("utf-8")
        return (
            len(version_bytes).to_bytes(1, "big")
            + version_bytes
            + len(self.nonce).to_bytes(1, "big")
            + self.nonce
            + self.ciphertext
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> EncryptedPayload:
        """Deserialise from bytes previously produced by :meth:`to_bytes`.

        Parameters
        ----------
        data:
            Raw bytes in the wire format described in :meth:`to_bytes`.

        Returns
        -------
        EncryptedPayload
            The reconstructed payload.

        Raises
        ------
        ValueError
            If *data* is too short to contain a valid payload.
        """
        if len(data) < 2:
            raise ValueError("Payload too short to deserialise")
        offset = 0
        version_len = data[offset]
        offset += 1
        if len(data) < offset + version_len:
            raise ValueError("Payload truncated in version field")
        version = data[offset : offset + version_len].decode("utf-8")
        offset += version_len
        if len(data) < offset + 1:
            raise ValueError("Payload truncated before nonce_len")
        nonce_len = data[offset]
        offset += 1
        if len(data) < offset + nonce_len:
            raise ValueError("Payload truncated in nonce field")
        nonce = data[offset : offset + nonce_len]
        offset += nonce_len
        ciphertext = data[offset:]
        return cls(ciphertext=ciphertext, nonce=nonce, version=version)


# ---------------------------------------------------------------------------
# SessionEncryptor
# ---------------------------------------------------------------------------


class SessionEncryptor:
    """AES-256-GCM encryption and decryption for session dict payloads.

    Parameters
    ----------
    key:
        Exactly 32 bytes of key material.  Use :meth:`generate_key` to
        create a cryptographically random key.

    Raises
    ------
    ImportError
        If the ``cryptography`` package is not installed.
    ValueError
        If *key* is not exactly 32 bytes.
    """

    def __init__(self, key: bytes) -> None:
        if not _CRYPTO_AVAILABLE:
            raise ImportError(  # pragma: no cover
                "Install cryptography>=41.0: pip install agent-session-linker[crypto]"
            )
        if len(key) != _KEY_LENGTH:
            raise ValueError(
                f"Key must be {_KEY_LENGTH} bytes (AES-256), got {len(key)}"
            )
        self._aesgcm = AESGCM(key)

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_key() -> bytes:
        """Generate a random 32-byte AES-256 key using ``os.urandom``.

        Returns
        -------
        bytes
            32 bytes of cryptographically random key material.
        """
        return os.urandom(_KEY_LENGTH)

    # ------------------------------------------------------------------
    # Encrypt / decrypt
    # ------------------------------------------------------------------

    def encrypt(self, payload: dict[str, object]) -> EncryptedPayload:
        """Encrypt *payload* using AES-256-GCM.

        The payload is JSON-serialised (keys sorted for determinism) before
        encryption.  A fresh 12-byte nonce is generated for every call.

        Parameters
        ----------
        payload:
            Arbitrary JSON-serialisable dict to encrypt.

        Returns
        -------
        EncryptedPayload
            The ciphertext and nonce wrapped in an :class:`EncryptedPayload`.
        """
        plaintext = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        nonce = os.urandom(_NONCE_LENGTH)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)
        return EncryptedPayload(ciphertext=ciphertext, nonce=nonce)

    def decrypt(self, encrypted: EncryptedPayload) -> dict[str, object]:
        """Decrypt *encrypted* and return the original dict.

        Parameters
        ----------
        encrypted:
            An :class:`EncryptedPayload` produced by :meth:`encrypt`.

        Returns
        -------
        dict[str, object]
            The decrypted and JSON-parsed payload.

        Raises
        ------
        cryptography.exceptions.InvalidTag
            If the ciphertext or nonce has been tampered with.
        """
        plaintext = self._aesgcm.decrypt(encrypted.nonce, encrypted.ciphertext, None)
        result: dict[str, object] = json.loads(plaintext.decode("utf-8"))
        return result
