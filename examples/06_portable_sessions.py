#!/usr/bin/env python3
"""Example: Portable Sessions and USF Format

Demonstrates exporting sessions to the Universal Session Format (USF)
and importing them back, with optional encryption.

Usage:
    python examples/06_portable_sessions.py

Requirements:
    pip install agent-session-linker
"""
from __future__ import annotations

import agent_session_linker
from agent_session_linker import (
    ContextSegment,
    EncryptedPayload,
    InMemoryBackend,
    OpenAIExporter,
    OpenAIImporter,
    SessionEncryptor,
    SessionExporter,
    SessionImporter,
    SessionManager,
    SessionState,
    USFMessage,
    UniversalSession,
    USFVersion,
)


def main() -> None:
    print(f"agent-session-linker version: {agent_session_linker.__version__}")

    # Build a session to export
    state = SessionState(
        session_id="export-demo-001",
        segments=[
            ContextSegment(text="User asked about product roadmap.", importance=0.8),
            ContextSegment(text="Agent retrieved Q4 release plan.", importance=0.9),
            ContextSegment(text="Next steps: draft announcement email.", importance=0.7),
        ],
    )

    # Export to USF
    exporter = SessionExporter()
    usf: UniversalSession = exporter.export(state)
    print(f"Exported to USF version {usf.version.value}")
    print(f"  Messages: {len(usf.messages)}")

    # Import back from USF
    importer = SessionImporter()
    restored = importer.import_session(usf)
    print(f"  Imported session: {restored.session_id} "
          f"({len(restored.segments)} segments)")

    # Export in OpenAI format
    oai_exporter = OpenAIExporter()
    oai_messages = oai_exporter.export(state)
    print(f"\nOpenAI format: {len(oai_messages)} messages")
    for msg in oai_messages[:2]:
        print(f"  [{msg['role']}] {str(msg['content'])[:60]}")

    # Import from OpenAI format
    oai_importer = OpenAIImporter()
    from_oai = oai_importer.import_messages(
        messages=oai_messages,
        session_id="from-oai-001",
    )
    print(f"  Imported from OpenAI: {len(from_oai.segments)} segments")

    # Encrypt a session payload
    encryptor = SessionEncryptor()
    key = encryptor.generate_key()
    payload = exporter.export_bytes(state)
    encrypted: EncryptedPayload = encryptor.encrypt(payload, key=key)
    decrypted = encryptor.decrypt(encrypted, key=key)
    print(f"\nEncryption: payload={len(payload)}B -> "
          f"ciphertext={len(encrypted.ciphertext)}B -> "
          f"decrypted={len(decrypted)}B (match={payload == decrypted})")


if __name__ == "__main__":
    main()
