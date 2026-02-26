"""Unit tests for agent_session_linker.storage.s3.S3Backend.

All tests mock boto3 at the import level so no AWS credentials or
actual S3 buckets are needed.
"""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(
    bucket_name: str = "test-bucket",
    prefix: str = "agent-sessions/",
    endpoint_url: str | None = None,
) -> Any:
    """Return an S3Backend whose internal s3 client is a MagicMock."""
    mock_boto3 = MagicMock()
    mock_session = MagicMock()
    mock_client = MagicMock()

    mock_boto3.session.Session.return_value = mock_session
    mock_session.client.return_value = mock_client

    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        from agent_session_linker.storage.s3 import S3Backend
        backend = S3Backend(
            bucket_name=bucket_name,
            prefix=prefix,
            endpoint_url=endpoint_url,
        )
    backend._mock_s3 = mock_client  # type: ignore[attr-defined]
    return backend


def _make_fresh_backend_module() -> Any:
    """Reload S3Backend module to get a fresh import for each test."""
    import importlib
    import agent_session_linker.storage.s3 as mod
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestS3BackendImportGuard:
    def test_import_error_when_boto3_missing(self) -> None:
        with patch.dict(sys.modules, {"boto3": None}):  # type: ignore[dict-item]
            mod = _make_fresh_backend_module()
            with pytest.raises(ImportError, match="boto3"):
                mod.S3Backend("my-bucket")

    def test_import_error_contains_install_hint(self) -> None:
        with patch.dict(sys.modules, {"boto3": None}):  # type: ignore[dict-item]
            mod = _make_fresh_backend_module()
            with pytest.raises(ImportError, match="pip install boto3"):
                mod.S3Backend("my-bucket")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestS3BackendConstruction:
    def test_bucket_name_stored(self) -> None:
        backend = _make_backend(bucket_name="my-bucket")
        assert backend._bucket == "my-bucket"

    def test_default_prefix_stored(self) -> None:
        backend = _make_backend()
        assert backend._prefix == "agent-sessions/"

    def test_custom_prefix_stored(self) -> None:
        backend = _make_backend(prefix="sessions/prod/")
        assert backend._prefix == "sessions/prod/"

    def test_session_client_created_with_endpoint_url(self) -> None:
        mock_boto3 = MagicMock()
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_boto3.session.Session.return_value = mock_session
        mock_session.client.return_value = mock_client

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            import importlib
            import agent_session_linker.storage.s3 as mod
            importlib.reload(mod)
            mod.S3Backend("bucket", endpoint_url="http://localhost:4566")
            mock_session.client.assert_called_once_with(
                "s3", endpoint_url="http://localhost:4566"
            )


# ---------------------------------------------------------------------------
# _object_key helper
# ---------------------------------------------------------------------------


class TestS3BackendObjectKey:
    def test_object_key_combines_prefix_and_suffix(self) -> None:
        backend = _make_backend(prefix="pfx/")
        assert backend._object_key("sess-1") == "pfx/sess-1.json"

    def test_object_key_default_prefix(self) -> None:
        backend = _make_backend()
        assert backend._object_key("abc") == "agent-sessions/abc.json"


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestS3BackendSave:
    def test_save_calls_put_object(self) -> None:
        backend = _make_backend()
        backend.save("s1", '{"a": 1}')
        backend._mock_s3.put_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="agent-sessions/s1.json",
            Body=b'{"a": 1}',
            ContentType="application/json",
        )

    def test_save_encodes_payload_as_utf8(self) -> None:
        backend = _make_backend()
        backend.save("s1", "unicode: \u00e9")
        call_kwargs = backend._mock_s3.put_object.call_args[1]
        assert call_kwargs["Body"] == "unicode: \u00e9".encode("utf-8")

    def test_save_uses_correct_content_type(self) -> None:
        backend = _make_backend()
        backend.save("s1", "data")
        call_kwargs = backend._mock_s3.put_object.call_args[1]
        assert call_kwargs["ContentType"] == "application/json"


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class _S3NoSuchKey(Exception):
    """Stand-in for the real boto3 S3 NoSuchKey exception in tests."""


class TestS3BackendLoad:
    def test_load_returns_decoded_body(self) -> None:
        backend = _make_backend()
        mock_body = MagicMock()
        mock_body.read.return_value = b'{"loaded": true}'
        backend._mock_s3.get_object.return_value = {"Body": mock_body}
        result = backend.load("s1")
        assert result == '{"loaded": true}'

    def test_load_raises_key_error_on_no_such_key_via_response(self) -> None:
        backend = _make_backend()
        # Make exceptions.NoSuchKey a real exception class so Python can catch it.
        backend._mock_s3.exceptions.NoSuchKey = _S3NoSuchKey
        exc = _S3NoSuchKey("NoSuchKey")
        backend._mock_s3.get_object.side_effect = exc
        with pytest.raises(KeyError, match="s1"):
            backend.load("s1")

    def test_load_raises_key_error_on_404_code(self) -> None:
        backend = _make_backend()
        # Make exceptions.NoSuchKey a real exception class (not MagicMock).
        backend._mock_s3.exceptions.NoSuchKey = _S3NoSuchKey
        exc = Exception("Not Found")
        exc.response = MagicMock()  # type: ignore[attr-defined]
        exc.response.Error = {"Code": "404"}  # type: ignore[attr-defined]
        backend._mock_s3.get_object.side_effect = exc
        with pytest.raises(KeyError):
            backend.load("s1")

    def test_load_re_raises_unexpected_exceptions(self) -> None:
        backend = _make_backend()
        # Make exceptions.NoSuchKey a real exception class (not MagicMock).
        backend._mock_s3.exceptions.NoSuchKey = _S3NoSuchKey
        exc = RuntimeError("network failure")
        backend._mock_s3.get_object.side_effect = exc
        with pytest.raises(RuntimeError, match="network failure"):
            backend.load("s1")

    def test_load_calls_get_object_with_correct_key(self) -> None:
        backend = _make_backend(prefix="pfx/")
        mock_body = MagicMock()
        mock_body.read.return_value = b"data"
        backend._mock_s3.get_object.return_value = {"Body": mock_body}
        backend.load("my-session")
        backend._mock_s3.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="pfx/my-session.json"
        )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestS3BackendList:
    def _setup_paginator(self, backend: Any, pages: list[dict]) -> None:
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = iter(pages)
        backend._mock_s3.get_paginator.return_value = mock_paginator

    def test_list_returns_empty_when_no_objects(self) -> None:
        backend = _make_backend()
        self._setup_paginator(backend, [{"Contents": []}])
        assert backend.list() == []

    def test_list_strips_prefix_and_json_suffix(self) -> None:
        backend = _make_backend(prefix="agent-sessions/")
        self._setup_paginator(
            backend,
            [{"Contents": [{"Key": "agent-sessions/abc.json"}]}],
        )
        assert backend.list() == ["abc"]

    def test_list_handles_multiple_pages(self) -> None:
        backend = _make_backend(prefix="pfx/")
        self._setup_paginator(
            backend,
            [
                {"Contents": [{"Key": "pfx/s1.json"}, {"Key": "pfx/s2.json"}]},
                {"Contents": [{"Key": "pfx/s3.json"}]},
            ],
        )
        result = backend.list()
        assert sorted(result) == ["s1", "s2", "s3"]

    def test_list_skips_objects_without_json_suffix(self) -> None:
        backend = _make_backend(prefix="pfx/")
        self._setup_paginator(
            backend,
            [{"Contents": [{"Key": "pfx/s1.json"}, {"Key": "pfx/s2.txt"}]}],
        )
        result = backend.list()
        assert "s1" in result
        assert "s2" not in result

    def test_list_handles_empty_page_contents(self) -> None:
        backend = _make_backend()
        self._setup_paginator(backend, [{}])
        assert backend.list() == []


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestS3BackendDelete:
    def test_delete_calls_delete_object(self) -> None:
        backend = _make_backend()
        # exists() check calls head_object — make it return successfully.
        backend._mock_s3.head_object.return_value = {}
        backend.delete("s1")
        backend._mock_s3.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="agent-sessions/s1.json"
        )

    def test_delete_raises_key_error_when_not_exists(self) -> None:
        backend = _make_backend()
        exc = Exception("Not Found")
        exc.response = MagicMock()  # type: ignore[attr-defined]
        exc.response.Error = {"Code": "404"}  # type: ignore[attr-defined]
        backend._mock_s3.head_object.side_effect = exc
        with pytest.raises(KeyError, match="s1"):
            backend.delete("s1")


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestS3BackendExists:
    def test_exists_returns_true_when_head_object_succeeds(self) -> None:
        backend = _make_backend()
        backend._mock_s3.head_object.return_value = {"ContentLength": 100}
        assert backend.exists("s1") is True

    def test_exists_returns_false_on_404_error(self) -> None:
        backend = _make_backend()
        exc = Exception("Not Found")
        exc.response = MagicMock()  # type: ignore[attr-defined]
        exc.response.Error = {"Code": "404"}  # type: ignore[attr-defined]
        backend._mock_s3.head_object.side_effect = exc
        assert backend.exists("s1") is False

    def test_exists_returns_false_on_no_such_key(self) -> None:
        backend = _make_backend()
        exc = Exception("NoSuchKey")
        exc.response = MagicMock()  # type: ignore[attr-defined]
        exc.response.Error = {"Code": "NoSuchKey"}  # type: ignore[attr-defined]
        backend._mock_s3.head_object.side_effect = exc
        assert backend.exists("s1") is False

    def test_exists_re_raises_unexpected_exception(self) -> None:
        backend = _make_backend()
        exc = RuntimeError("AWS timeout")
        backend._mock_s3.head_object.side_effect = exc
        with pytest.raises(RuntimeError, match="AWS timeout"):
            backend.exists("s1")

    def test_exists_uses_correct_key(self) -> None:
        backend = _make_backend(prefix="x/")
        backend._mock_s3.head_object.return_value = {}
        backend.exists("session-id")
        backend._mock_s3.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="x/session-id.json"
        )


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestS3BackendRepr:
    def test_repr_contains_bucket_name(self) -> None:
        backend = _make_backend(bucket_name="my-bucket")
        assert "my-bucket" in repr(backend)

    def test_repr_contains_prefix(self) -> None:
        backend = _make_backend(prefix="sessions/")
        assert "sessions/" in repr(backend)
