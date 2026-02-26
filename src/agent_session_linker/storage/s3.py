"""AWS S3 storage backend.

Import-guarded: ``boto3`` is an optional dependency.  Attempting to
instantiate ``S3Backend`` without ``boto3`` installed raises ``ImportError``.

Each session is stored as a separate S3 object whose key is
``<prefix><session_id>.json``.

Classes
-------
- S3Backend  — AWS S3 object-storage backend
"""
from __future__ import annotations

from agent_session_linker.storage.base import StorageBackend

_BOTO3_IMPORT_ERROR = (
    "The 'boto3' package is required for S3Backend. "
    "Install it with: pip install boto3"
)


class S3Backend(StorageBackend):
    """Stores sessions as individual JSON objects in an S3 bucket.

    Parameters
    ----------
    bucket_name:
        Name of the target S3 bucket.
    prefix:
        Key prefix for all session objects.  Defaults to
        ``"agent-sessions/"``  (note the trailing slash).
    region_name:
        AWS region for the bucket.
    aws_access_key_id:
        Optional explicit AWS access key ID.
    aws_secret_access_key:
        Optional explicit AWS secret key.
    endpoint_url:
        Optional custom endpoint URL (e.g. for LocalStack).
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "agent-sessions/",
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(_BOTO3_IMPORT_ERROR) from exc

        session = boto3.session.Session(
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self._s3 = session.client("s3", endpoint_url=endpoint_url)
        self._bucket = bucket_name
        self._prefix = prefix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _object_key(self, session_id: str) -> str:
        """Return the full S3 object key for ``session_id``.

        Parameters
        ----------
        session_id:
            Raw session identifier.

        Returns
        -------
        str
            S3 object key string.
        """
        return f"{self._prefix}{session_id}.json"

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, session_id: str, payload: str) -> None:
        """Write ``payload`` to S3.

        Parameters
        ----------
        session_id:
            Storage key.
        payload:
            UTF-8 string (typically JSON).
        """
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._object_key(session_id),
            Body=payload.encode("utf-8"),
            ContentType="application/json",
        )

    def load(self, session_id: str) -> str:
        """Download and return the payload for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to retrieve.

        Returns
        -------
        str
            Decoded payload string.

        Raises
        ------
        KeyError
            If the object does not exist in S3.
        """
        try:
            response = self._s3.get_object(
                Bucket=self._bucket, Key=self._object_key(session_id)
            )
            return response["Body"].read().decode("utf-8")
        except self._s3.exceptions.NoSuchKey:
            raise KeyError(
                f"Session {session_id!r} not found in S3Backend "
                f"(bucket={self._bucket!r})."
            ) from None
        except Exception as exc:
            # Wrap unexpected S3/network errors in a descriptive KeyError.
            error_code = getattr(getattr(exc, "response", None), "Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise KeyError(
                    f"Session {session_id!r} not found in S3Backend."
                ) from exc
            raise

    def list(self) -> list[str]:
        """List all session IDs stored under the configured prefix.

        Uses paginated ``list_objects_v2`` to handle large buckets.

        Returns
        -------
        list[str]
            Session IDs with prefix and ``.json`` suffix stripped.
        """
        prefix_len = len(self._prefix)
        session_ids: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                # Strip prefix and .json suffix.
                if key.startswith(self._prefix) and key.endswith(".json"):
                    session_ids.append(key[prefix_len:-5])
        return session_ids

    def delete(self, session_id: str) -> None:
        """Remove the S3 object for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to delete.

        Raises
        ------
        KeyError
            If the object does not exist.
        """
        if not self.exists(session_id):
            raise KeyError(
                f"Session {session_id!r} not found in S3Backend "
                f"(bucket={self._bucket!r})."
            )
        self._s3.delete_object(Bucket=self._bucket, Key=self._object_key(session_id))

    def exists(self, session_id: str) -> bool:
        """Return True if the S3 object for ``session_id`` exists."""
        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._object_key(session_id))
            return True
        except Exception as exc:
            error_code = getattr(getattr(exc, "response", None), "Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            # Unexpected errors bubble up.
            raise

    def __repr__(self) -> str:
        return (
            f"S3Backend(bucket={self._bucket!r}, prefix={self._prefix!r})"
        )
