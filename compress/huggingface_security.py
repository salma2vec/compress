"""Fail-closed revision handling for Hugging Face model downloads."""

import re
from pathlib import Path

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def secure_pretrained_kwargs(source: str, revision: str | None) -> dict:
    """Return safe loader arguments for a local path or immutable Hub commit.

    Existing local paths are forced offline and do not need a Hub revision.
    Any non-local identifier must carry a full 40-character commit SHA.
    """
    if Path(source).expanduser().exists():
        return {"local_files_only": True, "revision": None}
    normalized = (revision or "").lower()
    if not _COMMIT_SHA.fullmatch(normalized):
        raise ValueError("Remote Hugging Face sources require a full 40-character commit SHA")
    return {"revision": normalized}
