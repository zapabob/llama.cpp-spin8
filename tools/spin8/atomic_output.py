"""Publish a complete local/DriveFS file without requiring hardlinks."""
from __future__ import annotations

import os
from pathlib import Path


def publish_file(temporary: str | Path, target: str | Path, *, overwrite: bool = False) -> None:
    temporary, target = Path(temporary), Path(target)
    if temporary.parent.resolve() != target.parent.resolve():
        raise ValueError("Atomic publication requires a temporary file beside the target")
    lock = target.with_name(target.name + ".publish.lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("Output publication is locked: " + str(lock)
                              + "; remove a stale lock only after confirming the previous writer has stopped") from error
    try:
        with os.fdopen(descriptor, "w") as marker:
            marker.write(str(os.getpid()) + "\n")
        if not overwrite and (target.exists() or target.is_symlink()):
            raise FileExistsError(target)
        with temporary.open("rb") as complete:
            os.fsync(complete.fileno())
        os.replace(temporary, target)
    finally:
        lock.unlink()
