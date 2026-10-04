"""Pequenas operações de arquivos para manter os estágios legíveis."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
from uuid import uuid4


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_run_directory(root: Path, stage: str) -> Path:
    """Nunca sobrescreva uma execução anterior."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "runs" / f"{stage}_{timestamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
