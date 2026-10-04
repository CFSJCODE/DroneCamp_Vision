"""Operações pequenas de arquivos usadas por todos os estágios.

Função no projeto: utilidades comuns, para os outros arquivos ficarem legíveis.

O que faz:
- ``resolve_local_path``: acha neste clone um arquivo gravado com caminho de outra máquina.
- ``file_hash``: SHA-256 de um arquivo (identidade de fotos, labels, pesos e registros).
- ``make_run_directory``: cria ``runs/<etapa>_<data>_<id>`` sem sobrescrever nada.
- ``write_json``: grava JSON em UTF-8 legível.

Quando mexer: raramente; mudanças aqui afetam todos os comandos.
"""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
from uuid import uuid4

from .config import PROJECT_ROOT


def resolve_local_path(value: str | Path, root: Path) -> Path:
    """Reancore no projeto local um caminho gravado em outra máquina ou clone.

    Registros guardam caminhos absolutos (ex.: E:\\...\\sistema_ia\\data\\...).
    Se o original não existir aqui, procura o mesmo trecho após a pasta do
    projeto, na raiz configurada e depois neste clone (uma raiz temporária de
    saída não contém os originais). Quem chama confere o SHA-256 do arquivo.
    """
    # Caminho que já existe nesta máquina: usa direto.
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    # Separa as partes aceitando "\" (Windows) e "/" (Linux/macOS).
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    # Troca tudo até a última pasta "sistema_ia" pela pasta do projeto local.
    for base in dict.fromkeys((root, PROJECT_ROOT)):
        if base.name in parts:
            anchor = len(parts) - 1 - parts[::-1].index(base.name)
            candidate = base.joinpath(*parts[anchor + 1:])
            if candidate.exists():
                return candidate
    return path


def file_hash(path: Path) -> str:
    """SHA-256 lido em blocos de 1 MB (não carrega arquivos grandes inteiros na RAM)."""
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
    """Grava JSON com acentos legíveis; NaN/infinito são recusados (allow_nan=False)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
