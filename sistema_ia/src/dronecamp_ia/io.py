"""Operações pequenas de arquivos usadas por todos os estágios.

Função no projeto: utilidades comuns, para os outros arquivos ficarem legíveis.

O que faz:
- ``resolve_local_path``: acha neste clone um arquivo gravado com caminho de outra máquina.
- ``file_hash``: SHA-256 de um arquivo (identidade de fotos, labels, pesos e registros).
- ``make_run_directory``: cria ``runs/<etapa>_<data>_<id>`` sem sobrescrever nada.
- ``write_json``: grava JSON em UTF-8 legível.

Quando mexer: raramente; mudanças aqui afetam todos os comandos.

Quem chama: praticamente todos os módulos. ``file_hash`` e ``write_json`` estão
em quase todo comando; ``make_run_directory`` em training, prediction,
calibration, cycle, hparam_bandit, model_gate e model_report;
``resolve_local_path`` em quem lê registros de revisão (pilot, review_render,
exports, vlm_assist, calibration).

O que lê e grava no disco:
- Lê qualquer arquivo pedido (``file_hash``) e consulta a existência de caminhos.
- Cria pastas novas em ``runs/`` (``make_run_directory``) e grava arquivos JSON
  onde o chamador pedir (``write_json``).

Onde mexer (casos concretos):
- Para mudar o formato do nome das pastas de ``runs/`` (data, sufixo), vá em
  ``make_run_directory``; atenção a quem procura ``runs/pilot_train_*`` por padrão
  de nome (operations.py, plataforma).
- Para um formato de saída novo (ex.: CSV com a mesma garantia de UTF-8), crie
  uma função irmã de ``write_json`` aqui e use-a nos módulos.
- Para aceitar outro separador ou outra pasta-âncora em caminhos vindos de outra
  máquina, vá em ``resolve_local_path``.
"""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
from uuid import uuid4

from .config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Caminhos e identidade de arquivos
# ---------------------------------------------------------------------------

def resolve_local_path(value: str | Path, root: Path) -> Path:
    """Reancore no projeto local um caminho gravado em outra máquina ou clone.

    Registros guardam caminhos absolutos (ex.: E:\\...\\sistema_ia\\data\\...).
    Se o original não existir aqui, procura o mesmo trecho após a pasta do
    projeto, na raiz configurada e depois neste clone (uma raiz temporária de
    saída não contém os originais). Quem chama confere o SHA-256 do arquivo.

    Parâmetros:
        value (str | Path): caminho gravado num registro (Windows ou POSIX,
            absoluto ou relativo), ex.: ``"E:\\DroneCamp\\sistema_ia\\data\\reviews\\x\\foto.jpg"``.
        root (Path): raiz do projeto em uso (normalmente ``config.root``).

    Retorna:
        Path: o caminho que existe nesta máquina; se nenhum candidato existir,
        devolve ``Path(value)`` sem alteração (quem chama decide o erro).

    Exemplo: ``E:\\...\\sistema_ia\\data\\reviews\\a\\foto.jpg`` com ``root`` =
    ``/home/user/DroneCamp_Vision/sistema_ia`` vira
    ``/home/user/DroneCamp_Vision/sistema_ia/data/reviews/a/foto.jpg`` se existir.

    ATENCAO: a âncora é o NOME da pasta raiz (``root.name``, normalmente
    ``sistema_ia``). Se o projeto for renomeado numa máquina, caminhos gravados
    nas outras não são mais encontrados por aqui.
    """
    # Caminho que já existe nesta máquina: usa direto.
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    # Separa as partes aceitando "\" (Windows) e "/" (Linux/macOS).
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    # Troca tudo até a última pasta "sistema_ia" pela pasta do projeto local.
    # dict.fromkeys remove a duplicata quando root == PROJECT_ROOT, mantendo a ordem
    # (primeiro a raiz configurada, depois este clone).
    for base in dict.fromkeys((root, PROJECT_ROOT)):
        if base.name in parts:
            # parts[::-1].index acha a ÚLTIMA ocorrência do nome da pasta (lista invertida);
            # anchor é o índice dela na lista original.
            anchor = len(parts) - 1 - parts[::-1].index(base.name)
            # O que vem depois da âncora (data/reviews/...) é pendurado na base local.
            candidate = base.joinpath(*parts[anchor + 1:])
            if candidate.exists():
                return candidate
    # EXTENSAO: outra estratégia de busca (ex.: procurar pelo nome do arquivo numa pasta
    # conhecida) entraria aqui, antes de devolver o caminho original.
    return path


# ATENCAO: o SHA-256 é a identidade de fotos, labels, pesos, taxonomia e registros; quase
# todo comando confere o hash gravado contra este cálculo. Não troque o algoritmo sem
# migrar todos os registros e manifestos gravados.
def file_hash(path: Path) -> str:
    """SHA-256 lido em blocos de 1 MB (não carrega arquivos grandes inteiros na RAM).

    Parâmetros:
        path (Path): arquivo a ler (foto, label, peso ``.pt``/``.onnx``, JSON...).

    Retorna:
        str: o hash em hexadecimal minúsculo (64 caracteres).

    Erros:
        OSError: se o arquivo não existir ou não puder ser lido.
    """
    digest = sha256()
    with path.open("rb") as stream:
        # iter(função, b"") chama stream.read(1 MB) até voltar bytes vazios (fim do arquivo).
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Pastas de execução e gravação de JSON
# ---------------------------------------------------------------------------

def make_run_directory(root: Path, stage: str) -> Path:
    """Nunca sobrescreva uma execução anterior.

    Cria ``<root>/runs/<stage>_<AAAAMMDDTHHMMSSZ>_<8 hex>``. A data é em UTC
    (sufixo ``Z``) e o final é um trecho aleatório de UUID, para duas execuções
    no mesmo segundo não colidirem.

    Parâmetros:
        root (Path): pasta do projeto (``config.root``).
        stage (str): nome da etapa, ex.: ``"pilot_train"``, ``"train"``,
            ``"evaluate_val"``, ``"tune"``.

    Retorna:
        Path: a pasta recém-criada, ex.: ``runs/pilot_train_20261012T153000Z_1a2b3c4d``.

    Erros:
        FileExistsError: se a pasta já existir (``exist_ok=False``): nunca reaproveita.

    ATENCAO: a plataforma e outros módulos reconhecem execuções pelo prefixo
    ``<stage>_`` (ex.: ``runs/pilot_train_*``). Mudar o formato quebra essa busca.
    """
    # Data/hora em UTC no formato compacto ISO 8601 (ex.: 20261012T153000Z).
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # uuid4().hex[:8]: 8 caracteres hexadecimais aleatórios.
    directory = root / "runs" / f"{stage}_{timestamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path: Path, value: dict) -> None:
    """Grava JSON com acentos legíveis; NaN/infinito são recusados (allow_nan=False).

    Parâmetros:
        path (Path): arquivo de destino; as pastas que faltarem são criadas.
            Um arquivo existente é SOBRESCRITO (quem chama garante a pasta nova).
        value (dict): conteúdo serializável em JSON (str, números finitos, listas,
            dicts, bool, None). ``Path`` e ``datetime`` não são aceitos: converta antes.

    Retorna:
        None.

    Erros:
        ValueError: se houver NaN ou infinito (JSON padrão não os representa).
        TypeError: se houver um valor não serializável.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False mantém "ç" e "ã" legíveis; indent=2 facilita ler e comparar no git.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
