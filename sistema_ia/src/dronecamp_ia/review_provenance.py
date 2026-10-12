"""Guarda de procedência do treino de PRODUÇÃO, executada logo antes de treinar.

Função no projeto: ``training.prepare_dataset`` chama
``validate_training_provenance`` para provar que cada foto e label do dataset
de produção veio de uma aprovação humana registrada e não foi alterada.

O que faz:
- reconcilia cada amostra (foto, label, split, grupo) com o registro de revisão
  de origem e com ``configs/review_groups.json``;
- confere SHA-256 de fotos, labels e documentos de origem;
- não confia no indicador de prontidão salvo em disco: recalcula tudo.

Um YAML válido não comprova revisão humana. O piloto tem a sua própria guarda
(``pilot.validate_pilot_dataset``).

Quando mexer: só se o formato de ``provenance.json`` mudar em ``review_dataset.py``.

Funções (em ordem no arquivo):
- ``_json(path)``: lê um JSON de origem exigindo ``schema_version == 1``.
- ``_source_document(provenance, name)``: abre o registro ou o arquivo de grupos
  citado em ``provenance.json`` e confere o hash.
- ``_relative_path(root, value, prefix, split)``: valida um caminho relativo de amostra.
- ``_group_rows(root)``: lê ``groups.csv``.
- ``_check_sample(root, sample, reviewed, assignments, groups)``: confere UMA amostra.
- ``validate_training_provenance(config, data_path)``: a guarda completa (pública).

Quem chama: ``training.prepare_dataset`` (treino de produção). Se esta função
levanta ValueError, o motivo é gravado em ``runs/<treino>/training_provenance.json``
e o treino não começa.

O que lê (somente leitura; não grava nada):
- ``dataset.yaml``, ``provenance.json`` e ``groups.csv`` da pasta do dataset
  (gerados por ``review_dataset.build_approved_dataset``);
- as fotos e labels de ``images/<split>/`` e ``labels/<split>/``;
- o ``registry.json`` e o arquivo de grupos indicados em ``provenance.json``
  (``registry_path``/``assignments_path``, com ``registry_sha256``/``assignments_sha256``);
- as fotos originais (``source_path`` de cada foto do registro);
- ``configs/taxonomy.json``.

Campos de cada amostra em ``provenance.json`` conferidos aqui: ``image_sha256``,
``image``, ``label``, ``label_sha256``, ``source_path``, ``building_group``, ``split``,
``width``, ``height``, ``status``, ``human_approved``, ``technical_status``,
``approval_author``. Campos de cada foto do registro usados: ``image_sha256``,
``human_approved``, ``technical_status``, ``human_reviewer``, ``status``,
``building_group``, ``width``, ``height``, ``source_path`` e ``boxes`` (formato completo
no cabeçalho de ``review_data.py``).

Onde mexer:
- Nova checagem por amostra: fim de ``_check_sample``.
- Nova regra de prontidão (ex.: número mínimo de fotos por split): etapa 4 de
  ``validate_training_provenance``; mantenha a regra equivalente em
  ``review_dataset._readiness`` para os dois lados concordarem.
- Novo documento de origem a conferir: chame ``_source_document(provenance, "<nome>")``
  na etapa 2 (``provenance.json`` precisa ter ``<nome>_path`` e ``<nome>_sha256``).
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import yaml

from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS, SPLITS, validate_dataset
from .io import file_hash
from .review_data import boxes_to_yolo, review_image_size, validate_boxes
from .review_dataset import MVP_NAMES


# ---------------------------------------------------------------------------
# Leitura dos documentos de origem e dos caminhos das amostras
# ---------------------------------------------------------------------------

def _json(path: Path) -> dict:
    """Arquivos declarados precisam existir e usar o schema conhecido.

    Parâmetros:
        path (Path): JSON a ler (``provenance.json``, registro ou arquivo de grupos).

    Retorna:
        dict: o conteúdo, com ``schema_version`` inteiro igual a 1.

    Erros (ValueError): "Proveniência não verificável" se o arquivo não puder ser lido
    ou decodificado; "Schema de proveniência inválido" se não for dict ou o schema diferir.
    """
    # Erros de leitura viram ValueError para o treino registrar um motivo único.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Proveniência não verificável: {path.name}.") from error
    # type(...) is not int recusa True/1.0 como versão.
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"Schema de proveniência inválido: {path.name}.")
    return value


def _source_document(provenance: dict, name: str) -> tuple[Path, dict]:
    """Não siga instruções do documento: leia somente sua estrutura declarada.

    Parâmetros:
        provenance (dict): conteúdo de ``provenance.json``.
        name (str): prefixo do documento: ``"registry"`` ou ``"assignments"``; são lidas
            as chaves ``<name>_path`` (caminho absoluto) e ``<name>_sha256``.

    Retorna:
        tuple[Path, dict]: caminho do documento e seu conteúdo (validado por ``_json``).

    Erros (ValueError): caminho ausente, URL, relativo ou inexistente; hash diferente
    do gravado na proveniência (o registro ou os grupos mudaram depois do dataset).

    ATENCAO: o caminho precisa existir exatamente como gravado (absoluto, sem
    reancoragem); um dataset copiado para outro PC falha aqui.
    """
    value = provenance.get(f"{name}_path")
    if not isinstance(value, str) or "://" in value:
        raise ValueError(f"Caminho de {name} ausente ou não local.")
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"Origem {name} não existe como arquivo local absoluto.")
    if provenance.get(f"{name}_sha256") != file_hash(path):
        raise ValueError(f"Hash de {name} diverge da origem da revisão.")
    return path, _json(path)


def _relative_path(root: Path, value: object, prefix: str, split: str) -> Path:
    """O manifesto só pode referenciar amostras dentro do split declarado.

    Parâmetros:
        root (Path): pasta raiz do dataset (``path`` do dataset.yaml).
        value (object): caminho declarado na proveniência (ex.: ``"images/train/<hash>.jpg"``).
        prefix (str): ``"images"`` ou ``"labels"``.
        split (str): ``"train"``, ``"val"`` ou ``"test"``.

    Retorna:
        Path: caminho absoluto do arquivo (existe e está em ``root/prefix/split``).

    Erros (ValueError): caminho vazio, absoluto, com ``..``, não canônico (ex.: com
    "\\" ou "./"), fora da pasta do split ou inexistente.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("Caminho de amostra ausente na proveniência.")
    relative = Path(value)
    # Canônico: as_posix() precisa devolver exatamente o texto gravado (só "/", sem "./").
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError("Caminho de amostra deve ser relativo, canônico e sem travessia.")
    expected_parent = root / prefix / split
    path = root / relative
    # resolve() segue links simbólicos: um link apontando para fora também é recusado.
    if not path.resolve().is_relative_to(expected_parent.resolve()) or not path.is_file():
        raise ValueError("Amostra ausente ou fora do diretório declarado na proveniência.")
    return path


def _group_rows(root: Path) -> dict[str, tuple[str, str]]:
    """O validador estrutural já confere schema, duplicatas e vazamento.

    Parâmetros:
        root (Path): pasta raiz do dataset (contém ``groups.csv``).

    Retorna:
        dict: ``{"images/<split>/<arquivo>": (group_id, split)}``; barras invertidas
        do Windows são trocadas por "/".
    """
    # utf-8-sig aceita CSV salvo com BOM (Excel).
    with (root / "groups.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["image"].replace("\\", "/"): (row["group_id"], row["split"]) for row in csv.DictReader(stream)}


# ---------------------------------------------------------------------------
# Conferência de cada amostra
# ---------------------------------------------------------------------------

def _check_sample(root: Path, sample: dict, reviewed: dict, assignments: dict, groups: dict) -> None:
    """Confirme autoria humana, identidade da foto e anotação aprovada.

    Parâmetros:
        root (Path): pasta raiz do dataset.
        sample (dict): entrada de ``provenance.json["images"]`` (veja o cabeçalho).
        reviewed (dict): a foto correspondente (mesmo hash) no ``registry.json``.
        assignments (dict): ``{building_group: split}`` do arquivo de grupos.
        groups (dict): linhas de ``groups.csv`` (``_group_rows``).

    Retorna:
        None; só retorna se a amostra passar em todas as checagens.

    Erros (ValueError), em ordem: falta aprovação/estado humano; autor ausente ou
    diferente; status não positivo/negativo ou diferente do registro; grupo/split
    divergente; caminhos inválidos; label com nome diferente da foto; hash da foto ou
    da label diferente; dimensões divergentes; original ausente/alterado; caixas
    inválidas; status incompatível com caixas; conteúdo da label diferente do esperado.

    EXTENSAO: nova checagem por amostra: acrescente um ``if ...: raise ValueError(...)``
    no fim da função (o campo novo precisa ser gravado em review_dataset._write_samples).
    """
    # 1. Aprovação humana nos DOIS lados (proveniência e registro) e mesmo autor.
    author = sample.get("approval_author")
    if sample.get("human_approved") is not True or reviewed.get("human_approved") is not True:
        raise ValueError("Treino exige aprovação humana explícita para cada amostra.")
    if sample.get("technical_status") != "human_visual_reviewed" or reviewed.get("technical_status") != "human_visual_reviewed":
        raise ValueError("Amostra ainda não possui estado de revisão humana.")
    if not isinstance(author, str) or not author.strip() or author != reviewed.get("human_reviewer"):
        raise ValueError("Autor da aprovação ausente ou diferente do registro humano.")
    # 2. Decisão positiva/negativa igual à do registro.
    status = sample.get("status")
    if not isinstance(status, str) or status not in {"positive", "negative"} or status != reviewed.get("status"):
        raise ValueError("Decisão humana incompatível com a proveniência.")
    # 3. Edificação e split coerentes entre proveniência, registro, arquivo de grupos e groups.csv.
    group, split = sample.get("building_group"), sample.get("split")
    if not isinstance(group, str) or not group.strip() or group != reviewed.get("building_group") or split not in SPLITS:
        raise ValueError("Grupo ou split diverge da imagem revisada.")
    if assignments.get(group) != split or groups.get(sample.get("image")) != (group, split):
        raise ValueError("Grupos de captura divergem da revisão ou dos assignments.")
    # 4. Foto e label dentro da pasta do split; label = labels/<split>/<mesmo nome>.txt.
    image = _relative_path(root, sample.get("image"), "images", split)
    label = _relative_path(root, sample.get("label"), "labels", split)
    expected_label = Path("labels") / image.relative_to(root / "images").with_suffix(".txt")
    if label.relative_to(root) != expected_label:
        raise ValueError("Label não corresponde à imagem declarada.")
    # 5. Hashes: foto copiada = hash da amostra = hash do registro; label = hash gravado.
    if file_hash(image) != sample.get("image_sha256") or sample.get("image_sha256") != reviewed.get("image_sha256"):
        raise ValueError("Hash da imagem diverge da amostra aprovada.")
    if file_hash(label) != sample.get("label_sha256"):
        raise ValueError("Hash da label diverge da anotação exportada.")
    # 6. Dimensões reais da cópia iguais às do registro e da proveniência (todas inteiras).
    width, height = review_image_size(image)
    if any(type(item.get(key)) is not int for item in (reviewed, sample) for key in ("width", "height")) or (width, height) != (reviewed.get("width"), reviewed.get("height")) or (width, height) != (sample.get("width"), sample.get("height")):
        raise ValueError("Dimensões da imagem divergem da aprovação.")
    # 7. Original revisado: local, absoluto, existente, mesmo caminho e mesmo hash/tamanho.
    source_value = reviewed.get("source_path")
    if not isinstance(source_value, str) or "://" in source_value:
        raise ValueError("Caminho da imagem original não verificável.")
    source = Path(source_value)
    if not source.is_absolute() or not source.is_file():
        raise ValueError("Imagem original revisada ausente.")
    if sample.get("source_path") != str(source.resolve()) or file_hash(source) != reviewed["image_sha256"]:
        raise ValueError("Hash ou caminho da imagem original diverge da revisão.")
    if review_image_size(source) != (width, height):
        raise ValueError("Dimensões da imagem original divergem da revisão.")
    # 8. Caixas aprovadas válidas e coerentes com o status.
    boxes = reviewed.get("boxes")
    validate_boxes(boxes, width, height, len(MVP_NAMES))
    if (status == "positive" and not boxes) or (status == "negative" and boxes):
        raise ValueError("Decisão humana incompatível com as caixas aprovadas.")
    # 9. A label em disco tem de ser exatamente a conversão YOLO das caixas aprovadas.
    # Recalcular impede editar a label e apenas atualizar seu hash no manifesto.
    # ATENCAO: depende do formato fixo de review_data.boxes_to_yolo (8 casas decimais).
    expected_contents = boxes_to_yolo(boxes, width, height)
    if label.read_text(encoding="utf-8") != expected_contents:
        raise ValueError("Label normalizada diverge das caixas do registro humano.")


# ---------------------------------------------------------------------------
# Guarda completa (chamada por training.prepare_dataset)
# ---------------------------------------------------------------------------

# ENTRADA: training.prepare_dataset chama esta função antes de qualquer treino de produção.
# ATENCAO: não relaxe estas regras para "fazer o treino rodar": é a última barreira que
# impede propostas de IA ou labels editadas à mão de entrarem no modelo de produção.
def validate_training_provenance(config: ProjectConfig, data_path: Path) -> dict:
    """Retorne o audit de prontidão ou ValueError antes de carregar um modelo.

    A função é somente leitura. A integridade documental não autentica uma pessoa;
    a aprovação tem de ter sido obtida no fluxo humano de revisão do projeto.

    Parâmetros:
        config (ProjectConfig): configuração (taxonomia).
        data_path (Path): ``dataset.yaml`` do dataset de produção.

    Retorna:
        dict: ``ready_for_training`` (sempre True quando retorna), ``dataset_root``,
        hashes de taxonomia/proveniência/registro/grupos/dataset.yaml/groups.csv,
        ``counts``, ``train_boxes_by_class`` ({"0": n, ...}) e ``groups_by_split``.

    Erros (ValueError): qualquer divergência; a mensagem diz qual. As principais:
    taxonomia diferente do MVP; dataset estruturalmente inválido; proveniência ou
    documentos de origem alterados; amostra duplicada/sem origem aprovada; arquivos a
    mais ou a menos; foto aprovada no registro que não está no dataset; classe sem
    positivo no treino; menos de três splits/edificações.
    """
    # 1. Taxonomia do MVP e estrutura do dataset.
    data_path = Path(data_path).resolve()
    taxonomy = load_taxonomy(config.taxonomy_path)
    names = detection_names(taxonomy)
    if tuple(names) != MVP_NAMES or [item["id"] for item in taxonomy["classes"] if item["phase"] == 1] != list(range(len(MVP_NAMES))):
        raise ValueError("A guarda requer a taxonomia das classes ativas do MVP.")
    validation = validate_dataset(data_path, names)
    if not validation["valid"]:
        raise ValueError("Dataset estruturalmente inválido: " + "; ".join(validation["errors"]))
    # 1a. Raiz do dataset: "path" do YAML, relativo à pasta do YAML quando não for absoluto.
    config_data = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    root_value = Path(config_data.get("path", ".")).expanduser()
    root = (data_path.parent / root_value).resolve() if not root_value.is_absolute() else root_value.resolve()
    # 2. provenance.json e documentos de origem (registro humano e grupos) intactos.
    provenance_path = root / "provenance.json"
    provenance = _json(provenance_path)
    taxonomy_digest = file_hash(config.taxonomy_path)
    if provenance.get("taxonomy_sha256") != taxonomy_digest:
        raise ValueError("Taxonomia diverge da proveniência do dataset.")
    # EXTENSAO: novo documento de origem a conferir: outra chamada _source_document(provenance, "<nome>").
    registry_path, registry = _source_document(provenance, "registry")
    assignments_path, assignments_document = _source_document(provenance, "assignments")
    if registry.get("taxonomy_sha256") != taxonomy_digest:
        raise ValueError("Taxonomia diverge do registro humano.")
    assignments = assignments_document.get("groups")
    if not isinstance(assignments, dict) or any(not isinstance(group, str) or not group.strip() or split not in SPLITS for group, split in assignments.items()):
        raise ValueError("Assignments inválidos na origem da revisão.")
    reviewed_images = registry.get("images")
    samples = provenance.get("images")
    if not isinstance(reviewed_images, list) or not isinstance(samples, list) or not samples:
        raise ValueError("Registro e proveniência devem listar as imagens aprovadas.")
    # 2a. Índice das fotos do registro por hash (64 hex minúsculos, sem duplicata).
    by_digest = {}
    for item in reviewed_images:
        if not isinstance(item, dict):
            raise ValueError("Cada imagem do registro deve ser um objeto estruturado.")
        digest = item.get("image_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or digest in by_digest:
            raise ValueError("Imagem inválida ou duplicada no registro humano.")
        by_digest[digest] = item
    # 3. Inventário real das pastas e conferência amostra a amostra.
    rows = _group_rows(root)
    inventory = set()
    # 3a. Toda foto que realmente existe nas pastas dos splits (caminhos relativos com "/").
    # ATENCAO: config_data[split] exige os três splits no dataset.yaml (KeyError se faltar um);
    # na prática validate_dataset e a regra de três splits recusam antes um dataset assim.
    for split in SPLITS:
        split_path = Path(config_data[split]).expanduser()
        split_root = (root / split_path).resolve() if not split_path.is_absolute() else split_path.resolve()
        inventory.update(path.relative_to(root).as_posix() for path in split_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    declared, seen_hashes = set(), set()
    train_counts = Counter()
    split_groups = {split: set() for split in SPLITS}
    # 3b. Cada amostra declarada: única, com origem no registro, conferida por _check_sample.
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Cada amostra da proveniência deve ser um objeto estruturado.")
        digest, relative = sample.get("image_sha256"), sample.get("image")
        if not isinstance(digest, str) or not isinstance(relative, str) or relative in declared or digest in seen_hashes or digest not in by_digest:
            raise ValueError("Inventário de proveniência duplicado ou sem origem aprovada.")
        declared.add(relative)
        seen_hashes.add(digest)
        reviewed = by_digest[digest]
        _check_sample(root, sample, reviewed, assignments, rows)
        split_groups[sample["split"]].add(sample["building_group"])
        # Caixas por classe só do treino (usadas na regra de cobertura de classes).
        if sample["split"] == "train":
            train_counts.update(box["class_id"] for box in reviewed["boxes"])
    # 4. Nada a mais ou a menos; todas as classes no treino; três edificações independentes.
    # EXTENSAO: nova regra de prontidão: acrescente aqui (e em review_dataset._readiness).
    if declared != inventory or set(rows) != declared:
        raise ValueError("Inventário de imagens/grupos diverge da proveniência; há amostras extras ou ausentes.")
    # Toda foto aprovada no registro precisa estar no dataset (e só elas).
    approved_digests = {digest for digest, item in by_digest.items() if item.get("human_approved") is True and item.get("status") in {"positive", "negative"}}
    if seen_hashes != approved_digests:
        raise ValueError("Inventário não corresponde à versão de imagens aprovadas do registro.")
    missing_classes = [name for index, name in enumerate(MVP_NAMES) if not train_counts[index]]
    if missing_classes:
        raise ValueError("Treino sem positivos aprovados para todas as classes ativas: " + ", ".join(missing_classes) + ".")
    # Os três splits com pelo menos uma edificação e ao menos três edificações no total.
    if not all(split_groups.values()) or len(set.union(*split_groups.values())) < 3:
        raise ValueError("São necessários três splits e ao menos três grupos de edificações independentes.")
    # 5. Relatório gravado pelo treino em runs/<treino>/training_provenance.json.
    return {
        "schema_version": 1, "ready_for_training": True, "dataset_root": str(root),
        "taxonomy_sha256": taxonomy_digest, "provenance_sha256": file_hash(provenance_path),
        "registry_sha256": file_hash(registry_path), "assignments_sha256": file_hash(assignments_path),
        "dataset_yaml_sha256": file_hash(data_path), "groups_sha256": file_hash(root / "groups.csv"),
        "counts": validation["counts"], "train_boxes_by_class": {str(index): train_counts[index] for index in range(len(MVP_NAMES))},
        "groups_by_split": {split: sorted(groups) for split, groups in split_groups.items()},
        "note": "Prontidão recalculada com revisão humana e integridade; nenhum modelo foi carregado.",
    }
