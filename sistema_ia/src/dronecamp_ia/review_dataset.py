"""Dataset de PRODUÇÃO a partir da revisão humana (não treina modelos).

Função no projeto: equivalente de produção do ``pilot.py``. Gera
``data/versions/<versão>/`` com fotos, labels, ``groups.csv`` e
``provenance.json``, usando a divisão por edificação de ``configs/review_groups.json``.
É o comando ``build-reviewed-data``.

O que faz:
- ``_approved_images``: fotos com aprovação humana e grupo de edificação conhecido.
- ``_write_samples``: copia fotos e escreve labels por split.
- ``_readiness``: diz se o dataset já pode treinar (exige três edificações e
  exemplos das classes); hoje, com só o CEASA, a resposta é "não".
- ``build_approved_dataset``: executa tudo e devolve o relatório.

Mais imagens só ajudam quando seus rótulos são confiáveis. Este módulo mantém
aprovação, origem e grupos de edificações separados das propostas feitas por IA.

Quando mexer: regras de prontidão para produção ou a lista ``MVP_NAMES``
(sempre acrescentando classes ao fim, junto com a taxonomia).

Quem chama:
- ``cli.py``, comando ``build-reviewed-data`` (sai com código 2 quando o dataset não
  está pronto para treinar).
- ``review_provenance.py`` importa ``MVP_NAMES`` e confere, antes de treinar, os
  arquivos gravados aqui.

O que lê:
- ``registry.json`` de uma revisão (formato descrito no cabeçalho de ``review_data.py``;
  aqui importam ``image_sha256``, ``status``, ``human_approved``, ``technical_status``,
  ``human_reviewer``, ``human_review_at``, ``building_group``, ``source_path``,
  ``width``, ``height``, ``boxes`` e ``visual_review_status``).
- O arquivo de grupos (``configs/review_groups.json``):
  ``{"schema_version": 1, "groups": {"<building_group>": "train" | "val" | "test"}}``.
- ``configs/taxonomy.json`` e as fotos originais.

O que grava (pasta nova ``output_dir``; nunca sobrescreve):
- ``images/<split>/<hash>.<ext>`` e ``labels/<split>/<hash>.txt`` (YOLO);
- ``groups.csv`` (colunas ``image``, ``group_id``, ``split``);
- ``dataset.yaml`` (``path``, splits presentes, ``nc``, ``names``);
- ``provenance.json`` (hashes de entrada e uma entrada por amostra; lido por
  ``review_provenance``);
- ``readiness.json`` (o relatório devolvido por ``build_approved_dataset``).

Onde mexer:
- Nova classe ativa: ``MVP_NAMES`` (ao fim) + ``configs/taxonomy.json`` + ``COLORS`` em
  review_render.py.
- Nova regra de prontidão: ``_readiness`` (lista ``reasons``).
- Novo campo por amostra em ``provenance.json``: ``_write_samples`` e, se ele precisar
  ser conferido antes do treino, ``review_provenance._check_sample``.
- Nova coluna em ``groups.csv``: ``fieldnames`` e ``writer.writerow`` em ``_write_samples``
  (e a leitura em ``review_provenance._group_rows`` / ``dataset.validate_dataset``).
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import yaml

from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS, SPLITS, validate_dataset
from .io import file_hash, write_json
from .review_data import boxes_to_yolo, review_image_size, validate_boxes

# ---------------------------------------------------------------------------
# Classes do MVP (contrato com a taxonomia, os pesos e as labels)
# ---------------------------------------------------------------------------

# As 14 classes ativas, na ordem dos IDs (precisa coincidir com configs/taxonomy.json).
# ATENCAO: a posição na tupla É o ID da classe (0, 1, 2...). build_approved_dataset e
# review_provenance recusam a taxonomia se os slugs/IDs ativos não forem exatamente estes.
# EXTENSAO: nova classe ativa: acrescente o slug ao FIM (nunca no meio nem reordene), na
# mesma posição em que ela aparece com phase 1 em configs/taxonomy.json, e acrescente a
# cor em review_render.COLORS. Depois migre as revisões (migrate-review).
MVP_NAMES = (
    "telha_quebrada", "telha_ausente", "residuos_telha", "reparo_telha",
    "rufo_ausente", "rufo_quebrado", "residuos_calha", "vegetacao_calha",
    "pedaco_telha",
    "rufo_deslocado",
    # Acrescente classes ao fim: os rótulos já revisados mantêm seus IDs.
    "pedaco_telha_sobreposto",
    "fixador_telha_frouxo",
    "reparo_rufo",
    "pedaco_calha_sobre_telha",
)


# ---------------------------------------------------------------------------
# Leitura e seleção das fotos aprovadas
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    """Falhe antes de criar a versão se o contrato não puder ser lido.

    Parâmetros:
        path (Path): JSON a ler (registro ou arquivo de grupos).

    Retorna:
        dict: o conteúdo, já conferido como dict com ``schema_version == 1``.

    Erros:
        ValueError("Schema inválido: <nome>...") se não for dict ou o schema diferir.
        Erros de leitura/JSON (OSError, JSONDecodeError) sobem sem tratamento.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Schema inválido: {path.name}; esperado schema_version: 1.")
    return value


def _approved_images(registry: dict, groups: dict) -> tuple[list[dict], list[dict]]:
    """Selecione decisões explícitas; uma segunda revisão de IA não aprova nada.

    Parâmetros:
        registry (dict): ``registry.json`` carregado.
        groups (dict): ``{building_group: split}`` do arquivo de grupos.

    Retorna:
        tuple: ``(approved, excluded)``.
        ``approved``: fotos do registro aprovadas, cada uma com ``source_path`` resolvido
        e a chave extra ``split``. ``excluded``: ``[{image_sha256, status, reason}]`` das
        fotos sem aprovação humana (ficam registradas em provenance.json).

    Erros (ValueError): hash inválido ou duplicado; aprovação sem autor/estado humano;
    grupo ausente ou sem split no arquivo de grupos; ``source_path`` não local, não
    absoluto, inexistente ou com extensão não suportada; hash/dimensões divergentes;
    caixas inválidas; status incompatível; nenhuma foto aprovada.

    ATENCAO: o ``source_path`` precisa existir EXATAMENTE como gravado (caminho absoluto
    deste PC). Diferente de render-review/import-review, aqui não há reancoragem por
    ``io.resolve_local_path``; registros copiados de outro PC falham com
    "source_path deve ser um arquivo de imagem absoluto e existente".
    """
    approved, excluded = [], []
    seen = set()
    if not isinstance(registry.get("images"), list):
        raise ValueError("O registro deve conter images como lista.")
    for item in registry["images"]:
        # 1. Identidade: SHA-256 em 64 caracteres hexadecimais minúsculos, sem repetição.
        digest = item.get("image_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Hash de imagem inválido no registro.")
        if digest in seen:
            raise ValueError("Imagem duplicada no registro de revisão.")
        seen.add(digest)
        # 2. Sem aprovação humana positiva/negativa: fica de fora (não é erro).
        # ATENCAO: este é o filtro que impede proposta de IA de entrar no treino.
        if item.get("human_approved") is not True or item.get("status") not in {"positive", "negative"}:
            excluded.append({"image_sha256": digest, "status": item.get("status"),
                             "reason": "Sem aprovação humana explícita ou decisão ainda ambígua/excluída."})
            continue
        # 3. Aprovação coerente: estado técnico humano e autor preenchido.
        reviewer = item.get("human_reviewer")
        if item.get("technical_status") != "human_visual_reviewed" or not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("Aprovação precisa de autor e estado de revisão humana explícitos.")
        # 4. Edificação com split definido no arquivo de grupos.
        group = item.get("building_group")
        if not isinstance(group, str) or not group.strip() or group not in groups:
            raise ValueError("Cada imagem aprovada precisa de building_group e assignment explícito.")
        # 5. Original local (sem URL), absoluto, existente e com extensão de imagem suportada.
        source_value = item.get("source_path")
        if not isinstance(source_value, str) or "://" in source_value:
            raise ValueError("source_path deve indicar um arquivo local conhecido.")
        source = Path(source_value)
        if not source.is_absolute() or not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("source_path deve ser um arquivo de imagem absoluto e existente.")
        # 6. Bytes e tamanho do original iguais aos revisados.
        if file_hash(source) != digest:
            raise ValueError(f"Hash do original alterado: {source.name}.")
        size = review_image_size(source)
        if any(type(item.get(key)) is not int for key in ("width", "height")) or size != (item["width"], item["height"]):
            raise ValueError(f"Dimensões do original divergem do registro: {source.name}.")
        # 7. Caixas válidas na taxonomia do MVP e coerentes com o status.
        boxes = item.get("boxes")
        validate_boxes(boxes, *size, len(MVP_NAMES))
        if (item["status"] == "positive" and not boxes) or (item["status"] == "negative" and boxes):
            raise ValueError("Decisão positiva/negativa incompatível com as caixas.")
        # 8. Aprovada: caminho absoluto canônico + split da edificação.
        approved.append({**item, "source_path": str(source.resolve()), "split": groups[group]})
    if not approved:
        raise ValueError("Nenhuma imagem tem aprovação humana explícita; nenhum dataset foi criado.")
    return approved, excluded


# ---------------------------------------------------------------------------
# Gravação das amostras e prontidão
# ---------------------------------------------------------------------------

def _write_samples(directory: Path, images: list[dict]) -> list[dict]:
    """Copie bytes; nomes por hash evitam colisões entre campanhas diferentes.

    Parâmetros:
        directory (Path): pasta do dataset (já criada, vazia).
        images (list[dict]): fotos aprovadas de ``_approved_images`` (com ``split``).

    Retorna:
        list[dict]: uma entrada por amostra para ``provenance.json``: ``image_sha256``,
        ``image``/``label`` (caminhos relativos à pasta), ``label_sha256``,
        ``source_path``, ``building_group``, ``split``, ``width``, ``height``, ``status``,
        ``human_approved`` (True), ``technical_status``, ``approval_author``,
        ``approved_at`` e ``visual_review_status``.

    Erros:
        ValueError se a cópia não tiver o mesmo hash do original (a origem mudou).

    Grava: ``images/<split>/<hash>.<ext>``, ``labels/<split>/<hash>.txt`` e ``groups.csv``.
    """
    provenance_images = []
    # groups.csv é escrito junto com as cópias: uma linha por foto.
    with (directory / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
        # EXTENSAO: nova coluna em groups.csv: acrescente em fieldnames e em writer.writerow
        # abaixo (e confira a leitura em review_provenance._group_rows e dataset.validate_dataset).
        writer = csv.DictWriter(stream, fieldnames=["image", "group_id", "split"])
        writer.writeheader()
        for item in images:
            # 1. Pastas do split e nome do arquivo = hash + extensão original em minúsculas.
            split, digest = item["split"], item["image_sha256"]
            source = Path(item["source_path"])
            filename = digest + source.suffix.lower()
            image_dir, label_dir = directory / "images" / split, directory / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            # 2. Cópia dos bytes e reconferência do hash.
            copied = image_dir / filename
            shutil.copyfile(source, copied)
            # Reconfira os bytes copiados: a origem pode mudar após o preflight.
            if file_hash(copied) != digest:
                raise ValueError("Original mudou durante a cópia; versão incompleta não deve ser utilizada.")
            # 3. Label YOLO (vazia para negativas) com o mesmo nome-base da foto.
            label = label_dir / (digest + ".txt")
            label.write_text(boxes_to_yolo(item["boxes"], item["width"], item["height"]), encoding="utf-8")
            # 4. Linha do groups.csv e entrada da proveniência (caminhos sempre com "/").
            relative = copied.relative_to(directory).as_posix()
            writer.writerow({"image": relative, "group_id": item["building_group"], "split": split})
            # EXTENSAO: novo campo por amostra em provenance.json: acrescente neste dict; se
            # ele tiver de ser conferido antes do treino, inclua a checagem em
            # review_provenance._check_sample.
            provenance_images.append({
                "image_sha256": digest, "image": relative,
                "label": label.relative_to(directory).as_posix(), "label_sha256": file_hash(label),
                "source_path": item["source_path"], "building_group": item["building_group"],
                "split": split, "width": item["width"], "height": item["height"],
                "status": item["status"], "human_approved": True,
                "technical_status": item.get("technical_status"),
                "approval_author": item.get("human_reviewer"), "approved_at": item.get("human_review_at"),
                "visual_review_status": item.get("visual_review_status"),
            })
    return provenance_images


def _readiness(directory: Path, images: list[dict]) -> dict:
    """Prontidão é estrutura, cobertura de classes e separação por edificação.

    Parâmetros:
        directory (Path): pasta do dataset já gravado (com ``dataset.yaml``).
        images (list[dict]): fotos aprovadas (com ``split``, ``building_group``, ``boxes``).

    Retorna:
        dict (gravado em ``readiness.json``): ``ready_for_training`` (bool), ``state``
        (``"ready"`` ou ``"draft"``), ``reasons`` (lista de motivos em português),
        ``validation`` (relatório de ``dataset.validate_dataset``), ``groups_by_split``,
        ``counts``, ``train_boxes_by_class`` ({"0": n, ...}) e caminhos.

    EXTENSAO: nova regra de prontidão: acrescente um ``reasons.append(...)`` depois das
    regras existentes; qualquer motivo deixa o dataset em "draft". A guarda equivalente
    antes do treino fica em review_provenance.validate_training_provenance (etapa 4).
    """
    # 1. Estrutura do dataset (pastas, labels, groups.csv) pelo validador comum.
    validation = validate_dataset(directory / "dataset.yaml", list(MVP_NAMES))
    # 2. Edificações por split e caixas por classe no treino.
    groups = {split: sorted({item["building_group"] for item in images if item["split"] == split}) for split in SPLITS}
    train_counts = Counter(box["class_id"] for item in images if item["split"] == "train" for box in item["boxes"])
    missing_classes = [index for index in range(len(MVP_NAMES)) if not train_counts[index]]
    reasons = list(validation["errors"])
    # 3. Regras: toda classe com positivo no treino; os três splits preenchidos e
    #    pelo menos três edificações distintas no total.
    if missing_classes:
        reasons.append("Faltam positivos no treino para as classes: " + ", ".join(MVP_NAMES[index] for index in missing_classes) + ".")
    if not all(groups.values()) or len({group for values in groups.values() for group in values}) < 3:
        reasons.append("Treino, validação e teste exigem ao menos três grupos de edificações independentes; um único CEASA não pode ocupar os três splits.")
    return {
        "schema_version": 1, "ready_for_training": validation["valid"] and not reasons,
        "state": "ready" if validation["valid"] and not reasons else "draft",
        "dataset_path": str(directory / "dataset.yaml"), "provenance_path": str(directory / "provenance.json"),
        "validation": validation, "reasons": reasons, "groups_by_split": groups,
        "counts": validation["counts"], "train_boxes_by_class": {str(index): train_counts[index] for index in range(len(MVP_NAMES))},
        "note": "A checagem não prova diversidade real ou qualidade técnica dos rótulos; não iniciou treinamento.",
    }


# ---------------------------------------------------------------------------
# Fluxo principal (build-reviewed-data)
# ---------------------------------------------------------------------------

# ENTRADA: comando build-reviewed-data (cli.py). ATENCAO: a pasta de saída nunca pode existir;
# datasets são imutáveis (uma correção gera uma versão nova em data/versions/).
def build_approved_dataset(config: ProjectConfig, registry_path: Path, assignments_path: Path, output_dir: Path) -> dict:
    """Exporte uma nova versão, recusando sobrescrita e imagens sem aprovação.

    ``assignments`` usa {schema_version: 1, groups: {building_group: split}}.
    Splits sem amostras ficam ausentes: um rascunho nunca simula diversidade.

    Parâmetros:
        config (ProjectConfig): configuração (taxonomia).
        registry_path (Path): ``registry.json`` com decisões humanas importadas.
        assignments_path (Path): arquivo de grupos (em geral ``configs/review_groups.json``).
        output_dir (Path): pasta NOVA do dataset (ex.: ``data/versions/producao_v1``).

    Retorna:
        dict: o relatório de prontidão (``_readiness``), também gravado em ``readiness.json``.

    Erros (ValueError): saída já existe; schema inválido; taxonomia ativa diferente de
    ``MVP_NAMES``; registro de outra taxonomia; arquivo de grupos inválido; e os erros de
    ``_approved_images`` e ``_write_samples``.
    """
    # 1. Saída nova, taxonomia do MVP e registro na mesma taxonomia.
    registry_path, assignments_path, directory = Path(registry_path), Path(assignments_path), Path(output_dir)
    # is_symlink(): um link quebrado não "existe", mas também não pode ser usado como saída.
    if directory.exists() or directory.is_symlink():
        raise ValueError("Saída já existe; datasets são imutáveis. Escolha uma nova versão.")
    registry, assignments = _read_json(registry_path), _read_json(assignments_path)
    taxonomy = load_taxonomy(config.taxonomy_path)
    # ATENCAO: nomes e IDs ativos da taxonomia precisam ser exatamente MVP_NAMES (0..13).
    if tuple(detection_names(taxonomy)) != MVP_NAMES or [item["id"] for item in taxonomy["classes"] if item["phase"] == 1] != list(range(len(MVP_NAMES))):
        raise ValueError("A taxonomia ativa deve manter os IDs e nomes do MVP.")
    if registry.get("taxonomy_sha256") != file_hash(config.taxonomy_path):
        raise ValueError("Taxonomia mudou; migre e revise o registro antes de construir dados.")
    # 2. Cada edificação (building_group) vai inteira para um split, conforme o arquivo de grupos.
    #    Isso impede vazamento: fotos da mesma edificação nunca ficam em treino e teste ao mesmo tempo.
    groups = assignments.get("groups")
    if not isinstance(groups, dict) or any(not isinstance(group, str) or not group.strip() or split not in SPLITS for group, split in groups.items()):
        raise ValueError("Assignments deve mapear grupos para train, val ou test.")
    images, excluded = _approved_images(registry, groups)
    # Hashes das entradas, gravados na proveniência e reconferidos antes do treino.
    input_hashes = {"registry_sha256": file_hash(registry_path), "assignments_sha256": file_hash(assignments_path),
                    "taxonomy_sha256": file_hash(config.taxonomy_path)}
    directory = directory.resolve()
    # Toda aprovação, geometria e origem foi conferida antes da primeira escrita.
    # exist_ok=False: se outra execução criou a pasta nesse meio-tempo, falha aqui.
    directory.mkdir(parents=True, exist_ok=False)
    # 3. Fotos, labels, dataset.yaml e provenance.json; por fim, a prontidão.
    provenance_images = _write_samples(directory, images)
    present_splits = {item["split"] for item in images}
    # dataset.yaml no formato Ultralytics: só os splits que têm fotos.
    dataset_yaml = {"path": str(directory), **{split: f"images/{split}" for split in SPLITS if split in present_splits},
                    "nc": len(MVP_NAMES), "names": list(MVP_NAMES)}
    (directory / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # provenance.json: entradas (hashes e caminhos absolutos), amostras, excluídas e política.
    # ATENCAO: review_provenance lê registry_path/assignments_path e os *_sha256 daqui.
    write_json(directory / "provenance.json", {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "version": directory.name, **input_hashes, "registry_path": str(registry_path.resolve()),
        "assignments_path": str(assignments_path.resolve()), "images": provenance_images, "excluded_images": excluded,
        "human_feedback": registry.get("human_feedback"), "policy": "Somente revisão humana explícita; arquivos anteriores preservados."})
    readiness = _readiness(directory, images)
    write_json(directory / "readiness.json", readiness)
    return readiness
