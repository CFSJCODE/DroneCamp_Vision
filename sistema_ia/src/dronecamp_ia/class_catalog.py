"""Cadastro das classes pela plataforma: nomes de exibição, cores, ativo/inativo e classes propostas.

Função no projeto: rotas ``/api/classes`` do comando ``platform`` (seção Classes).
As edições ficam em ``configs/taxonomy_overrides.json``, ao lado da taxonomia, e
**nunca reescrevem** ``taxonomy.json``. Motivo: revisões, datasets piloto,
calibradores e laudos guardam o SHA-256 de ``taxonomy.json``; trocar a cor de
uma classe dentro dele invalidaria todos esses arquivos de uma vez.

O que cada operação faz:
- ``list_classes``: taxonomia + edições + classes propostas, com o que cada uma
  exige (``in_detector``: classe ativa que os pesos atuais já aprenderam;
  ``requires_training``: só passa a ser detectada depois de dados aprovados e
  novo treino, mesmo quando já é ativa na taxonomia, como o ID 13 na v6).
- ``create_class``: classe proposta com ID depois do maior da taxonomia, sempre
  ``requires_training``. Ela não entra no detector nem nas caixas da revisão:
  isso exige uma nova versão da taxonomia (como v5 → v6) e ``migrate-review``.
- ``update_class``: nome de exibição, cor e ativo/inativo de qualquer classe;
  nome técnico (slug) e critério só das propostas. O slug de uma classe da
  taxonomia é o nome que os pesos treinados conferem (``check_domain_names``).
- ``deactivate_class``: exclusão lógica (``active: false``); nada é apagado e o
  histórico registra cada mudança.

Inativa não muda o modelo nem as caixas já feitas: a página só deixa de
oferecer a classe para caixas novas.

As edições das classes da taxonomia ficam guardadas **pelo slug**, não pelo ID:
uma migração desloca IDs do catálogo (na v6 o antigo ID 13 virou 23) e o slug é
o que não muda. As propostas são conferidas contra a taxonomia a cada leitura
(``_reconcile``): proposta cujo slug entrou na taxonomia é dada como absorvida e
sai da lista; proposta cujo ID uma expansão da taxonomia ocupou recebe o
próximo ID livre. Tudo fica no ``history``.

Quando mexer: para levar uma proposta à detecção, crie a nova versão da
taxonomia com snapshot e migração (ver ``docs/analise_taxonomia_classes.md``);
não edite IDs existentes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import threading

from .config import ProjectConfig, load_taxonomy
from .io import file_hash

OVERRIDES_FILE = "taxonomy_overrides.json"
TRAINING_NOTICE = "Mudanças nas classes exigem novo ciclo de treinamento para ter efeito nos modelos."
# Português em snake_case, como os slugs da taxonomia (ex.: pedaco_calha_sobre_telha).
SLUG = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
MAX_NAME = 80
MAX_CRITERION = 1000
MAX_HISTORY = 500
_LOCK = threading.RLock()


class ClassNotFound(LookupError):
    """ID que não existe na taxonomia nem entre as propostas."""


def overrides_path(config: ProjectConfig) -> Path:
    return config.taxonomy_path.with_name(OVERRIDES_FILE)


def _empty() -> dict:
    return {"schema_version": 1, "note": "Edições feitas pela plataforma; taxonomy.json não é alterado. Chaves de classes = slug.",
            "classes": {}, "proposed": [], "history": []}


def _reconcile(taxonomy: dict, overrides: dict) -> bool:
    """Confere as propostas contra a taxonomia atual; devolve True se algo mudou (e precisa gravar)."""
    slugs = {item["slug"]: item["id"] for item in taxonomy["classes"]}
    ids = {item["id"] for item in taxonomy["classes"]}
    changed = False
    kept = []
    for item in overrides["proposed"]:
        if item["slug"] in slugs:
            # Entrou na taxonomia (promovida numa nova versão): a proposta cumpriu o papel.
            overrides["history"].append({"at": _now(), "action": "absorbed", "id": item["id"], "by": None,
                                         "changes": {"name": item["slug"], "taxonomy_id": slugs[item["slug"]]}})
            changed = True
            continue
        if item["id"] in ids:
            # A taxonomia cresceu até este ID com outra classe: a proposta recebe o próximo livre.
            new_id = max(ids | {value["id"] for value in kept}) + 1
            overrides["history"].append({"at": _now(), "action": "renumbered", "id": new_id, "by": None,
                                         "changes": {"name": item["slug"], "previous_id": item["id"]}})
            item["id"] = new_id
            changed = True
        kept.append(item)
    overrides["proposed"] = kept
    return changed


def load_overrides(config: ProjectConfig, taxonomy: dict | None = None) -> dict:
    """Lê as edições e as concilia com a taxonomia; grava se a conciliação mudou algo."""
    path = overrides_path(config)
    if not path.is_file():
        return _empty()
    with _LOCK:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("classes"), dict)
                or not isinstance(value.get("proposed"), list)):
            raise ValueError(f"{path.name} fora do formato esperado (schema_version 1).")
        value.setdefault("history", [])
        if _reconcile(taxonomy or load_taxonomy(config.taxonomy_path), value):
            _save(config, value)
    return value


# ATENCAO: a seção Classes grava SÓ em configs/taxonomy_overrides.json. Nunca escreva em
# taxonomy.json daqui: o hash dele está em revisões, datasets e laudos.
def _save(config: ProjectConfig, value: dict) -> None:
    """Grava em arquivo temporário e troca de uma vez: uma queda no meio não deixa JSON pela metade."""
    path = overrides_path(config)
    value["history"] = value["history"][-MAX_HISTORY:]
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def default_color(class_id: int) -> str:
    from .review_render import COLORS
    return COLORS[class_id % len(COLORS)]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text(body: dict, key: str, limit: int, required: bool = False) -> str | None:
    value = body.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Informe {key}.")
    value = " ".join(value.split())
    if len(value) > limit:
        raise ValueError(f"{key} passa de {limit} caracteres.")
    return value


def _color(body: dict) -> str | None:
    value = body.get("color")
    if value is None:
        return None
    if not isinstance(value, str) or not COLOR.match(value):
        raise ValueError("Cor no formato #RRGGBB.")
    return value.upper()


def _slug(body: dict) -> str:
    value = body.get("name")
    if not isinstance(value, str) or not SLUG.match(value) or not 3 <= len(value) <= 60:
        raise ValueError("Nome técnico em português snake_case: letras minúsculas sem acento, números e _ (3 a 60 caracteres).")
    return value


def _entry(item: dict, edits: dict, proposed: bool) -> dict:
    class_id = item["id"]
    phase = item.get("phase")
    # Ativa na taxonomia (entra em detection_names) não é o mesmo que já aprendida pelos pesos:
    # o ID 13 da v6 é ativo e ainda exige treino (os pesos piloto têm 13 classes).
    requires_training = True if proposed else bool(item.get("requires_training", phase != 1))
    return {"id": class_id, "name": item["slug"],
            "display_name": edits.get("display_name") or item["label"], "original_display_name": item["label"],
            "color": edits.get("color") or item.get("color") or default_color(class_id),
            "active": edits.get("active", item.get("active", True)) is not False,
            "phase": "proposta" if proposed else phase, "source": "proposta" if proposed else "taxonomia",
            "in_detector": not proposed and phase == 1 and not requires_training,
            "requires_training": requires_training,
            "criterion": item.get("criterion"), "name_editable": proposed,
            "created_at": item.get("created_at"), "updated_at": edits.get("updated_at") or item.get("updated_at")}


def list_classes(config: ProjectConfig) -> dict:
    taxonomy = load_taxonomy(config.taxonomy_path)
    overrides = load_overrides(config, taxonomy)
    edits = overrides["classes"]  # chave = slug: sobrevive ao deslocamento de IDs nas migrações
    classes = [_entry(item, edits.get(item["slug"], {}), False) for item in taxonomy["classes"]]
    classes += [_entry(item, {}, True) for item in overrides["proposed"]]
    return {"taxonomy_version": taxonomy.get("version"), "taxonomy_sha256": file_hash(config.taxonomy_path),
            "overrides_file": overrides_path(config).relative_to(config.root).as_posix() if overrides_path(config).is_relative_to(config.root) else OVERRIDES_FILE,
            "notice": TRAINING_NOTICE, "classes": classes,
            "summary": {"total": len(classes), "in_detector": sum(item["in_detector"] for item in classes),
                        "inactive": sum(not item["active"] for item in classes),
                        "requires_training": sum(item["requires_training"] for item in classes)}}


def _find(config: ProjectConfig, overrides: dict, class_id: int) -> tuple[dict, bool]:
    taxonomy = load_taxonomy(config.taxonomy_path)
    for item in taxonomy["classes"]:
        if item["id"] == class_id:
            return item, False
    for item in overrides["proposed"]:
        if item["id"] == class_id:
            return item, True
    raise ClassNotFound(f"Classe {class_id} não existe.")


def _get(config: ProjectConfig, class_id: int) -> dict:
    return next(item for item in list_classes(config)["classes"] if item["id"] == class_id)


def create_class(config: ProjectConfig, body: dict, author: str | None = None) -> dict:
    slug = _slug(body)
    display = _text(body, "display_name", MAX_NAME, required=True)
    color = _color(body)
    criterion = _text(body, "criterion", MAX_CRITERION)
    with _LOCK:
        taxonomy = load_taxonomy(config.taxonomy_path)
        overrides = load_overrides(config, taxonomy)
        existing = taxonomy["classes"] + overrides["proposed"]
        if any(item["slug"] == slug for item in existing):
            raise ValueError(f"Já existe uma classe com o nome técnico {slug}.")
        class_id = max(item["id"] for item in existing) + 1
        item = {"id": class_id, "slug": slug, "label": display, "color": color or default_color(class_id), "active": True,
                "requires_training": True, "criterion": criterion, "created_at": _now(), "created_by": author}
        overrides["proposed"].append(item)
        overrides["history"].append({"at": item["created_at"], "action": "create", "id": class_id, "by": author,
                                     "changes": {"name": slug, "display_name": display}})
        _save(config, overrides)
    return _get(config, class_id)


def update_class(config: ProjectConfig, class_id: int, body: dict, author: str | None = None) -> dict:
    allowed = {"display_name", "color", "active", "name", "criterion"}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(f"Campos não editáveis: {', '.join(sorted(unknown))}.")
    if not body:
        raise ValueError("Nada para alterar.")
    if "active" in body and not isinstance(body["active"], bool):
        raise ValueError("active deve ser true ou false.")
    changes = {}
    if "display_name" in body:
        changes["display_name"] = _text(body, "display_name", MAX_NAME, required=True)
    if "color" in body:
        changes["color"] = _color({"color": body["color"]}) if body["color"] is not None else None
    if "active" in body:
        changes["active"] = body["active"]
    with _LOCK:
        overrides = load_overrides(config)
        item, proposed = _find(config, overrides, class_id)
        if ("name" in body or "criterion" in body) and not proposed:
            raise ValueError("Nome técnico e critério das classes da taxonomia só mudam numa nova versão da taxonomia "
                             "(os pesos treinados conferem o nome). Altere o nome de exibição.")
        if "name" in body:
            slug = _slug(body)
            taken = load_taxonomy(config.taxonomy_path)["classes"] + [value for value in overrides["proposed"] if value is not item]
            if any(value["slug"] == slug for value in taken):
                raise ValueError(f"Já existe uma classe com o nome técnico {slug}.")
            changes["name"] = slug
        if "criterion" in body:
            changes["criterion"] = _text(body, "criterion", MAX_CRITERION)
        stamp = _now()
        if proposed:
            mapping = {"display_name": "label", "name": "slug"}
            for key, value in changes.items():
                item[mapping.get(key, key)] = value if not (key == "color" and value is None) else default_color(class_id)
            item["updated_at"] = stamp
        else:
            edits = overrides["classes"].setdefault(item["slug"], {})
            for key, value in changes.items():
                if value is None or (key == "display_name" and value == item["label"]):
                    edits.pop(key, None)  # cor vazia ou nome igual ao original = volta ao padrão da taxonomia
                else:
                    edits[key] = value
            edits["updated_at"] = stamp
        overrides["history"].append({"at": stamp, "action": "update", "id": class_id, "by": author, "changes": changes})
        _save(config, overrides)
    return _get(config, class_id)


def deactivate_class(config: ProjectConfig, class_id: int, author: str | None = None) -> dict:
    """Exclusão lógica: a classe some das opções de caixa nova; nada é apagado."""
    return update_class(config, class_id, {"active": False}, author)


def apply_to_page_classes(config: ProjectConfig, classes: list[dict]) -> list[dict]:
    """Nome de exibição, cor e ativo/inativo nas classes embutidas na página servida.

    Só as classes ativas da taxonomia entram (as caixas da revisão usam esses
    IDs); um arquivo de edições inválido deixa a página com os dados originais.
    """
    try:
        current = {item["id"]: item for item in list_classes(config)["classes"] if item["source"] == "taxonomia" and item["phase"] == 1}
    except (OSError, ValueError, KeyError):
        return classes
    result = []
    for category in classes:
        entry = current.get(category.get("id"))
        if entry is None or entry["name"] != (category.get("slug") or entry["name"]):
            result.append(category)
            continue
        result.append({**category, "label": entry["display_name"], "color": entry["color"], "active": entry["active"],
                       "requires_training": entry["requires_training"]})
    return result
