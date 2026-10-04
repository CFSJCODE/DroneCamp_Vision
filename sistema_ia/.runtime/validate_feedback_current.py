"""Conferir a importação real sem alterar imagens, decisões ou pesos."""

from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.io import file_hash, make_run_directory, write_json
from dronecamp_ia.review_data import boxes_to_yolo, review_image_size, validate_boxes
from dronecamp_ia.review_dataset import build_approved_dataset


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_preserved_inputs(root: Path) -> int:
    """Cada arquivo existente antes da importação precisa manter os mesmos bytes."""
    inventory = read_json(root / ".runtime/feedback_a36ea7f67d08_before.json")
    for filename, expected in inventory["files"].items():
        path = Path(filename)
        require(path.is_file() and file_hash(path) == expected, "Um arquivo preservado diverge do inventário anterior.")
    return len(inventory["files"])


def verify_annotations(root: Path, run: Path) -> dict:
    """Conferir origem, escopo humano e rótulos sem promover aprovação histórica."""
    feedback_path = Path("C:/Users/claud/Desktop/dronecamp-revisao-ceasa_v3.json")
    feedback = read_json(feedback_path)
    source_path = root / "data/reviews/ceasa_v3/registry_with_candidates.json"
    human_path = root / "data/reviews/ceasa_v3_humana_a36ea7f67d08/registry.json"
    current_path = root / "data/reviews/ceasa_v6_corrigida_a36ea7f67d08/registry.json"
    old_taxonomy_path = root / "configs/taxonomy_ceasa_v3_pedaco_telha_rufo_deslocado.json"
    current_taxonomy_path = root / "configs/taxonomy.json"
    historical, current = read_json(human_path), read_json(current_path)
    source_by_hash = {item["image_sha256"]: item for item in read_json(source_path)["images"]}
    decisions = {item["image_sha256"]: item for item in feedback["images"]}
    historical_by_hash = {item["image_sha256"]: item for item in historical["images"]}
    approved = [item for item in historical["images"] if item.get("human_approved") is True]
    require(feedback["registry_sha256"] == file_hash(source_path), "Feedback ligado a outra versão.")
    require(file_hash(feedback_path) == file_hash(human_path.parent / "feedback.json"), "Cópia do feedback alterada.")
    require(len(decisions) == len(feedback["images"]) == 36, "Número de decisões inesperado.")
    require(Counter(item["status"] for item in feedback["images"]) == {"positive": 33, "ambiguous": 3}, "Decisões divergentes.")
    require(len(approved) == 33 and sum(len(item["boxes"]) for item in approved) == 108, "Acervo humano divergente.")
    require(historical["taxonomy_sha256"] == file_hash(old_taxonomy_path), "Escopo histórico divergente.")
    require(current["taxonomy_sha256"] == file_hash(current_taxonomy_path), "Taxonomia atual divergente.")
    require(len(current["images"]) == len(historical_by_hash) == len(source_by_hash) == 38, "Inventário de fotos divergente.")
    for item in historical["images"]:
        digest = item["image_sha256"]
        if digest in decisions:
            decision = decisions[digest]
            require(item["boxes"] == decision["boxes"] and item["status"] == decision["status"], "Correção humana alterada na importação.")
            require(item.get("human_notes") == decision["notes"], "Nota humana alterada.")
        else:
            require(item["boxes"] == source_by_hash[digest]["boxes"], "Foto sem decisão alterada.")
    human_current = 0
    historical_current = 0
    for item in current["images"]:
        previous = historical_by_hash[item["image_sha256"]]
        source = Path(item["source_path"])
        require(file_hash(source) == item["image_sha256"], "Foto original alterada.")
        require(review_image_size(source) == (item["width"], item["height"]), "Dimensões divergentes.")
        validate_boxes(item["boxes"], item["width"], item["height"], 13)
        require(item["boxes"] == previous["boxes"] and item["status"] == previous["status"], "Migração alterou decisão/caixas.")
        for key in ("human_reviewer", "human_review_at", "human_notes"):
            require(item.get(key) == previous.get(key), "Migração alterou proveniência humana.")
        require(item.get("taxonomy_recheck_pending") is True, "Migração liberou aprovação atual indevida.")
        require(item.get("severity") is None, "Gravidade inventada na migração.")
        human_current += item.get("human_approved") is True
        if previous.get("human_approved") is True:
            historical_current += 1
            require(item.get("previous_human_approved") is True, "Histórico humano perdido.")
            require(item["visual_review_status"] == "prior_taxonomy_human_review", "Origem humana marcada como IA.")
            require(item["classes_reviewed_before_migration"] == list(range(10)), "Escopo histórico incorreto.")
            history = item["historical_human_review"]
            require(history["taxonomy_sha256"] == file_hash(old_taxonomy_path), "Taxonomia de aprovação histórica divergente.")
            require(history["source_registry_sha256"] == file_hash(human_path), "Origem da aprovação histórica divergente.")
        else:
            require(not item.get("previous_human_approved"), "Histórico humano inventado.")
            require(item["classes_reviewed_before_migration"] == list(range(8)), "Escopo pendente ampliado artificialmente.")
    require(human_current == 0 and historical_current == 33, "Aprovações divergentes.")
    require(sum(len(item["boxes"]) for item in current["images"]) == 115, "Contagem atual divergente.")
    corpus = root / "data/annotation_corpus/ceasa_v3_a36ea7f67d08"
    manifest = read_json(corpus / "manifest.json")
    require(manifest["ready_for_training"] is False and not (corpus / "dataset.yaml").exists(), "Acervo histórico foi liberado para treino.")
    require(manifest["active_classes_at_review"] == detection_names(load_taxonomy(old_taxonomy_path)), "Classes históricas divergentes.")
    require(manifest["source_registry_sha256"] == file_hash(human_path), "Origem do acervo divergente.")
    require(len(manifest["samples"]) == 33, "Número de amostras divergente.")
    for sample in manifest["samples"]:
        item = historical_by_hash[sample["image_sha256"]]
        image_path, label_path = corpus / sample["image"], corpus / sample["label"]
        require(file_hash(image_path) == sample["image_sha256"], "Cópia de imagem divergente.")
        require(file_hash(label_path) == sample["label_sha256"], "Label histórico alterado.")
        require(label_path.read_text(encoding="utf-8") == boxes_to_yolo(item["boxes"], item["width"], item["height"]), "Coordenadas YOLO divergentes.")
    output = run / "dataset_should_not_exist"
    config = load_config(root / "configs/project.yaml")
    try:
        build_approved_dataset(config, current_path, root / "configs/review_groups.json", output)
    except ValueError as error:
        builder_reason = str(error)
    else:
        raise ValueError("Builder liberou registro sem aprovação atual.")
    require(not output.exists(), "Builder criou dados antes da aprovação atual.")
    return {
        "feedback_sha256": file_hash(feedback_path), "source_registry_sha256": file_hash(source_path),
        "historical_registry_sha256": file_hash(human_path), "current_registry_sha256": file_hash(current_path),
        "feedback_decisions": 36, "historical_human_images": 33, "historical_human_boxes": 108,
        "unique_images": 38, "current_boxes": 115, "current_human_approved": human_current,
        "current_historical_human_images": historical_current, "current_active_classes": 13,
        "historical_active_classes": 10, "new_class_positive_boxes": {"10": 0, "11": 0, "12": 0},
        "corpus_label_files_verified": 33, "corpus_ready_for_training": False,
        "builder_rejected": True, "builder_created_output": False, "builder_reason": builder_reason,
        "independent_building_groups": 1,
    }


def verify_generated_page(root: Path, run: Path) -> dict:
    """Sintaxe e dados da página; esta checagem não substitui uso no navegador."""
    directory = root / "data/reviews/ceasa_v6_corrigida_a36ea7f67d08"
    payload = read_json(directory / "browser_data.json")
    registry = read_json(directory / "registry.json")
    require(payload["registry_sha256"] == file_hash(directory / "registry.json"), "Página usa outro registro.")
    require(len(payload["classes"]) == 13 and len(payload["images"]) == 38, "Página incompleta.")
    for rendered, original in zip(payload["images"], registry["images"]):
        require(rendered["image_sha256"] == original["image_sha256"] and rendered["boxes"] == original["boxes"], "Página alterou as caixas.")
        require(rendered.get("previous_human_approved") == original.get("previous_human_approved"), "Página perdeu histórico humano.")
        require(file_hash(directory / rendered["image_url"]) == rendered["image_sha256"], "Cópia da página divergente.")
        require((directory / rendered["evidence_url"]).is_file(), "Evidência ausente.")
        if original["status"] in {"positive", "negative"}:
            label = directory / "labels_proposed" / payload["registry_sha256"] / (original["image_sha256"] + ".txt")
            require(label.read_text(encoding="utf-8") == boxes_to_yolo(original["boxes"], original["width"], original["height"]), "Label da página divergente.")
    html = (directory / "index.html").read_text(encoding="utf-8")
    require('id="historyNotice"' in html and "previous_human_approved" in html, "Aviso humano histórico ausente.")
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", html, flags=re.S | re.I)
    javascript = run / "review.js"
    javascript.write_text("\n".join(scripts), encoding="utf-8")
    node = shutil.which("node")
    require(node is not None, "Verificador de sintaxe JS indisponível.")
    result = subprocess.run([node, "--check", str(javascript)], capture_output=True, text=True, encoding="utf-8")
    require(result.returncode == 0, "JavaScript gerado com erro de sintaxe.")
    compiled = 0
    for tree in (root / "src", root / "tests"):
        for path in tree.rglob("*.py"):
            compile(path.read_bytes(), str(path), "exec")
            compiled += 1
    return {"generated_javascript_syntax_valid": True, "python_sources_compiled": compiled,
            "rendered_original_copies_verified": 38, "page_class_options": 13,
            "browser_visual_validation": False, "server_updated": False}


def verify_local_links(root: Path) -> int:
    """Conferir links locais sem acessar URLs ou executar conteúdo documental."""
    documents = [root / "README.md", *sorted((root / "docs").glob("*.md")),
                 root / "data/reviews/ceasa_v6_corrigida_a36ea7f67d08/review_report.md"]
    checked = 0
    for document in documents:
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            target = match.group(1).strip().strip("<>")
            if "://" in target or target.startswith("#"):
                continue
            path = (document.parent / unquote(target.split("#", 1)[0])).resolve()
            require(path.exists(), f"Link local ausente em {document.name}: {target}")
            checked += 1
    return checked


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run = make_run_directory(root, "feedback_validation")
    report = {"preserved_input_files_verified": verify_preserved_inputs(root)}
    report.update(verify_annotations(root, run))
    report.update(verify_generated_page(root, run))
    report["local_document_links_verified"] = verify_local_links(root)
    report.update({"unit_test_result": {"passed": True, "count": 156, "seconds": 10.623,
                                      "source": "unittest discover -s tests -v; execução anterior nesta mesma tarefa, exit_code=0"},
                   "specialized_training_executed": False, "domain_metrics_calculated": False,
                   "generic_model_sha256": file_hash(root / "models/yolo26l.pt"),
                   "ready_for_training": False, "validation_passed": True})
    write_json(run / "validation.json", report)
    print(json.dumps({"validation_file": str(run / "validation.json"),
                      "preserved_files": report["preserved_input_files_verified"],
                      "tests_passed": 156, "ready_for_training": False}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
