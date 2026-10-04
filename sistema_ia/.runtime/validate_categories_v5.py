"""Auditoria da expansão solicitada; nenhum teste usa aprovação nos dados reais."""

from collections import Counter
import compileall
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

import yaml

from dronecamp_ia.config import detection_names, load_config, load_taxonomy
from dronecamp_ia.io import file_hash, make_run_directory, write_json
from dronecamp_ia.review_data import build_review_registry, review_image_size, validate_boxes
from dronecamp_ia.review_dataset import MVP_NAMES, build_approved_dataset


def validate_preservation(root: Path) -> int:
    """Conferir todas as versões antigas, originais e pesos pelo hash anterior."""
    before = json.loads((root / ".runtime/category_update_v5_before.json").read_text(encoding="utf-8"))["files"]
    taxonomy_path = root / "configs/taxonomy.json"
    snapshot = root / "configs/taxonomy_ceasa_v4_fragmento_sobreposto_fixador_solto.json"
    for value, digest in before.items():
        path = Path(value)
        preserved = snapshot if path == taxonomy_path else path
        assert file_hash(preserved) == digest, f"Arquivo anterior mudou: {path.name}"
    return len(before)


def validate_registry(root: Path, config) -> dict:
    """Conferir o contrato atual, as caixas e a rastreabilidade de cada foto."""
    directory = root / "data/reviews/ceasa_v5"
    registry = json.loads((directory / "registry.json").read_text(encoding="utf-8"))
    previous = json.loads((root / "data/reviews/ceasa_v4/registry.json").read_text(encoding="utf-8"))
    old_images = {item["image_sha256"]: item for item in previous["images"]}
    names = detection_names(load_taxonomy(config.taxonomy_path))
    assert names == list(MVP_NAMES) and len(names) == 13
    assert registry["taxonomy_sha256"] == file_hash(config.taxonomy_path)
    assert registry["taxonomy_migration"]["new_class_ids"] == [12]
    assert len(registry["images"]) == len(old_images) == 38
    for item in registry["images"]:
        source = Path(item["source_path"])
        assert file_hash(source) == item["image_sha256"]
        assert review_image_size(source) == (item["width"], item["height"])
        assert item["boxes"] == old_images[item["image_sha256"]]["boxes"]
        assert item["status"] == old_images[item["image_sha256"]]["status"]
        assert item["classes_reviewed_before_migration"] == list(range(8))
        assert item["historical_visual_review_status"] == "second_pass_ai"
        assert item["taxonomy_recheck_pending"] and not item["human_approved"]
        assert item["severity"] is None
        validate_boxes(item["boxes"], item["width"], item["height"], len(names))
        assert file_hash(directory / "images" / (item["image_sha256"] + source.suffix.lower())) == item["image_sha256"]
    boxes = Counter(box["class_id"] for item in registry["images"] for box in item["boxes"])
    assert sum(boxes.values()) == 109 and boxes[10] == boxes[11] == boxes[12] == 0
    template = yaml.safe_load(config.dataset_path.read_text(encoding="utf-8"))
    assert [template["names"][i] for i in range(13)] == names
    browser_data = json.loads((directory / "browser_data.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in browser_data["classes"]] == list(range(13))
    assert browser_data["classes_requiring_review"] == [item["label"] for item in browser_data["classes"][8:]]
    assert browser_data["registry_sha256"] == file_hash(directory / "registry.json")
    summary = json.loads((directory / "review_summary.json").read_text(encoding="utf-8"))
    assert summary["prior_taxonomy_second_visual_review_images"] == 38
    return {"original_images_and_copies_verified": 38, "previous_109_boxes_preserved": True,
            "active_classes": names, "new_class_boxes": {"12": boxes[12]},
            "total_proposed_boxes": 109, "human_approved_images": 0,
            "taxonomy_recheck_pending_images": 38, "actual_previous_review_scope": list(range(8))}


def validate_guards(root: Path, config, run: Path) -> None:
    """Executar bloqueios reais sem criar dataset nem carregar o modelo."""
    output = run / "unapproved_dataset_must_not_exist"
    try:
        build_approved_dataset(config, root / "data/reviews/ceasa_v5/registry.json",
                               root / "configs/review_groups.json", output)
    except ValueError as error:
        assert "aprovação humana" in str(error)
    else:
        raise AssertionError("Dados reais sem aprovação foram aceitos.")
    assert not output.exists()
    try:
        build_review_registry(config, root / "data/reviews/ceasa_v1")
    except ValueError as error:
        assert "migrate-review" in str(error)
    else:
        raise AssertionError("Lotes antigos foram reinterpretados sob classes novas.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/project.yaml")
    run = make_run_directory(root, "review_validation_v5")
    report = validate_registry(root, config)
    validate_guards(root, config, run)
    report["preserved_files_verified"] = validate_preservation(root)
    assert compileall.compile_dir(root / "src", quiet=1)
    html = (root / "data/reviews/ceasa_v5/index.html").read_text(encoding="utf-8")
    javascript = run / "review_page_syntax.js"
    javascript.write_text("\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)), encoding="utf-8")
    node = shutil.which("node")
    assert node, "Runtime JavaScript indisponível para checar a página."
    syntax = subprocess.run([node, "--check", str(javascript)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    log = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"))
    results = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (run / "tests.log").write_text(log.getvalue(), encoding="utf-8")
    report.update(tests_run=results.testsRun, tests_passed=results.wasSuccessful(),
                  failures=len(results.failures), errors=len(results.errors), skipped=len(results.skipped),
                  python_and_javascript_syntax_ok=True, unapproved_dataset_blocked_without_output=True,
                  legacy_batches_blocked=True, model_weights_unchanged=True,
                  model_sha256=file_hash(root / "models/yolo26l.pt"), specialized_training_executed=False,
                  run_directory=str(run))
    write_json(run / "validation.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not results.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
