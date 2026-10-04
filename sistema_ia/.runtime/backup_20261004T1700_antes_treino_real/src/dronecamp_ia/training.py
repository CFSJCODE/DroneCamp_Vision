"""Fine-tuning, avaliação e tuning: dataset validado é condição de entrada."""

from pathlib import Path
import csv
import json
import math

import yaml

from .backend import check_domain_names, load_detector, runtime_info
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import validate_dataset
from .io import file_hash, make_run_directory, write_json


def prepare_dataset(config: ProjectConfig, data_path: Path, run: Path) -> tuple[Path, dict]:
    """Registre a revisão e normalize caminhos antes de entregar dados ao YOLO."""
    names = detection_names(load_taxonomy(config.taxonomy_path))
    report = validate_dataset(data_path, names)
    write_json(run / "dataset_validation.json", report)
    if not report["valid"]:
        raise ValueError(f"Dataset reprovado; consulte {run / 'dataset_validation.json'}.")
    from .review_provenance import validate_training_provenance
    try:
        provenance = validate_training_provenance(config, data_path)
    except ValueError as error:
        write_json(run / "training_provenance.json", {"valid": False, "reason": str(error)})
        raise
    write_json(run / "training_provenance.json", {"valid": True, **provenance})
    original = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    root = Path(original.get("path", ".")).expanduser()
    root = (data_path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    normalized = {"path": str(root), "names": names}
    for split in ("train", "val", "test"):
        split_path = Path(original[split]).expanduser()
        normalized[split] = str((root / split_path).resolve())
    # Não copie campos download ou outras instruções do YAML recebido.
    normalized_path = run / "dataset_resolved.yaml"
    normalized_path.write_text(yaml.safe_dump(normalized, allow_unicode=True), encoding="utf-8")
    snapshot = []
    class_counts = {split: {name: 0 for name in names} for split in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        image_dir = Path(normalized[split])
        label_dir = root / "labels" / image_dir.relative_to(root / "images")
        for directory in (image_dir, label_dir):
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    snapshot.append({"path": path.relative_to(root).as_posix(), "sha256": file_hash(path)})
        for path in label_dir.rglob("*.txt"):
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    class_counts[split][names[int(line.split()[0])]] += 1
    write_json(run / "dataset_snapshot.json", {
        "source_yaml_sha256": file_hash(data_path), "groups_sha256": file_hash(root / "groups.csv"),
        "taxonomy_sha256": file_hash(config.taxonomy_path), "files": snapshot,
        "class_counts": class_counts,
    })
    # Uma classe ativa sem exemplos não foi aprendida, mesmo que o treino termine.
    missing = [name for name, count in class_counts["train"].items() if count == 0]
    if missing:
        raise ValueError(f"Treino sem exemplos de classes: {', '.join(missing)}. Anote dados ou proponha nova taxonomia/versionamento antes de treinar.")
    return normalized_path, class_counts


def train(config: ProjectConfig, data_path: Path, weights: str | None = None) -> Path:
    """Ajuste os pesos pré-treinados; parâmetros explícitos ficam auditáveis."""
    run = make_run_directory(config.root, "train")
    _write_training_summary(run, ["A preparação e o treinamento ainda não concluíram."], state="running")
    try:
        dataset, counts = prepare_dataset(config, data_path.resolve(), run)
        expected_names = detection_names(load_taxonomy(config.taxonomy_path))
        model = load_detector(config, weights)
        parameters = dict(config.training)
        parameters["nms"] = config.prediction["nms"]
        write_json(run / "execution.json", {"runtime": runtime_info(model, config), "parameters": parameters,
                                           "class_counts": counts, "stage": "fine_tuning"})
        model.train(data=str(dataset), device=config.device, project=str(run), name="fit",
                    exist_ok=False, **parameters)
    except Exception as error:
        # Uma falha de preparação, treino ou callback nunca deixa uma execução aprovada.
        reason = f"Execução interrompida por {type(error).__name__}: {error}"
        _write_training_summary(run, [reason], state="failed")
        raise ValueError(f"Treino incompleto: {reason}. Revise {run}.") from error
    review_training_result(run, model, expected_names)
    return run


DETECTION_METRIC_COLUMNS = (
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
)


def _write_training_summary(run: Path, reasons: list[str], **details) -> dict:
    """Conclusão é execução auditada; aprovação técnica continua sendo humana."""
    summary = {
        "schema_version": 1, "stage": "fine_tuning", "complete": not reasons,
        "reasons": reasons, "weights_directory": str(run / "fit/weights"),
        "test_evaluated": False, "human_acceptance": "pendente",
        "checkpoint_functional_validation": False, "export_parity_verified": False,
        "note": "Arquivos .pt não vazios não comprovam checkpoint funcional. Avaliar por classe e revisar erros antes de usar em inspeções.",
        **details,
    }
    write_json(run / "summary.json", summary)
    return summary


def _review_training_csv(path: Path) -> tuple[list[str], dict]:
    """Confira métricas e perdas reais, aceitando parada antecipada com épocas válidas."""
    reasons, rows = [], []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = [name.strip() for name in (reader.fieldnames or [])]
            if len(columns) != len(set(columns)):
                return ["results.csv contém colunas duplicadas."], {"epochs_recorded": 0}
            reader.fieldnames = columns
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        return [f"Não foi possível ler results.csv: {type(error).__name__}."], {"epochs_recorded": 0}

    required = {"epoch", *DETECTION_METRIC_COLUMNS, "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss"}
    missing = sorted(required.difference(columns))
    if missing:
        reasons.append("results.csv sem colunas obrigatórias: " + ", ".join(missing) + ".")
    # A biblioteca local nomeia a distância l1_loss no YOLO26 e dfl_loss em outras cabeças.
    if not any(f"train/{name}" in columns and f"val/{name}" in columns for name in ("l1_loss", "dfl_loss")):
        reasons.append("results.csv precisa de perdas de distância correspondentes no treino e na validação (l1_loss ou dfl_loss).")
    loss_columns = [name for name in columns if name.startswith(("train/", "val/")) and name.endswith("_loss")]
    if not rows:
        reasons.append("results.csv não contém nenhuma época de treinamento.")

    last_epoch = 0
    last_metrics = {}
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            reasons.append(f"results.csv linha {row_number} tem valores além das colunas declaradas.")
        numeric = {}
        for name in ["epoch", *DETECTION_METRIC_COLUMNS, *loss_columns]:
            if name not in columns:
                continue
            try:
                value = float(row[name])
            except (TypeError, ValueError):
                reasons.append(f"results.csv linha {row_number}: {name} não é um número.")
                continue
            if not math.isfinite(value):
                reasons.append(f"results.csv linha {row_number}: {name} não é finito.")
                continue
            if name in DETECTION_METRIC_COLUMNS and not 0 <= value <= 1:
                reasons.append(f"results.csv linha {row_number}: {name} precisa estar entre 0 e 1.")
                continue
            numeric[name] = value
        epoch = numeric.get("epoch")
        if epoch is not None:
            if epoch < 1 or not epoch.is_integer() or epoch <= last_epoch:
                reasons.append(f"results.csv linha {row_number}: época deve ser inteira, positiva e crescente.")
            else:
                last_epoch = int(epoch)
        # Métricas zeradas são um resultado possível, não aprovação de qualidade.
        last_metrics = {name: numeric[name] for name in DETECTION_METRIC_COLUMNS if name in numeric}
    return reasons, {"epochs_recorded": len(rows), "last_epoch": last_epoch, "last_metrics": last_metrics,
                     "loss_columns_checked": loss_columns}


def review_training_result(run: Path, model, expected_names: list[str]) -> dict:
    """Recuse retorno sem artefatos, métricas finitas ou contrato das classes ativas."""
    run = Path(run).resolve()
    reasons, artifacts = [], {}
    paths = {"best_weights": run / "fit/weights/best.pt", "last_weights": run / "fit/weights/last.pt",
             "results_csv": run / "fit/results.csv"}
    for name, path in paths.items():
        try:
            # Um link não pode redirecionar a auditoria para artefatos de outra execução.
            contained = path.resolve().is_relative_to(run)
            present = contained and path.is_file()
            size = path.stat().st_size if present else 0
            artifacts[name] = {"path": str(path), "present": present, "size_bytes": size}
            if not contained:
                reasons.append(f"{name} aponta para fora da execução.")
            elif not present:
                reasons.append(f"{name} ausente ou não é um arquivo.")
            elif size == 0:
                reasons.append(f"{name} está vazio.")
        except (OSError, RuntimeError):
            artifacts[name] = {"path": str(path), "present": False, "size_bytes": 0}
            reasons.append(f"{name} não pôde ser verificado.")
    csv_details = {"epochs_recorded": 0}
    if artifacts["results_csv"]["present"] and artifacts["results_csv"]["size_bytes"] > 0:
        csv_reasons, csv_details = _review_training_csv(paths["results_csv"])
        reasons.extend(csv_reasons)
    class_contract_verified = False
    try:
        check_domain_names(model, expected_names)
        class_contract_verified = True
    except (ValueError, AttributeError, KeyError, TypeError) as error:
        reasons.append(f"Contrato das classes após o treino inválido: {error}")
    summary = _write_training_summary(
        run, reasons, state="failed" if reasons else "completed", artifacts=artifacts,
        model_class_contract_verified=class_contract_verified, expected_class_names=expected_names, **csv_details,
    )
    if reasons:
        raise ValueError(f"Treino incompleto: {' '.join(reasons)} Revise {run}.")
    return summary


def evaluate(config: ProjectConfig, data_path: Path, weights: str, split: str = "val") -> Path:
    """Avalie checkpoint especializado; use test apenas para decisão final."""
    run = make_run_directory(config.root, f"evaluate_{split}")
    dataset, counts = prepare_dataset(config, data_path.resolve(), run)
    model = load_detector(config, weights)
    check_domain_names(model, detection_names(load_taxonomy(config.taxonomy_path)))
    # Não reutilize conf=.25 do predict: AP precisa incluir candidatos de baixa confiança.
    metrics = model.val(data=str(dataset), split=split, device=config.device,
                        imgsz=config.prediction["imgsz"], conf=0.001, rect=False,
                        nms=config.prediction["nms"], batch=config.training["batch"],
                        workers=0, project=str(run), name="metrics", plots=True)
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(index)
        per_class.append({"class_name": model.names[int(class_id)], "precision": float(precision),
                          "recall": float(recall), "ap50": float(ap50), "ap50_95": float(ap)})
    write_json(run / "evaluation.json", {
        "runtime": runtime_info(model, config), "split": split, "class_counts": counts,
        "global_metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class": per_class, "human_acceptance": "pendente",
    })
    return run


def tune(config: ProjectConfig, data_path: Path, iterations: int, epochs: int, weights: str | None = None) -> Path:
    """Procure hiperparâmetros na validação; não consuma o teste final."""
    if iterations < 1 or epochs < 1:
        raise ValueError("iterations e epochs precisam ser positivos.")
    run = make_run_directory(config.root, "tune")
    dataset, _ = prepare_dataset(config, data_path.resolve(), run)
    model = load_detector(config, weights)
    parameters = dict(config.training)
    parameters.update(epochs=epochs, close_mosaic=min(10, epochs), nms=config.prediction["nms"])
    write_json(run / "execution.json", {"runtime": runtime_info(model, config), "iterations": iterations,
                                       "parameters": parameters, "test_used": False})
    model.tune(data=str(dataset), iterations=iterations, device=config.device, val=True,
               project=str(run), name="search", space={"lr0": (0.0001, 0.003), "mosaic": (0.0, 0.5)},
               **parameters)
    review_tuning_result(run, iterations)
    return run


def review_tuning_result(run: Path, expected_iterations: int) -> dict:
    """O tuner pode retornar normalmente mesmo se todos os treinos falharem."""
    search = run / "search"
    history = search / "tune_results.ndjson"
    records = []
    if history.is_file():
        records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
    successful = 0
    for record in records:
        datasets = record.get("datasets", {})
        failed = set(record.get("failed_datasets", []))
        if datasets and all(bool(metrics) and name not in failed for name, metrics in datasets.items()):
            successful += 1
    best_yaml = search / "best_hyperparameters.yaml"
    best_weights = search / "weights/best.pt"
    complete = (len(records) == expected_iterations and successful == expected_iterations
                and best_yaml.is_file() and best_weights.is_file())
    summary = {"attempted": len(records), "successful": successful,
               "expected": expected_iterations, "complete": complete,
               "best_parameters_present": best_yaml.is_file(), "best_weights_present": best_weights.is_file(),
               "test_used": False, "human_acceptance": "pendente"}
    write_json(run / "summary.json", summary)
    if not complete:
        raise ValueError(f"Tuning incompleto: {successful}/{expected_iterations} tentativas com métricas e artefatos. Revise {run}.")
    return summary
