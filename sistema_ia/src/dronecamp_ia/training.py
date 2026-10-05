"""Treinamento, avaliação e busca de hiperparâmetros do detector YOLO.

Função no projeto: é o arquivo que de fato treina o modelo. Ele chama
``model.train`` da Ultralytics e audita o resultado antes de dar a execução
como concluída.

O que faz:
- ``train_pilot``: treino piloto com as revisões humanas do CEASA (o usado hoje);
  os pesos só sugerem caixas na revisão.
- ``train``: fine-tuning de produção; exige dataset aprovado e três edificações.
- ``evaluate``: métricas por classe em val ou test.
- ``tune``: vários treinos curtos para procurar hiperparâmetros.
- ``review_training_result``: confere pesos, ``results.csv`` e classes no fim.

Quando mexer: épocas, imgsz, batch, patience, lr e aumentos ficam em
``configs/project.yaml`` (seções ``training`` e ``pilot.training``), não aqui.
Mexa neste arquivo para mudar o fluxo: o que é validado antes, quais parâmetros
chegam ao ``model.train`` ou como o resultado é auditado. Para mudar como o
dataset piloto é montado (splits, duplicatas), o arquivo é ``pilot.py``.
"""

from pathlib import Path
import csv
import json
import math

import yaml

from .backend import check_domain_names, load_detector, runtime_info
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import validate_dataset
from .io import file_hash, make_run_directory, write_json


# ---------------------------------------------------------------------------
# Preparação dos dados: tudo o que é conferido antes de o YOLO ver o dataset.
# ---------------------------------------------------------------------------

def prepare_dataset(config: ProjectConfig, data_path: Path, run: Path) -> tuple[Path, dict]:
    """Registre a revisão e normalize caminhos antes de entregar dados ao YOLO."""
    # 1. Estrutura: pastas, labels, classes e groups.csv (dataset.py).
    names = detection_names(load_taxonomy(config.taxonomy_path))
    report = validate_dataset(data_path, names)
    write_json(run / "dataset_validation.json", report)
    if not report["valid"]:
        raise ValueError(f"Dataset reprovado; consulte {run / 'dataset_validation.json'}.")
    # 2. Procedência: cada foto/label precisa vir de uma aprovação humana registrada.
    from .review_provenance import validate_training_provenance
    try:
        provenance = validate_training_provenance(config, data_path)
    except ValueError as error:
        write_json(run / "training_provenance.json", {"valid": False, "reason": str(error)})
        raise
    write_json(run / "training_provenance.json", {"valid": True, **provenance})
    # 3. Caminhos absolutos e foto (hash) do que será treinado.
    normalized_path, class_counts = _resolve_and_snapshot(config, data_path, run, names)
    # Uma classe ativa sem exemplos não foi aprendida, mesmo que o treino termine.
    missing = [name for name, count in class_counts["train"].items() if count == 0]
    if missing:
        raise ValueError(f"Treino sem exemplos de classes: {', '.join(missing)}. Anote dados ou proponha nova taxonomia/versionamento antes de treinar.")
    return normalized_path, class_counts


def _resolve_and_snapshot(config: ProjectConfig, data_path: Path, run: Path, names: list[str]) -> tuple[Path, dict]:
    """Normalize caminhos e registre hashes e contagens do que será treinado."""
    # Lê o dataset.yaml recebido e transforma path/train/val/test em caminhos absolutos.
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
    # Registra o SHA-256 de cada foto/label e conta as caixas por classe e split.
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
    return normalized_path, class_counts


# ---------------------------------------------------------------------------
# Treinos: produção (train) e piloto (train_pilot). Ambos chamam model.train.
# ---------------------------------------------------------------------------

def train(config: ProjectConfig, data_path: Path, weights: str | None = None) -> Path:
    """Ajuste os pesos pré-treinados; parâmetros explícitos ficam auditáveis."""
    # Pasta nova runs/train_<data>_<id>; o summary começa como "running".
    run = make_run_directory(config.root, "train")
    _write_training_summary(run, ["A preparação e o treinamento ainda não concluíram."], state="running")
    try:
        dataset, counts = prepare_dataset(config, data_path.resolve(), run)
        expected_names = detection_names(load_taxonomy(config.taxonomy_path))
        model = load_detector(config, weights)
        # Hiperparâmetros vêm da seção training do configs/project.yaml.
        parameters = dict(config.training)
        parameters["nms"] = config.prediction["nms"]
        write_json(run / "execution.json", {"runtime": runtime_info(model, config), "parameters": parameters,
                                           "class_counts": counts, "stage": "fine_tuning"})
        # Treino de verdade (Ultralytics); pesos e results.csv vão para run/fit.
        model.train(data=str(dataset), device=config.device, project=str(run), name="fit",
                    exist_ok=False, **parameters)
    except Exception as error:
        # Uma falha de preparação, treino ou callback nunca deixa uma execução aprovada.
        reason = f"Execução interrompida por {type(error).__name__}: {error}"
        _write_training_summary(run, [reason], state="failed")
        raise ValueError(f"Treino incompleto: {reason}. Revise {run}.") from error
    review_training_result(run, model, expected_names)
    return run


def train_pilot(config: ProjectConfig, data_path: Path, weights: str | None = None, overrides: dict | None = None) -> Path:
    """Treino real e exploratório com o dataset piloto; nunca aprova o modelo para uso."""
    from .pilot import validate_pilot_dataset

    # 1. Pasta nova runs/pilot_train_<data>_<id>; summary marcado como piloto.
    run = make_run_directory(config.root, "pilot_train")
    pilot_details = {"pilot": True, "production_ready": False}
    _write_training_summary(run, ["A preparação e o treinamento piloto ainda não concluíram."], state="running", **pilot_details)
    try:
        # 2. Reconfere hashes e labels do dataset piloto contra os registros humanos.
        names = detection_names(load_taxonomy(config.taxonomy_path))
        audit = validate_pilot_dataset(config, data_path.resolve())
        write_json(run / "pilot_provenance.json", audit)
        dataset, counts = _resolve_and_snapshot(config, data_path.resolve(), run, names)
        # 3. Modelo inicial: --weights, ou pilot.model, ou model do project.yaml.
        model = load_detector(config, weights or config.pilot.get("model") or config.model)
        # 4. Parâmetros: training < pilot.training < opções da linha de comando.
        parameters = {**config.training, **config.pilot.get("training", {}), **(overrides or {})}
        parameters["nms"] = config.prediction["nms"]
        # 4b. Classes raras: lista de treino com repetições (sampling.py). Não é
        # parâmetro da Ultralytics, por isso sai de ``parameters`` antes do train.
        threshold = parameters.pop("repeat_factor_threshold", None)
        if threshold is None:
            threshold = (config.pilot.get("sampling") or {}).get("repeat_factor_threshold")
        if threshold is not None:
            pilot_details["sampling"] = _apply_repeat_factor_sampling(run, dataset, float(threshold),
                                                                     int(parameters.get("seed", 0)))
        pilot_details["classes_without_training_boxes"] = [name for name, count in counts["train"].items() if count == 0]
        pilot_details["warning"] = audit["warning"]
        write_json(run / "execution.json", {"runtime": runtime_info(model, config), "parameters": parameters,
                                           "class_counts": counts, "stage": "fine_tuning_piloto", **pilot_details})
        # 5. Treino de verdade (Ultralytics); best.pt/last.pt vão para run/fit/weights.
        model.train(data=str(dataset), device=config.device, project=str(run), name="fit",
                    exist_ok=False, **parameters)
    except Exception as error:
        reason = f"Execução interrompida por {type(error).__name__}: {error}"
        _write_training_summary(run, [reason], state="failed", **pilot_details)
        raise ValueError(f"Treino piloto incompleto: {reason}. Revise {run}.") from error
    # 6. Auditoria final: só aqui o summary passa para "completed".
    review_training_result(run, model, names, **pilot_details)
    return run


def _apply_repeat_factor_sampling(run: Path, dataset: Path, threshold: float, seed: int) -> dict:
    """Troca ``train:`` do dataset_resolved.yaml por uma lista com fotos raras repetidas.

    Validação e teste não mudam: as métricas continuam medindo cada foto uma vez.
    """
    from .sampling import write_repeat_factor_list

    resolved = yaml.safe_load(dataset.read_text(encoding="utf-8"))
    image_dir = Path(resolved["train"])
    label_dir = Path(resolved["path"]) / "labels" / image_dir.relative_to(Path(resolved["path"]) / "images")
    summary = write_repeat_factor_list(image_dir, label_dir, run / "train_repeat_factor.txt", threshold, seed)
    resolved["train"] = summary["list"]
    dataset.write_text(yaml.safe_dump(resolved, allow_unicode=True), encoding="utf-8")
    write_json(run / "sampling.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Pesos piloto: identificação e tamanho de imagem usado na inferência.
# ---------------------------------------------------------------------------

def is_pilot_checkpoint(weights: str | Path | None) -> bool:
    """Pesos de runs/pilot_train_*/fit/weights carregam o aviso do piloto."""
    if not weights:
        return False
    summary = Path(weights).resolve().parent.parent.parent / "summary.json"
    try:
        return json.loads(summary.read_text(encoding="utf-8")).get("pilot") is True
    except (OSError, ValueError):
        return False


def pilot_inference_config(config: ProjectConfig, model) -> ProjectConfig:
    """Pesos piloto inferem no tamanho em que foram treinados, não no da produção."""
    from dataclasses import replace

    if not is_pilot_checkpoint(getattr(model, "ckpt_path", None)):
        return config
    trained = (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz")
    if not isinstance(trained, int) or trained < 32:
        return config
    return replace(config, prediction={**config.prediction, "imgsz": trained})


# ---------------------------------------------------------------------------
# Auditoria do resultado: summary.json, results.csv e contrato das classes.
# ---------------------------------------------------------------------------

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

    # Colunas que toda execução de detecção precisa registrar.
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

    # Linha a linha: números finitos, métricas entre 0 e 1, épocas crescentes.
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


def review_training_result(run: Path, model, expected_names: list[str], **details) -> dict:
    """Recuse retorno sem artefatos, métricas finitas ou contrato das classes ativas."""
    run = Path(run).resolve()
    reasons, artifacts = [], {}
    # 1. Arquivos obrigatórios da execução: pesos e histórico de épocas.
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
    # 2. Épocas, perdas e métricas finitas no results.csv.
    csv_details = {"epochs_recorded": 0}
    if artifacts["results_csv"]["present"] and artifacts["results_csv"]["size_bytes"] > 0:
        csv_reasons, csv_details = _review_training_csv(paths["results_csv"])
        reasons.extend(csv_reasons)
    # 3. O modelo treinado precisa ter exatamente as classes ativas da taxonomia.
    class_contract_verified = False
    try:
        check_domain_names(model, expected_names)
        class_contract_verified = True
    except (ValueError, AttributeError, KeyError, TypeError) as error:
        reasons.append(f"Contrato das classes após o treino inválido: {error}")
    # 4. summary.json final: "completed" só sem nenhum motivo de falha.
    summary = _write_training_summary(
        run, reasons, state="failed" if reasons else "completed", artifacts=artifacts,
        model_class_contract_verified=class_contract_verified, expected_class_names=expected_names, **csv_details,
        **details,
    )
    if reasons:
        raise ValueError(f"Treino incompleto: {' '.join(reasons)} Revise {run}.")
    return summary


# ---------------------------------------------------------------------------
# Avaliação e busca de hiperparâmetros (produção).
# ---------------------------------------------------------------------------

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
    # Precisão, recall e AP de cada classe medida no split.
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
    # Cada tentativa treina poucas épocas e varia lr0 e mosaic dentro do espaço abaixo.
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
