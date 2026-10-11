"""Ciclo de retreino: dataset → treino (CPU+GPU) → exportação → gate → medição → relatório.

Função no projeto: depois que o revisor exporta as decisões e o ``import-review``
as grava nos registros, o próximo modelo exige uma sequência longa de comandos
na ordem certa. Este módulo encadeia essa sequência (comando ``retrain-cycle``)
e grava ``runs/cycle_*/cycle.json`` com o caminho de cada etapa, para que a
comparação com os modelos anteriores saia sempre nas mesmas fotos.

O que faz:
- ``run_cycle``: executa as etapas abaixo e para na primeira que falhar.
  1. ``build_pilot_dataset`` (pilot.py), com ``sequence_block`` para as fotos de voo.
  2. ``train_pilot`` (training.py) com ``tile`` (janelas das fotos grandes) e o
     worker de avaliação na GPU (gpu_eval.py).
  3. ``export_onnx`` (exporting.py) do ``best.pt`` sem paridade (o gate já roda o ONNX).
  4. ``compare_models`` (model_gate.py) candidato × cada modelo anterior.
  5. ``measure_models`` (model_gate.py): todos os modelos nas mesmas fotos.
  6. ``build_model_report`` (model_report.py): gráficos comparativos.

Nada aqui aprova pesos: ``gate.json`` de cada comparação diz se há ganho medido.

Quando mexer: para acrescentar uma etapa ao ciclo ou mudar a ordem.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .config import ProjectConfig
from .io import make_run_directory, write_json


def _write(cycle: Path, record: dict) -> None:
    write_json(cycle / "cycle.json", record)


def run_cycle(config: ProjectConfig, registries: list[Path], output: Path, baselines: list[str] = (),
              weights: str | None = None, overrides: dict | None = None, seed: int = 42,
              sequence_block: int = 0, gpu_eval: dict | None = None, onnx_provider: str = "directml",
              splits: tuple[str, ...] = ("test",), title: str | None = None) -> Path:
    """Executa o ciclo completo e devolve ``runs/cycle_*`` com ``cycle.json``."""
    from .exporting import export_onnx
    from .model_gate import compare_models, measure_models
    from .model_report import build_model_report
    from .pilot import build_pilot_dataset
    from .training import train_pilot

    cycle = make_run_directory(config.root, "cycle")
    record = {"started_at": datetime.now(timezone.utc).isoformat(), "state": "running", "steps": {},
              "registries": [str(Path(path).resolve()) for path in registries], "baselines": list(baselines),
              "overrides": dict(overrides or {}), "sequence_block": sequence_block}
    _write(cycle, record)
    try:
        # 1. Dataset piloto (fotos inteiras; as janelas são geradas no treino com tile).
        dataset = build_pilot_dataset(config, list(registries), Path(output), seed, sequence_block=sequence_block)
        record["steps"]["dataset"] = dataset
        _write(cycle, record)
        data_path = Path(dataset["dataset"])
        # 2. Treino na CPU com avaliação paralela na GPU.
        run = train_pilot(config, data_path, weights, overrides, gpu_eval)
        execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
        best = run / "fit" / "weights" / "best.pt"
        candidate = best if best.is_file() else run / "fit" / "weights" / "last.pt"
        record["steps"]["training"] = {"run": str(run), "candidate": str(candidate), "tiling": execution.get("tiling"),
                                       "gpu_eval": execution.get("gpu_eval")}
        _write(cycle, record)
        # 3. ONNX do candidato (o gate e a medição rodam o ONNX na GPU).
        export = export_onnx(config, str(candidate), parity_images=[])
        onnx = export / "model.onnx"
        record["steps"]["export"] = {"run": str(export), "onnx": str(onnx)}
        _write(cycle, record)
        # O gate e a medição avaliam o ONNX recém-exportado (o artefato que roda na GPU),
        # para que a decisão valide o mesmo arquivo usado nas sugestões.
        candidate = onnx if onnx.is_file() else candidate
        # 4. Gate candidato × cada anterior, nas fotos fora do treino de ambos.
        gates = []
        for baseline in baselines:
            gate_run = compare_models(config, data_path, baseline, str(candidate), splits, onnx_provider=onnx_provider)
            gate = json.loads((gate_run / "gate.json").read_text(encoding="utf-8"))
            gates.append({"baseline": baseline, "run": str(gate_run), "adopt": gate.get("adopt"),
                          "reasons": gate.get("reasons"), "images_evaluated": gate.get("images_evaluated")})
        record["steps"]["gates"] = gates
        _write(cycle, record)
        # 5. Medição de todos os modelos nas mesmas fotos e 6. relatório.
        measurement = measure_models(config, data_path, [*baselines, str(candidate)], splits, None, onnx_provider)
        record["steps"]["measurement"] = {"run": str(measurement)}
        report = build_model_report(config, None, [Path(item["run"]) / "gate.json" for item in gates], (), (), [run],
                                    title or f"Ciclo {cycle.name}", measurements=[measurement / "measurement.json"])
        record["steps"]["report"] = {"run": str(report), "html": str(report / "report.html")}
        record["state"] = "completed"
        record["adopt_any"] = any(item["adopt"] for item in gates) if gates else None
        record["note"] = ("Sem baseline não há comparação: o candidato só pode ser adotado após um compare-models."
                          if not gates else "adopt vem de cada gate.json (fotos fora do treino de ambos os modelos).")
    except Exception as error:
        record["state"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        _write(cycle, record)
        raise
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write(cycle, record)
    return cycle
