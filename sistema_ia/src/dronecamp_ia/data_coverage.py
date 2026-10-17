"""Cobertura dos dados: o que o modelo realmente viu, por classe e por conjunto.

Função no projeto: separar "o modelo melhorou porque viu exemplos novos" de "o
modelo melhorou porque viu os mesmos recortes mais vezes". Para cada divisão
(treino, validação, teste) e cada classe conta:

- fotos originais com a classe, objetos anotados (caixas), cenas e edificações;
- com um treino em janelas (``train-pilot --tile``): recortes com a classe e
  objetos projetados nos recortes (uma caixa na sobreposição de duas janelas
  aparece nas duas);
- com reamostragem de classes raras (``train_repeat_factor.txt``): quantas vezes
  os recortes da classe entram numa época (aparições) e a maior repetição;
- negativos: fotos e recortes sem nenhuma caixa.

Classe sem nenhuma caixa numa divisão aparece com ``status = "sem_exemplos"``.
``compare_coverage`` diz, classe a classe, quanto cresceram os objetos ÚNICOS e
quanto cresceram só as aparições (repetição).

Quem chama: comando ``data-coverage`` (``cli.py``).

O que lê e grava no disco:
- Lê: ``pilot.json`` e as labels YOLO do dataset piloto; com ``--run``:
  ``execution.json`` (pasta das janelas), ``tiling_manifest.json``, as labels das
  janelas e ``train_repeat_factor.txt``.
- Grava: ``runs/coverage_<data>_<id>/coverage.json`` e ``coverage.md`` (pasta nova).

ATENCAO: a pasta das janelas pode estar na área de gravação de outro disco
(``DRONECAMP_FIT_STAGING``). Se ela não existir mais, a seção de recortes fica
``indisponivel`` e só os totais do ``tiling_manifest.json`` aparecem.

Onde mexer:
- Outro agrupamento (ex.: por voo): acrescente o campo em ``_photo_rows`` e em ``KEYS``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .io import file_hash, make_run_directory, write_json

SPLITS = ("train", "val", "test")
# Nome das janelas gravado por tiling.build_tiled_dataset: <foto>_p<índice>_<x>_<y>.
TILE_NAME = re.compile(r"^(?P<stem>.+)_p\d{2,}_\d+_\d+$")


def _label_classes(path: Path) -> list[int]:
    """IDs de classe de cada linha de uma label YOLO (arquivo ausente = lista vazia)."""
    if not path.is_file():
        return []
    return [int(line.split()[0]) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _photo_rows(root: Path) -> list[dict]:
    """Uma linha por foto do ``pilot.json``: divisão, cena, edificação e classes de cada caixa."""
    manifest = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
    rows = []
    for sample in manifest["images"]:
        rows.append({"stem": Path(sample["image"]).stem, "split": sample["split"],
                     "scene": sample.get("scene_group") or "sem_cena",
                     "building": sample.get("building_group") or "sem_edificacao",
                     "classes": _label_classes(root / sample["label"])})
    return rows


def _tiles(run: Path) -> tuple[dict | None, str]:
    """Labels das janelas do treino ``run``: ``{divisão: {nome_janela: [classes]}}`` e o estado."""
    execution = run / "execution.json"
    if not execution.is_file():
        return None, "run_sem_execution_json"
    tiling = json.loads(execution.read_text(encoding="utf-8")).get("tiling")
    if not tiling:
        return None, "treino_sem_janelas"
    folder = Path(tiling.get("tiles_directory", ""))
    if not (folder / "labels").is_dir():
        return None, f"indisponivel ({folder} não existe nesta máquina)"
    result = {}
    for split in SPLITS:
        labels = folder / "labels" / split
        if labels.is_dir():
            result[split] = {path.stem: _label_classes(path) for path in sorted(labels.glob("*.txt"))}
    return result, "ok"


def _repeats(run: Path) -> Counter | None:
    """Quantas vezes cada imagem aparece na lista de treino reamostrada (``None`` sem reamostragem)."""
    path = run / "train_repeat_factor.txt"
    if not path.is_file():
        return None
    return Counter(Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def coverage_report(data_path: Path, names: list[str], run: Path | None = None) -> dict:
    """Monta a cobertura por divisão e classe (sem gravar nada).

    Parâmetros:
        data_path: ``dataset.yaml`` do dataset piloto (``pilot.json`` na mesma pasta).
        names: classes na ordem dos IDs.
        run: treino (``runs/pilot_train_*``) para contar janelas e repetições.

    Retorna ``{"dataset", "pilot_sha256", "unit", "splits": {divisão: {"totals", "classes"}}, ...}``.
    """
    root = Path(data_path).resolve().parent
    photos = _photo_rows(root)
    tiles, tiles_state = _tiles(Path(run)) if run else (None, "sem_run")
    repeats = _repeats(Path(run)) if run else None
    report = {"schema_version": 1, "dataset": str(Path(data_path).resolve()), "pilot_sha256": file_hash(root / "pilot.json"),
              "run": str(run) if run else None, "tiles": tiles_state, "repeat_list": repeats is not None,
              "generated_at": datetime.now(timezone.utc).isoformat(), "splits": {}}
    if run and (Path(run) / "tiling_manifest.json").is_file():
        report["tiling_manifest"] = json.loads((Path(run) / "tiling_manifest.json").read_text(encoding="utf-8")).get("stats")
    for split in SPLITS:
        rows = [row for row in photos if row["split"] == split]
        classes = {}
        for class_id, name in enumerate(names):
            with_class = [row for row in rows if class_id in row["classes"]]
            entry = {"photos": len(with_class), "objects": sum(row["classes"].count(class_id) for row in rows),
                     "scenes": len({row["scene"] for row in with_class}),
                     "buildings": len({row["building"] for row in with_class})}
            entry["status"] = "com_exemplos" if entry["objects"] else "sem_exemplos"
            classes[name] = entry
        totals = {"photos": len(rows), "negative_photos": sum(1 for row in rows if not row["classes"]),
                  "objects": sum(len(row["classes"]) for row in rows),
                  "scenes": len({row["scene"] for row in rows}), "buildings": len({row["building"] for row in rows}),
                  "classes_without_examples": [name for name, entry in classes.items() if not entry["objects"]]}
        # Janelas (recortes) desta divisão, ligadas à foto de origem pelo nome.
        split_tiles = (tiles or {}).get(split)
        if split_tiles is not None:
            stems_split = {row["stem"] for row in rows}
            unmatched = 0
            for name in names:
                classes[name].update({"tiles": 0, "tile_objects": 0, "photos_via_tiles": 0})
            photos_by_class = defaultdict(set)
            for tile, tile_classes in split_tiles.items():
                match = TILE_NAME.match(tile)
                stem = match.group("stem") if match else None
                if stem not in stems_split:
                    unmatched += 1
                for class_id in set(tile_classes):
                    if 0 <= class_id < len(names):
                        classes[names[class_id]]["tiles"] += 1
                        photos_by_class[class_id].add(stem)
                for class_id in tile_classes:
                    if 0 <= class_id < len(names):
                        classes[names[class_id]]["tile_objects"] += 1
            for class_id, stems in photos_by_class.items():
                classes[names[class_id]]["photos_via_tiles"] = len(stems)
            totals.update({"tiles": len(split_tiles),
                           "negative_tiles": sum(1 for value in split_tiles.values() if not value),
                           "tiles_without_source_photo": unmatched})
        # Repetições da reamostragem: só o treino usa a lista.
        if split == "train" and repeats is not None:
            unit = split_tiles if split_tiles is not None else {row["stem"]: row["classes"] for row in rows}
            for class_id, name in enumerate(names):
                members = [stem for stem, values in unit.items() if class_id in values]
                appearances = sum(repeats.get(stem, 0) for stem in members)
                classes[name].update({"sampled_appearances": appearances,
                                      "max_repeat": max((repeats.get(stem, 0) for stem in members), default=0),
                                      "repetition_factor": round(appearances / len(members), 3) if members else None})
            totals["sampled_items"] = sum(repeats.values())
        report["splits"][split] = {"totals": totals, "classes": classes}
    report["unit"] = "fotos e janelas" if tiles else "fotos"
    return report


def compare_coverage(old: dict, new: dict) -> dict:
    """Diferença por classe no TREINO: objetos únicos novos × aparições extras por repetição.

    Retorna ``{classe: {"objects_delta", "tiles_delta", "appearances_delta", "verdict"}}`` com
    ``verdict`` = ``exemplos_novos``, ``so_repeticao``, ``menos_exemplos`` ou ``igual``.
    """
    result = {}
    old_classes = old["splits"]["train"]["classes"]
    for name, entry in new["splits"]["train"]["classes"].items():
        before = old_classes.get(name, {})
        objects = entry["objects"] - before.get("objects", 0)
        appearances = (entry.get("sampled_appearances") or entry.get("tiles") or entry["objects"]) - \
            (before.get("sampled_appearances") or before.get("tiles") or before.get("objects", 0))
        verdict = ("exemplos_novos" if objects > 0 else "menos_exemplos" if objects < 0
                   else "so_repeticao" if appearances > 0 else "igual")
        result[name] = {"objects_delta": objects, "tiles_delta": (entry.get("tiles") or 0) - (before.get("tiles") or 0),
                        "appearances_delta": appearances, "verdict": verdict}
    return result


def _markdown(report: dict) -> str:
    """Tabelas da cobertura em Markdown."""
    lines = [f"# Cobertura dos dados", "", f"- Dataset: `{report['dataset']}` (pilot.json {report['pilot_sha256'][:12]}).",
             f"- Unidade: {report['unit']}; janelas: {report['tiles']}; reamostragem: "
             f"{'sim' if report['repeat_list'] else 'não'}.", ""]
    for split, block in report["splits"].items():
        totals = block["totals"]
        lines += [f"## {split}", "", f"Fotos: {totals['photos']} (negativas: {totals['negative_photos']}); objetos: "
                  f"{totals['objects']}; cenas: {totals['scenes']}; edificações: {totals['buildings']}"
                  + (f"; janelas: {totals['tiles']} (negativas: {totals['negative_tiles']})" if "tiles" in totals else "")
                  + ".", "", "| Classe | Status | Fotos | Objetos | Cenas | Edificações | Janelas | Objetos em janelas | Aparições | Repetição máx. |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, entry in block["classes"].items():
            lines.append(f"| {name} | {entry['status']} | {entry['photos']} | {entry['objects']} | {entry['scenes']} | "
                         f"{entry['buildings']} | {entry.get('tiles', '—')} | {entry.get('tile_objects', '—')} | "
                         f"{entry.get('sampled_appearances', '—')} | {entry.get('max_repeat', '—')} |")
        lines.append("")
    if report.get("comparison"):
        lines += ["## Comparação com a cobertura anterior (treino)", "", "| Classe | Objetos únicos | Janelas | Aparições | Leitura |",
                  "|---|---|---|---|---|"]
        for name, entry in report["comparison"].items():
            lines.append(f"| {name} | {entry['objects_delta']:+d} | {entry['tiles_delta']:+d} | "
                         f"{entry['appearances_delta']:+d} | {entry['verdict']} |")
        lines.append("")
    return "\n".join(lines)


def write_coverage(config, data_path: Path, run: Path | None = None, baseline: Path | None = None,
                   output: Path | None = None) -> Path:
    """ENTRADA de ``data-coverage``: grava ``coverage.json`` e ``coverage.md`` numa pasta nova.

    Parâmetros:
        config: ``ProjectConfig`` (taxonomia e pasta ``runs/``).
        data_path: ``dataset.yaml`` do dataset piloto.
        run: treino para contar janelas e repetições (opcional).
        baseline: ``coverage.json`` anterior para a comparação (opcional).
        output: pasta de saída (padrão ``runs/coverage_<data>_<id>``).
    """
    from .config import detection_names, load_taxonomy

    names = detection_names(load_taxonomy(config.taxonomy_path))
    report = coverage_report(data_path, names, run)
    if baseline:
        report["comparison"] = compare_coverage(json.loads(Path(baseline).read_text(encoding="utf-8")), report)
    folder = Path(output) if output else make_run_directory(config.root, "coverage")
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "coverage.json", report)
    (folder / "coverage.md").write_text(_markdown(report), encoding="utf-8")
    return folder
