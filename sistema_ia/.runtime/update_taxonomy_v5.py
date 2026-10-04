"""Cadastrar reparo em rufo, preservando IDs ativos e evidências anteriores."""

import json
from pathlib import Path
import shutil

from dronecamp_ia.io import file_hash, write_json


def preserve_previous_version(root: Path, taxonomy_path: Path, snapshot: Path) -> None:
    """Guardar bytes da taxonomia e hashes das fontes antes da expansão."""
    manifest_path = root / ".runtime/category_update_v5_before.json"
    if snapshot.exists() or manifest_path.exists():
        raise ValueError("Atualização já iniciada; não substituir a auditoria anterior.")
    paths = [taxonomy_path, root / "models/yolo26l.pt"]
    paths.extend(root.glob("configs/taxonomy_ceasa_*.json"))
    for version in ("ceasa_v1", "ceasa_v3", "ceasa_v4"):
        paths.extend(path for path in (root / "data/reviews" / version).rglob("*") if path.is_file())
    paths.extend(path for path in (root / "data/reference/ceasa").rglob("*") if path.is_file())
    write_json(manifest_path, {"files": {str(path.resolve()): file_hash(path) for path in paths}})
    shutil.copy2(taxonomy_path, snapshot)


def expand_taxonomy(taxonomy: dict) -> dict:
    """Acrescentar ID12 sem atribuir diagnóstico ou gravidade automaticamente."""
    if taxonomy["version"] != "ceasa-v4-proposta-fragmento-sobreposto-fixador-solto":
        raise ValueError("Taxonomia de origem diferente da versão esperada.")
    future_category = next(item for item in taxonomy["classes"] if item["slug"] == "reparo_selante_fixador_telha")
    future_category["id"] = 22  # Categoria futura sem labels ativos: manter IDs0–11.
    taxonomy["classes"].append({
        "id": 12, "slug": "reparo_rufo", "label": "Reparo em rufo", "phase": 1,
        "pages": [], "reference_severity": None,
        "origin": "categoria_adicionada_por_solicitacao_do_usuario_em_2026-10-03",
        "criterion": "Intervenção visualmente identificável no próprio rufo: remendo, fita, manta, aplicação de selante ou peça de reparo. Confirmar a identidade do rufo e diferenciar vedação original regular, junta prevista, sujeira, sombra e material solto apenas apoiado sobre o elemento.",
        "annotation_policy": "Delimitar a área de intervenção visível, sem incluir todo o rufo. Reparo sobre telha permanece ID3 reparo_telha; a categoria depende do elemento reparado. Se a foto não distinguir rufo de telha ou reparo de acabamento original, manter ambíguo. Não converter caixas antigas nem duplicar uma caixa idêntica como reparo e quebra/deslocamento sem evidências distintas.",
        "physical_diagnosis": "Presença de reparo não comprova falha ou não conformidade por si só. Não inferir aderência, estanqueidade, causa, eficácia, risco, urgência ou severidade pela fotografia.",
    })
    taxonomy["classes"].sort(key=lambda item: item["id"])
    taxonomy["version"] = "ceasa-v5-proposta-reparo-rufo"
    taxonomy["migration"] = {
        "from_version": "ceasa-v4-proposta-fragmento-sobreposto-fixador-solto",
        "preserved_active_ids": list(range(12)), "new_active_class_ids": [12],
        "catalog_id_mapping": {"12": 22},
        "prior_catalog_id_mapping": {"8": 19, "9": 20, "10": 21},
        "note": "IDs ativos 0–11 preservados. Nova classe 12 de reparo em rufo; o antigo ID12 de reparo em fixador de telha, fase2, passa a22. Snapshots e propostas anteriores preservados, sem aprendizado ou conversão automática de caixas.",
    }
    return taxonomy


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/taxonomy.json"
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    updated = expand_taxonomy(taxonomy)
    snapshot = root / "configs/taxonomy_ceasa_v4_fragmento_sobreposto_fixador_solto.json"
    preserve_previous_version(root, path, snapshot)
    write_json(path, updated)
    dataset_path = root / "configs/dataset.yaml"
    content = dataset_path.read_text(encoding="utf-8")
    dataset_path.write_text(content.rstrip() + "\n  12: reparo_rufo\n", encoding="utf-8")
    print("Taxonomia v5 salva: 13 classes ativas; snapshot v4 preservado.")


if __name__ == "__main__":
    main()
