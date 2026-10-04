"""Atualização pontual solicitada pelo usuário; preservar fontes antes de editar."""

import json
from pathlib import Path
import shutil

from dronecamp_ia.io import file_hash, write_json

root = Path(__file__).resolve().parents[1]
taxonomy_path = root / "configs/taxonomy.json"
snapshot = root / "configs/taxonomy_ceasa_v3_pedaco_telha_rufo_deslocado.json"
before_path = root / ".runtime/category_update_v4_before.json"
if snapshot.exists() or before_path.exists():
    raise SystemExit("Atualização já iniciada: preservar os arquivos existentes.")

taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
if taxonomy["version"] != "ceasa-v3-proposta-pedaco-telha-rufo-deslocado":
    raise SystemExit("Taxonomia de origem diferente da versão esperada.")

# A auditoria cobre as versões antigas inteiras, originais e pesos, sem gravar conteúdo.
preserved_paths = [taxonomy_path, root / "models/yolo26l.pt"]
for version in ("ceasa_v1", "ceasa_v3"):
    preserved_paths.extend(path for path in (root / "data/reviews" / version).rglob("*") if path.is_file())
preserved_paths.extend(path for path in (root / "data/reference/ceasa").rglob("*") if path.is_file())
write_json(before_path, {"files": {str(path.resolve()): file_hash(path) for path in preserved_paths}})
shutil.copy2(taxonomy_path, snapshot)

corroded = next(item for item in taxonomy["classes"] if item["slug"] == "fixador_telha_corroido")
corroded["id"] = 21  # Categoria futura sem labels ativos: liberar ID10 sem mudar IDs0–9.
loose = next(item for item in taxonomy["classes"] if item["slug"] == "fixador_telha_frouxo")
loose.update(
    label="Elemento de fixação de telha solto/frouxo", phase=1,
    reference_severity=None, previous_reference_severity="media",
    origin="categoria_fase2_ativada_por_solicitacao_do_usuario_em_2026-10-03",
    criterion="Elemento de fixação associado à telha com soltura, desprendimento, deslocamento ou folga visualmente identificável. Diferenciar fixador instalado corretamente, arruela prevista, sombra e objeto avulso sem vínculo identificável com a telha.",
    annotation_policy="Delimitar o fixador e incluir contexto suficiente para identificar a evidência de soltura. Cabeça elevada ou falta de contato aparente, isoladamente, pode ser dúvida: manter ambíguo quando não houver confirmação visual. Ausência completa permanece fixador_telha_ausente; corrosão não comprova folga.",
    physical_diagnosis="Não inferir torque, aperto, estabilidade mecânica, causa, estanqueidade, risco ou severidade pela fotografia.",
)
taxonomy["classes"].append({
    "id": 10, "slug": "pedaco_telha_sobreposto", "label": "Pedaço de telha sobreposto à telha",
    "phase": 1, "pages": [], "reference_severity": None,
    "origin": "categoria_adicionada_por_solicitacao_do_usuario_em_2026-10-03",
    "criterion": "Fragmento solto de telha visualmente reconhecível apoiado sobre uma telha instalada, com limites e superfície de apoio identificáveis. Diferenciar a sobreposição regular entre telhas inteiras, remendo aderido, sombra e ruptura ainda pertencente à telha instalada.",
    "annotation_policy": "Delimitar o fragmento, sem incluir toda a telha de apoio. Preferir esta classe ao ID8 quando o apoio sobre uma telha instalada estiver confirmado; não duplicar o mesmo objeto nas duas classes. Usar ID8 para fragmento reconhecível em outro local ou sem apoio identificável. Fragmento ou apoio incerto permanece ambíguo.",
    "physical_diagnosis": "Não inferir material, origem, ausência de travamento, aderência, risco ou severidade pelo rótulo visual.",
})
taxonomy["classes"].sort(key=lambda item: item["id"])
taxonomy["version"] = "ceasa-v4-proposta-fragmento-sobreposto-fixador-solto"
taxonomy["migration"] = {
    "from_version": "ceasa-v3-proposta-pedaco-telha-rufo-deslocado",
    "preserved_active_ids": list(range(10)), "new_active_class_ids": [10, 11],
    "catalog_id_mapping": {"10": 21}, "prior_catalog_id_mapping": {"8": 19, "9": 20},
    "promoted_catalog_class": {"id": 11, "slug": "fixador_telha_frouxo", "previous_phase": 2},
    "note": "IDs ativos0–9 preservados. Nova classe10 de fragmento sobre telha e classe11 promovida da fase2. Antigo ID10 de fixador corroído passa a21. Snapshots e revisões anteriores preservados; pesos e anotações não aprendem automaticamente as novas classes.",
}
write_json(taxonomy_path, taxonomy)
dataset_path = root / "configs/dataset.yaml"
dataset_text = dataset_path.read_text(encoding="utf-8")
dataset_path.write_text(dataset_text.rstrip() + "\n  10: pedaco_telha_sobreposto\n  11: fixador_telha_frouxo\n", encoding="utf-8")
print("Taxonomia v4: 12 classes ativas, 22 no catálogo. Snapshot e auditoria salvos.")
