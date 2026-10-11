"""Exportações comerciais: pacote de evidências (ZIP) e laudo preliminar.

Dados sintéticos (três fotos de 120×80 px): confere que o zip traz as entradas
esperadas, que decisões da página sobrepõem o registro, que filtros e redução
funcionam, que foto alterada é recusada, que o laudo nunca preenche gravidade,
diagnóstico ou recomendação, que sugestões pendentes só aparecem no anexo e que
nada é gravado fora de ``output_dir``.
"""

from copy import deepcopy
from pathlib import Path
import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageColor

from dronecamp_ia.config import ProjectConfig
from dronecamp_ia.exports import (
    OCCURRENCE_COLUMNS, PENDING_TEXT, build_evidence_package, build_report, draw_boxes, occurrences_csv, resolve_decisions,
)
from dronecamp_ia.io import file_hash
from dronecamp_ia.laudo_abnt import DEFAULT_TITLE, table_of_contents
from dronecamp_ia.review_render import COLORS

HAS_REPORTLAB = importlib.util.find_spec("reportlab") is not None


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class ExportsFixture(unittest.TestCase):
    """Taxonomia com 2 classes; fotos: positiva no registro, negativa e ambígua."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="dronecamp exportações á ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.taxonomy_path = self.root / "configs" / "taxonomy.json"
        write_json(self.taxonomy_path, {"version": "fixture", "classes": [
            {"id": 0, "slug": "telha_quebrada", "label": "Telha quebrada", "phase": 1,
             "criterion": "Ruptura visível do corpo da telha.", "reference_severity": "gravissima"},
            {"id": 1, "slug": "residuos_calha", "label": "Resíduos em calha", "phase": 1,
             "criterion": "Detritos no canal da calha.", "reference_severity": "media"}]})
        self.config = ProjectConfig(root=self.root, model="fixture.pt", requested_model="fixture.pt",
                                    taxonomy_path=self.taxonomy_path, dataset_path=self.root / "dataset.yaml",
                                    device="cpu", prediction={}, training={})
        self.directory = self.root / "data" / "reviews" / "fixture_v1"
        self.photos = []
        for name, color in (("foto_a", (90, 90, 90)), ("foto_b", (120, 100, 80)), ("foto_c", (60, 80, 120))):
            path = self.root / "data" / "reference" / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (120, 80), color=color).save(path)
            self.photos.append(path)
        self.box = {"class_id": 0, "bbox_xyxy": [20, 10, 60, 50], "note": "Sintética."}
        statuses = (("positive", [deepcopy(self.box)]), ("negative", []), ("ambiguous", []))
        self.registry = {"schema_version": 1, "version": "fixture_v1", "taxonomy_sha256": file_hash(self.taxonomy_path),
                         "taxonomy_version": "fixture", "images": [
                             {"image_sha256": file_hash(path), "filename": path.name, "source_path": str(path), "width": 120, "height": 80,
                              "pages": [index + 1], "status": status, "boxes": boxes, "human_approved": False,
                              "technical_status": "pending_human_review", "visual_review_status": "second_pass_ai",
                              "building_group": "galpao_b", "scene_group": f"cena_{index}", "notes": "Proposta sintética.", "severity": None}
                             for index, (path, (status, boxes)) in enumerate(zip(self.photos, statuses))]}
        self.registry_path = self.directory / "registry.json"
        write_json(self.registry_path, self.registry)
        self.sha = [item["image_sha256"] for item in self.registry["images"]]
        self.output = self.root / "saidas"
        # Sugestões do modelo para a foto negativa: uma aceita, uma descartada e uma pendente.
        self.suggestions = {self.sha[1]: [
            {"id": "s001", "class_id": 1, "bbox_xyxy": [10, 10, 30, 30], "confidence": 0.4},
            {"id": "s002", "class_id": 0, "bbox_xyxy": [50, 20, 90, 60], "confidence": 0.2},
            {"id": "s003", "class_id": 1, "bbox_xyxy": [5, 5, 15, 15], "confidence": 0.3}]}
        self.decision = {self.sha[1]: {"status": "positive", "notes": "Conferida.", "confirmed_complete": True,
                                       "boxes": [{"class_id": 1, "bbox_xyxy": [10, 10, 30, 30], "note": "Sugestão aceita."}],
                                       "dismissed_suggestions": ["s002"]}}

    def snapshot(self) -> dict:
        """Tudo que existe fora de output_dir (para provar que nada mais foi tocado)."""
        return {path: path.read_bytes() for path in self.root.rglob("*")
                if path.is_file() and self.output not in path.parents}

    def csv_rows(self, text: str) -> list[list[str]]:
        import csv
        return list(csv.reader(io.StringIO(text.lstrip("﻿"))))


class DrawBoxesTests(ExportsFixture):
    def test_draws_class_colored_outline_without_mutating_input(self) -> None:
        image = Image.new("RGB", (120, 80), (90, 90, 90))
        before = image.tobytes()
        drawn = draw_boxes(image, [deepcopy(self.box)], {0: "Telha quebrada", 1: "Resíduos em calha"}, COLORS)
        self.assertEqual(image.tobytes(), before)
        self.assertEqual(drawn.size, (120, 80))
        self.assertEqual(drawn.getpixel((20, 40)), ImageColor.getrgb(COLORS[0]), "Borda esquerda na cor da classe.")
        self.assertEqual(drawn.getpixel((40, 30)), (90, 90, 90), "Interior da caixa fica intacto.")

    def test_scale_maps_original_coordinates_onto_reduced_image(self) -> None:
        small = Image.new("RGB", (60, 40), (90, 90, 90))
        drawn = draw_boxes(small, [deepcopy(self.box)], {0: "Telha quebrada"}, COLORS, scale=0.5)
        self.assertEqual(drawn.getpixel((10, 20)), ImageColor.getrgb(COLORS[0]), "Caixa [20,10,60,50] vira [10,5,30,25].")
        self.assertEqual(drawn.getpixel((20, 21)), (90, 90, 90), "Interior (abaixo da placa do rótulo) intacto.")
        self.assertEqual(drawn.getpixel((45, 30)), (90, 90, 90), "Nada desenhado nas coordenadas originais.")

    def test_suggestions_are_dashed_in_class_color(self) -> None:
        image = Image.new("RGB", (400, 300), (90, 90, 90))
        suggestion = {"class_id": 1, "bbox_xyxy": [50, 50, 350, 250], "confidence": 0.42}
        drawn = draw_boxes(image, [], {0: "Telha quebrada", 1: "Resíduos em calha"}, COLORS, suggestions=[suggestion])
        edge = [drawn.getpixel((x, 50)) for x in range(50, 350)]
        color = ImageColor.getrgb(COLORS[1])
        self.assertIn(color, edge, "Segmentos na cor da classe.")
        self.assertIn((90, 90, 90), edge, "Lacunas do tracejado.")
        with self.assertRaises(ValueError):
            draw_boxes(image, [], {}, COLORS, scale=0)


class ResolveDecisionsTests(ExportsFixture):
    def test_registry_is_used_when_there_is_no_decision(self) -> None:
        resolved = resolve_decisions(self.registry, None, 2)
        self.assertEqual([entry["source"] for entry in resolved], ["registro"] * 3)
        self.assertEqual([entry["status"] for entry in resolved], ["positive", "negative", "ambiguous"])
        self.assertEqual(resolved[0]["boxes"], [self.box])
        self.assertEqual(resolved[0]["pending_suggestions"], 0)

    def test_decision_overrides_registry_and_counts_pending_suggestions(self) -> None:
        resolved = resolve_decisions(self.registry, self.decision, 2, suggestions=self.suggestions)
        second = resolved[1]
        self.assertEqual((second["source"], second["status"]), ("revisao_humana", "positive"))
        self.assertEqual(second["boxes"][0]["class_id"], 1)
        self.assertEqual(second["notes"], "Conferida.")
        self.assertEqual(second["pending_suggestions"], 1, "Aceita e descartada não contam; sobra s003.")
        self.assertTrue(second["confirmed_complete"])
        without = resolve_decisions(self.registry, None, 2, suggestions=self.suggestions)
        self.assertEqual(without[1]["pending_suggestions"], 3)

    def test_invalid_status_boxes_and_unknown_images_are_rejected(self) -> None:
        for decision in ({"status": "approved", "boxes": []}, {"status": "positive", "boxes": []},
                         {"status": "negative", "boxes": [deepcopy(self.box)]},
                         {"status": "positive", "boxes": [{"class_id": 5, "bbox_xyxy": [1, 1, 5, 5]}]},
                         {"status": "positive", "boxes": [{"class_id": 0, "bbox_xyxy": [0, 0, 500, 5]}]}, "positive"):
            with self.subTest(decision=decision):
                with self.assertRaises(ValueError):
                    resolve_decisions(self.registry, {self.sha[0]: decision}, 2)
        with self.assertRaises(ValueError):
            resolve_decisions(self.registry, {"0" * 64: {"status": "negative", "boxes": []}}, 2)
        with self.assertRaises(ValueError):
            resolve_decisions(self.registry, None, 2, suggestions={self.sha[0]: [{"class_id": 9, "bbox_xyxy": [1, 1, 2, 2], "confidence": 0.5}]})

    def test_pending_draft_keeps_boxes_without_counting_as_positive(self) -> None:
        resolved = resolve_decisions(self.registry, {self.sha[2]: {"status": "pending", "boxes": [deepcopy(self.box)]}}, 2)
        self.assertEqual(resolved[2]["status"], "pending")
        self.assertEqual(len(resolved[2]["boxes"]), 1)
        self.assertFalse(resolved[2]["confirmed_complete"])


class EvidencePackageTests(ExportsFixture):
    def test_zip_has_expected_entries_hashes_and_writes_nothing_else(self) -> None:
        before = self.snapshot()
        result = build_evidence_package(self.config, self.registry_path, self.output)
        self.assertEqual(self.snapshot(), before, "Nada fora de output_dir.")
        path = Path(result["zip"])
        self.assertEqual(path.parent, self.output)
        self.assertRegex(path.name, r"^\d{8}T\d{6}Z_evidencias_fixture_v1\.zip$")
        self.assertEqual((result["images"], result["boxes"], result["bytes"], result["sha256"]),
                         (3, 1, path.stat().st_size, file_hash(path)))
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(archive.namelist(), ["evidencias/001_foto_a.jpg", "evidencias/002_foto_b.jpg", "evidencias/003_foto_c.jpg",
                                                  "ocorrencias.csv", "fotos.csv", "LEIA-ME.txt", "manifesto.json"])
            self.assertIsNone(archive.testzip())
            manifest = json.loads(archive.read("manifesto.json"))
            self.assertEqual(manifest["registry_sha256"], file_hash(self.registry_path))
            self.assertEqual(manifest["taxonomy_sha256"], file_hash(self.taxonomy_path))
            self.assertEqual(manifest["version"], "fixture_v1")
            self.assertEqual(manifest["counts"]["by_status"], {"positive": 1, "negative": 1, "ambiguous": 1, "excluded": 0, "pending": 0})
            for name in archive.namelist():
                if name != "manifesto.json":
                    from hashlib import sha256
                    self.assertEqual(manifest["files"][name]["sha256"], sha256(archive.read(name)).hexdigest(), name)
            with archive.open("evidencias/001_foto_a.jpg") as stream:
                self.assertEqual(Image.open(stream).size, (120, 80), "max_side 2048 não reduz foto pequena.")
            readme = archive.read("LEIA-ME.txt").decode("utf-8")
            self.assertIn("NÃO é laudo técnico", readme)
            self.assertIn("gravidade", readme)
            occurrences = archive.read("ocorrencias.csv")
            self.assertTrue(occurrences.startswith("﻿".encode("utf-8")), "UTF-8 com BOM para o Excel.")
            rows = self.csv_rows(occurrences.decode("utf-8"))
            self.assertEqual(rows[0], OCCURRENCE_COLUMNS)
            self.assertEqual(rows[1], ["1", self.sha[0], "foto_a.png", "positive", "registro", "1", "0", "Telha quebrada", "20", "10", "60", "50", "Sintética."])
            photos = self.csv_rows(archive.read("fotos.csv").decode("utf-8"))
            self.assertEqual(len(photos), 4)
            self.assertEqual(photos[2][8:12], ["negative", "registro", "0", "0"])

    def test_decisions_override_registry_with_human_origin(self) -> None:
        result = build_evidence_package(self.config, self.registry_path, self.output, decisions=self.decision, suggestions=self.suggestions)
        self.assertEqual(result["boxes"], 2)
        with zipfile.ZipFile(result["zip"]) as archive:
            rows = self.csv_rows(archive.read("ocorrencias.csv").decode("utf-8"))
            self.assertEqual([row[4] for row in rows[1:]], ["registro", "revisao_humana"])
            self.assertEqual(rows[2][5:8], ["1", "1", "Resíduos em calha"])
            photos = self.csv_rows(archive.read("fotos.csv").decode("utf-8"))
            self.assertEqual(photos[2][8:13], ["positive", "revisao_humana", "1", "1", "Conferida."])
            self.assertEqual(json.loads(archive.read("manifesto.json"))["counts"]["pending_suggestions"], 1)

    def test_only_filters_and_keeps_registry_numbering(self) -> None:
        positive = build_evidence_package(self.config, self.registry_path, self.output, only="positive")
        with zipfile.ZipFile(positive["zip"]) as archive:
            self.assertEqual([name for name in archive.namelist() if name.endswith(".jpg")], ["evidencias/001_foto_a.jpg"])
        decided = build_evidence_package(self.config, self.registry_path, self.output, only="decided", decisions=self.decision)
        with zipfile.ZipFile(decided["zip"]) as archive:
            self.assertEqual([name for name in archive.namelist() if name.endswith(".jpg")], ["evidencias/002_foto_b.jpg"])
            rows = self.csv_rows(archive.read("fotos.csv").decode("utf-8"))
            self.assertEqual(rows[1][0], "2", "Numeração segue a posição no registro.")
        with self.assertRaises(ValueError):
            build_evidence_package(self.config, self.registry_path, self.output, only="tudo")

    def test_max_side_reduces_and_none_keeps_original(self) -> None:
        reduced = build_evidence_package(self.config, self.registry_path, self.output, max_side=60, include_suggestions=True,
                                         suggestions=self.suggestions)
        with zipfile.ZipFile(reduced["zip"]) as archive:
            with archive.open("evidencias/002_foto_b.jpg") as stream:
                self.assertEqual(Image.open(stream).size, (60, 40))
            # Foto reduzida à metade, sem caixas e com as três sugestões pendentes tracejadas (scale=0.5).
            with Image.open(self.photos[1]) as photo:
                small = photo.convert("RGB").resize((60, 40), Image.Resampling.LANCZOS)
            expected = draw_boxes(small, [], {0: "Telha quebrada", 1: "Resíduos em calha"}, COLORS,
                                  suggestions=self.suggestions[self.sha[1]], scale=0.5)
            buffer = io.BytesIO()
            expected.save(buffer, "JPEG", quality=92)
            self.assertEqual(archive.read("evidencias/002_foto_b.jpg"), buffer.getvalue())
            manifest = json.loads(archive.read("manifesto.json"))
            self.assertEqual((manifest["files"]["evidencias/002_foto_b.jpg"]["width"], manifest["parameters"]["max_side"]), (60, 60))
        original = build_evidence_package(self.config, self.registry_path, self.output, max_side=None)
        with zipfile.ZipFile(original["zip"]) as archive:
            with archive.open("evidencias/001_foto_a.jpg") as stream:
                self.assertEqual(Image.open(stream).size, (120, 80))
        self.assertNotEqual(reduced["zip"], original["zip"], "Cada exportação é um arquivo novo.")

    def test_changed_photo_is_rejected_and_leaves_no_partial_zip(self) -> None:
        Image.new("RGB", (120, 80), color="white").save(self.photos[2])
        with self.assertRaises(ValueError):
            build_evidence_package(self.config, self.registry_path, self.output)
        self.assertEqual(list(self.output.glob("*.zip")), [])

    def test_changed_taxonomy_and_invalid_decisions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_evidence_package(self.config, self.registry_path, self.output, decisions={self.sha[0]: {"status": "ok", "boxes": []}})
        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        taxonomy["version"] = "outra"
        write_json(self.taxonomy_path, taxonomy)
        with self.assertRaises(ValueError):
            build_evidence_package(self.config, self.registry_path, self.output)
        self.assertFalse(self.output.exists() and any(self.output.iterdir()))


class OccurrencesCsvTests(ExportsFixture):
    def test_csv_text_matches_registry_and_decisions(self) -> None:
        rows = self.csv_rows(occurrences_csv(self.config, self.registry_path))
        self.assertEqual(rows[0], OCCURRENCE_COLUMNS)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][:5], ["1", self.sha[0], "foto_a.png", "positive", "registro"])
        decided = self.csv_rows(occurrences_csv(self.config, self.registry_path, decisions=self.decision))
        self.assertEqual(len(decided), 3)
        self.assertEqual(decided[2][:6], ["2", self.sha[1], "foto_b.png", "positive", "revisao_humana", "1"])
        self.assertEqual(decided[2][8:13], ["10", "10", "30", "30", "Sugestão aceita."])
        # O texto é o mesmo do zip (sem o BOM, que é detalhe da gravação).
        with zipfile.ZipFile(build_evidence_package(self.config, self.registry_path, self.output, decisions=self.decision)["zip"]) as archive:
            self.assertEqual(archive.read("ocorrencias.csv").decode("utf-8-sig"), occurrences_csv(self.config, self.registry_path, decisions=self.decision))


class ReportTests(ExportsFixture):
    def build(self, **options) -> tuple[dict, str, dict]:
        options.setdefault("pdf", False)
        result = build_report(self.config, self.registry_path, self.output, **options)
        html = Path(result["html"]).read_text(encoding="utf-8")
        data = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
        return result, html, data

    def test_report_files_manifest_and_nothing_outside_output_dir(self) -> None:
        before = self.snapshot()
        result, html, data = self.build(reviewer="rev01", client="Cliente X", site="Galpão B", inspector="Insp. Y", notes="Obs. <teste>")
        self.assertEqual(self.snapshot(), before)
        directory = Path(result["directory"])
        self.assertEqual(directory.parent, self.output)
        self.assertRegex(directory.name, r"^\d{8}T\d{6}Z_laudo_fixture_v1$")
        self.assertEqual(sorted(path.name for path in directory.iterdir()), ["laudo.html", "laudo.json", "manifesto.json"])
        self.assertIsNone(result["pdf"])
        self.assertIn("pdf=False", result["pdf_reason"])
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["files"]), {"laudo.html", "laudo.json"})
        self.assertEqual(manifest["files"]["laudo.html"]["sha256"], file_hash(directory / "laudo.html"))
        self.assertEqual(manifest["registry_sha256"], file_hash(self.registry_path))
        self.assertEqual((result["images"], result["occurrences"], result["pending_suggestions"]), (1, 1, 0))
        # Cabeçalho: título padrão, solicitante (cliente), local, revisores, responsável, hash, texto preliminar e critério.
        self.assertIn(DEFAULT_TITLE, html)
        for text in ("Cliente X", "Galpão B", "rev01", "Insp. Y", "Obs. &lt;teste&gt;", "galpao_b", file_hash(self.registry_path),
                     "preliminar, sujeito a validação por responsável técnico", "Ruptura visível do corpo da telha.",
                     "não aplicada automaticamente", "@page", "data:image/jpeg;base64,", "4.4 Limitações"):
            self.assertIn(text, html, text)
        self.assertNotIn("<link", html, "HTML autossuficiente.")
        self.assertEqual(data["header"]["reviewers"], ["rev01"])
        self.assertEqual((data["details"]["requester"], data["details"]["responsible_name"]), ("Cliente X", "Insp. Y"))
        self.assertEqual(data["summary"]["by_class"][0]["reference_severity"], "gravissima")
        self.assertEqual(data["summary"]["positive"], 1)

    def test_abnt_structure_sections_norms_and_checklist_items(self) -> None:
        _, html, data = self.build(decisions=self.decision, suggestions=self.suggestions)
        # Elementos pré-textuais, seções numeradas, referências e anexos na ordem da NBR 10719.
        anchors = ["capa", "folha-rosto", "sumario", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "referencias",
                   "anexo-a", "anexo-b", "anexo-c", "anexo-d", "anexo-e"]
        positions = [html.index(f'<section id="{anchor}"') for anchor in anchors]
        self.assertEqual(positions, sorted(positions))
        for anchor, _, _ in table_of_contents(data):
            self.assertIn(f'href="#{anchor}"', html)
            self.assertIn(f'id="{anchor}"', html)
        for norm in ("NBR 16747", "NBR 5674", "NBR 7196", "NBR 10844", "NBR 10719", "RBAC-E nº 94"):
            self.assertIn(norm, html.split('<section id="s3">')[1].split('<section id="s4">')[0], norm)
        self.assertIn("ASSOCIAÇÃO BRASILEIRA DE NORMAS TÉCNICAS. NBR 16747: Inspeção predial", html)
        # Um item por categoria confirmada, com a pergunta do laudo CEASA e figuras numeradas na ordem.
        self.assertEqual([(item["label"], item["question"]) for item in data["items"]],
                         [("Telha quebrada", "Existem telhas quebradas?"), ("Resíduos em calha", "Há acúmulo de resíduos nas calhas?")])
        self.assertEqual([(figure["number"], figure["numero_foto"], figure["item"]) for figure in data["figures"]], [(1, 1, 1), (2, 2, 2)])
        self.assertEqual(html.count('<figure id="figura-'), 2)
        self.assertIn("Figura 1</b> – Telha quebrada (fotografia 001)", html)
        self.assertEqual(data["items_without_occurrence"], [])
        self.assertTrue(data["manual_checklist"])
        self.assertIn("não avaliado pelo sistema", html)
        self.assertEqual([record["figura"] for record in data["occurrences"]], [1, 2])
        # Dados do imóvel não informados não são estimados.
        self.assertIn("Área total estimada</th><td>não informado", html)

    def test_details_are_shown_as_given_and_unknown_keys_are_rejected(self) -> None:
        details = {"inspection_date": "06/08/2025", "roof_type": "Telhas de fibrocimento tipo canaleta", "area": "6040 m²",
                   "responsible_name": "Eng. Fulano", "responsible_registry": "CREA-MG 123", "art_number": "MG2025000", "equipment": "DJI Air 3"}
        _, html, data = self.build(details=details, inspector="ignorado")
        for value in details.values():
            self.assertIn(value, html)
        self.assertEqual(data["technical_responsibility"]["name"], "Eng. Fulano")
        self.assertNotIn("ignorado", html)
        with self.assertRaises(ValueError):
            build_report(self.config, self.registry_path, self.output, pdf=False, details={"area_calculada": "1"})

    def test_evaluation_fields_are_always_pending(self) -> None:
        result, html, data = self.build(decisions=self.decision, suggestions=self.suggestions)
        self.assertEqual(result["images"], 2)
        self.assertEqual(html.count('<article class="item"'), 2)
        for field in ("Conformidade", "Risco (gravidade)", "Diagnóstico", "Recomendação", "Prioridade (NBR 16747)"):
            values = re.findall(rf"<tr><th>{re.escape(field)}</th><td class=\"pendente\">(.*?)</td>", html)
            self.assertEqual(values, [PENDING_TEXT] * 2, field)
        self.assertNotIn("gravissima</td></tr><tr><th>Diagn", html, "Gravidade de referência nunca vira avaliação.")
        for item in data["items"]:
            self.assertEqual({item[key] for key in ("conformidade", "risco", "diagnostico", "recomendacao", "prioridade")}, {PENDING_TEXT})
        for occurrence in data["occurrences"]:
            self.assertEqual((occurrence["gravidade"], occurrence["diagnostico"], occurrence["recomendacao"]), (PENDING_TEXT,) * 3)
        self.assertEqual(len(data["occurrences"]), 2)
        self.assertEqual([occurrence["origem"] for occurrence in data["occurrences"]], ["registro", "revisao_humana"])
        body = html.split('<section id="s6">')[1].split('<section id="s7">')[0]
        for forbidden in ("gravissima", "Não conformidade gravíssima", "Substituir as telhas"):
            self.assertNotIn(forbidden, body, "Texto do laudo-modelo e gravidade de referência ficam só no Anexo E.")
        annex = html.split('<section id="anexo-e">')[1]
        self.assertIn("Não conformidade gravíssima", annex)
        self.assertIn("não foram aplicadas", annex)
        self.assertEqual(data["conclusion"]["technical_conclusion"], PENDING_TEXT)

    def test_pending_suggestions_only_in_annex_b(self) -> None:
        result, html, data = self.build(decisions=self.decision, suggestions=self.suggestions)
        self.assertEqual(result["pending_suggestions"], 1)
        body = html.split('<section id="s6">')[1].split('<section id="s7">')[0]
        self.assertNotIn("sugestões", body.lower(), "Só a nota da caixa aceita cita sugestão; nada de pendências no corpo.")
        self.assertNotIn("IA 1", body)
        # A figura da foto_b é exatamente o desenho das caixas da decisão, sem a sugestão pendente.
        import base64
        encoded = re.findall(r'data-foto="002".*?src="data:image/jpeg;base64,([^"]+)"', body, re.S)[0]
        with Image.open(self.photos[1]) as photo:
            expected = draw_boxes(photo, self.decision[self.sha[1]]["boxes"], {0: "Telha quebrada", 1: "Resíduos em calha"}, COLORS)
        buffer = io.BytesIO()
        expected.save(buffer, "JPEG", quality=82)
        self.assertEqual(base64.b64decode(encoded), buffer.getvalue())
        annex = html.split('<section id="anexo-b">')[1].split('<section id="anexo-c">')[0]
        self.assertIn("não constituem ocorrências", annex)
        self.assertIn("Resíduos em calha", annex)
        self.assertEqual(data["pending_suggestions"], {"total": 1, "by_class": [{"class_id": 1, "label": "Resíduos em calha", "count": 1, "images": 1}],
                                                       "note": data["pending_suggestions"]["note"]})
        self.assertEqual(data["summary"]["confirmed_boxes"], 2, "Sugestões pendentes não entram nas contagens de ocorrências.")

    def test_annex_a_lists_non_positive_photos_and_all_photos_can_be_shown(self) -> None:
        _, html, data = self.build()
        annex = html.split('<section id="anexo-a">')[1].split('<section id="anexo-b">')[0]
        self.assertIn("foto_b.png", annex)
        self.assertIn("foto_c.png", annex)
        self.assertNotIn("foto_a.png", annex)
        self.assertEqual([image["included_in_report"] for image in data["images"]], [True, False, False])
        self.assertEqual([row["label"] for row in data["items_without_occurrence"]], ["Resíduos em calha"])
        result, html, data = self.build(only_positive_images=False, max_side=60)
        self.assertEqual((result["images"], html.count('<figure id="figura-')), (3, 3))
        annex = html.split('<section id="anexo-a">')[1].split('<section id="anexo-b">')[0]
        self.assertEqual(annex.count('<figure id="figura-'), 2, "Fotos sem ocorrência confirmada vão ao Anexo A.")
        self.assertEqual(data["images"][1]["status_label"], "negativa")

    def test_ai_notes_of_this_registry_become_pending_support(self) -> None:
        notes_path = self.directory / "ai_notes.json"
        write_json(notes_path, {"schema_version": 1, "kind": "vlm_notes", "registry_sha256": file_hash(self.registry_path),
                                "created_at": "2026-10-11T00:00:00+00:00", "settings": {"model_requested": "openrouter/free"},
                                "described": 1, "total_boxes": 1,
                                "agreement": {"human_boxes_described": 1, "agree": 1, "agreement_rate": 1.0,
                                              "per_class": {"telha_quebrada": {"boxes": 1, "agree": 1, "agreement_rate": 1.0}}},
                                "images": {self.sha[0]: [{"source": "humana", "class_id": 0, "bbox_xyxy": [20.2, 10, 60, 49.8],
                                                          "description": "Telha com fratura transversal visível.",
                                                          "model_used": "vlm-x:free", "class_guess": "telha_quebrada",
                                                          "severity": None, "status": "pendente"}]}})
        _, html, data = self.build()
        self.assertEqual(data["items"][0]["ai_support"][0]["text"], "Telha com fratura transversal visível.")
        self.assertIn("Subsídio de IA", html)
        self.assertIn("não constitui constatação técnica", html)
        self.assertIn("100%", html.split('<section id="anexo-c">')[1])
        self.assertTrue(data["ai_support"]["available"])
        self.assertEqual(data["items"][0]["conformidade"], PENDING_TEXT, "Descrição da IA nunca preenche a avaliação.")
        # Notas de outra versão do registro são ignoradas.
        stale = json.loads(notes_path.read_text(encoding="utf-8"))
        stale["registry_sha256"] = "0" * 64
        write_json(notes_path, stale)
        _, html, data = self.build()
        self.assertFalse(data["ai_support"]["available"])
        self.assertIn("outra versão", data["ai_support"]["note"])
        self.assertEqual(data["items"][0]["ai_support"], [])

    def test_suggestion_file_next_to_registry_is_used_only_for_this_version(self) -> None:
        suggestions_path = self.directory / "suggestions.json"
        write_json(suggestions_path, {"schema_version": 1, "registry_sha256": file_hash(self.registry_path), "model_label": "Modelo piloto (não validado)",
                                      "weights_sha256": "a" * 64, "conf": 0.15, "pilot": True, "images": self.suggestions})
        _, html, data = self.build()
        self.assertIn("Modelo piloto (não validado)", html)
        self.assertIn("a" * 64, html)
        self.assertEqual(data["pending_suggestions"]["total"], 3)
        stale = json.loads(suggestions_path.read_text(encoding="utf-8"))
        stale["registry_sha256"] = "0" * 64
        write_json(suggestions_path, stale)
        _, html, data = self.build()
        self.assertEqual(data["pending_suggestions"]["total"], 0)
        self.assertIn("outra versão", data["provenance"]["suggestion_note"])

    def test_changed_photo_leaves_no_partial_report(self) -> None:
        Image.new("RGB", (120, 80), color="white").save(self.photos[0])
        with self.assertRaises(ValueError):
            build_report(self.config, self.registry_path, self.output, pdf=False)
        self.assertEqual(list(self.output.iterdir()), [])

    @unittest.skipUnless(HAS_REPORTLAB, "reportlab não instalado (extra 'reports')")
    def test_pdf_is_generated_with_reportlab(self) -> None:
        result = build_report(self.config, self.registry_path, self.output, decisions=self.decision, suggestions=self.suggestions,
                              reviewer="rev01", client="Cliente X", pdf=True)
        self.assertIsNone(result["pdf_reason"])
        pdf = Path(result["pdf"])
        self.assertTrue(pdf.is_file())
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["files"]["laudo.pdf"]["sha256"], file_hash(pdf))
        if importlib.util.find_spec("pypdf") is not None:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            for expected in ("SUMÁRIO", "1 IDENTIFICAÇÃO DO LAUDO", "NBR 16747", "6.1 Telha quebrada", "REFERÊNCIAS", PENDING_TEXT):
                self.assertIn(expected, text, expected)


if __name__ == "__main__":
    unittest.main()
