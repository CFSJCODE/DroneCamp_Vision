"""Extração de imagens incorporadas no PDF com origem e revisão pendente."""

from pathlib import Path
from hashlib import sha256
import csv

from .io import file_hash, write_json


def extract_report_images(pdf: Path, output: Path) -> dict:
    """Extraia material de referência; não crie labels ou positivos de treino."""
    from pypdf import PdfReader

    pdf = pdf.resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("Informe um PDF local existente.")
    if output.exists():
        raise ValueError("A pasta de saída já existe; escolha uma nova para preservar a revisão anterior.")
    output.mkdir(parents=True)
    reader = PdfReader(pdf)
    records, hashes = [], {}
    for page_number, page in enumerate(reader.pages, start=1):
        for index, embedded in enumerate(page.images, start=1):
            width, height = embedded.image.size
            # Filtra ícones pequenos; a seleção final de fotografias é manual.
            if width < 300 or height < 200:
                continue
            digest = sha256(embedded.data).hexdigest()
            suffix = Path(embedded.name).suffix.lower() or ".png"
            filename = f"p{page_number:03d}_img{index:02d}_{digest[:8]}{suffix}"
            (output / filename).write_bytes(embedded.data)
            duplicate = hashes.get(digest)
            hashes.setdefault(digest, filename)
            records.append({
                "filename": filename, "page": page_number, "width": width, "height": height,
                "sha256": digest, "duplicate_of": duplicate,
                "annotation_status": "nao_anotada", "training_approved": False,
                "review_note": "Verificar marcações, compressão, contexto e cenas repetidas antes de anotar.",
            })
    manifest = {
        "source_pdf": str(pdf), "source_pdf_sha256": file_hash(pdf),
        "purpose": "referencia_e_demonstracao", "training_dataset": False,
        "count": len(records), "unique_binary_images": len(hashes), "images": records,
        "note": "Páginas e proximidade textual não determinam automaticamente a classe ou a caixa. Desenhos vetoriais do PDF podem não acompanhar a imagem extraída; marcações gravadas nos pixels permanecem.",
    }
    write_json(output / "manifest.json", manifest)
    with (output / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "page", "candidate_classes", "review_status", "notes"])
        for item in records:
            writer.writerow([item["filename"], item["page"], "", "pendente", ""])
    return manifest
