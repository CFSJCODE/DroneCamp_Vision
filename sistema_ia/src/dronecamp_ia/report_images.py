"""Extração das fotos incorporadas no PDF do laudo (comando ``extract-report``).

Função no projeto: foi o ponto de partida do acervo. Extrai as fotografias do
laudo do CEASA para ``data/reference/ceasa/`` com página de origem e SHA-256.

O que faz (``extract_report_images``): percorre as páginas, ignora ícones
pequenos, salva cada imagem, marca duplicatas e grava ``manifest.json`` e
``review_queue.csv``. Não cria labels nem aprova nada para treino.

Quando mexer: para mudar o tamanho mínimo das imagens extraídas (300×200 px).

O que o arquivo contém: só a função pública ``extract_report_images``.

Quem chama: cli.py, comando ``python -m dronecamp_ia extract-report --pdf <arquivo.pdf>
--output <pasta nova>``.

Disco:
    - Lê: o PDF informado em ``--pdf`` (com ``pypdf``; as dimensões vêm do Pillow).
    - Grava, numa pasta NOVA (``--output``):
        * ``p<página>_img<n>_<hash8>.<ext>``: cada imagem extraída, byte a byte como estava no PDF;
        * ``manifest.json``: origem (PDF e seu SHA-256) e a lista de imagens com hash e duplicatas;
        * ``review_queue.csv``: fila de anotação (UTF-8 com BOM, para abrir no Excel).

Onde mexer:
    - Tamanho mínimo das imagens: o ``if width < 300 or height < 200`` no laço de páginas.
    - Campo novo por imagem no manifesto: o dicionário ``records.append({...})``.
    - Coluna nova na fila de anotação: o cabeçalho ``writer.writerow([...])`` e a linha de
      cada imagem logo abaixo, na mesma posição.

ATENCAO: nada aqui vira dado de treino. ``training_approved`` sai sempre False e
``training_dataset`` sempre False; classes e caixas são definidas depois, na revisão humana.
"""

from pathlib import Path
from hashlib import sha256
import csv

from .io import file_hash, write_json


# ENTRADA: comando ``extract-report`` da CLI (cli.py).
def extract_report_images(pdf: Path, output: Path) -> dict:
    """Extraia material de referência; não crie labels ou positivos de treino.

    Percorre todas as páginas do PDF, salva cada imagem incorporada que tenha pelo menos
    300×200 px, registra página, tamanho e SHA-256 e marca as repetidas (mesmo conteúdo
    binário) com ``duplicate_of``.

    Parâmetros:
        pdf: caminho do PDF do laudo de referência (ex.: ``CEASA Minas - Pavilão A - Laudo.pdf``).
        output: pasta de saída; precisa NÃO existir (é criada aqui, com as pastas-mãe).

    Retorna:
        dict do ``manifest.json``: ``source_pdf``, ``source_pdf_sha256``, ``purpose``,
        ``training_dataset`` (False), ``count`` (imagens salvas), ``unique_binary_images``
        (hashes distintos), ``images`` (lista de registros por imagem) e ``note``.

    Erros:
        ValueError("Informe um PDF local existente.") se o arquivo não existir ou não terminar em .pdf;
        ValueError("A pasta de saída já existe; ...") se ``output`` já existir.
        ImportError se ``pypdf`` não estiver instalado (o import é feito só aqui dentro).

    Chamada por: cli.py (``extract-report``).
    """
    # Import tardio: pypdf só é necessário para este comando.
    from pypdf import PdfReader

    # 0. Validações: PDF existente e pasta de saída nova.
    pdf = pdf.resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError("Informe um PDF local existente.")
    # ATENCAO: nunca sobrescreve uma extração anterior (ela pode já ter revisão em andamento).
    if output.exists():
        raise ValueError("A pasta de saída já existe; escolha uma nova para preservar a revisão anterior.")
    output.mkdir(parents=True)
    # 1. Cada imagem de cada página: salva com nome p<página>_img<n>_<hash>.
    reader = PdfReader(pdf)
    # records: um dict por imagem salva; hashes: SHA-256 → nome do PRIMEIRO arquivo com esse conteúdo.
    records, hashes = [], {}
    # Páginas e imagens numeradas a partir de 1 (como aparecem para quem lê o PDF).
    for page_number, page in enumerate(reader.pages, start=1):
        for index, embedded in enumerate(page.images, start=1):
            # Dimensões em pixels da imagem decodificada (Pillow).
            width, height = embedded.image.size
            # Filtra ícones pequenos; a seleção final de fotografias é manual.
            # EXTENSAO: tamanho mínimo das imagens extraídas (largura 300 px, altura 200 px).
            if width < 300 or height < 200:
                continue
            # Hash dos bytes como estão no PDF (identifica duplicatas exatas).
            digest = sha256(embedded.data).hexdigest()
            # Extensão do nome interno da imagem no PDF; sem extensão, assume .png.
            suffix = Path(embedded.name).suffix.lower() or ".png"
            # Ex.: p012_img03_1a2b3c4d.jpg (página com 3 dígitos, índice com 2, 8 primeiros do hash).
            filename = f"p{page_number:03d}_img{index:02d}_{digest[:8]}{suffix}"
            (output / filename).write_bytes(embedded.data)
            # Duplicata: se o hash já apareceu, aponta para o primeiro arquivo; o arquivo repetido
            # é salvo mesmo assim (a página de origem também importa para a revisão).
            duplicate = hashes.get(digest)
            hashes.setdefault(digest, filename)
            # EXTENSAO: campo novo por imagem no manifest.json entra neste dicionário.
            # ATENCAO: ``annotation_status`` começa "nao_anotada" e ``training_approved`` False;
            # a extração nunca aprova imagem para treino.
            records.append({
                "filename": filename, "page": page_number, "width": width, "height": height,
                "sha256": digest, "duplicate_of": duplicate,
                "annotation_status": "nao_anotada", "training_approved": False,
                "review_note": "Verificar marcações, compressão, contexto e cenas repetidas antes de anotar.",
            })
    # 2. manifest.json (origem e hashes) e review_queue.csv (fila de anotação).
    # ``purpose`` e ``training_dataset`` deixam claro que é material de referência, não de treino.
    manifest = {
        "source_pdf": str(pdf), "source_pdf_sha256": file_hash(pdf),
        "purpose": "referencia_e_demonstracao", "training_dataset": False,
        "count": len(records), "unique_binary_images": len(hashes), "images": records,
        "note": "Páginas e proximidade textual não determinam automaticamente a classe ou a caixa. Desenhos vetoriais do PDF podem não acompanhar a imagem extraída; marcações gravadas nos pixels permanecem.",
    }
    write_json(output / "manifest.json", manifest)
    # CSV em UTF-8 com BOM (``utf-8-sig``) para o Excel reconhecer os acentos; ``newline=""``
    # evita linhas em branco extras no Windows.
    with (output / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        # EXTENSAO: coluna nova na fila = nome aqui e valor na mesma posição da linha abaixo.
        writer.writerow(["filename", "page", "candidate_classes", "review_status", "notes"])
        # Uma linha por imagem salva, com classes vazias e status "pendente" (preenchidos à mão).
        for item in records:
            writer.writerow([item["filename"], item["page"], "", "pendente", ""])
    return manifest
