"""Exportação para ONNX e conferência de paridade com o .pt.

Função no projeto: gera ``model.onnx`` para rodar fora do Python/Ultralytics
(ex.: ONNX Runtime com DirectML na APU) e prova que ele dá as mesmas caixas
que o checkpoint PyTorch. É o comando ``export``.

O que faz:
- ``export_onnx``: copia o ``.pt`` para ``runs/export_*``, exporta FP32 com
  entrada fixa e grava ``export.json``.
- ``verify_onnx_parity`` / ``compare_outputs``: roda ``.pt`` e ``.onnx`` nas
  mesmas fotos e pareia as caixas (IoU ≥ 0,9 ou ≤ 1 px; Δconfiança ≤ 0,02).

Criar um ONNX não certifica a qualidade do modelo; só a equivalência numérica.

Quando mexer: tolerâncias da paridade (constantes abaixo) ou opções de
exportação (tamanho, FP16/INT8, NMS embutido).

Conteúdo do arquivo (na ordem):
- Constantes ``PARITY_IOU``, ``PARITY_CONFIDENCE``, ``PARITY_PIXELS``.
- ``_iou(a, b)``: interseção sobre união de duas caixas xyxy.
- ``_boxes(model, image, config, conf)``: caixas de um modelo numa foto.
- ``compare_outputs(reference, candidate, conf)``: pareia as caixas do .pt e
  do .onnx e diz se são equivalentes.
- ``default_parity_images(config, limit)``: fotos padrão da paridade.
- ``_onnx_imgsz(path)``: tamanho de entrada gravado no ONNX.
- ``verify_onnx_parity(...)``: paridade em várias fotos (vai para parity.json).
- ``export_onnx(...)``: o comando inteiro.

Quem chama:
- ``cli.py``, comando ``python -m dronecamp_ia export --weights <.pt>
  [--parity-source <pasta>] [--demo]`` → ``export_onnx``.
- ``cycle.py`` e ``gpu_eval.py`` → ``export_onnx`` (este último com ``output``
  apontando para uma pasta própria, uma exportação por época).
- ``training.pilot_inference_config`` → ``_onnx_imgsz`` (modelo ONNX carregado
  para inferência usa o tamanho gravado no arquivo).

O que lê do disco: o checkpoint ``.pt``, ``configs/taxonomy.json`` e as fotos
de paridade (padrão: ``data/reference/ceasa``).

O que grava (pasta NOVA ``runs/export_<UTC>_<id>/`` ou ``export_demo_*``, ou
a pasta vazia passada em ``output``):
- ``model.pt``: cópia do checkpoint (o original nunca é tocado);
- ``model.onnx``: gerado pela Ultralytics ao lado da cópia;
- ``parity.json``: comparação foto a foto (só se houver onnxruntime e fotos);
- ``export.json``: resumo (hash do ONNX, parâmetros, resultado da paridade).

Onde mexer:
- Afrouxar/apertar a paridade: constantes ``PARITY_*`` (marcadores EXTENSAO).
- Exportar em FP16/INT8, com eixo dinâmico ou outro formato: ``export_onnx``,
  chamada ``model.export(...)`` (etapa 2) e os campos ``format``/``quantize``
  do ``export.json`` (etapa 3), que precisam refletir o que foi feito.
- Comparar em outro dispositivo (ex.: DirectML): ``_boxes`` usa ``device="cpu"``
  fixo; ``load_exported_detector`` (backend.py) escolhe o provedor do ONNX.
- Trocar as fotos padrão da paridade: ``default_parity_images``.
- Novo campo no export.json: dicionário do ``write_json(run / "export.json", ...)``.

ATENCAO: ``runtime_parity_validated`` só prova que .pt e .onnx dão as mesmas
caixas NA CPU deste computador. Não diz nada sobre a qualidade do modelo nem
sobre outro hardware; ``human_acceptance`` sai sempre "pendente".
"""

# ---------------------------------------------------------------------------
# Importações
# ---------------------------------------------------------------------------

# replace: cópia de um dataclass (ProjectConfig) trocando um campo, sem alterar o original.
from dataclasses import replace
from pathlib import Path
# importlib.util.find_spec: confere se onnx/onnxruntime estão instalados sem importá-los.
import importlib.util
import shutil

# Módulos do pacote: carregar .pt e .onnx, conferir classes, versões;
# taxonomia; extensões de foto; hash, pasta de execução e JSON.
from .backend import check_domain_names, load_detector, load_exported_detector, runtime_info
from .config import ProjectConfig, detection_names, load_taxonomy
from .dataset import IMAGE_EXTENSIONS
from .io import file_hash, make_run_directory, write_json

# ---------------------------------------------------------------------------
# Tolerâncias da paridade .pt × .onnx
# ---------------------------------------------------------------------------

# Tolerâncias para considerar duas caixas (.pt × .onnx) iguais.
# PARITY_IOU: IoU mínimo (0..1) entre as caixas de um par. Subir para 0.95
# torna a paridade mais exigente; baixar aceita desvios maiores de posição.
# EXTENSAO: tolerância de IoU da paridade: mude o valor aqui (vale para
# compare_outputs e é gravado em parity.json como "iou_required").
PARITY_IOU = 0.9
# PARITY_CONFIDENCE: diferença máxima de confiança (0..1) num par, e também a
# margem acima do limiar ``conf`` para uma caixa sem par contar como falha.
# EXTENSAO: tolerância de confiança: mude aqui (gravada como "confidence_tolerance").
PARITY_CONFIDENCE = 0.02
# Caixas finíssimas mudam muito de IoU com décimos de pixel; aceite até 1 px.
# PARITY_PIXELS: diferença máxima (em pixels da foto) em CADA coordenada para
# o par ser aceito mesmo com IoU baixo (gravada como "pixel_tolerance").
PARITY_PIXELS = 1.0


# ---------------------------------------------------------------------------
# Comparação de saídas: caixas do .pt contra caixas do .onnx.
# ---------------------------------------------------------------------------

def _iou(a: list[float], b: list[float]) -> float:
    """Interseção sobre união (IoU) de duas caixas.

    Parâmetros:
        a, b (list[float]): caixas ``[x1, y1, x2, y2]`` em pixels (x1 < x2,
            y1 < y2; origem no canto superior esquerdo).

    Retorna:
        float: 0.0 (sem sobreposição) a 1.0 (caixas idênticas); 0.0 se a união
        tiver área zero (caixas degeneradas).

    Chamada por: ``compare_outputs``.
    """
    # Largura e altura da interseção: da maior borda esquerda/superior até a
    # menor borda direita/inferior; max(0, ...) zera quando não se tocam.
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    # União = área de a + área de b - interseção (contada duas vezes).
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    # Proteção contra divisão por zero.
    return intersection / union if union > 0 else 0.0


def _boxes(model, image: Path, config: ProjectConfig, conf: float) -> list[tuple[int, float, list[float]]]:
    """Caixas (classe, confiança, xyxy) de um modelo .pt ou .onnx numa foto.

    Os dois modelos passam exatamente pelos mesmos parâmetros, para que a
    única diferença entre as saídas seja o runtime (PyTorch × ONNX Runtime).

    Parâmetros:
        model: modelo Ultralytics (``load_detector`` para .pt ou
            ``load_exported_detector`` para .onnx).
        image (Path): uma foto.
        config (ProjectConfig): usa ``prediction`` (imgsz, nms, etc.).
        conf (float): limiar de confiança (0..1); substitui o do project.yaml.

    Retorna:
        list[tuple[int, float, list[float]]]: ``(class_id, confiança,
        [x1, y1, x2, y2])`` em pixels da foto original; lista vazia se não
        houver caixas.

    Chamada por: ``verify_onnx_parity``.
    """
    # Parâmetros do project.yaml com o "conf" trocado pelo da paridade.
    parameters = {**config.prediction, "conf": conf}
    # rect=False: o .pt recebe a mesma entrada quadrada fixa do ONNX exportado.
    # device="cpu" fixo: a paridade é medida na CPU (ver ATENCAO do cabeçalho).
    # next(iter(...)): com stream=True o predict devolve um gerador; uma foto
    # rende um único resultado.
    result = next(iter(model.predict(source=str(image), device="cpu", save=False, verbose=False, stream=True,
                                     rect=False, **parameters)))
    if result.boxes is None:
        return []
    # Mesma conversão de prediction.serialize_detections: tensores → listas
    # Python; zip anda nas três listas em paralelo (c = classe, s = confiança).
    return [(int(c), float(s), [float(v) for v in xyxy]) for xyxy, s, c in
            zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())]


def compare_outputs(reference: list, candidate: list, conf: float) -> dict:
    """Pareie caixas pelo melhor IoU global; tolere só oscilações junto ao limiar.

    Algoritmo (guloso): monta todos os pares possíveis (mesma classe e
    confiança dentro de ``PARITY_CONFIDENCE``), ordena do mais parecido para o
    menos parecido e aceita cada par se nenhuma das duas caixas já tiver par.
    Caixas que sobram só reprovam a paridade se a confiança estiver
    claramente acima do limiar (uma caixa com confiança 0,101 pode existir
    num runtime e não no outro com limiar 0,1 sem ser erro).

    Parâmetros:
        reference (list): caixas do .pt, no formato de ``_boxes``.
        candidate (list): caixas do .onnx, mesmo formato.
        conf (float): limiar de confiança usado nas duas inferências.

    Retorna:
        dict com ``reference_boxes``/``exported_boxes`` (quantidades),
        ``matched`` (pares aceitos), ``missing_in_export``/``extra_in_export``
        (listas de ``{"class_id", "confidence"}`` sem par), ``worst_matched_iou``,
        ``worst_confidence_delta`` e ``equivalent`` (True se nada sobrou).

    Chamada por: ``verify_onnx_parity`` (e testes).
    """
    def similarity(a: list[float], b: list[float]) -> float:
        """Nota de semelhança de duas caixas xyxy: o IoU, ou 1.0 se todas as
        coordenadas diferirem no máximo ``PARITY_PIXELS``. Retorna float 0..1."""
        # close: a maior diferença entre coordenadas correspondentes ≤ 1 px.
        close = max(abs(x - y) for x, y in zip(a, b)) <= PARITY_PIXELS
        return max(_iou(a, b), 1.0 if close else 0.0)

    # Candidatos a par: mesma classe e confiança próxima; os mais parecidos primeiro.
    # Cada item: (semelhança, índice em reference, índice em candidate).
    # a[0] = classe, a[1] = confiança, a[2] = xyxy. reverse=True: maior semelhança primeiro.
    pairs = sorted(((similarity(a[2], b[2]), i, j) for i, a in enumerate(reference) for j, b in enumerate(candidate)
                    if a[0] == b[0] and abs(a[1] - b[1]) <= PARITY_CONFIDENCE), reverse=True)
    # Índices já pareados e os piores valores entre os pares aceitos
    # (começam no melhor caso: IoU 1.0 e Δconfiança 0.0).
    used_reference, used_candidate, worst_iou, worst_confidence = set(), set(), 1.0, 0.0
    # Pareamento guloso.
    for iou, i, j in pairs:
        # Lista ordenada: a partir daqui todos os pares são piores que o mínimo.
        if iou < PARITY_IOU:
            break
        # Uma caixa só pode estar em um par.
        if i in used_reference or j in used_candidate:
            continue
        used_reference.add(i)
        used_candidate.add(j)
        # Pior IoU real: se o par entrou pela tolerância de pixels (nota 1.0),
        # registra o IoU geométrico verdadeiro; senão a própria nota (que já é o IoU).
        worst_iou = min(worst_iou, _iou(reference[i][2], candidate[j][2]) if iou < 1.0 else iou)
        worst_confidence = max(worst_confidence, abs(reference[i][1] - candidate[j][1]))
    # Caixas sem par só contam se estiverem claramente acima do limiar.
    # missing: o .pt viu e o .onnx não; extra: o .onnx viu e o .pt não.
    missing = [{"class_id": c, "confidence": round(s, 4)} for index, (c, s, _) in enumerate(reference)
               if index not in used_reference and s - conf > PARITY_CONFIDENCE]
    extra = [{"class_id": c, "confidence": round(s, 4)} for index, (c, s, _) in enumerate(candidate)
             if index not in used_candidate and s - conf > PARITY_CONFIDENCE]
    # EXTENSAO: novo critério de reprovação (ex.: worst_matched_iou mínimo):
    # acrescente a condição em "equivalent" e o valor no dicionário.
    return {"reference_boxes": len(reference), "exported_boxes": len(candidate), "matched": len(used_reference),
            "missing_in_export": missing, "extra_in_export": extra,
            "worst_matched_iou": round(worst_iou, 4), "worst_confidence_delta": round(worst_confidence, 4),
            "equivalent": not missing and not extra}


# ---------------------------------------------------------------------------
# Paridade e exportação.
# ---------------------------------------------------------------------------

def default_parity_images(config: ProjectConfig, limit: int = 5) -> list[Path]:
    """Sem --parity-source, compara nas 5 primeiras fotos do CEASA.

    Parâmetros:
        config (ProjectConfig): usa ``root`` (pasta sistema_ia).
        limit (int): quantas fotos no máximo (padrão 5).

    Retorna:
        list[Path]: fotos de ``data/reference/ceasa`` (só o nível de cima, sem
        subpastas), em ordem alfabética; lista vazia se a pasta não existir,
        o que faz ``export_onnx`` pular a paridade.

    Chamada por: ``export_onnx`` quando ``parity_images`` é None.
    """
    # EXTENSAO: outra pasta padrão de fotos de paridade: troque o caminho aqui.
    reference = config.root / "data" / "reference" / "ceasa"
    # glob("*") não desce em subpastas; filtro pelas extensões de foto do dataset.
    return sorted(path for path in reference.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)[:limit]


def _onnx_imgsz(path: Path) -> int | None:
    """Tamanho de entrada gravado nos metadados do ONNX (ex.: 640).

    A Ultralytics grava ``imgsz`` como texto (ex.: "[640, 640]") nos metadados
    customizados do arquivo ONNX. Como o ONNX exportado tem entrada FIXA, toda
    inferência com ele precisa usar exatamente esse tamanho.

    Parâmetros:
        path (Path): arquivo ``.onnx``.

    Retorna:
        int | None: o lado da entrada (primeiro valor da lista), ou None se o
        metadado não existir ou não for uma lista/tupla legível.

    Erros: falhas do onnxruntime ao abrir o arquivo propagam (arquivo inválido).

    Chamada por: ``verify_onnx_parity`` e ``training.pilot_inference_config``.
    """
    # Imports locais: onnxruntime é dependência opcional (extra "export").
    import ast
    import onnxruntime

    # Sessão só na CPU: aqui só se leem metadados, não se infere.
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    value = session.get_modelmeta().custom_metadata_map.get("imgsz")
    # ast.literal_eval converte o texto "[640, 640]" em lista com segurança
    # (não executa código, ao contrário de eval).
    try:
        size = ast.literal_eval(value) if value else None
    except (ValueError, SyntaxError):
        return None
    # Entrada quadrada: o primeiro valor basta.
    return int(size[0]) if isinstance(size, (list, tuple)) and size else None


def verify_onnx_parity(config: ProjectConfig, checkpoint: Path, onnx_path: Path, images: list[Path], conf: float) -> dict:
    """Rode o .pt e o .onnx nas mesmas fotos: só assim a exportação é comparável.

    Parâmetros:
        config (ProjectConfig): configuração (usa ``prediction``).
        checkpoint (Path): o ``.pt`` de referência (a cópia em ``runs/export_*``).
        onnx_path (Path): o ``.onnx`` exportado.
        images (list[Path]): fotos de comparação (ao menos uma).
        conf (float): limiar de confiança usado nos dois modelos.

    Retorna:
        dict (conteúdo do ``parity.json``): ``conf``, as três tolerâncias,
        ``images`` (uma entrada por foto com ``image``, ``sha256`` e o
        resultado de ``compare_outputs``), ``runtime_parity_validated`` (True
        só se TODAS as fotos forem equivalentes) e ``scope`` (texto do alcance).

    Erros:
        ValueError("Informe ao menos uma foto ...") se ``images`` estiver vazia;
        erros de carga dos modelos propagam de ``backend``.

    Chamada por: ``export_onnx`` (etapa 3).
    """
    # Validação: paridade sem fotos não prova nada.
    if not images:
        raise ValueError("Informe ao menos uma foto para verificar a paridade da exportação.")
    # 1. Carrega os dois modelos.
    reference = load_detector(config, str(checkpoint))
    exported = load_exported_detector(config, onnx_path)
    # O ONNX tem entrada fixa: compare sempre no tamanho gravado nos metadados.
    # 2. Cópia do config com prediction["imgsz"] trocado (o original não muda).
    imgsz = _onnx_imgsz(onnx_path)
    if imgsz:
        config = replace(config, prediction={**config.prediction, "imgsz": imgsz})
    # 3. Uma comparação por foto, com o hash da foto para rastreabilidade.
    per_image = []
    for image in images:
        comparison = compare_outputs(_boxes(reference, image, config, conf), _boxes(exported, image, config, conf), conf)
        per_image.append({"image": str(image), "sha256": file_hash(image), **comparison})
    # 4. Resultado: aprovado só se todas as fotos forem equivalentes.
    # EXTENSAO: novo campo no parity.json: acrescente neste dicionário.
    return {"conf": conf, "iou_required": PARITY_IOU, "confidence_tolerance": PARITY_CONFIDENCE, "pixel_tolerance": PARITY_PIXELS,
            "images": per_image, "runtime_parity_validated": all(item["equivalent"] for item in per_image),
            "scope": "Paridade numérica PyTorch × ONNX Runtime na CPU deste computador; não avalia qualidade nem outro hardware."}


# ENTRADA: comando "python -m dronecamp_ia export --weights X.pt [--parity-source <pasta>] [--demo]" (cli.py);
# também chamado por cycle.py e gpu_eval.py.
def export_onnx(config: ProjectConfig, weights: str, demo: bool = False, parity_images: list[Path] | None = None,
                output: Path | None = None) -> Path:
    """Exporte ONNX FP32, batch 1 e tamanho fixo; confira a paridade com o .pt.

    ``parity_images=[]`` pula a paridade; ``output`` troca ``runs/export_*`` por
    uma pasta já existente e vazia (avaliação por época na GPU, ``gpu_eval.py``).

    Parâmetros:
        config (ProjectConfig): configuração do projeto (usa ``root``,
            ``taxonomy_path`` e ``prediction``: ``imgsz``, ``nms``, ``conf``).
        weights (str): caminho do checkpoint ``.pt`` a exportar.
        demo (bool): True aceita pesos genéricos (não confere as classes da
            taxonomia nem aplica o imgsz do piloto); pasta ``export_demo_*``.
        parity_images (list[Path] | None): fotos da paridade. None = fotos
            padrão (``default_parity_images``); ``[]`` = sem paridade.
        output (Path | None): pasta de saída já existente e VAZIA; None cria
            ``runs/export_<UTC>_<id>/``.

    Retorna:
        Path: a pasta da exportação (com ``model.pt``, ``model.onnx``,
        ``export.json`` e, se houve paridade, ``parity.json``).

    Erros:
        ValueError: pacote ``onnx`` ausente; classes do checkpoint diferentes
        das da taxonomia (sem ``demo``); ``output`` não vazia; ``.pt`` inválido.
        Paridade reprovada NÃO levanta erro: vai como
        ``runtime_parity_validated: false`` no export.json.

    Chamada por: ``cli.py`` (comando ``export``), ``cycle.py``, ``gpu_eval.py``.
    """
    # 0. Pré-requisito: o exportador ONNX da Ultralytics precisa do pacote "onnx"
    # (extra "export" do pyproject). Falha cedo, antes de criar pastas.
    if importlib.util.find_spec("onnx") is None:
        raise ValueError("Dependência ONNX ausente. Instale o extra export documentado no README antes de exportar.")
    # 1. Modelo com as classes ativas; pesos piloto exportam no tamanho de treino.
    model = load_detector(config, weights)
    if not demo:
        from .training import pilot_inference_config

        # ATENCAO: nomes do modelo = slugs das classes ativas, na mesma ordem.
        check_domain_names(model, detection_names(load_taxonomy(config.taxonomy_path)))
        # Pesos piloto: troca prediction["imgsz"] pelo do treino (ex.: 640).
        config = pilot_inference_config(config, model)
    # 2. Cópia do .pt na pasta da exportação e exportação FP32, batch 1, tamanho fixo.
    run = Path(output) if output is not None else make_run_directory(config.root, "export_demo" if demo else "export")
    # ATENCAO: nunca exportar por cima de arquivos existentes numa pasta passada pelo chamador.
    if output is not None and any(run.iterdir()):
        raise ValueError(f"A pasta de exportação precisa estar vazia: {run}")
    checkpoint = run / "model.pt"
    # copy2 preserva datas; ckpt_path é o .pt resolvido por load_detector.
    shutil.copy2(model.ckpt_path, checkpoint)
    # O exportador grava ao lado do checkpoint: uma cópia evita tocar o original.
    model = load_detector(config, str(checkpoint))
    # Opções da exportação:
    # - device="cpu": exportação reprodutível em qualquer máquina;
    # - imgsz: lado da entrada fixa (do project.yaml ou do treino do piloto);
    # - batch=1 e dynamic=False: forma de entrada fixa (1, 3, imgsz, imgsz);
    # - simplify=False: não passa pelo onnxslim/onnxsim;
    # - nms: embute o NMS no grafo conforme prediction.nms (YOLO26 é
    #   end-to-end e usa false);
    # - quantize=32: FP32 (sem meia precisão nem INT8).
    # "output" passa a ser o caminho do .onnx devolvido pela Ultralytics
    # (o parâmetro original não é mais necessário daqui em diante).
    # EXTENSAO: FP16/INT8/eixo dinâmico/outro formato: mude os argumentos aqui
    # e os campos "format"/"quantize" do export.json abaixo; confira também se
    # backend.load_exported_detector e training.pilot_inference_config
    # continuam válidos para o novo arquivo.
    output = model.export(format="onnx", device="cpu", imgsz=config.prediction["imgsz"],
                          batch=1, dynamic=False, simplify=False, nms=config.prediction["nms"], quantize=32)
    # 3. Paridade .pt × .onnx (parity.json) e resumo da exportação (export.json).
    images = parity_images if parity_images is not None else default_parity_images(config)
    parity = None
    # Paridade só se o onnxruntime estiver instalado e houver fotos; senão
    # export.json registra runtime_parity_validated=false e parity_images=0.
    # O limiar é no máximo 0.1 (mais caixas na comparação = teste mais rigoroso).
    if importlib.util.find_spec("onnxruntime") is not None and images:
        parity = verify_onnx_parity(config, checkpoint, Path(output), images, min(0.1, float(config.prediction["conf"])))
        write_json(run / "parity.json", parity)
    # export.json: resumo para auditoria e para quem consome o ONNX.
    # EXTENSAO: novo campo no export.json: acrescente neste dicionário.
    write_json(run / "export.json", {
        # Versões, dispositivo e hash do .pt; caminho e hash do .onnx gerado.
        "runtime": runtime_info(model, config), "artifact": str(output),
        "sha256": file_hash(Path(output)), "format": "onnx", "quantize": 32,
        # Parâmetros de entrada efetivos e origem dos pesos.
        "nms": config.prediction["nms"], "imgsz": config.prediction["imgsz"], "demo": demo, "source_weights": str(Path(weights)),
        # bool(...) dá False quando a paridade não rodou (parity None).
        "runtime_parity_validated": bool(parity and parity["runtime_parity_validated"]),
        # ATENCAO: aceite humano nunca é automático.
        "parity_images": len(images) if parity else 0, "human_acceptance": "pendente",
    })
    return run
