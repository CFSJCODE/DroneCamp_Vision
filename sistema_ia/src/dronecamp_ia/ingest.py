"""Importação de fotos e vídeos enviados pela plataforma, sem modelo.

Função no projeto: é a porta de entrada da mídia da plataforma comercial. O
servidor recebe os arquivos do cliente e chama ``ingest_media``, que congela
cada foto por SHA-256 em ``originals/``, transforma vídeos em quadros e grava
um ``registry.json`` no mesmo formato das revisões criadas por
``suggestions.create_review_for_new_images``. É o comando ``import-media``.

O que faz:
- ``slugify_review_name``: nome digitado pelo cliente vira nome de pasta seguro.
- ``extract_video_frames``: quadros JPEG a cada N segundos (OpenCV ou ffmpeg).
- ``canonicalize_photo``: foto com rotação EXIF vira cópia girada com hash novo;
  as demais são copiadas byte a byte (hash preservado).
- ``ingest_media``: monta a revisão nova (registro + página) e devolve o resumo.

Nada aqui roda modelo, aprova fotos ou altera uma revisão existente: toda saída
é uma pasta nova, e as sugestões vêm depois com ``suggest --registry``. O
registro gerado é aceito por ``suggestions.suggest_for_registry``,
``review_render.render_review_package`` e ``pilot.build_pilot_dataset``.

Quando mexer: formatos aceitos (``VIDEO_EXTENSIONS``), limites da amostragem de
quadros ou os campos do bloco ``ingest`` gravado por imagem.

Conteúdo do arquivo (na ordem em que aparece):
- Constantes: ``VIDEO_EXTENSIONS``, ``WEB_IMAGE_SUFFIXES``, limites de amostragem,
  ``INGEST_SCOPE``, ``FRAME_EXTRACTOR_ERROR``, ``MEDIA_ERRORS``.
- Nomes e tipos: ``media_kind``, ``slugify_review_name``, ``_validate_sampling``.
- Quadros de vídeo: ``_frame_extractor``, ``_sample_times``, ``_frame_path``,
  ``_frame_record``, ``_extract_with_opencv``, ``_ffmpeg_duration``,
  ``_extract_with_ffmpeg``, ``extract_video_frames``.
- Foto congelada: ``_TRANSPOSE_BY_ORIENTATION``, ``_transposed``, ``canonicalize_photo``.
- Importação: ``_move_into``, ``_collect_inputs``, ``_optional_text``, ``_Importer``,
  ``ingest_media``.

Quem chama:
- CLI (cli.py): ``import-media`` → ``ingest_media`` (arquivos do próprio usuário,
  ``keep_sources=False``).
- ``platform_server.py``: envio pela página → ``slugify_review_name`` (nome da pasta),
  ``media_kind`` (filtra o upload) e ``ingest_media(..., keep_sources=True)``.

O que lê: as fotos e vídeos informados e ``configs/taxonomy.json``.
O que grava (sempre numa pasta NOVA ``output``):
- ``output/originals/<sha256>.<ext>``: fotos congeladas (nome = hash do conteúdo);
- ``output/midia_enviada/``: arquivos enviados, quando ``keep_sources=True``;
- ``output/registry.json`` e a página da revisão (``render_review_package``);
- ``output/INCOMPLETO.txt`` se a importação falhar no meio.
Pastas temporárias ``quadros_*`` e ``ffmpeg_*`` dentro de ``output`` são apagadas ao fim.

Onde mexer:
- Aceitar um formato de vídeo novo: ``VIDEO_EXTENSIONS``. Formato de foto novo:
  ``dataset.IMAGE_EXTENSIONS`` (e ``WEB_IMAGE_SUFFIXES`` se o navegador exibe).
- Mudar limites do intervalo/quantidade de quadros: ``EVERY_SECONDS_RANGE`` e
  ``MAX_FRAMES_RANGE`` (os padrões 2.0 s e 200 ficam nas assinaturas de
  ``extract_video_frames``/``ingest_media`` e nos argumentos da CLI em cli.py).
- Novo campo por imagem no registro: dicionário em ``_Importer.add``
  (mantenha os campos de ``suggestions.create_review_for_new_images``).
- Novo campo por arquivo de origem: dicionário ``entry`` em ``ingest_media``.
- Novo parâmetro do envio (ex.: responsável): assinatura de ``ingest_media``,
  bloco ``"ingest"`` do registro e a chamada em ``platform_server.py``/cli.py.
- Outro extrator de quadros: ``_frame_extractor`` e o despacho em ``extract_video_frames``.

ATENCAO: toda foto entra com ``human_approved: False`` e sem caixas; nada aqui aprova.
ATENCAO: nunca sobrescreve: a saída precisa ser uma pasta nova, quadros e originais
não substituem arquivos existentes.
"""

# ---------------------------------------------------------------------------
# Importações
# ---------------------------------------------------------------------------

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
from uuid import uuid4

from PIL import Image, ImageOps

# ProjectConfig: caminhos do projeto; load_taxonomy: lê configs/taxonomy.json (versão vai ao registro).
from .config import ProjectConfig, load_taxonomy
# Extensões de foto aceitas em todo o projeto (fonte única; não duplique aqui).
from .dataset import IMAGE_EXTENSIONS
# file_hash: SHA-256 do arquivo; write_json: grava o registry.json.
from .io import file_hash, write_json
# review_name: nome exibido da revisão (nome digitado ou nome da pasta).
from .naming import review_name
# REVIEW_SCHEMA: versão do formato do registro; review_image_size: largura/altura e checagem da foto.
from .review_data import REVIEW_SCHEMA, review_image_size
# Gera a página HTML da revisão ao lado do registro.
from .review_render import render_review_package

# Nomes públicos do módulo (os importados por outros módulos e pelos testes).
# EXTENSAO: nova função pública: acrescente o nome aqui.
__all__ = ["IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "WEB_IMAGE_SUFFIXES", "media_kind", "slugify_review_name",
           "extract_video_frames", "canonicalize_photo", "ingest_media"]

# ---------------------------------------------------------------------------
# Constantes do módulo (formatos aceitos, limites e mensagens)
# ---------------------------------------------------------------------------

# Contêineres de vídeo aceitos (a decodificação fica com OpenCV ou ffmpeg).
# EXTENSAO: formato de vídeo novo = extensão aqui; formato de foto novo = dataset.IMAGE_EXTENSIONS
# (e WEB_IMAGE_SUFFIXES abaixo, se o navegador mostra o formato sem conversão).
# ATENCAO: extensões em minúsculas e com ponto; media_kind compara com suffix.lower().
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"})
# Formatos que o navegador exibe e o servidor serve; fotos nos demais (TIFF, BMP) são regravadas como JPEG ao congelar.
# EXTENSAO: formato que o navegador passou a exibir: acrescente aqui para copiá-lo byte a byte
# (hash preservado) em vez de regravá-lo como JPEG.
WEB_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
# Limites da amostragem de quadros: intervalo em segundos e quadros por vídeo.
# EVERY_SECONDS_RANGE = (mínimo, máximo) do intervalo entre quadros, em segundos.
# ATENCAO: o mínimo 0.2 s garante nomes de quadro distintos (_frame_path usa 1 casa decimal);
# baixar para menos de 0.1 s faria dois quadros caírem no mesmo nome e o vídeo falhar.
EVERY_SECONDS_RANGE = (0.2, 600.0)
# MAX_FRAMES_RANGE = (mínimo, máximo) de quadros extraídos por vídeo.
MAX_FRAMES_RANGE = (1, 2000)
# Valor do campo "scope" do registry.json: diz que a revisão veio de mídia importada e está pendente.
INGEST_SCOPE = "midia_importada_pela_plataforma_pendente_de_revisao"
# Mensagem quando não há OpenCV nem ffmpeg para decodificar vídeo.
FRAME_EXTRACTOR_ERROR = "Extração de quadros requer OpenCV (vem com a Ultralytics) ou ffmpeg no PATH."
# Erros de leitura de um arquivo de mídia: o arquivo é pulado e registrado, a importação segue.
# ImportError entra porque a Ultralytics, quando importada no mesmo processo, troca ``Image.open`` por uma
# versão que tenta instalar ``pi-heif`` ao falhar na leitura; sem rede, um arquivo ilegível vira ImportError.
# EXTENSAO: novo tipo de erro de leitura que deve só pular o arquivo: acrescente-o nesta tupla.
# ATENCAO: ValueError está na tupla; qualquer ValueError dentro da leitura de UM arquivo
# (inclusive "Quadro já existe") vira "arquivo_ilegivel" em vez de interromper a importação.
MEDIA_ERRORS = (OSError, ValueError, SyntaxError, ImportError, Image.DecompressionBombError)


# ---------------------------------------------------------------------------
# Nomes e tipos de arquivo.
# ---------------------------------------------------------------------------

def media_kind(path: Path) -> str | None:
    """``"photo"``, ``"video"`` ou None (extensão não suportada).

    Parâmetros:
    - ``path``: caminho ou nome do arquivo (só a extensão é olhada; o conteúdo não é lido).

    Retorna: ``"photo"`` se a extensão está em ``IMAGE_EXTENSIONS``, ``"video"`` se
    está em ``VIDEO_EXTENSIONS``, senão None.

    Chamada por: ``_collect_inputs`` e ``platform_server.py`` (filtro do upload).
    EXTENSAO: novo tipo de mídia (ex.: ``"pdf"``): acrescente um teste aqui e trate o
    tipo no laço de ``ingest_media``.
    """
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "photo"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def slugify_review_name(name: str) -> str:
    """Nome de pasta da revisão: minúsculas, sem acentos, só ``[a-z0-9_]``.

    Começa com letra (recebe o prefixo ``r_`` quando o nome começa com número)
    e tem no máximo 60 caracteres. ValueError se não sobrar nada.

    Parâmetros:
    - ``name``: nome digitado (ex.: ``"Galpão B · outubro"``).

    Retorna: slug (ex.: ``"galpao_b_outubro"``; ``"2026 lote"`` vira ``"r_2026_lote"``).

    Erros: ``ValueError`` se ``name`` não for texto ou se nada sobrar após a limpeza.

    Chamada por: ``platform_server.py`` (nome da pasta da inspeção nova).
    """
    if not isinstance(name, str):
        raise ValueError("Nome da revisão deve ser texto.")
    # NFKD separa letra e acento; os acentos (marcas combinantes) são descartados.
    plain = "".join(char for char in unicodedata.normalize("NFKD", name) if not unicodedata.combining(char))
    # Qualquer sequência fora de [a-z0-9] (espaço, ponto, "·", "ç" sem decomposição...) vira um "_";
    # "_" nas pontas é removido.
    slug = re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")
    if not slug:
        raise ValueError("Nome da revisão vazio após a normalização; use letras ou números.")
    # Nome de pasta começa com letra (convenção das revisões).
    if not slug[0].isalpha():
        slug = f"r_{slug}"
    # Corta em 60 caracteres e tira um "_" que tenha ficado no fim do corte.
    return slug[:60].rstrip("_")


def _validate_sampling(every_seconds: float, max_frames: int) -> tuple[float, int]:
    """Intervalo entre quadros e máximo de quadros dentro dos limites.

    Parâmetros:
    - ``every_seconds``: intervalo entre quadros, em segundos (aceita texto numérico).
    - ``max_frames``: máximo de quadros por vídeo (inteiro; aceita float inteiro como 10.0).

    Retorna: ``(every como float, max_frames como int)``.

    Erros: ``ValueError`` se o intervalo não for número finito em ``EVERY_SECONDS_RANGE``
    ou se ``max_frames`` não for inteiro em ``MAX_FRAMES_RANGE`` (bool é recusado).

    Chamada por: ``extract_video_frames`` e ``ingest_media``.
    """
    # Intervalo: converte e confere faixa (math.isfinite recusa NaN e infinito).
    try:
        every = float(every_seconds)
    except (TypeError, ValueError):
        raise ValueError("every_seconds deve ser um número de segundos.") from None
    if not math.isfinite(every) or not EVERY_SECONDS_RANGE[0] <= every <= EVERY_SECONDS_RANGE[1]:
        raise ValueError(f"every_seconds deve estar entre {EVERY_SECONDS_RANGE[0]} e {EVERY_SECONDS_RANGE[1]:.0f} segundos.")
    # Máximo de quadros: bool é subclasse de int em Python e precisa ser recusado à parte;
    # int(x) != x pega valores fracionários como 2.5.
    if (isinstance(max_frames, bool) or not isinstance(max_frames, (int, float)) or not math.isfinite(max_frames)
            or int(max_frames) != max_frames or not MAX_FRAMES_RANGE[0] <= int(max_frames) <= MAX_FRAMES_RANGE[1]):
        raise ValueError(f"max_frames deve ser um inteiro entre {MAX_FRAMES_RANGE[0]} e {MAX_FRAMES_RANGE[1]}.")
    return every, int(max_frames)


# ---------------------------------------------------------------------------
# Quadros de vídeo (OpenCV quando importável; senão ffmpeg do PATH).
# ---------------------------------------------------------------------------

def _frame_extractor() -> tuple[str, str | None]:
    """``("opencv", None)`` ou ``("ffmpeg", caminho)``; ValueError sem nenhum dos dois.

    Retorna: o extrator disponível, em ordem de preferência: OpenCV (``cv2``) e,
    na falta dele, o executável ``ffmpeg`` encontrado no PATH.

    Erros: ``ValueError(FRAME_EXTRACTOR_ERROR)``.

    Chamada por: ``extract_video_frames`` e ``ingest_media`` (checagem antecipada).
    EXTENSAO: outro extrator (ex.: PyAV): acrescente um bloco de detecção aqui com
    outro nome de backend e o despacho correspondente em ``extract_video_frames``.
    """
    # 1. OpenCV (instalado junto com a Ultralytics); importação só para testar a presença.
    try:
        import cv2  # noqa: F401  (vem com a Ultralytics; opcional aqui)
        return "opencv", None
    except ImportError:
        pass
    # 2. ffmpeg do sistema.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return "ffmpeg", ffmpeg
    raise ValueError(FRAME_EXTRACTOR_ERROR)


def _sample_times(duration: float, every_seconds: float, max_frames: int) -> tuple[list[float], float]:
    """Instantes amostrados (0, N, 2N… < duração) e o intervalo efetivamente usado.

    Se o vídeo renderia mais de ``max_frames`` quadros, o intervalo cresce para
    ``duração / max_frames``, mantendo a amostragem uniforme do início ao fim.

    Parâmetros:
    - ``duration``: duração do vídeo em segundos.
    - ``every_seconds``: intervalo pedido, em segundos.
    - ``max_frames``: máximo de quadros.

    Retorna: ``(lista de instantes em segundos com 3 casas, intervalo usado)``.
    Exemplo: 10 s a cada 2 s → ``([0.0, 2.0, 4.0, 6.0, 8.0], 2.0)``.

    Erros: ``ValueError("Vídeo sem duração legível.")`` se ``duration`` não for > 0
    (``not duration > 0`` também pega NaN).
    """
    if not duration > 0:
        raise ValueError("Vídeo sem duração legível.")
    # Quantidade = teto(duração / intervalo); o "- 1e-9" evita um quadro a mais por erro
    # de ponto flutuante quando a divisão é exata (10 / 2 = 5.0000000001 → 5, não 6).
    count = max(1, math.ceil(duration / every_seconds - 1e-9))
    every_used = every_seconds
    # Teto de quadros: espaça os instantes por igual em vez de cortar o fim do vídeo.
    if count > max_frames:
        count, every_used = max_frames, duration / max_frames
    return [round(index * every_used, 3) for index in range(count)], every_used


def _frame_path(output_dir: Path, stem: str, time_s: float) -> Path:
    """``<stem>_t<segundos com 1 casa>.jpg``; nunca sobrescreve um quadro existente.

    Parâmetros:
    - ``output_dir``: pasta dos quadros.
    - ``stem``: nome do vídeo sem extensão.
    - ``time_s``: instante do quadro, em segundos.

    Retorna: caminho do quadro (ex.: ``voo1_t12.0.jpg``).

    Erros: ``ValueError`` se o arquivo já existir.
    ATENCAO: o nome usa 1 casa decimal; dois instantes com a mesma casa geram conflito
    (por isso o intervalo mínimo é 0.2 s em ``EVERY_SECONDS_RANGE``).
    """
    path = output_dir / f"{stem}_t{time_s:.1f}.jpg"
    if path.exists():
        raise ValueError(f"Quadro já existe; use uma pasta de saída nova: {path.name}.")
    return path


def _frame_record(path: Path, time_s: float, index: int, every_used: float) -> dict:
    """Item da lista devolvida por ``extract_video_frames``.

    Parâmetros:
    - ``path``: arquivo JPEG gravado.
    - ``time_s``: instante do quadro, em segundos.
    - ``index``: posição do quadro na amostragem (0 = primeiro).
    - ``every_used``: intervalo efetivamente usado, em segundos.

    Retorna: ``{"path", "time_s", "frame_index", "every_seconds_used"}``.
    EXTENSAO: novo dado por quadro (ex.: número do quadro no vídeo): acrescente a chave
    aqui e passe o valor nas duas chamadas (OpenCV e ffmpeg).
    """
    return {"path": path, "time_s": float(time_s), "frame_index": index, "every_seconds_used": float(every_used)}


def _extract_with_opencv(video: Path, output_dir: Path, every_seconds: float, max_frames: int) -> list[dict]:
    """Leitura sequencial (sem seek, que é impreciso em alguns contêineres).

    Parâmetros:
    - ``video``: arquivo de vídeo.
    - ``output_dir``: pasta onde os JPEG serão gravados.
    - ``every_seconds``: intervalo pedido, em segundos (já validado).
    - ``max_frames``: máximo de quadros (já validado).

    Retorna: lista de ``_frame_record``.

    Erros: ``ValueError`` se o vídeo não abrir, se não tiver fps legível, se não
    tiver duração (via ``_sample_times``) ou se nenhum quadro for lido.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    try:
        # 1. Abre o vídeo e lê fps e número de quadros declarados no contêiner.
        if not capture.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo {video.name}.")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not fps > 0:
            raise ValueError(f"Vídeo sem taxa de quadros legível: {video.name}.")
        if total <= 0:
            # Contêiner sem contagem declarada: conta os quadros numa primeira passada.
            while capture.grab():
                total += 1
            # Reabre para voltar ao início (sem seek).
            capture.release()
            capture = cv2.VideoCapture(str(video))
        # 2. Instantes desejados; duração = quadros / fps.
        times, every_used = _sample_times(total / fps, every_seconds, max_frames)
        # Um quadro de origem por instante; instantes que caem no mesmo quadro
        # (vídeos com fps muito baixo) rendem um só arquivo.
        # targets: {número do quadro no vídeo: instante em segundos}; round(t * fps) = quadro mais
        # próximo, limitado ao último quadro (total - 1).
        targets: dict[int, float] = {}
        for time_s in times:
            targets.setdefault(min(total - 1, round(time_s * fps)), time_s)
        # 3. Percorre o vídeo quadro a quadro: grab() avança sem decodificar; retrieve() só
        #    decodifica os quadros escolhidos. Para depois do último alvo.
        frames: list[dict] = []
        position, last = 0, max(targets)
        while position <= last and capture.grab():
            if position in targets:
                ok, pixels = capture.retrieve()
                if not ok:
                    break
                time_s = targets[position]
                path = _frame_path(output_dir, video.stem, time_s)
                # OpenCV entrega BGR; o Pillow espera RGB. JPEG qualidade 95.
                Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).save(path, "JPEG", quality=95)
                frames.append(_frame_record(path, time_s, len(frames), every_used))
            position += 1
    finally:
        # Libera o arquivo de vídeo mesmo em caso de erro.
        capture.release()
    if not frames:
        raise ValueError(f"Nenhum quadro legível em {video.name}.")
    return frames


def _ffmpeg_duration(video: Path, ffmpeg: str) -> float:
    """Duração em segundos via ffprobe ou, na falta dele, pelo cabeçalho que o ffmpeg imprime.

    Parâmetros:
    - ``video``: arquivo de vídeo.
    - ``ffmpeg``: caminho do executável ``ffmpeg`` (o ``ffprobe`` é procurado no PATH
      e na mesma pasta dele).

    Retorna: duração em segundos (float).

    Erros: ``ValueError`` se nenhum dos dois métodos conseguir ler a duração.
    """
    # 1. ffprobe: imprime só o número da duração (ex.: "12.345000").
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe", path=str(Path(ffmpeg).parent))
    if ffprobe:
        probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", str(video)],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL)
        try:
            return float(probe.stdout.strip())
        except ValueError:
            # Saída vazia ou "N/A": tenta o cabeçalho do ffmpeg.
            pass
    # 2. ffmpeg -i sem saída imprime no stderr uma linha "Duration: HH:MM:SS.ss".
    header = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-i", str(video)],
                            capture_output=True, text=True, stdin=subprocess.DEVNULL)
    # Regex: horas, minutos e segundos (com fração opcional).
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", header.stderr)
    if not match:
        raise ValueError(f"Não foi possível ler a duração do vídeo {video.name}.")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _extract_with_ffmpeg(video: Path, output_dir: Path, every_seconds: float, max_frames: int, ffmpeg: str) -> list[dict]:
    """``-vf fps=1/N`` grava BMP (sem perda) numa pasta temporária; o JPEG q95 é do Pillow.

    ``round=up`` faz cada slot receber o quadro no instante k·N (ou o imediatamente
    anterior), como a leitura por OpenCV; o padrão (``near``) adiantaria meio intervalo.
    ``start_time=0`` mantém o slot 0 em vídeos cujo primeiro quadro não começa em 0 s.

    Parâmetros:
    - ``video``: arquivo de vídeo.
    - ``output_dir``: pasta dos JPEG finais.
    - ``every_seconds``: intervalo pedido (já validado).
    - ``max_frames``: máximo de quadros (já validado).
    - ``ffmpeg``: caminho do executável.

    Retorna: lista de ``_frame_record``.

    Erros: ``ValueError`` se a duração não for legível, se o ffmpeg sair com erro
    (inclui os 300 primeiros caracteres do stderr) ou se nenhum quadro for gerado.
    """
    # 1. Instantes desejados e taxa de amostragem como fração exata (fps = 1/N).
    times, every_used = _sample_times(_ffmpeg_duration(video, ffmpeg), every_seconds, max_frames)
    rate = 1 / Fraction(every_used).limit_denominator(10_000)  # racional exato: evita 1/1.4999999
    output_dir.mkdir(parents=True, exist_ok=True)
    # 2. BMPs numa pasta temporária dentro da saída (apagada ao sair do with).
    with tempfile.TemporaryDirectory(prefix="ffmpeg_", dir=output_dir) as temporary:
        pattern = Path(temporary) / "quadro_%06d.bmp"
        # -frames:v limita ao número de instantes; -start_number 0 alinha o nome com o índice.
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
                   "-vf", f"fps={rate.numerator}/{rate.denominator}:start_time=0:round=up",
                   "-frames:v", str(len(times)), "-start_number", "0", str(pattern)]
        completed = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if completed.returncode != 0:
            raise ValueError(f"ffmpeg não conseguiu extrair quadros de {video.name}: {completed.stderr.strip()[:300]}")
        # 3. Converte cada BMP (em ordem de nome = ordem temporal) em JPEG q95 com o nome final.
        frames: list[dict] = []
        for index, bitmap in enumerate(sorted(Path(temporary).glob("quadro_*.bmp"))):
            # Instante do slot; se o ffmpeg gerar quadros além da lista, calcula pelo intervalo.
            time_s = times[index] if index < len(times) else round(index * every_used, 3)
            path = _frame_path(output_dir, video.stem, time_s)
            with Image.open(bitmap) as frame:
                frame.convert("RGB").save(path, "JPEG", quality=95)
            frames.append(_frame_record(path, time_s, index, every_used))
    if not frames:
        raise ValueError(f"Nenhum quadro legível em {video.name}.")
    return frames


def extract_video_frames(video: Path, output_dir: Path, every_seconds: float = 2.0, max_frames: int = 200) -> list[dict]:
    """Grave quadros JPEG (qualidade 95) a cada ``every_seconds`` e devolva a lista.

    Arquivos: ``output_dir/<stem>_t<segundos com 1 casa>.jpg``. Cada item é
    ``{"path": Path, "time_s": float, "frame_index": int, "every_seconds_used": float}``:
    ``frame_index`` é a ordem do quadro na amostragem (0 = primeiro quadro extraído),
    e ``every_seconds_used`` é o intervalo realmente aplicado — igual a
    ``every_seconds``, ou ``duração / max_frames`` quando o vídeo renderia mais
    de ``max_frames`` quadros (a amostragem continua uniforme, do início ao fim).
    Usa OpenCV se importável; senão ``ffmpeg`` do PATH; senão ValueError.

    Parâmetros:
    - ``video``: arquivo de vídeo.
    - ``output_dir``: pasta de saída (criada se não existir).
    - ``every_seconds``: intervalo entre quadros, em segundos (padrão 2.0).
    - ``max_frames``: máximo de quadros (padrão 200).

    Erros: ``ValueError`` para parâmetros fora dos limites, vídeo inexistente,
    nenhum extrator disponível ou vídeo ilegível.

    Chamada por: ``ingest_media`` (cada vídeo enviado).
    """
    # 1. Validação dos parâmetros e da existência do vídeo.
    every, limit = _validate_sampling(every_seconds, max_frames)
    video, output_dir = Path(video), Path(output_dir)
    if not video.is_file():
        raise ValueError(f"Vídeo não encontrado: {video}")
    # 2. Escolhe o extrator e despacha.
    # EXTENSAO: novo backend de extração: acrescente um ``if backend == "nome":`` aqui.
    backend, ffmpeg = _frame_extractor()
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend == "opencv":
        return _extract_with_opencv(video, output_dir, every, limit)
    return _extract_with_ffmpeg(video, output_dir, every, limit, ffmpeg)


# ---------------------------------------------------------------------------
# Foto congelada por hash (rotação EXIF aplicada de forma explícita).
# ---------------------------------------------------------------------------

# Mesma tabela de ImageOps.exif_transpose, usada se o EXIF da foto não puder ser regravado.
# Chave = valor da tag EXIF Orientation (274); valor = operação que deixa a foto "em pé".
# 1 (normal) não aparece: não precisa de transformação.
# ATENCAO: mantenha igual à tabela do Pillow; uma entrada errada gira a foto e desalinha as caixas.
_TRANSPOSE_BY_ORIENTATION = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def _transposed(image: Image.Image, orientation: int) -> Image.Image:
    """Aplica a rotação EXIF; mantém o resto do EXIF (GPS, data) quando ele pode ser regravado.

    Parâmetros:
    - ``image``: foto aberta pelo Pillow.
    - ``orientation``: valor da tag EXIF 274 (1 a 8).

    Retorna: nova imagem já girada, em modo RGB ou L (tons de cinza), pronta para JPEG.
    Quando o EXIF é exótico demais, gira pela tabela ``_TRANSPOSE_BY_ORIENTATION``
    e descarta o EXIF (a foto não se perde).

    Chamada por: ``canonicalize_photo``.
    """
    # Caminho normal: o Pillow gira e zera a tag de orientação no EXIF copiado.
    try:
        transposed = ImageOps.exif_transpose(image)
    except Exception:  # EXIF exótico (maker notes): gira sem metadados, mas não perde a foto
        method = _TRANSPOSE_BY_ORIENTATION.get(orientation)
        transposed = image.transpose(method) if method else image.copy()
        # Sem EXIF: senão a orientação antiga continuaria gravada e a foto giraria duas vezes.
        transposed.info.pop("exif", None)
    # JPEG só aceita RGB ou L; RGBA, P (paleta), CMYK etc. são convertidos.
    if transposed.mode not in ("RGB", "L"):
        transposed = transposed.convert("RGB")
    return transposed


def canonicalize_photo(source: Path, destination_dir: Path) -> tuple[Path, dict]:
    """Congele a foto em ``destination_dir`` com o nome igual ao seu SHA-256.

    JPG/PNG/WEBP (``WEB_IMAGE_SUFFIXES``) com orientação EXIF (tag 274) igual
    a 1: cópia byte a byte em ``<sha256><extensão minúscula>`` e ``{}``. Nos
    outros casos (foto girada pelo EXIF, ou formato que o navegador não exibe,
    como TIFF e BMP) a rotação é aplicada (``ImageOps.exif_transpose``), o
    resultado é salvo como JPEG (qualidade 95) em ``<novo sha256>.jpg`` e o
    retorno traz ``{"canonicalized_from_sha256": sha_original,
    "exif_orientation": n}``. A foto final sempre passa em ``review_image_size``.

    Parâmetros:
    - ``source``: foto de origem.
    - ``destination_dir``: pasta ``originals/`` da revisão (criada se não existir).

    Retorna: ``(caminho da foto congelada, metadados da canonicalização ou {})``.
    O nome do arquivo (sem extensão) é o ``image_sha256`` usado no registro.

    Erros:
    - ``ValueError`` se outro arquivo com conteúdo diferente já ocupar o nome do hash;
    - erros do Pillow/``review_image_size`` para fotos ilegíveis (capturados em
      ``ingest_media`` por ``MEDIA_ERRORS``).

    ATENCAO: o hash é a identidade da foto em todo o projeto (registro, sugestões,
    notas de IA); fotos giradas mudam de hash de propósito, e o hash original fica
    em ``canonicalized_from_sha256``. As caixas são sempre desenhadas sobre a foto congelada.
    """
    source, destination_dir = Path(source), Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    original_digest = file_hash(source)
    with Image.open(source) as image:
        # Tag 274 = Orientation; ausente equivale a 1 (normal).
        orientation = int(image.getexif().get(274, 1))
        # Caso 1: foto já "em pé" e em formato web: cópia exata, hash preservado.
        if orientation == 1 and source.suffix.lower() in WEB_IMAGE_SUFFIXES:
            target = destination_dir / f"{original_digest}{source.suffix.lower()}"
            if not target.exists():
                shutil.copyfile(source, target)
            # ATENCAO: nunca sobrescreve; mesmo nome com conteúdo diferente é erro.
            elif file_hash(target) != original_digest:
                raise ValueError(f"Arquivo diferente já ocupa {target.name}.")
            review_image_size(target)  # cópia ilegível ou truncada falha aqui
            return target, {}
        # Caso 2: precisa girar e/ou converter; carrega a imagem girada antes de fechar o arquivo.
        transposed = _transposed(image, orientation)
    # Opções do JPEG: qualidade 95 e o perfil de cor original, se houver.
    options = {"quality": 95}
    if transposed.info.get("icc_profile"):
        options["icc_profile"] = transposed.info["icc_profile"]
    # Grava primeiro num nome temporário (o hash só é conhecido depois de gravar).
    temporary = destination_dir / f".canonical_{uuid4().hex}.jpg"
    try:
        try:
            transposed.save(temporary, "JPEG", exif=transposed.info.get("exif") or b"", **options)
        except Exception:
            transposed.save(temporary, "JPEG", **options)  # sem EXIF, se ele não puder ser regravado
        # Renomeia para <hash>.jpg; se já existir (mesma foto enviada duas vezes), descarta o temporário.
        digest = file_hash(temporary)
        target = destination_dir / f"{digest}.jpg"
        if target.exists():
            temporary.unlink()  # mesmo conteúdo já congelado
        else:
            temporary.replace(target)
    except BaseException:
        # Qualquer falha (inclusive Ctrl+C) remove o temporário antes de propagar.
        temporary.unlink(missing_ok=True)
        raise
    review_image_size(target)  # garante que a orientação foi removida junto com a rotação
    # EXTENSAO: novo metadado de canonicalização: acrescente a chave aqui e copie-a no bloco
    # "ingest" de _Importer.add.
    return target, {"canonicalized_from_sha256": original_digest, "exif_orientation": orientation}


# ---------------------------------------------------------------------------
# Importação: revisão nova com fotos, quadros de vídeo, registro e página.
# ---------------------------------------------------------------------------

def _move_into(path: Path, directory: Path) -> Path:
    """Move ``path`` para ``directory`` sem sobrescrever (``nome_2.ext`` se o nome já existir).

    Parâmetros:
    - ``path``: arquivo enviado (na área temporária do lote).
    - ``directory``: destino (``output/midia_enviada``), criado se não existir.

    Retorna: caminho absoluto do arquivo movido.

    Chamada por: ``ingest_media`` quando ``keep_sources=True``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    # Procura o primeiro nome livre: foto.jpg, foto_2.jpg, foto_3.jpg...
    counter = 1
    while target.exists():
        counter += 1
        target = directory / f"{path.stem}_{counter}{path.suffix}"
    shutil.move(str(path), str(target))
    return target.resolve()


def _collect_inputs(inputs) -> tuple[list[tuple[Path, str]], list[dict]]:
    """Arquivos suportados (na ordem informada; pastas em ordem alfabética) e os ignorados.

    Parâmetros:
    - ``inputs``: um caminho (str/Path) ou lista de caminhos de arquivos e pastas.

    Retorna: ``(arquivos, ignorados)``:
    - ``arquivos``: lista de ``(caminho absoluto, "photo" | "video")`` sem repetição;
    - ``ignorados``: lista de ``{"file", "reason": "extensao_nao_suportada"}`` só para
      arquivos informados diretamente (em pastas, extensões estranhas passam em silêncio).

    Erros: ``ValueError("Entrada não encontrada: ...")`` se um caminho não existir.
    """
    # Aceita um único caminho além de listas.
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    files, skipped, seen = [], [], set()
    for value in inputs:
        path = Path(value).expanduser()
        if not path.exists():
            raise ValueError(f"Entrada não encontrada: {path}")
        # Arquivo: ele mesmo. Pasta: todos os arquivos das subpastas, em ordem alfabética
        # sem diferenciar maiúsculas (casefold), para a ordem não depender do sistema.
        candidates = [path] if path.is_file() else sorted(
            (item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix().casefold())
        for candidate in candidates:
            kind = media_kind(candidate)
            if kind is None:
                # Em pastas, outros arquivos (notas, planilhas) passam em silêncio.
                if path.is_file():
                    skipped.append({"file": candidate.name, "reason": "extensao_nao_suportada"})
                continue
            # O mesmo arquivo informado duas vezes (ou dentro de duas pastas) entra uma vez.
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append((resolved, kind))
    return files, skipped


def _optional_text(value, label: str, limit: int) -> str | None:
    """Valida um texto opcional.

    Parâmetros:
    - ``value``: valor recebido (None ou texto).
    - ``label``: nome do campo, para a mensagem de erro.
    - ``limit``: máximo de caracteres.

    Retorna: o texto sem espaços nas pontas, ou None se vazio/ausente.

    Erros: ``ValueError`` se não for texto ou passar do limite.
    """
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{label} deve ser texto com até {limit} caracteres.")
    return value.strip() or None


class _Importer:
    """Estado da importação: originais congelados, imagens do registro e repetidas.

    Atributos:
    - ``originals``: pasta ``output/originals``.
    - ``group``: grupo da edificação/campanha (``building_group``).
    - ``images``: itens do registro, na ordem de importação.
    - ``skipped``: arquivos ignorados (``{"file", "reason", ...}``).
    - ``seen``: ``{sha256 final: nome exibido}`` para detectar fotos repetidas.
    - ``counts``: ``Counter`` por tipo (``"photo"``, ``"video_frame"``).

    Usado só por ``ingest_media``.
    """

    def __init__(self, originals: Path, group: str) -> None:
        """Cria o estado vazio.

        Parâmetros:
        - ``originals``: pasta onde as fotos são congeladas.
        - ``group``: ``building_group`` gravado em cada imagem.
        """
        self.originals, self.group = originals, group
        self.images: list[dict] = []
        self.skipped: list[dict] = []
        self.seen: dict[str, str] = {}      # sha256 final → nome exibido
        self.counts: Counter = Counter()

    def add(self, photo: Path, *, kind: str, display: str, source_file: str, source_digest: str,
            original_path: Path, notes: str, scene: str | None = None, frame: dict | None = None) -> None:
        """Congela a foto e a acrescenta ao registro; repetida (mesmo SHA-256 final) vai para ``skipped``.

        Parâmetros:
        - ``photo``: foto enviada ou quadro extraído.
        - ``kind``: ``"photo"`` ou ``"video_frame"`` (entra em ``counts`` e em ``ingest.kind``).
        - ``display``: nome exibido nas mensagens (ex.: ``"voo.mp4 @ 4.0s"``).
        - ``source_file``: nome do arquivo enviado (foto ou vídeo).
        - ``source_digest``: SHA-256 do arquivo enviado.
        - ``original_path``: caminho do arquivo enviado (gravado em ``original_path``).
        - ``notes``: texto inicial do campo ``notes`` da imagem.
        - ``scene``: ``scene_group`` (o do vídeo para quadros); None cria uma cena por foto.
        - ``frame``: item de ``extract_video_frames`` (quadros) ou None (fotos).

        Retorna: nada; altera ``images``, ``skipped``, ``seen`` e ``counts``.

        Erros: os de ``canonicalize_photo``/``review_image_size`` (tratados em ``ingest_media``).
        """
        # 1. Congela; o nome do arquivo sem extensão é o hash final (image_sha256).
        frozen, canonical = canonicalize_photo(photo, self.originals)
        digest = frozen.stem
        # 2. Repetida (mesmo conteúdo final): só registra quem ela repete.
        if digest in self.seen:
            self.skipped.append({"file": display, "reason": "foto_repetida", "sha256": digest,
                                 "duplicate_of": self.seen[digest]})
            return
        width, height = review_image_size(frozen)
        self.seen[digest] = display
        self.counts[kind] += 1
        # Mesmos campos de suggestions.create_review_for_new_images, mais o bloco "ingest".
        # Cena: a do vídeo para quadros; para fotos, uma por foto (hash final, como no suggest).
        # ATENCAO: estado inicial obrigatório: human_approved False, boxes vazio, severity None
        # (a IA e a importação nunca aprovam nem atribuem gravidade).
        # ATENCAO: mantenha os mesmos campos de suggestions.create_review_for_new_images; o registro
        # é lido por suggest_for_registry, review_render e pilot.
        # EXTENSAO: novo campo por imagem: acrescente-o neste dict (e em
        # suggestions.create_review_for_new_images, para os dois registros continuarem iguais).
        self.images.append({
            "image_sha256": digest, "filename": photo.name, "pages": [], "width": width, "height": height,
            "status": "ambiguous", "boxes": [], "annotator": "sem_anotacao_humana", "notes": notes,
            "visual_review_status": "pending_model_suggestions", "technical_status": "pending_human_review",
            "human_approved": False, "building_group": self.group,
            # scene_group: grupo + 12 primeiros caracteres do hash (da foto, ou do vídeo para quadros).
            "scene_group": scene or f"{self.group}_{digest[:12]}",
            "source_path": str(frozen), "original_path": str(original_path), "severity": None,
            # Rastreabilidade da importação: de qual arquivo, de que instante, se foi girada.
            # EXTENSAO: novo dado de origem por imagem: acrescente a chave neste bloco "ingest".
            "ingest": {
                "kind": kind, "source_file": source_file, "source_sha256": source_digest,
                "frame_time_s": None if frame is None else frame["time_s"],
                "frame_index": None if frame is None else frame["frame_index"],
                "canonicalized_from_sha256": canonical.get("canonicalized_from_sha256"),
                "exif_orientation": canonical.get("exif_orientation"),
            },
        })


def ingest_media(config: ProjectConfig, inputs: list[Path], output: Path, building_group: str, *,
                 every_seconds: float = 2.0, max_frames: int = 200, uploaded_by: str | None = None,
                 campaign_notes: str = "", keep_sources: bool = False, display_name: str | None = None) -> dict:
    """Fotos e vídeos enviados viram uma revisão nova em ``output`` (registro + página).

    ``inputs``: arquivos ou pastas (pastas são varridas recursivamente; só
    extensões suportadas). Vídeos viram quadros a cada ``every_seconds`` (no
    máximo ``max_frames`` por vídeo), todos na mesma cena (``scene_group``).
    Fotos repetidas (mesmo SHA-256 final) entram uma vez; arquivos ilegíveis e
    repetidos ficam em ``skipped``. ValueError se nada for suportado, se
    ``output`` já existir ou se ``building_group`` for vazio ou > 100 caracteres.

    ``keep_sources`` (envio pela plataforma): os arquivos de entrada são
    MOVIDOS para ``output/midia_enviada/`` antes de serem lidos, para que
    ``sources[].path`` e ``original_path`` continuem válidos depois que a área
    temporária do lote for apagada e para que o vídeo inteiro fique guardado.
    Com arquivos do próprio usuário (``import-media``), deixe ``False``.

    ``display_name``: nome digitado no envio ("Galpão B · outubro"), guardado no
    registro para a plataforma mostrar no lugar do nome da pasta. Ignorado
    quando igual à pasta.

    ENTRADA: comando ``import-media`` (cli.py) e o envio de mídia de ``platform_server.py``.

    Parâmetros:
    - ``config``: ``ProjectConfig`` do projeto.
    - ``inputs``: arquivos/pastas de entrada.
    - ``output``: pasta NOVA da revisão (o nome da pasta vira ``version`` do registro).
    - ``building_group``: grupo da edificação/campanha (ex.: ``galpao_b_2026_10``), até 100 caracteres.
    - ``every_seconds`` / ``max_frames``: amostragem dos vídeos (ver ``_validate_sampling``).
    - ``uploaded_by``: quem enviou (até 100 caracteres) ou None.
    - ``campaign_notes``: observações da campanha (até 5000 caracteres).
    - ``keep_sources``: move os arquivos de entrada para dentro da revisão.
    - ``display_name``: nome amigável (até 80 caracteres gravados).

    Retorna: resumo ``{"review", "display_name", "registry", "images", "photos",
    "video_frames", "videos", "skipped", "page"}`` (a CLI imprime este dict).

    Erros: ``ValueError`` para parâmetros inválidos, saída existente, nenhuma
    mídia suportada, nenhum extrator de vídeo ou nenhuma foto importada. Qualquer
    falha depois de criar a pasta deixa ``INCOMPLETO.txt`` nela e propaga o erro.
    """
    # 1. Validações antes de criar qualquer pasta.
    every, limit = _validate_sampling(every_seconds, max_frames)
    if not isinstance(building_group, str) or not building_group.strip() or len(building_group) > 100:
        raise ValueError("Informe o grupo da edificação/campanha (ex.: galpao_b_2026_10).")
    group = building_group.strip()
    uploaded_by = _optional_text(uploaded_by, "uploaded_by", 100)
    if not isinstance(campaign_notes, str) or len(campaign_notes) > 5000:
        raise ValueError("campaign_notes deve ser texto com até 5000 caracteres.")
    # ATENCAO: nunca reaproveita uma pasta existente (não mistura com outra revisão).
    output = Path(output).expanduser()
    if output.exists():
        raise ValueError("Saída já existe; escolha uma nova versão da revisão.")
    files, skipped = _collect_inputs(inputs)
    if not files:
        raise ValueError("Nenhuma foto ou vídeo suportado nas entradas informadas.")
    if any(kind == "video" for _, kind in files):
        _frame_extractor()  # falha antes de criar a pasta se não há OpenCV nem ffmpeg
    taxonomy = load_taxonomy(config.taxonomy_path)
    # 1. Pasta nova; a partir daqui, qualquer falha deixa a marca INCOMPLETO.txt.
    output = output.resolve()
    originals = output / "originals"
    # mkdir sem exist_ok: se outra importação criou a pasta no meio tempo, falha em vez de misturar.
    originals.mkdir(parents=True)
    importer = _Importer(originals, group)
    importer.skipped.extend(skipped)
    # sources: um item por arquivo de entrada (foto ou vídeo), gravado em registry["sources"].
    sources: list[dict] = []
    # seen_videos: {sha256 do vídeo: nome}, para não decodificar o mesmo vídeo duas vezes.
    seen_videos: dict[str, str] = {}
    try:
        # Envio pela plataforma: move os arquivos para dentro da revisão antes de ler.
        if keep_sources:
            files = [(_move_into(path, output / "midia_enviada"), kind) for path, kind in files]
        # 2. Um arquivo de entrada por vez.
        for path, kind in files:
            digest = file_hash(path)
            # EXTENSAO: novo dado por arquivo de origem: acrescente a chave neste dict.
            entry = {"file": path.name, "path": str(path), "kind": kind, "bytes": path.stat().st_size,
                     "sha256": digest, "frames_extracted": None}
            sources.append(entry)
            # Erros de leitura (MEDIA_ERRORS) pulam só este arquivo; a importação continua.
            try:
                # EXTENSAO: novo tipo de mídia (ver media_kind): trate-o aqui com outro
                # ``if kind == "...":`` antes do ramo de vídeo.
                if kind == "photo":
                    # 2a. Foto: congelada por hash (girada antes, se o EXIF pedir).
                    importer.add(path, kind="photo", display=path.name, source_file=path.name, source_digest=digest,
                                 original_path=path,
                                 notes="Foto enviada pela plataforma: nenhuma caixa revisada. "
                                       "Peça as sugestões do modelo (suggest) antes de revisar.")
                    continue
                if digest in seen_videos:
                    # O mesmo vídeo enviado duas vezes não é decodificado de novo.
                    importer.skipped.append({"file": path.name, "reason": "video_repetido", "sha256": digest,
                                             "duplicate_of": seen_videos[digest]})
                    continue
                seen_videos[digest] = path.name
                # 2b. Vídeo: quadros numa pasta temporária dentro da saída; todos na cena do vídeo.
                temporary = Path(tempfile.mkdtemp(prefix="quadros_", dir=output))
                try:
                    frames = extract_video_frames(path, temporary, every, limit)
                    # Cada quadro é congelado como foto; a cena usa o hash do VÍDEO (todos juntos).
                    for frame in frames:
                        importer.add(frame["path"], kind="video_frame", display=f"{path.name} @ {frame['time_s']:.1f}s",
                                     source_file=path.name, source_digest=digest, original_path=path,
                                     scene=f"{group}_{digest[:12]}", frame=frame,
                                     notes=f"Quadro aos {frame['time_s']:.1f} s do vídeo {path.name}: "
                                           "nenhuma caixa revisada.")
                finally:
                    # Os JPEG temporários já foram copiados para originals/; apaga a pasta.
                    shutil.rmtree(temporary, ignore_errors=True)
                entry["frames_extracted"] = len(frames)
                entry["every_seconds_used"] = frames[0]["every_seconds_used"]
            except MEDIA_ERRORS as error:
                # ATENCAO: se um quadro falhar no meio do vídeo, os quadros já adicionados
                # continuam no registro, mas "frames_extracted" deste vídeo fica None.
                importer.skipped.append({"file": path.name, "reason": "arquivo_ilegivel", "detail": str(error)[:300]})
        # Sem nenhuma foto válida, a revisão não faz sentido: lista até 10 motivos.
        if not importer.images:
            raise ValueError("Nenhuma foto importada: " + "; ".join(
                f"{item['file']} ({item['reason']})" for item in importer.skipped[:10]))
        # 3. Registro no schema das revisões e página gerada ao lado.
        # ATENCAO: taxonomy_sha256 amarra o registro à taxonomia atual (IDs de classe = contrato).
        # EXTENSAO: novo campo no topo do registro: acrescente a chave neste dict; novo
        # parâmetro do envio: acrescente-o no bloco "ingest".
        registry_path = output / "registry.json"
        write_json(registry_path, {
            "schema_version": REVIEW_SCHEMA, "version": output.name,
            # display_name só entra se for diferente do nome da pasta (cortado em 80 caracteres).
            **({"display_name": display_name.strip()[:80]} if display_name and display_name.strip() and display_name.strip() != output.name else {}),
            "taxonomy_sha256": file_hash(config.taxonomy_path), "taxonomy_version": taxonomy["version"],
            "created_at": datetime.now(timezone.utc).isoformat(), "scope": INGEST_SCOPE,
            "sources": sources,
            "ingest": {"tool": "platform-upload", "every_seconds": every, "max_frames": limit,
                       "uploaded_by": uploaded_by, "notes": campaign_notes},
            "images": importer.images,
        })
        page = render_review_package(config, registry_path)
    except BaseException:
        # Uma pasta pela metade não pode ser confundida com uma revisão válida.
        (output / "INCOMPLETO.txt").write_text("Importação interrompida; não utilize esta pasta.\n", encoding="utf-8")
        raise
    # 4. Resumo devolvido à CLI e ao servidor.
    # EXTENSAO: nova informação no resumo: acrescente a chave neste dict (o servidor repassa à página).
    return {"review": output.name, "display_name": review_name(output.name, display_name if display_name and display_name.strip() else None),
            "registry": str(registry_path), "images": len(importer.images),
            "photos": importer.counts["photo"], "video_frames": importer.counts["video_frame"],
            "videos": sum(item["kind"] == "video" for item in sources), "skipped": importer.skipped,
            "page": str(page)}
