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
"""

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

from .config import ProjectConfig, load_taxonomy
from .dataset import IMAGE_EXTENSIONS
from .io import file_hash, write_json
from .review_data import REVIEW_SCHEMA, review_image_size
from .review_render import render_review_package

__all__ = ["IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "WEB_IMAGE_SUFFIXES", "media_kind", "slugify_review_name",
           "extract_video_frames", "canonicalize_photo", "ingest_media"]

# Contêineres de vídeo aceitos (a decodificação fica com OpenCV ou ffmpeg).
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"})
# Formatos que o navegador exibe e o servidor serve; fotos nos demais (TIFF, BMP) são regravadas como JPEG ao congelar.
WEB_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
# Limites da amostragem de quadros: intervalo em segundos e quadros por vídeo.
EVERY_SECONDS_RANGE = (0.2, 600.0)
MAX_FRAMES_RANGE = (1, 2000)
INGEST_SCOPE = "midia_importada_pela_plataforma_pendente_de_revisao"
FRAME_EXTRACTOR_ERROR = "Extração de quadros requer OpenCV (vem com a Ultralytics) ou ffmpeg no PATH."
# Erros de leitura de um arquivo de mídia: o arquivo é pulado e registrado, a importação segue.
# ImportError entra porque a Ultralytics, quando importada no mesmo processo, troca ``Image.open`` por uma
# versão que tenta instalar ``pi-heif`` ao falhar na leitura; sem rede, um arquivo ilegível vira ImportError.
MEDIA_ERRORS = (OSError, ValueError, SyntaxError, ImportError, Image.DecompressionBombError)


# ---------------------------------------------------------------------------
# Nomes e tipos de arquivo.
# ---------------------------------------------------------------------------

def media_kind(path: Path) -> str | None:
    """``"photo"``, ``"video"`` ou None (extensão não suportada)."""
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
    """
    if not isinstance(name, str):
        raise ValueError("Nome da revisão deve ser texto.")
    # NFKD separa letra e acento; os acentos (marcas combinantes) são descartados.
    plain = "".join(char for char in unicodedata.normalize("NFKD", name) if not unicodedata.combining(char))
    slug = re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")
    if not slug:
        raise ValueError("Nome da revisão vazio após a normalização; use letras ou números.")
    if not slug[0].isalpha():
        slug = f"r_{slug}"
    return slug[:60].rstrip("_")


def _validate_sampling(every_seconds: float, max_frames: int) -> tuple[float, int]:
    """Intervalo entre quadros e máximo de quadros dentro dos limites."""
    try:
        every = float(every_seconds)
    except (TypeError, ValueError):
        raise ValueError("every_seconds deve ser um número de segundos.") from None
    if not math.isfinite(every) or not EVERY_SECONDS_RANGE[0] <= every <= EVERY_SECONDS_RANGE[1]:
        raise ValueError(f"every_seconds deve estar entre {EVERY_SECONDS_RANGE[0]} e {EVERY_SECONDS_RANGE[1]:.0f} segundos.")
    if (isinstance(max_frames, bool) or not isinstance(max_frames, (int, float)) or not math.isfinite(max_frames)
            or int(max_frames) != max_frames or not MAX_FRAMES_RANGE[0] <= int(max_frames) <= MAX_FRAMES_RANGE[1]):
        raise ValueError(f"max_frames deve ser um inteiro entre {MAX_FRAMES_RANGE[0]} e {MAX_FRAMES_RANGE[1]}.")
    return every, int(max_frames)


# ---------------------------------------------------------------------------
# Quadros de vídeo (OpenCV quando importável; senão ffmpeg do PATH).
# ---------------------------------------------------------------------------

def _frame_extractor() -> tuple[str, str | None]:
    """``("opencv", None)`` ou ``("ffmpeg", caminho)``; ValueError sem nenhum dos dois."""
    try:
        import cv2  # noqa: F401  (vem com a Ultralytics; opcional aqui)
        return "opencv", None
    except ImportError:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return "ffmpeg", ffmpeg
    raise ValueError(FRAME_EXTRACTOR_ERROR)


def _sample_times(duration: float, every_seconds: float, max_frames: int) -> tuple[list[float], float]:
    """Instantes amostrados (0, N, 2N… < duração) e o intervalo efetivamente usado.

    Se o vídeo renderia mais de ``max_frames`` quadros, o intervalo cresce para
    ``duração / max_frames``, mantendo a amostragem uniforme do início ao fim.
    """
    if not duration > 0:
        raise ValueError("Vídeo sem duração legível.")
    count = max(1, math.ceil(duration / every_seconds - 1e-9))
    every_used = every_seconds
    if count > max_frames:
        count, every_used = max_frames, duration / max_frames
    return [round(index * every_used, 3) for index in range(count)], every_used


def _frame_path(output_dir: Path, stem: str, time_s: float) -> Path:
    """``<stem>_t<segundos com 1 casa>.jpg``; nunca sobrescreve um quadro existente."""
    path = output_dir / f"{stem}_t{time_s:.1f}.jpg"
    if path.exists():
        raise ValueError(f"Quadro já existe; use uma pasta de saída nova: {path.name}.")
    return path


def _frame_record(path: Path, time_s: float, index: int, every_used: float) -> dict:
    return {"path": path, "time_s": float(time_s), "frame_index": index, "every_seconds_used": float(every_used)}


def _extract_with_opencv(video: Path, output_dir: Path, every_seconds: float, max_frames: int) -> list[dict]:
    """Leitura sequencial (sem seek, que é impreciso em alguns contêineres)."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    try:
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
            capture.release()
            capture = cv2.VideoCapture(str(video))
        times, every_used = _sample_times(total / fps, every_seconds, max_frames)
        # Um quadro de origem por instante; instantes que caem no mesmo quadro
        # (vídeos com fps muito baixo) rendem um só arquivo.
        targets: dict[int, float] = {}
        for time_s in times:
            targets.setdefault(min(total - 1, round(time_s * fps)), time_s)
        frames: list[dict] = []
        position, last = 0, max(targets)
        while position <= last and capture.grab():
            if position in targets:
                ok, pixels = capture.retrieve()
                if not ok:
                    break
                time_s = targets[position]
                path = _frame_path(output_dir, video.stem, time_s)
                Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).save(path, "JPEG", quality=95)
                frames.append(_frame_record(path, time_s, len(frames), every_used))
            position += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Nenhum quadro legível em {video.name}.")
    return frames


def _ffmpeg_duration(video: Path, ffmpeg: str) -> float:
    """Duração em segundos via ffprobe ou, na falta dele, pelo cabeçalho que o ffmpeg imprime."""
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe", path=str(Path(ffmpeg).parent))
    if ffprobe:
        probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", str(video)],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL)
        try:
            return float(probe.stdout.strip())
        except ValueError:
            pass
    header = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-i", str(video)],
                            capture_output=True, text=True, stdin=subprocess.DEVNULL)
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
    """
    times, every_used = _sample_times(_ffmpeg_duration(video, ffmpeg), every_seconds, max_frames)
    rate = 1 / Fraction(every_used).limit_denominator(10_000)  # racional exato: evita 1/1.4999999
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ffmpeg_", dir=output_dir) as temporary:
        pattern = Path(temporary) / "quadro_%06d.bmp"
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video),
                   "-vf", f"fps={rate.numerator}/{rate.denominator}:start_time=0:round=up",
                   "-frames:v", str(len(times)), "-start_number", "0", str(pattern)]
        completed = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if completed.returncode != 0:
            raise ValueError(f"ffmpeg não conseguiu extrair quadros de {video.name}: {completed.stderr.strip()[:300]}")
        frames: list[dict] = []
        for index, bitmap in enumerate(sorted(Path(temporary).glob("quadro_*.bmp"))):
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
    """
    every, limit = _validate_sampling(every_seconds, max_frames)
    video, output_dir = Path(video), Path(output_dir)
    if not video.is_file():
        raise ValueError(f"Vídeo não encontrado: {video}")
    backend, ffmpeg = _frame_extractor()
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend == "opencv":
        return _extract_with_opencv(video, output_dir, every, limit)
    return _extract_with_ffmpeg(video, output_dir, every, limit, ffmpeg)


# ---------------------------------------------------------------------------
# Foto congelada por hash (rotação EXIF aplicada de forma explícita).
# ---------------------------------------------------------------------------

# Mesma tabela de ImageOps.exif_transpose, usada se o EXIF da foto não puder ser regravado.
_TRANSPOSE_BY_ORIENTATION = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def _transposed(image: Image.Image, orientation: int) -> Image.Image:
    """Aplica a rotação EXIF; mantém o resto do EXIF (GPS, data) quando ele pode ser regravado."""
    try:
        transposed = ImageOps.exif_transpose(image)
    except Exception:  # EXIF exótico (maker notes): gira sem metadados, mas não perde a foto
        method = _TRANSPOSE_BY_ORIENTATION.get(orientation)
        transposed = image.transpose(method) if method else image.copy()
        transposed.info.pop("exif", None)
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
    """
    source, destination_dir = Path(source), Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    original_digest = file_hash(source)
    with Image.open(source) as image:
        orientation = int(image.getexif().get(274, 1))
        if orientation == 1 and source.suffix.lower() in WEB_IMAGE_SUFFIXES:
            target = destination_dir / f"{original_digest}{source.suffix.lower()}"
            if not target.exists():
                shutil.copyfile(source, target)
            elif file_hash(target) != original_digest:
                raise ValueError(f"Arquivo diferente já ocupa {target.name}.")
            review_image_size(target)  # cópia ilegível ou truncada falha aqui
            return target, {}
        transposed = _transposed(image, orientation)
    options = {"quality": 95}
    if transposed.info.get("icc_profile"):
        options["icc_profile"] = transposed.info["icc_profile"]
    temporary = destination_dir / f".canonical_{uuid4().hex}.jpg"
    try:
        try:
            transposed.save(temporary, "JPEG", exif=transposed.info.get("exif") or b"", **options)
        except Exception:
            transposed.save(temporary, "JPEG", **options)  # sem EXIF, se ele não puder ser regravado
        digest = file_hash(temporary)
        target = destination_dir / f"{digest}.jpg"
        if target.exists():
            temporary.unlink()  # mesmo conteúdo já congelado
        else:
            temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    review_image_size(target)  # garante que a orientação foi removida junto com a rotação
    return target, {"canonicalized_from_sha256": original_digest, "exif_orientation": orientation}


# ---------------------------------------------------------------------------
# Importação: revisão nova com fotos, quadros de vídeo, registro e página.
# ---------------------------------------------------------------------------

def _move_into(path: Path, directory: Path) -> Path:
    """Move ``path`` para ``directory`` sem sobrescrever (``nome_2.ext`` se o nome já existir)."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    counter = 1
    while target.exists():
        counter += 1
        target = directory / f"{path.stem}_{counter}{path.suffix}"
    shutil.move(str(path), str(target))
    return target.resolve()


def _collect_inputs(inputs) -> tuple[list[tuple[Path, str]], list[dict]]:
    """Arquivos suportados (na ordem informada; pastas em ordem alfabética) e os ignorados."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    files, skipped, seen = [], [], set()
    for value in inputs:
        path = Path(value).expanduser()
        if not path.exists():
            raise ValueError(f"Entrada não encontrada: {path}")
        candidates = [path] if path.is_file() else sorted(
            (item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix().casefold())
        for candidate in candidates:
            kind = media_kind(candidate)
            if kind is None:
                # Em pastas, outros arquivos (notas, planilhas) passam em silêncio.
                if path.is_file():
                    skipped.append({"file": candidate.name, "reason": "extensao_nao_suportada"})
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append((resolved, kind))
    return files, skipped


def _optional_text(value, label: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{label} deve ser texto com até {limit} caracteres.")
    return value.strip() or None


class _Importer:
    """Estado da importação: originais congelados, imagens do registro e repetidas."""

    def __init__(self, originals: Path, group: str) -> None:
        self.originals, self.group = originals, group
        self.images: list[dict] = []
        self.skipped: list[dict] = []
        self.seen: dict[str, str] = {}      # sha256 final → nome exibido
        self.counts: Counter = Counter()

    def add(self, photo: Path, *, kind: str, display: str, source_file: str, source_digest: str,
            original_path: Path, notes: str, scene: str | None = None, frame: dict | None = None) -> None:
        """Congela a foto e a acrescenta ao registro; repetida (mesmo SHA-256 final) vai para ``skipped``."""
        frozen, canonical = canonicalize_photo(photo, self.originals)
        digest = frozen.stem
        if digest in self.seen:
            self.skipped.append({"file": display, "reason": "foto_repetida", "sha256": digest,
                                 "duplicate_of": self.seen[digest]})
            return
        width, height = review_image_size(frozen)
        self.seen[digest] = display
        self.counts[kind] += 1
        # Mesmos campos de suggestions.create_review_for_new_images, mais o bloco "ingest".
        # Cena: a do vídeo para quadros; para fotos, uma por foto (hash final, como no suggest).
        self.images.append({
            "image_sha256": digest, "filename": photo.name, "pages": [], "width": width, "height": height,
            "status": "ambiguous", "boxes": [], "annotator": "sem_anotacao_humana", "notes": notes,
            "visual_review_status": "pending_model_suggestions", "technical_status": "pending_human_review",
            "human_approved": False, "building_group": self.group,
            "scene_group": scene or f"{self.group}_{digest[:12]}",
            "source_path": str(frozen), "original_path": str(original_path), "severity": None,
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
                 campaign_notes: str = "", keep_sources: bool = False) -> dict:
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
    """
    every, limit = _validate_sampling(every_seconds, max_frames)
    if not isinstance(building_group, str) or not building_group.strip() or len(building_group) > 100:
        raise ValueError("Informe o grupo da edificação/campanha (ex.: galpao_b_2026_10).")
    group = building_group.strip()
    uploaded_by = _optional_text(uploaded_by, "uploaded_by", 100)
    if not isinstance(campaign_notes, str) or len(campaign_notes) > 5000:
        raise ValueError("campaign_notes deve ser texto com até 5000 caracteres.")
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
    originals.mkdir(parents=True)
    importer = _Importer(originals, group)
    importer.skipped.extend(skipped)
    sources: list[dict] = []
    seen_videos: dict[str, str] = {}
    try:
        if keep_sources:
            files = [(_move_into(path, output / "midia_enviada"), kind) for path, kind in files]
        for path, kind in files:
            digest = file_hash(path)
            entry = {"file": path.name, "path": str(path), "kind": kind, "bytes": path.stat().st_size,
                     "sha256": digest, "frames_extracted": None}
            sources.append(entry)
            try:
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
                    for frame in frames:
                        importer.add(frame["path"], kind="video_frame", display=f"{path.name} @ {frame['time_s']:.1f}s",
                                     source_file=path.name, source_digest=digest, original_path=path,
                                     scene=f"{group}_{digest[:12]}", frame=frame,
                                     notes=f"Quadro aos {frame['time_s']:.1f} s do vídeo {path.name}: "
                                           "nenhuma caixa revisada.")
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
                entry["frames_extracted"] = len(frames)
                entry["every_seconds_used"] = frames[0]["every_seconds_used"]
            except MEDIA_ERRORS as error:
                importer.skipped.append({"file": path.name, "reason": "arquivo_ilegivel", "detail": str(error)[:300]})
        if not importer.images:
            raise ValueError("Nenhuma foto importada: " + "; ".join(
                f"{item['file']} ({item['reason']})" for item in importer.skipped[:10]))
        # 3. Registro no schema das revisões e página gerada ao lado.
        registry_path = output / "registry.json"
        write_json(registry_path, {
            "schema_version": REVIEW_SCHEMA, "version": output.name,
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
    return {"review": output.name, "registry": str(registry_path), "images": len(importer.images),
            "photos": importer.counts["photo"], "video_frames": importer.counts["video_frame"],
            "videos": sum(item["kind"] == "video" for item in sources), "skipped": importer.skipped,
            "page": str(page)}
