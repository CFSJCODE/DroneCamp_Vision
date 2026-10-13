"""Ponte com a Ultralytics: carrega o modelo YOLO e confere suas classes.

Função no projeto: é o único arquivo que importa a biblioteca Ultralytics. Treino,
predição, sugestões e exportação pedem o modelo por aqui.

O que faz:
- ``load_detector``: abre um checkpoint ``.pt`` local (ou baixa um YOLO26 oficial).
- ``_ultralytics``: aponta configurações e caches da biblioteca para dentro do projeto.
- ``load_exported_detector``: abre um ``.onnx`` exportado, só para conferir paridade.
- ``load_inference_model``: abre ``.pt`` ou ``.onnx`` para sugerir/avaliar; o ONNX
  roda por padrão na GPU via DirectML (gancho em ``ONNXBackend.load_model``).
- ``onnx_source_checkpoint``: acha (e confere pelo SHA-256) o ``.pt`` que gerou um ``.onnx``.
- ``check_domain_names``: recusa pesos cujas classes não são as ativas do projeto (aceita pesos v6 pelo mapa de fusão da v7).
- ``runtime_info``: versões de Python/torch/Ultralytics e hash do checkpoint.
- Internas da GPU: ``_directml_session`` (sessão ONNX Runtime no DirectML) e
  ``_install_directml_hook`` (troca a sessão criada pela Ultralytics).

Quando mexer: para trocar de biblioteca. A família/versão do YOLO (26 hoje, 27
quando sair) fica em ``model_family.py``; trocar de versão ou de tamanho
(n/s/m/l/x) é mudar ``model`` e ``pilot.model`` em ``configs/project.yaml``.

Quem chama:
- ``load_detector``: training (train, train_pilot, resume_pilot, tune),
  prediction, calibration, exporting e ``load_inference_model``.
- ``load_inference_model``: suggestions, model_gate, gpu_eval, autolabel,
  training.evaluate (comandos ``suggest``, ``compare-models``, ``evaluate``...).
- ``check_domain_names``: quem carrega pesos especializados antes de usar.
- ``runtime_info``: quem grava ``execution.json``/``evaluation.json``.

O que lê e grava no disco:
- Lê pesos ``.pt``/``.onnx`` e o ``export.json`` ao lado do ``.onnx``.
- Pode BAIXAR um peso oficial para ``sistema_ia/models/`` (só nomes oficiais).
- Grava as configurações da Ultralytics em ``sistema_ia/.runtime/``.

Onde mexer (casos concretos):
- Outro provedor de GPU para o ONNX (ex.: CUDA): acrescente em ``ONNX_PROVIDERS``,
  crie a sessão correspondente ao lado de ``_directml_session`` e trate o novo
  valor em ``load_inference_model`` (ver EXTENSAO:). Mude também as opções
  ``--onnx-provider`` em ``cli.py`` e ``gpu_eval.PROVIDERS`` se for o caso.
- Outro campo de reprodutibilidade no ``execution.json``: acrescente em ``runtime_info``.
- Aceitar outro formato de peso além de ``.pt``/``.onnx``: ``load_detector`` e
  ``load_inference_model``.
"""

from pathlib import Path
import json
import os
import platform

from .config import ProjectConfig
from .io import file_hash
from .model_family import describe_checkpoint, ensure_downloadable, parse_model_name


# ---------------------------------------------------------------------------
# Carregamento do modelo PyTorch (.pt) e configuração da Ultralytics
# ---------------------------------------------------------------------------

def load_detector(config: ProjectConfig, weights: str | None = None):
    """Carregue checkpoint local ou peso oficial de uma família publicada; sem fallback silencioso.

    Parâmetros:
        config (ProjectConfig): configuração do projeto (``root``, ``model``).
        weights (str | None): caminho do ``.pt`` (absoluto ou relativo a
            ``sistema_ia``) ou só o nome oficial (ex.: ``"yolo26l.pt"``).
            ``None`` usa ``config.model``.

    Retorna:
        ultralytics.YOLO: o modelo carregado, com ``task == "detect"``.

    Erros:
        ValueError: se a extensão não for ``.pt``; se um caminho com pasta não
            existir ("Checkpoint local não encontrado..."); se o nome não for
            oficial/baixável (via ``model_family.ensure_downloadable``); se o
            checkpoint for de outra tarefa (segmentação, pose...).

    Chamada por: training, prediction, calibration, exporting e ``load_inference_model``.
    """
    # Só checkpoints PyTorch: o ONNX é conferido à parte (exporting.py).
    selected = weights or config.model
    if Path(selected).suffix.lower() != ".pt":
        raise ValueError("Este pipeline usa checkpoints PyTorch .pt. Para ONNX, valide o runtime de destino separadamente.")
    # Caminho relativo é resolvido a partir da pasta sistema_ia.
    local = Path(selected).expanduser()
    if not local.is_absolute():
        local = config.root / local
    # Sem arquivo local, só aceita baixar pesos oficiais de detecção de família
    # publicada (model_family.py) para models/; YOLO27 falha com explicação até sair.
    if not local.is_file():
        # Um caminho de checkpoint que não existe é erro: nunca trocar por pesos genéricos
        # só porque o nome do arquivo parece oficial (ex.: runs/anterior/yolo26l.pt).
        if "/" in str(selected) or "\\" in str(selected):
            raise ValueError(f"Checkpoint local não encontrado: {local}. Para baixar um peso oficial, informe só o nome "
                             "(ex.: yolo26l.pt).")
        ensure_downloadable(parse_model_name(selected), local)
        # Nome oficial sem pasta: o download vai para sistema_ia/models/<nome>.
        local = config.root / "models" / selected
        local.parent.mkdir(parents=True, exist_ok=True)
    # Abre o modelo pela Ultralytics e garante que é de detecção (caixas).
    YOLO = _ultralytics(config)
    model = YOLO(str(local), task="detect")
    if model.task != "detect":
        raise ValueError(f"Checkpoint é da tarefa {model.task!r}; esperado detect.")
    return model


# ATENCAO: toda importação da Ultralytics deve passar por aqui; importar ``ultralytics`` antes
# de definir YOLO_CONFIG_DIR faz a biblioteca gravar configurações fora do projeto.
def _ultralytics(config: ProjectConfig):
    """Restrinja ajustes/cache à pasta do projeto antes de importar a biblioteca.

    Parâmetros:
        config (ProjectConfig): usa ``config.root`` (pasta ``sistema_ia``).

    Retorna:
        type: a classe ``ultralytics.YOLO``, pronta para instanciar.

    Efeitos: define as variáveis de ambiente ``YOLO_CONFIG_DIR`` e
    ``YOLO_AUTOINSTALL`` (se ainda não definida) e atualiza ``settings`` da
    Ultralytics (``datasets_dir``, ``weights_dir``, ``runs_dir``, ``sync``).
    """
    # Configurações da Ultralytics em sistema_ia/.runtime; sem instalar pacotes sozinha.
    config_directory = config.root / ".runtime"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
    # setdefault: quem quiser permitir a autoinstalação pode definir a variável antes.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    
    # Importação tardia: só quem realmente usa o modelo paga o tempo de importar torch/Ultralytics.
    from ultralytics import YOLO
    from ultralytics import settings

    # Datasets, pesos e execuções sempre dentro de sistema_ia; sem telemetria (sync).
    # EXTENSAO: outro ajuste global da Ultralytics (ex.: desligar o TensorBoard) entra neste dict.
    settings.update({"sync": False, "datasets_dir": str(config.root / "data"),
                     "weights_dir": str(config.root / "models"), "runs_dir": str(config.root / "runs")})
    return YOLO


def load_exported_detector(config: ProjectConfig, path: Path):
    """Carregue um ONNX local já exportado, somente para conferir paridade.

    Parâmetros:
        config (ProjectConfig): configuração (para ``_ultralytics``).
        path (Path): arquivo ``.onnx`` existente.

    Retorna:
        ultralytics.YOLO: o modelo ONNX aberto pela Ultralytics (ONNX Runtime CPU,
        a não ser que o gancho DirectML esteja instalado para este arquivo).

    Erros:
        ValueError: "Artefato ONNX local não encontrado: ..." se a extensão não for
            ``.onnx`` ou o arquivo não existir.

    Chamada por: exporting (paridade) e ``load_inference_model``.
    """
    path = Path(path).resolve()
    if path.suffix.lower() != ".onnx" or not path.is_file():
        raise ValueError(f"Artefato ONNX local não encontrado: {path}.")
    return _ultralytics(config)(str(path), task="detect")


# ---------------------------------------------------------------------------
# Inferência na GPU (DirectML): ONNX exportado + ONNX Runtime com DmlExecutionProvider.
# A Ultralytics só cria sessões ONNX com CPU/CUDA/CoreML e recria a sessão sempre
# que o predictor é refeito; por isso o gancho fica em ONNXBackend.load_model.
# Medido na APU Ryzen 5 4600G (Vega): 115 ms por inferência 640×640 no DirectML,
# 356 ms no ONNX Runtime CPU e 535 ms no PyTorch CPU; caixas iguais às do .pt.
# ---------------------------------------------------------------------------

# Caminhos absolutos (str) dos .onnx que devem rodar no DirectML. O gancho só troca a
# sessão desses arquivos; os demais ONNX continuam na sessão padrão da Ultralytics.
# ATENCAO: é estado global do processo; um arquivo marcado aqui fica no DirectML até o fim do processo.
_DIRECTML_WEIGHTS: set[str] = set()
# Valores aceitos em onnx_provider (opção --onnx-provider): "directml" = GPU, "cpu" = ONNX Runtime CPU.
# EXTENSAO: provedor novo (ex.: "cuda") = valor novo aqui + sessão própria + ramo em load_inference_model.
ONNX_PROVIDERS = ("directml", "cpu")


def _directml_session(weight: str):
    """Sessão DirectML; falha alto se o provedor não estiver ativo (nada de CPU silenciosa).

    Parâmetros:
        weight (str): caminho do arquivo ``.onnx``.

    Retorna:
        onnxruntime.InferenceSession: sessão cujo primeiro provedor é o DirectML
        (o CPU fica só como reserva para operações que o DirectML não suporta).

    Erros:
        RuntimeError: "DirectML indisponível..." se o pacote instalado for o
            ``onnxruntime`` comum; "ONNX Runtime não ativou o DirectML..." se a
            sessão tiver caído para a CPU.

    Chamada por: ``load_inference_model`` (teste antecipado) e pelo gancho
    instalado em ``_install_directml_hook``.
    """
    import onnxruntime

    # Sem o pacote onnxruntime-directml o provedor nem aparece na lista.
    if "DmlExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("DirectML indisponível: instale onnxruntime-directml (não o onnxruntime comum) "
                           "ou use --onnx-provider cpu.")
    options = onnxruntime.SessionOptions()
    # Exigências do DirectML: sem padrão de memória e execução sequencial.
    options.enable_mem_pattern = False
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    # Ordem dos provedores = preferência: DirectML primeiro, CPU só como reserva.
    session = onnxruntime.InferenceSession(weight, options, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    # Confere que o DirectML ficou de fato como provedor principal.
    if session.get_providers()[0] != "DmlExecutionProvider":
        raise RuntimeError(f"ONNX Runtime não ativou o DirectML para {weight}.")
    return session


# ATENCAO: isto altera uma classe da Ultralytics em tempo de execução (monkey patch). Se uma
# versão nova da Ultralytics mudar ultralytics.nn.backends.onnx ou a assinatura de
# ONNXBackend.load_model, o import ou o gancho falham: confira aqui ao atualizar a biblioteca.
def _install_directml_hook() -> None:
    """Instala (uma vez por processo) o gancho que troca a sessão ONNX pela do DirectML.

    A Ultralytics cria a sessão ONNX Runtime dentro de ``ONNXBackend.load_model``
    sempre que monta o predictor. O gancho chama o método original e, se o arquivo
    estiver em ``_DIRECTML_WEIGHTS``, substitui ``self.session`` por uma sessão
    criada em ``_directml_session``.

    Parâmetros: nenhum.

    Retorna:
        None. É idempotente: a marca ``_dronecamp_directml`` no método evita
        instalar o gancho duas vezes (o que chamaria o original em cascata).

    Chamada por: ``load_inference_model`` quando ``onnx_provider == "directml"``.
    """
    from ultralytics.nn.backends import onnx as onnx_backend

    # Já instalado neste processo: nada a fazer.
    if getattr(onnx_backend.ONNXBackend.load_model, "_dronecamp_directml", False):
        return
    original = onnx_backend.ONNXBackend.load_model

    def load_model(self, weight):
        """Substituto de ``ONNXBackend.load_model``: carrega como a Ultralytics e troca a sessão.

        Parâmetros:
            self: a instância de ``ONNXBackend`` da Ultralytics.
            weight: caminho do ``.onnx`` que a Ultralytics está abrindo.

        Retorna:
            None (o efeito é ``self.session`` apontar para a sessão DirectML).
        """
        original(self, weight)
        # Só os arquivos registrados por load_inference_model trocam de sessão.
        if str(Path(weight).resolve()) in _DIRECTML_WEIGHTS:
            self.session = _directml_session(str(weight))
            # A linha "Using ONNX Runtime ... CPUExecutionProvider" acima é da Ultralytics, antes da troca.
            onnx_backend.LOGGER.info(f"DroneCamp: sessão ONNX trocada para {self.session.get_providers()[0]} (GPU).")

    # Marca o substituto (para a checagem de idempotência) e o instala na classe.
    load_model._dronecamp_directml = True
    onnx_backend.ONNXBackend.load_model = load_model


# ---------------------------------------------------------------------------
# Modelo para inferência (.pt ou .onnx) e procedência do ONNX
# ---------------------------------------------------------------------------

def onnx_source_checkpoint(config: ProjectConfig, onnx_path: Path) -> Path:
    """O .pt que gerou o ONNX (export.json), conferido pelo SHA-256 gravado na exportação.

    Lê ``export.json`` (gravado pelo comando ``export`` ao lado do ``.onnx``) e
    confere duas coisas: o ``.pt`` de origem ainda é o mesmo (hash em
    ``runtime.checkpoint_sha256``) e o ``.onnx`` não mudou (hash em ``sha256``).

    Parâmetros:
        config (ProjectConfig): usa ``config.root`` para caminhos relativos.
        onnx_path (Path): o arquivo ``.onnx``.

    Retorna:
        Path: caminho absoluto do ``.pt`` de origem (usado para reconhecer pesos
        piloto e as fotos de treino daquele modelo).

    Erros:
        ValueError: sem ``export.json`` legível; ``.pt`` ausente ou com hash
            diferente; ``.onnx`` com hash diferente do exportado.

    Chamada por: ``load_inference_model``.
    """
    manifest = Path(onnx_path).resolve().parent / "export.json"
    try:
        export = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"ONNX sem export.json ao lado ({manifest}); exporte com o comando export.") from error
    # source_weights pode ter sido gravado no Windows ("\"): normaliza para "/" antes de montar o Path.
    source = Path(str(export.get("source_weights", "")).replace("\\", "/"))
    source = source if source.is_absolute() else config.root / source
    # ATENCAO: os dois hashes impedem usar um ONNX cujo .pt de origem foi trocado ou que foi
    # regravado depois da exportação (a paridade medida no export deixaria de valer).
    if not source.is_file() or file_hash(source) != export.get("runtime", {}).get("checkpoint_sha256"):
        raise ValueError(f"O .pt de origem do ONNX não confere com export.json: {source}")
    if file_hash(Path(onnx_path)) != export.get("sha256"):
        raise ValueError(f"O ONNX mudou desde a exportação: {onnx_path}")
    return source.resolve()


# ENTRADA: modelo usado por suggest, compare-models/measure-models, evaluate e gpu-eval-worker.
def load_inference_model(config: ProjectConfig, weights: str, onnx_provider: str = "directml"):
    """Abre .pt (PyTorch) ou .onnx (ONNX Runtime; DirectML por padrão) para sugerir e avaliar.

    O modelo devolvido ganha ``dronecamp_source`` (o .pt de origem, para identificar
    piloto e fotos de treino) e ``dronecamp_runtime`` (registrado nas saídas).

    Parâmetros:
        config (ProjectConfig): configuração do projeto.
        weights (str): caminho do ``.pt`` ou do ``.onnx`` (ou nome oficial ``.pt``).
        onnx_provider (str): ``"directml"`` (GPU, padrão) ou ``"cpu"``; só vale para ``.onnx``.

    Retorna:
        ultralytics.YOLO com atributos extras:
        ``dronecamp_source`` (Path do ``.pt`` de origem), ``dronecamp_runtime``
        (``"pytorch-cpu"``, ``"onnx-directml"`` ou ``"onnx-cpu"``) e, só no ONNX,
        ``dronecamp_onnx`` (Path do ``.onnx``).

    Erros:
        ValueError: provedor inválido, ONNX sem ``export.json`` válido ou erros
            de ``load_detector``.
        RuntimeError: DirectML pedido mas indisponível (falha antes de qualquer foto).
    """
    path = Path(weights).expanduser()
    # Ramo PyTorch (.pt): o mesmo carregamento do treino.
    if path.suffix.lower() != ".onnx":
        model = load_detector(config, weights)
        model.dronecamp_source = Path(model.ckpt_path).resolve()
        # ATENCAO: o rótulo diz "cpu" porque config.device é "cpu" hoje; ele não lê config.device.
        model.dronecamp_runtime = "pytorch-cpu"
        return model
    # Ramo ONNX: valida o provedor, confere a procedência e abre o arquivo.
    if onnx_provider not in ONNX_PROVIDERS:
        raise ValueError(f"onnx_provider deve ser um de {ONNX_PROVIDERS}.")
    path = (path if path.is_absolute() else config.root / path).resolve()
    source = onnx_source_checkpoint(config, path)
    model = load_exported_detector(config, path)
    # EXTENSAO: provedor novo de GPU = novo ramo aqui (testar a sessão, instalar o gancho, registrar o arquivo).
    if onnx_provider == "directml":
        _directml_session(str(path))  # falha cedo, antes de qualquer foto
        _install_directml_hook()
        # Registra o arquivo: o gancho troca a sessão quando a Ultralytics montar o predictor.
        _DIRECTML_WEIGHTS.add(str(path))
    model.dronecamp_source = source
    model.dronecamp_onnx = path
    model.dronecamp_runtime = f"onnx-{onnx_provider}"
    return model


# ---------------------------------------------------------------------------
# Contrato das classes e registro do ambiente
# ---------------------------------------------------------------------------

# ATENCAO: se aparecer "Checkpoint não corresponde às N classes", os pesos foram treinados com
# outra lista de classes ativas (ex.: depois de ativar uma classe nova). Só um treino novo resolve.
def check_domain_names(model, expected: list[str], allow_prefix: bool = False, taxonomy: dict | None = None) -> list[str]:
    """Evite apresentar classes COCO como patologias: nomes precisam coincidir.

    Parâmetros:
        model: modelo carregado (``.names`` é dict ``{id: nome}`` ou lista).
        expected (list[str]): ``config.detection_names(...)``, na ordem dos IDs.
        allow_prefix (bool): aceita pesos treinados numa taxonomia anterior cujas
            classes são o início exato da atual (a migração só acrescenta classes no
            fim, então os IDs antigos continuam valendo). Só para sugestões: o modelo
            não propõe as classes novas, e quem chama deve avisar isso ao revisor.
        taxonomy (dict | None): taxonomia atual. Quando ela declara o mapa de fusão
            (``migration.active_class_map``, ex.: v6 → v7) e os nomes do peso são classes
            ativas antigas cobertas pelo mapa, o peso é aceito e ganha
            ``model.dronecamp_class_remap`` (ID antigo → ID novo), que ``detect_boxes``
            aplica, e ``model.dronecamp_partial_classes`` (classes ampliadas que o peso
            conhece só em parte). Só para comparação e sugestões; treino e exportação
            exigem igualdade.

    Retorna:
        list[str]: classes da taxonomia que o modelo não conhece (vazia quando os
        nomes coincidem exatamente).

    Erros:
        ValueError: "Checkpoint não corresponde às N classes ativas do projeto..."
            se a lista ou a ordem forem diferentes (ex.: peso COCO genérico).
    """
    names = model.names
    # A Ultralytics guarda names como dict {0: "a", 1: "b"}; converte para lista na ordem dos IDs.
    actual = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if actual == expected:
        return []
    # ATENCAO: prefixo exato (mesmos nomes, mesma ordem, nenhum a mais); qualquer troca de ID é recusada.
    if allow_prefix and actual and len(actual) < len(expected) and expected[:len(actual)] == actual:
        return expected[len(actual):]
    # Peso de uma taxonomia anterior com classes fundidas na atual (v6 → v7).
    if taxonomy is not None and actual:
        from .config import active_class_remap
        try:
            remap = active_class_remap(taxonomy, actual)
        except ValueError:
            remap = None  # nomes fora do mapa (ex.: COCO): cai na recusa abaixo
        if remap is not None:
            from .config import remap_partial_classes
            model.dronecamp_class_remap = [entry["to_id"] for entry in remap]
            # Classes ampliadas na fusão: os pesos antigos só conhecem parte delas.
            model.dronecamp_partial_classes = remap_partial_classes(taxonomy, actual)
            return [name for index, name in enumerate(expected) if index not in model.dronecamp_class_remap]
    raise ValueError(f"Checkpoint não corresponde às {len(expected)} classes ativas do projeto. Treine com o dataset revisado; para testar pesos gerais use predict --demo explicitamente.")


def runtime_info(model, config: ProjectConfig) -> dict:
    """Versões e hash do checkpoint gravados em execution.json (reprodutibilidade).

    Parâmetros:
        model: modelo ``.pt`` carregado (precisa de ``ckpt_path``).
        config (ProjectConfig): fornece ``device`` e ``requested_model``.

    Retorna:
        dict com ``python``, ``ultralytics``, ``torch``, ``device_requested``,
        ``checkpoint`` (caminho absoluto), ``checkpoint_sha256``,
        ``model_family`` (ver ``model_family.describe_checkpoint``) e
        ``requested_future_model``.

    Chamada por: training (train, train_pilot, tune, evaluate), prediction e outros
    que gravam ``execution.json``.
    """
    import torch
    import ultralytics

    checkpoint = Path(model.ckpt_path)
    # EXTENSAO: outro dado de ambiente para reprodutibilidade (ex.: versão do onnxruntime) entra neste dict.
    return {
        "python": platform.python_version(), "ultralytics": ultralytics.__version__,
        "torch": torch.__version__, "device_requested": config.device,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_hash(checkpoint),
        # Família/versão do YOLO (ex.: YOLO26L partindo de yolo26l.pt): procedência do modelo.
        "model_family": describe_checkpoint(checkpoint, (getattr(model, "ckpt", None) or {}).get("train_args")),
        "requested_future_model": config.requested_model,
    }
