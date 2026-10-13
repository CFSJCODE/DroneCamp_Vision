"""Configuração central: lê configs/project.yaml e a taxonomia de classes.

Função no projeto: todo comando começa aqui. Os parâmetros ficam no YAML, não
espalhados pelo código.

O que faz:
- ``ProjectConfig``: os valores lidos (modelo, dispositivo, predição, treino, piloto).
- ``load_config``: lê e valida ``configs/project.yaml``.
- ``load_taxonomy``: lê ``configs/taxonomy.json`` (IDs e nomes das classes).
- ``detection_names``: lista das classes ativas (fase 1), na ordem dos IDs.
- ``active_class_remap``: mapa das classes ativas de uma taxonomia anterior para a
  atual (fusões da v7), usado por ``migrate-review`` e pela comparação de pesos antigos.
- ``PROJECT_ROOT``: a pasta ``sistema_ia`` (base de todos os caminhos relativos).

Quando mexer: quase nunca. Para mudar valores, edite ``configs/project.yaml``.
Mexa aqui só para criar uma seção nova no YAML ou validar um parâmetro novo.

Quem chama:
- ``cli.main`` chama ``load_config(args.config)`` uma vez no início de cada comando
  e passa o ``ProjectConfig`` para todos os módulos.
- ``load_taxonomy`` e ``detection_names`` são usados por quase todos os módulos
  (training, pilot, suggestions, prediction, review_*, exports, model_gate...).
- ``PROJECT_ROOT`` é usado por ``io.resolve_local_path`` e onde não há config.

O que lê e grava no disco:
- Lê ``configs/project.yaml`` (ou o arquivo passado em ``--config``) e
  ``configs/taxonomy.json``. Não grava nada.

Onde mexer (casos concretos):
- Mudar épocas, imgsz, batch, conf, iou, dispositivo, peso base: só no
  ``configs/project.yaml``; nenhum código muda.
- Parâmetro novo dentro de ``training``/``pilot``/``prediction``: também só no
  YAML; ele chega como chave do dict (ex.: ``config.pilot["minha_chave"]``) e
  quem usa lê com ``.get``. Se precisar de validação, acrescente em ``load_config``
  (bloco "Limites básicos").
- Seção nova no topo do YAML (ex.: ``reporting:``): campo novo em
  ``ProjectConfig`` + leitura em ``load_config`` (ver os marcadores EXTENSAO:).
- Classe nova: NÃO é aqui; é versão nova de ``configs/taxonomy.json`` +
  ``migrate-review`` + treino novo (ver ATENCAO: em ``load_taxonomy``).
"""

from dataclasses import dataclass, field
from pathlib import Path
import json

import yaml

# Pasta sistema_ia (dois níveis acima de src/dronecamp_ia/config.py).
# parents[0] = src/dronecamp_ia, parents[1] = src, parents[2] = sistema_ia.
# ATENCAO: se este arquivo mudar de pasta, o índice de parents muda junto; todo caminho
# relativo do projeto (models/, runs/, configs/) depende deste valor.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Valores do project.yaml (objeto imutável passado a todos os comandos)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectConfig:
    """Valores do project.yaml já validados; caminhos absolutos a partir de root.

    É imutável (``frozen=True``): ninguém altera a configuração no meio de um
    comando. Para uma variação (ex.: imgsz do piloto na inferência), quem precisa
    cria uma cópia com ``dataclasses.replace`` (ver ``training.pilot_inference_config``).

    Atributos:
        root (Path): pasta ``sistema_ia``; base dos caminhos relativos.
        model (str): peso base (``model:`` do YAML), ex.: ``"yolo26l.pt"``.
        requested_model (str): modelo-alvo futuro (``requested_model:``), só
            registrado em ``execution.json`` e no ``model-info``.
        taxonomy_path (Path): caminho absoluto de ``configs/taxonomy.json``.
        dataset_path (Path): caminho absoluto de ``configs/dataset.yaml`` (produção).
        device (str): ``"cpu"`` hoje; passado como ``device=`` à Ultralytics.
        prediction (dict): seção ``prediction`` (imgsz, conf, iou, nms, max_det).
        training (dict): seção ``training`` (hiperparâmetros do treino de produção,
            também a base do piloto).
        pilot (dict): seção ``pilot`` inteira (model, suggestion_conf, tiling,
            duplicate_iou, gpu_eval, training, sampling...); ``{}`` se ausente.

    EXTENSAO: seção nova no topo do project.yaml = atributo novo aqui (com valor
    padrão via ``field(default_factory=...)`` se for opcional, e DEPOIS dos campos
    obrigatórios) + leitura no ``return ProjectConfig(...)`` de ``load_config``.
    """
    root: Path               # pasta sistema_ia
    model: str               # peso base (ex.: yolo26l.pt)
    requested_model: str     # modelo desejado no futuro (só registrado)
    taxonomy_path: Path      # configs/taxonomy.json
    dataset_path: Path       # configs/dataset.yaml (produção)
    device: str              # "cpu" (Ultralytics só treina em CUDA/MPS ou CPU)
    prediction: dict         # imgsz, conf, iou, nms, max_det da inferência
    training: dict           # hiperparâmetros do treino de produção
    # Piloto: treino exploratório com uma só edificação, nunca aprovado para uso.
    # ATENCAO: campos com valor padrão precisam ficar depois dos obrigatórios (regra do dataclass).
    pilot: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Leitura e validação do project.yaml
# ---------------------------------------------------------------------------

# EXTENSAO: seção nova no project.yaml = campo novo em ProjectConfig + leitura/validação aqui.
# Parâmetro novo dentro de training/pilot/prediction não precisa de código: já chega como dict.
# ENTRADA: chamada por cli.main no início de todo comando (opção --config).
def load_config(path: Path | None = None) -> ProjectConfig:
    """Resolva caminhos no projeto e rejeite configurações de outra tarefa.

    Lê o YAML, confere a versão do formato e a tarefa, valida os limites dos
    parâmetros de inferência e devolve um ``ProjectConfig`` com caminhos absolutos.

    Parâmetros:
        path (Path | None): arquivo YAML a ler. ``None`` (padrão) usa
            ``<PROJECT_ROOT>/configs/project.yaml``. Vem da opção ``--config``.

    Retorna:
        ProjectConfig: valores validados; ``root`` é a pasta-avó do YAML
        (o YAML fica em ``<root>/configs/``).

    Erros:
        ValueError: "Configuração deve ser um mapa com schema_version: 1." se o
            YAML não for um dicionário ou a versão não for 1;
            "Este estágio implementa apenas detecção (task: detect)." se
            ``task`` não for ``detect``; mensagens específicas se
            ``prediction.nms`` não for booleano ou ``conf``/``iou``/``imgsz``
            estiverem fora dos limites.
        KeyError: se faltar uma chave obrigatória (``prediction``, ``model``,
            ``requested_model``, ``taxonomy``, ``dataset``, ``device``, ``training``).
        OSError: se o arquivo não existir ou não puder ser lido.

    Chamada por: ``cli.main`` e testes.
    """
    # Usa configs/project.yaml do projeto, ou o arquivo passado em --config.
    config_path = (path or PROJECT_ROOT / "configs/project.yaml").resolve()
    # safe_load: lê o YAML sem executar construtores arbitrários (seguro para arquivos de fora).
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Versão do formato do arquivo: muda só se a estrutura do YAML mudar de forma incompatível.
    # EXTENSAO: um schema_version 2 exigiria aceitar 2 aqui e converter os campos antigos.
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Configuração deve ser um mapa com schema_version: 1.")
    # Só detecção com caixas está implementada (labels YOLO, métricas (B), check_domain_names).
    if raw.get("task") != "detect":
        raise ValueError("Este estágio implementa apenas detecção (task: detect).")
    # O arquivo de configuração fica em configs/; o seu pai define o projeto.
    root = config_path.parent.parent
    # Limites básicos dos parâmetros de inferência.
    # EXTENSAO: validação de um parâmetro novo de prediction (ou de training/pilot) entra
    # neste bloco, no mesmo estilo: condição + ValueError com o nome completo da chave.
    prediction = raw["prediction"]
    # nms precisa ser true/false do YAML (um 0, 1 ou "false" entre aspas é recusado).
    if not isinstance(prediction.get("nms"), bool):
        raise ValueError("prediction.nms deve ser booleano.")
    # conf: confiança mínima de uma caixa; 0 exclusivo (mostraria tudo), 1 inclusivo.
    if not 0 < float(prediction["conf"]) <= 1:
        raise ValueError("prediction.conf deve estar no intervalo (0, 1].")
    # iou: sobreposição para juntar caixas repetidas; mesmo intervalo da conf.
    if not 0 < float(prediction["iou"]) <= 1:
        raise ValueError("prediction.iou deve estar no intervalo (0, 1].")
    # imgsz: lado da imagem em pixels; o YOLO trabalha em múltiplos de 32 (stride máximo).
    if int(prediction["imgsz"]) < 32:
        raise ValueError("prediction.imgsz deve ser pelo menos 32.")
    # Monta o objeto final: caminhos relativos do YAML viram absolutos a partir de root;
    # as seções viram cópias (dict(...)) para ninguém alterar o dicionário lido do YAML.
    # EXTENSAO: campo novo de ProjectConfig = argumento novo nesta chamada (ex.: reporting=dict(raw.get("reporting") or {})).
    return ProjectConfig(
        root=root, model=str(raw["model"]), requested_model=str(raw["requested_model"]),
        taxonomy_path=(root / raw["taxonomy"]).resolve(),
        dataset_path=(root / raw["dataset"]).resolve(), device=str(raw["device"]),
        prediction=dict(prediction), training=dict(raw["training"]),
        # pilot é opcional no YAML: ausente ou vazio vira {}.
        pilot=dict(raw.get("pilot") or {}),
    )


# ---------------------------------------------------------------------------
# Taxonomia de classes (configs/taxonomy.json)
# ---------------------------------------------------------------------------

# ATENCAO: os IDs da taxonomia são o contrato entre labels YOLO, pesos treinados, registros
# de revisão e laudos. Nunca renumere nem apague uma classe; acrescente no fim e migre as
# revisões (migrate-review). Mudar este arquivo muda o SHA-256 que tudo confere.
def load_taxonomy(path: Path) -> dict:
    """IDs e slugs são contrato do dataset, checkpoint e saída.

    Lê a taxonomia e confere as duas regras que todo o resto do projeto supõe:
    IDs consecutivos começando em 0 (o ID é o número da classe na label YOLO) e
    slugs únicos (o slug é o nome da classe dentro do modelo e das revisões).

    Parâmetros:
        path (Path): caminho do JSON, normalmente ``config.taxonomy_path``.

    Retorna:
        dict: o JSON inteiro como foi lido; a chave principal é ``"classes"``,
        uma lista de dicts com pelo menos ``id`` (int), ``slug`` (str) e
        ``phase`` (int), além de nomes, cores e critérios.

    Erros:
        ValueError: "IDs da taxonomia devem ser consecutivos a partir de zero." ou
            "Slugs duplicados na taxonomia.".
        KeyError: se faltar ``classes``, ``id`` ou ``slug``.

    Chamada por: praticamente todos os módulos (training, pilot, suggestions,
    review_data, review_render, exports, model_gate...).
    """
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    classes = taxonomy["classes"]
    # Os IDs, na ordem do arquivo, precisam ser exatamente 0, 1, 2, ..., N-1.
    if [item["id"] for item in classes] != list(range(len(classes))):
        raise ValueError("IDs da taxonomia devem ser consecutivos a partir de zero.")
    # Conjunto de slugs com o mesmo tamanho da lista = nenhum slug repetido.
    if len({item["slug"] for item in classes}) != len(classes):
        raise ValueError("Slugs duplicados na taxonomia.")
    return taxonomy


def detection_names(taxonomy: dict) -> list[str]:
    """O MVP utiliza só fase 1; ampliar classes exige nova versão e novo treino.

    Devolve a lista de slugs das classes ativas (``phase == 1``), na ordem dos IDs.
    Essa lista é exatamente o ``names`` do dataset YOLO e do modelo treinado:
    a posição na lista é o ID da classe na label.

    Parâmetros:
        taxonomy (dict): resultado de ``load_taxonomy``.

    Retorna:
        list[str]: slugs das classes ativas, ex.: ``["telha_danificada", "telha_ausente", ...]``.

    Erros:
        ValueError: "IDs das classes ativas devem ser consecutivos a partir de zero."
            se não houver classe ativa ou se as ativas não forem 0..N-1 (uma
            classe inativa no meio quebraria a correspondência ID ↔ posição).

    Chamada por: training (nomes esperados no dataset e no modelo), pilot,
    suggestions, prediction, exports, model_gate e outros.

    ATENCAO: ``backend.check_domain_names`` compara esta lista com os nomes gravados
    nos pesos; mudar a fase de uma classe invalida todos os pesos treinados antes.
    """
    # EXTENSAO: uma fase 2 de classes entraria aqui (ex.: phase in (1, 2)), sempre com
    # taxonomia nova, migrate-review e treino novo; os IDs ativos devem continuar 0..N-1.
    active = [item for item in taxonomy["classes"] if item["phase"] == 1]
    if not active or [item["id"] for item in active] != list(range(len(active))):
        raise ValueError("IDs das classes ativas devem ser consecutivos a partir de zero.")
    return [item["slug"] for item in active]


# ENTRADA: mapa de classes entre versões da taxonomia (fusões, ex.: v6 → v7).
def active_class_remap(new_taxonomy: dict, old_names: list[str]) -> list[dict] | None:
    """Traduza cada classe ativa antiga para a classe ativa nova que a contém.

    A taxonomia nova declara em ``migration.active_class_map`` para onde vai cada
    slug ativo antigo (``{"to": slug_novo, "attributes": {...}}``). Só fusões são
    aceitas: cada classe antiga cabe inteira numa classe nova, então uma caixa
    muda de ID sem redesenho.

    Parâmetros:
        new_taxonomy (dict): taxonomia atual (``load_taxonomy``).
        old_names (list[str]): slugs ativos da taxonomia anterior, na ordem dos IDs
            (``detection_names`` da anterior ou ``names`` gravados nos pesos).

    Retorna:
        list[dict] | None: um item por ID antigo, ``{"to_id", "to_slug", "attributes"}``;
        ``None`` quando a taxonomia nova não declara mapa (expansão pura ou mesma versão).

    Erros (ValueError): mapa sem algum slug antigo, destino que não é classe ativa
    da taxonomia nova, ou atributo fora dos valores definidos na classe de destino.

    Chamada por: review_data.migrate_review_taxonomy (remapeamento das revisões) e
    model_gate/model_report (previsões de pesos antigos medidas nas classes novas).
    """
    mapping = (new_taxonomy.get("migration") or {}).get("active_class_map")
    if not mapping:
        return None
    new_names = detection_names(new_taxonomy)
    by_slug = {item["slug"]: item for item in new_taxonomy["classes"]}
    remap = []
    for slug in old_names:
        entry = mapping.get(slug)
        if not isinstance(entry, dict) or entry.get("to") not in new_names:
            raise ValueError(f"Mapa de classes da taxonomia não cobre a classe ativa antiga '{slug}'.")
        attributes = entry.get("attributes") or {}
        allowed = by_slug[entry["to"]].get("attributes") or {}
        # ATENCAO: atributo inventado no mapa viraria dado "revisado" sem ninguém ter visto.
        if any(value not in allowed.get(key, []) for key, value in attributes.items()):
            raise ValueError(f"Atributo inválido no mapa de '{slug}' para '{entry['to']}'.")
        remap.append({"to_id": new_names.index(entry["to"]), "to_slug": entry["to"], "attributes": dict(attributes)})
    return remap
