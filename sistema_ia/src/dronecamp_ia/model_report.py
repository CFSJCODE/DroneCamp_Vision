"""Relatório comparativo entre modelos: gráficos PNG, HTML autônomo e JSON normalizado.

Função no projeto: os números que dizem se um modelo piloto é melhor que outro
ficam espalhados em arquivos de formatos diferentes (``measurement.json`` do
``measure-models``, ``gate.json`` do ``compare-models``, ``runs/compare_*.json`` do
``scripts/compare_pilot_models.py``, ``runs/eval_por_classe_*.json`` e as curvas
``fit/results.csv`` de cada treino). Só uma medição (``measurement.json``) avalia
todos os modelos nas MESMAS fotos; por isso ela é a comparação principal e os
demais arquivos ficam numa seção "medições anteriores", não comparáveis entre si.
Este módulo junta tudo em uma pasta ``runs/report_*`` com gráficos comparativos
(um modelo = um rótulo e uma cor em todos eles), uma página HTML que abre do
disco sem internet e um ``models.json`` normalizado para quem quiser reprocessar.
Só lê arquivos: nada aqui treina, avalia ou aprova pesos.

O que faz:
- ``build_model_report``: monta a pasta do relatório a partir dos arquivos informados.
- ``load_measurement`` / ``load_gate`` / ``load_compare`` / ``load_eval`` / ``load_run``:
  leem cada formato e devolvem dicionários com as séries já identificadas no catálogo.
- ``ModelCatalog``: dá a cada série (SHA-256 dos pesos ou caminho, mais o regime de
  inferência quando há) um rótulo e uma cor fixos, na ordem em que aparece.
- ``wilson_interval``: intervalo de confiança de 95% do recall por classe.
- ``discover_runs``: acha em ``runs/`` os treinos piloto com ``fit/results.csv``.

Quem usa: o dono do projeto, para enxergar a diferença entre modelos piloto antes
de decidir o próximo treino; ``hparam_bandit.py`` pode gerar um ao fim da rodada.
Nunca importa Ultralytics/Torch: roda em qualquer máquina com matplotlib.

Quando mexer: para acrescentar um gráfico (escreva a função ``_chart_*`` e chame
em ``build_model_report``) ou para aceitar um formato novo de entrada.

Quem chama (no código atual):
- ``cli.py``, comando ``python -m dronecamp_ia report-models`` (opções ``--measurement``,
  ``--gate``, ``--compare``, ``--eval``, ``--run``, ``--output``, ``--title``).
- ``cycle.py`` (``retrain-cycle``), etapa 6 do ciclo de retreino.
- ``tests/test_model_report.py``.
  ATENCAO: apesar da frase acima, o ``hparam_bandit.py`` hoje NÃO chama este módulo;
  para ver os treinos do bandit, rode ``report-models --run <pasta>`` à mão.

O que lê do disco (só leitura): ``measurement.json``, ``gate.json``, ``runs/compare_*.json``,
``runs/eval_por_classe_*.json``, e em cada ``runs/pilot_train_*``: ``fit/results.csv``,
``execution.json``, ``summary.json``, ``gpu_eval/eval_curve.jsonl`` e o ``fit/weights/best.pt``
(só para calcular o SHA-256). Também lê ``configs/taxonomy.json`` (ordem das classes).

O que grava (pasta NOVA ``runs/report_<data>_<id>`` ou a pasta de ``--output``, que não pode existir):
- ``models.json``: tudo o que foi lido, já normalizado (``SCHEMA_VERSION``).
- ``*.png``: um arquivo por gráfico que tem dados (lista em ``models.json["charts"]``):
  ``overall_metrics.png``, ``per_class_recall.png``, ``per_class_errors.png``, ``strata_recall.png``,
  ``training_curves.png``, ``gpu_eval_curves.png``, ``compare_found.png`` e as versões ``*_arquivo.png``
  da seção "medições anteriores".
- ``report.html``: página autônoma (CSS e PNGs embutidos em base64).

Organização do arquivo (seções demarcadas com linhas de traços):
1. Constantes (cores, tamanhos, limites, estilo do matplotlib).
2. Identidade dos modelos: ``normalize_path``, ``run_id_of``, ``ModelCatalog``.
3. Leitura das entradas: ``load_measurement``, ``check_measurements_comparable``, ``load_gate``,
   ``load_compare``, ``load_eval``, ``load_run``, ``discover_runs`` e auxiliares.
4. Fontes dos gráficos: escolhem de onde vem cada número (medição > gate > eval_por_classe).
5. Gráficos: ``_chart_overall``, ``_chart_per_class_recall``, ``_chart_per_class_errors``,
   ``_chart_strata``, ``_chart_curves``, ``_chart_compare`` e auxiliares de desenho.
6. Notas e HTML: ``_notes``, ``_render_html`` e as funções de cada seção da página.
7. Montagem do relatório: ``build_model_report`` (ENTRADA).

Onde mexer (casos concretos):
- Gráfico novo: escreva ``_chart_<nome>(plt, path, ...)`` na seção "Gráficos", chame em
  ``build_model_report`` (procure ``EXTENSAO: gráfico novo``) acrescentando o nome do PNG em
  ``charts``, e mostre-o em ``_render_html`` com ``_figure`` (procure ``EXTENSAO: seção nova``).
- Formato de entrada novo: escreva ``load_<formato>(path, catalog)`` na seção "Leitura das
  entradas", acrescente o parâmetro em ``build_model_report`` e no subparser ``report-models``
  do cli.py, e decida em ``_overall_source``/``_per_class_source`` se ele vira fonte.
- Métrica nova por classe (ex.: AP50-95): acrescente a chave em ``CLASS_KEYS`` e use-a em
  ``_per_class_source`` e ``_per_class_section``.
- Novo estrato de fotos (``measure-models``): acrescente o nome legível em ``STRATA``.
- Mais cores para mais modelos: acrescente em ``PALETTE``.
- Estilo da página: ``CSS``; estilo dos gráficos: ``STYLE``.

ATENCAO: a ordem em que as entradas são lidas em ``build_model_report`` define a cor e o
rótulo de cada modelo em TODOS os gráficos. Mudar essa ordem muda as cores dos relatórios.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path, PurePosixPath

from .config import ProjectConfig, detection_names, load_taxonomy
from .io import file_hash, make_run_directory, write_json
from .operations import RUN_NAME, read_curves

# ---------------------------------------------------------------------------
# Constantes (cores, tamanhos, limites e estilo)
# ---------------------------------------------------------------------------

# Versão do formato de models.json. Suba o número se mudar a estrutura gravada,
# para que quem reprocessa o arquivo saiba distinguir os formatos.
SCHEMA_VERSION = 1
# Uma cor por modelo, nesta ordem fixa; do nono modelo em diante tudo vira cinza "outros".
# Cores hexadecimais "#rrggbb" (formato exigido por _lighten). Cada modelo novo recebe a
# próxima cor livre na ordem de registro no ModelCatalog.
# EXTENSAO: mais modelos com cor própria: acrescente cores no fim desta tupla (mudar a ordem
# muda a cor dos modelos já conhecidos nos próximos relatórios).
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# Cor e rótulo da legenda dos modelos além da paleta (do nono em diante).
OTHERS_COLOR, OTHERS_LABEL = "#8a8a8a", "outros"
# Cinza claro dos FN (caixas humanas não encontradas) no gráfico de acertos e erros.
FN_COLOR = "#bdbdbd"
# Resolução dos PNGs (pontos por polegada). Maior = imagem maior e HTML mais pesado
# (os PNGs vão embutidos em base64). Também converte pixels em pontos em _points.
DPI = 130
GROUP_WIDTH = 0.8        # largura total de um grupo de barras, em unidades do eixo
BAR_GAP_PX = 2           # espaço entre barras vizinhas do mesmo grupo
# Espessura das linhas das curvas de treino, em pixels (convertida por _points).
LINE_PX = 2
MAX_MARKER_POINTS = 30   # acima disso os marcadores só borram a linha
# Sufixo que o scripts/compare_pilot_models.py põe no nome do split quando as fotos
# foram vistas no treino (ex.: "val (visto no treino)"); separa memorização de generalização.
# ATENCAO: tem de ser idêntico ao texto gravado pelo script; se mudar lá, mude aqui.
SEEN_SUFFIX = " (visto no treino)"
# Ordem de preferência dos splits: teste (mais honesto), validação, treino.
SPLIT_ORDER = ("test", "val", "train")
FEW_BOXES = 3            # abaixo disso a classe é "pouco medida" (mesmo limite do gate)
# z da normal para o intervalo de Wilson (1,96 = 95% de confiança).
WILSON_Z = 1.96          # intervalo de 95%
# Campos por classe copiados de gate.json/measurement.json (ver _metrics_block).
# EXTENSAO: nova métrica por classe vinda do gate/medição: acrescente a chave aqui e use-a
# em _per_class_source e _per_class_section.
CLASS_KEYS = ("tp", "fp", "fn", "human_boxes", "precision", "recall", "ap50")
# Estratos da medição (chave gravada pelo measure-models → nome legível nos gráficos).
# A ordem do dict é a ordem das barras; estratos desconhecidos vão ao fim com o nome cru.
# EXTENSAO: novo estrato de fotos no measure-models: acrescente "chave": "Nome legível" aqui.
STRATA = {"fotos_fatiadas": "Fotos fatiadas (drone 20 MP)", "fotos_inteiras": "Fotos inteiras (laudo)"}
# Título da seção com gates/comparações/avaliações quando existe uma medição principal.
ARCHIVE_TITLE = "Medições anteriores (conjuntos de fotos diferentes; não comparáveis entre si)"
# Fundo claro, grade discreta e sem molduras: o leitor compara barras, não decoração.
STYLE = {
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "#fbfbfb",
    "axes.edgecolor": "#9a9a9a", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 8,
}


# ----------------------------------------------------------------------------
# Identidade dos modelos
# ----------------------------------------------------------------------------

def normalize_path(value: str | Path) -> str:
    """Caminhos gravados no Windows chegam com "\\"; o relatório compara com "/".

    Parâmetros:
        value: caminho (str ou Path), ex.: ``E:\\proj\\runs\\x\\best.pt``.

    Retorna:
        O mesmo caminho como texto, com todas as barras invertidas trocadas por ``/``.
    """
    return str(value).replace("\\", "/")


def run_id_of(path: str) -> str | None:
    """Id curto (8 hex) da pasta ``<etapa>_<data>_<id>`` presente no caminho, se houver.

    Parâmetros:
        path: caminho já normalizado com ``/``
            (ex.: ``runs/pilot_train_20261004T170212Z_aaaaaaaa/fit/weights/best.pt``).

    Retorna:
        O id da primeira pasta que casa com ``RUN_NAME`` (ex.: ``"aaaaaaaa"``) ou ``None``.

    Chamada por: ``ModelCatalog.register``.
    """
    # Percorre as pastas do caminho; RUN_NAME (operations.py) reconhece
    # "<etapa>_<AAAAMMDDTHHMMSSZ>_<8 hex>" e expõe o grupo "id".
    for part in path.split("/"):
        match = RUN_NAME.match(part)
        if match:
            return match["id"]
    return None


class ModelCatalog:
    """Rótulo e cor únicos por série, na ordem de primeira aparição nas entradas.

    Uma série é um modelo (SHA-256 dos pesos quando o arquivo de entrada o traz;
    senão o caminho normalizado) mais, numa medição, o regime de inferência: o
    mesmo ``best.pt`` em "janelas" e "inteira" são duas séries, com cores
    vizinhas. Sem SHA, o mesmo ``best.pt`` gravado com caminho absoluto numa
    máquina e relativo noutra ainda é o mesmo modelo: casa por (id do treino,
    nome do arquivo).

    Cada modelo registrado é um dict com as chaves ``id`` (SHA ou caminho, mais
    ``::<regime>`` quando há regime), ``label`` (texto da legenda), ``weights``
    (caminho normalizado), ``sha256``, ``run_id`` (8 hex do treino), ``regime`` e
    ``color`` (da ``PALETTE`` ou ``OTHERS_COLOR``).

    Atributos:
        models: lista dos modelos na ordem de registro (é a ordem das cores).
        _by_id / _by_sha / _by_path / _by_run_file: índices para achar um modelo
            já registrado por id, por (SHA, regime), por (caminho, regime) e por
            (id do treino, nome do arquivo, regime).

    ATENCAO: a cor depende só da ORDEM de registro; um mesmo modelo pode ter cores
    diferentes em relatórios montados com as entradas em outra ordem.

    Usada por: todas as funções ``load_*`` e pelos gráficos/HTML (rótulo e cor).
    """

    # --- Construtor -------------------------------------------------------
    def __init__(self) -> None:
        """Cria um catálogo vazio (nenhum modelo, índices vazios).

        Parâmetros: nenhum. Retorna: None.
        """
        self.models: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._by_sha: dict[tuple, dict] = {}
        self._by_path: dict[tuple, dict] = {}
        self._by_run_file: dict[tuple, dict] = {}

    def register(self, weights: str | Path, sha256: str | None = None, regime: str | None = None) -> str:
        """Registra uma série (ou reaproveita a já registrada) e devolve o id dela.

        A busca de um modelo já conhecido segue esta ordem: (1) mesmo SHA-256 e regime;
        (2) mesmo caminho normalizado e regime; (3) mesmo id de treino + nome de arquivo +
        regime. Se nada casar, cria um modelo novo com a próxima cor da paleta.

        Parâmetros:
            weights: caminho dos pesos como gravado no arquivo de entrada (``\\`` ou ``/``).
            sha256: SHA-256 dos pesos (64 hex) quando o arquivo de entrada traz; senão ``None``.
            regime: regime de inferência da medição (``"janelas"`` ou ``"inteira"``);
                ``None`` para gate, compare, eval e treinos.

        Retorna:
            O ``id`` do modelo (chave usada em todos os dicts ``{modelo: ...}`` do relatório).

        Chamada por: ``load_measurement``, ``load_gate``, ``load_compare``, ``load_eval``, ``load_run``.
        """
        # 1. Identificação: caminho com "/", id do treino (8 hex) e nome do arquivo (ex.: best.pt).
        path = normalize_path(weights)
        run_id, file_name = run_id_of(path), PurePosixPath(path).name
        # 2. Procura um modelo já registrado: primeiro pelo SHA (se houver), depois pelo caminho.
        model = (self._by_sha.get((sha256, regime)) if sha256 else None) or self._by_path.get((path, regime))
        # 3. Sem SHA nem caminho iguais: tenta (id do treino, nome do arquivo, regime), que casa o
        #    mesmo best.pt gravado com caminho absoluto numa máquina e relativo noutra.
        if model is None and run_id:
            model = self._by_run_file.get((run_id, file_name, regime))
        # 4. Modelo novo: id = SHA (ou caminho) + "::regime"; cor = próxima da paleta ou cinza.
        if model is None:
            index = len(self.models)
            base = sha256 or path
            model = {"id": f"{base}::{regime}" if regime else base,
                     "label": self._unique_label(run_id, file_name, sha256, regime), "weights": path,
                     "sha256": sha256, "run_id": run_id, "regime": regime,
                     "color": PALETTE[index] if index < len(PALETTE) else OTHERS_COLOR}
            self.models.append(model)
            self._by_id[model["id"]] = model
        # 5. Completa o SHA de um modelo que foi registrado antes sem ele (ex.: primeiro pelo
        #    caminho num compare, depois com SHA num gate) e atualiza os índices de busca.
        if sha256 and not model["sha256"]:
            model["sha256"] = sha256
        if sha256:
            self._by_sha[(sha256, regime)] = model
        self._by_path[(path, regime)] = model
        # setdefault: o primeiro modelo registrado para (treino, arquivo, regime) continua sendo
        # o escolhido nas próximas buscas por esse par.
        if run_id:
            self._by_run_file.setdefault((run_id, file_name, regime), model)
        return model["id"]

    def _unique_label(self, run_id: str | None, file_name: str, sha256: str | None, regime: str | None) -> str:
        """Monta o rótulo da legenda, sem repetir um rótulo já usado.

        Formato: ``"<id do treino> · <arquivo> · <regime>"`` (partes ausentes são omitidas),
        ex.: ``"aaaaaaaa · best.pt · janelas"``.

        Parâmetros:
            run_id: 8 hex do treino ou ``None``.
            file_name: nome do arquivo de pesos (ex.: ``best.pt``).
            sha256: SHA-256 dos pesos ou ``None`` (usado para desempatar).
            regime: regime de inferência ou ``None``.

        Retorna:
            O rótulo; se já existir, acrescenta ``" (<6 primeiros hex do SHA>)"`` ou
            ``" (<número do modelo>)"`` sem SHA.

        Chamada por: ``register``.
        """
        base = f"{run_id} · {file_name}" if run_id else file_name
        if regime:
            base = f"{base} · {regime}"
        used = {model["label"] for model in self.models}
        if base not in used:
            return base
        # Dois arquivos de mesmo nome sem treino distinguível: o SHA desempata.
        suffix = (sha256 or "")[:6] or str(len(self.models) + 1)
        return f"{base} ({suffix})"

    def by_id(self, model_id: str) -> dict:
        """Devolve o dict do modelo pelo id.

        Parâmetros:
            model_id: id devolvido por ``register``.

        Retorna:
            O dict do modelo (chaves no docstring da classe).

        Erros:
            ``KeyError`` se o id não foi registrado.
        """
        return self._by_id[model_id]

    def legend_label(self, model: dict) -> str:
        """Modelos além da paleta compartilham o cinza e a entrada "outros" na legenda.

        Parâmetros:
            model: dict de um modelo do catálogo.

        Retorna:
            ``model["label"]`` ou ``OTHERS_LABEL`` (``"outros"``) quando a cor é ``OTHERS_COLOR``.
        """
        return model["label"] if model["color"] != OTHERS_COLOR else OTHERS_LABEL

    def ordered(self, ids) -> list[dict]:
        """Modelos do catálogo presentes em ``ids``, na ordem de primeira aparição.

        Parâmetros:
            ids: qualquer iterável de ids (lista, set, chaves de dict).

        Retorna:
            Lista de dicts de modelo, na ordem do catálogo (que é a ordem das cores).
        """
        wanted = set(ids)
        return [model for model in self.models if model["id"] in wanted]


# ----------------------------------------------------------------------------
# Leitura das entradas
# ----------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    """Lê um JSON obrigatório (UTF-8).

    Parâmetros:
        path: caminho do arquivo.

    Retorna:
        O conteúdo decodificado.

    Erros:
        ``OSError`` se o arquivo não existir; ``json.JSONDecodeError`` (ValueError) se for inválido.
        Diferente de ``_optional_json``: aqui a falha derruba o relatório, de propósito,
        porque o arquivo foi pedido explicitamente pelo usuário.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict | None:
    """Arquivo opcional de um treino: ausente ou corrompido não derruba o relatório.

    Parâmetros:
        path: caminho do JSON (ex.: ``<treino>/execution.json``).

    Retorna:
        O dict lido, ou ``None`` se o arquivo faltar, for inválido ou não for um objeto JSON.

    Chamada por: ``load_run``.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _ratio(numerator, denominator) -> float | None:
    """Divisão arredondada a 4 casas; ``None`` quando o denominador é 0/None (não há medida).

    Parâmetros:
        numerator: numerador (ex.: caixas reencontradas).
        denominator: denominador (ex.: caixas humanas).

    Retorna:
        ``round(numerator / denominator, 4)`` ou ``None``.
    """
    return round(numerator / denominator, 4) if denominator else None


def _finite(value):
    """NaN/infinito viram None: ``write_json`` recusa NaN e o HTML não deve mostrar "nan".

    Parâmetros:
        value: qualquer valor.

    Retorna:
        O próprio valor quando não é número (ou é bool) ou quando é número finito; ``None``
        para NaN e infinito.
    """
    # bool é subclasse de int em Python: sai antes para True/False não serem tratados como número.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return value if math.isfinite(value) else None


def _clean(value):
    """Aplica ``_finite`` em toda a estrutura (dicts e listas aninhados) antes de gravar JSON.

    Parâmetros:
        value: dict, lista, tupla ou valor simples.

    Retorna:
        Cópia com as chaves dos dicts convertidas em texto, tuplas viradas listas e
        NaN/infinito trocados por ``None``.

    Chamada por: ``build_model_report`` (models.json).
    """
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return _finite(value)


def _metrics_block(metrics: dict) -> tuple[dict, dict]:
    """Dicionário por classe com ``_total`` (gate e medição): (totais, {classe: contagens}).

    Parâmetros:
        metrics: bloco ``metrics`` do gate.json/measurement.json, no formato
            ``{"_total": {"map50", "precision", "recall", ...}, "<classe>": {"tp", "fp", ...}}``.

    Retorna:
        Tupla ``(overall, per_class)``: ``overall`` = ``{"map50", "precision", "recall"}``;
        ``per_class`` = ``{classe: {chave: valor}}`` só com as chaves de ``CLASS_KEYS``.

    Chamada por: ``load_measurement`` e ``load_gate``.
    """
    # Totais: só as três métricas gerais usadas nos gráficos.
    total = metrics.get("_total") or {}
    overall = {"map50": total.get("map50"), "precision": total.get("precision"), "recall": total.get("recall")}
    # Por classe: todo item que não é "_total" e é um dict; só as chaves de CLASS_KEYS.
    per_class = {name: {key: values.get(key) for key in CLASS_KEYS}
                 for name, values in metrics.items() if name != "_total" and isinstance(values, dict)}
    return overall, per_class


def load_measurement(path: Path, catalog: ModelCatalog) -> dict:
    """``measurement.json`` de ``model_gate.measure_models``: N séries nas mesmas fotos.

    Parâmetros:
        path: caminho do ``measurement.json`` (``runs/measure_*/measurement.json``).
        catalog: catálogo onde cada (peso, regime) é registrado.

    Retorna:
        Dict com ``file``, ``dataset``, ``images`` (lista ordenada de SHA-256 das fotos, usada
        para conferir se duas medições são comparáveis), ``pilot_sha256``, ``splits``, ``conf``,
        ``ap_conf``, ``tiling``, ``images_evaluated``, ``images_excluded_seen_in_training``,
        ``strata_images`` ({estrato: nº de fotos}), ``models`` (ids das séries, em ordem),
        ``overall`` ({série: totais}), ``per_class`` ({classe: {série: contagens}}) e
        ``strata`` ({estrato: {série: totais}}).

    Erros:
        ``OSError``/``ValueError`` de ``_read_json``.

    Chamada por: ``build_model_report``.
    """
    raw = _read_json(path)
    series, overall, per_class, strata = [], {}, {}, {}
    # Uma entrada por (modelo, regime). Medição sem regime é tratada como foto inteira.
    # EXTENSAO: novo campo por série no measurement.json: leia aqui de "entry" e guarde num
    # dict próprio (como overall/strata), acrescentando a chave no dict devolvido abaixo.
    for entry in raw.get("models") or []:
        regime = entry.get("regime") or "inteira"
        series_id = catalog.register(entry.get("weights") or "modelo.pt", entry.get("sha256"), regime)
        series.append(series_id)
        overall[series_id], classes = _metrics_block(entry.get("metrics") or {})
        # Inverte para {classe: {série: contagens}}, o formato usado pelos gráficos por classe.
        for name, values in classes.items():
            per_class.setdefault(name, {})[series_id] = values
        # Estratos (ex.: fotos_fatiadas × fotos_inteiras): só os totais de cada estrato.
        for stratum, metrics in (entry.get("strata") or {}).items():
            strata.setdefault(stratum, {})[series_id] = _metrics_block(metrics or {})[0]
    # Dict normalizado. "images" guarda só os SHA-256 das fotos, ordenados, para comparar medições.
    return {"file": str(path), "dataset": normalize_path(raw.get("dataset") or ""),
            "images": sorted(str(item.get("image_sha256")) for item in (raw.get("images") or []) if isinstance(item, dict)),
            "pilot_sha256": raw.get("pilot_sha256"), "splits": list(raw.get("splits") or []),
            "conf": raw.get("conf"), "ap_conf": raw.get("ap_conf"), "tiling": raw.get("tiling"),
            "images_evaluated": raw.get("images_evaluated"),
            "images_excluded_seen_in_training": raw.get("images_excluded_seen_in_training"),
            "strata_images": dict(raw.get("strata") or {}), "models": series, "overall": overall,
            "per_class": per_class, "strata": strata}


def check_measurements_comparable(measurements: list[dict]) -> None:
    """Várias medições só entram no mesmo gráfico se avaliaram as MESMAS fotos.

    Compara cada medição com a primeira: mesmo ``pilot_sha256`` (dataset), mesma lista de
    fotos, mesma ``conf`` e mesma ``ap_conf``.

    Parâmetros:
        measurements: dicts devolvidos por ``load_measurement``.

    Retorna:
        None (só valida).

    Erros:
        ``ValueError("Medições incomparáveis: ...")`` com os dois arquivos em conflito.

    Chamada por: ``build_model_report``, logo depois de ler as medições.
    """
    # ATENCAO: esta é a garantia de que a "comparação principal" compara o mesmo conjunto de
    # fotos. Afrouxar esta regra mistura medições diferentes no mesmo gráfico.
    if len(measurements) < 2:
        return
    first = measurements[0]
    for other in measurements[1:]:
        same = (other["pilot_sha256"] == first["pilot_sha256"] and other["images"] == first["images"]
                and other["conf"] == first["conf"] and other["ap_conf"] == first["ap_conf"])
        if not same:
            raise ValueError(f"Medições incomparáveis: {first['file']} e {other['file']} avaliaram fotos, dataset ou "
                             "confiança diferentes. Gere um relatório por medição ou refaça o measure-models com todos os modelos.")


def load_gate(path: Path, catalog: ModelCatalog) -> dict:
    """``gate.json`` de ``model_gate.compare_models``; gates antigos podem não ter ``tiling``.

    Parâmetros:
        path: caminho do ``gate.json`` (``runs/compare_*/gate.json``).
        catalog: catálogo onde baseline e candidato são registrados (sem regime).

    Retorna:
        Dict com ``file``, ``conf``, ``splits``, ``images_evaluated``,
        ``images_excluded_seen_in_training``, ``tiling``, ``adopt`` (decisão do gate),
        ``reasons``, ``regressions``, ``baseline``/``candidate`` (ids no catálogo),
        ``overall`` ({modelo: totais}) e ``per_class`` ({classe: {modelo: contagens}}).

    Erros:
        ``OSError``/``ValueError`` de ``_read_json``.

    Chamada por: ``build_model_report``.
    """
    raw = _read_json(path)
    roles, overall, per_class = {}, {}, {}
    # Os dois papéis do gate; sem "weights" o nome vira "baseline.pt"/"candidate.pt" para não falhar.
    for role in ("baseline", "candidate"):
        entry = raw.get(role) or {}
        model_id = catalog.register(entry.get("weights") or f"{role}.pt", entry.get("sha256"))
        roles[role] = model_id
        overall[model_id], classes = _metrics_block(entry.get("metrics") or {})
        for name, values in classes.items():
            per_class.setdefault(name, {})[model_id] = values
    # ATENCAO: "adopt" é copiado do gate.json como está; o relatório nunca decide adoção.
    return {"file": str(path), "conf": raw.get("conf"), "splits": list(raw.get("splits") or []),
            "images_evaluated": raw.get("images_evaluated"),
            "images_excluded_seen_in_training": raw.get("images_excluded_seen_in_training"),
            "tiling": raw.get("tiling"), "adopt": raw.get("adopt"), "reasons": list(raw.get("reasons") or []),
            "regressions": list(raw.get("regressions") or []), "baseline": roles["baseline"],
            "candidate": roles["candidate"], "overall": overall, "per_class": per_class}


def load_compare(path: Path, catalog: ModelCatalog) -> dict:
    """``runs/compare_*.json`` de ``scripts/compare_pilot_models.py`` (contagens por split).

    Parâmetros:
        path: caminho do ``compare_*.json``.
        catalog: catálogo onde cada modelo é registrado (SHA em ``weights_sha256``).

    Retorna:
        Dict com ``file``, ``dataset``, ``pilot_sha256``, ``conf``, ``iou_match``, ``models``
        (ids) e ``splits`` = ``{split: {modelo: {"images", "truth", "found", "suggestions",
        "correct", "found_fraction", "correct_fraction", "seen_in_training"}}}``.

    Erros:
        ``KeyError`` se uma entrada de modelo não tiver ``weights``; ``OSError``/``ValueError``
        de ``_read_json``.

    Chamada por: ``build_model_report``.
    """
    raw = _read_json(path)
    models, splits = [], {}
    for entry in raw.get("models") or []:
        model_id = catalog.register(entry["weights"], entry.get("weights_sha256"))
        models.append(model_id)
        # Por split: "truth" = caixas humanas, "found" = reencontradas, "suggestions" = caixas
        # sugeridas, "correct" = sugestões certas. Valores ausentes viram 0.
        # found_fraction ~ recall; correct_fraction ~ precisão (None quando o total é 0).
        for split, counts in (entry.get("splits") or {}).items():
            truth, found = counts.get("truth") or 0, counts.get("found") or 0
            suggestions, correct = counts.get("suggestions") or 0, counts.get("correct") or 0
            splits.setdefault(split, {})[model_id] = {
                "images": counts.get("images"), "truth": truth, "found": found, "suggestions": suggestions,
                "correct": correct, "found_fraction": _ratio(found, truth),
                "correct_fraction": _ratio(correct, suggestions), "seen_in_training": split.endswith(SEEN_SUFFIX)}
    return {"file": str(path), "dataset": normalize_path(raw.get("dataset") or ""),
            "pilot_sha256": raw.get("pilot_sha256"), "conf": raw.get("conf"), "iou_match": raw.get("iou_match"),
            "models": models, "splits": splits}


def load_eval(path: Path, catalog: ModelCatalog) -> dict:
    """``runs/eval_por_classe_*.json``: ``model.val`` da Ultralytics mais contagens na confiança.

    Parâmetros:
        path: caminho do ``eval_por_classe_*.json``.
        catalog: catálogo onde cada peso é registrado (este formato não traz SHA).

    Retorna:
        Dict com ``file``, ``conf``, ``iou`` e ``models`` = ``{modelo: {split: {"overall":
        {"map50", "precision", "recall", "map50_95"}, "per_class_at_conf": {classe: {"truth",
        "tp", "fp", "fn", "precision", "recall"}}, "per_class_ultralytics": {classe:
        {"precision", "recall", "map50"}}}}}``.

    Erros:
        ``OSError``/``ValueError`` de ``_read_json``.

    Chamada por: ``build_model_report``.
    """
    raw = _read_json(path)
    models = {}
    # Formato de entrada: {"models": {"<caminho do peso>": {"<split>": {...}}}}.
    for weights, by_split in (raw.get("models") or {}).items():
        model_id = catalog.register(weights)
        models[model_id] = {}
        for split, values in (by_split or {}).items():
            overall = values.get("overall") or {}
            # Contagens na confiança das sugestões; "truth" ausente é reconstruído como tp + fn.
            at_conf = {}
            for name, counts in (values.get("per_class_at_conf") or {}).items():
                tp, fp, fn = counts.get("tp") or 0, counts.get("fp") or 0, counts.get("fn") or 0
                truth = counts.get("truth")
                at_conf[name] = {"truth": tp + fn if truth is None else truth, "tp": tp, "fp": fp, "fn": fn,
                                 "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}
            # Métricas por classe calculadas pela própria Ultralytics (P, R, mAP50).
            ultralytics = {name: {"precision": item.get("P"), "recall": item.get("R"), "map50": item.get("mAP50")}
                           for name, item in (values.get("per_class_ultralytics") or {}).items()}
            # Totais do model.val (nomes da Ultralytics "P", "R", "mAP50" → nomes do relatório).
            models[model_id][split] = {
                "overall": {"map50": overall.get("mAP50"), "precision": overall.get("P"), "recall": overall.get("R"),
                            "map50_95": overall.get("mAP50-95")},
                "per_class_at_conf": at_conf, "per_class_ultralytics": ultralytics}
    return {"file": str(path), "conf": raw.get("conf"), "iou": raw.get("iou"), "models": models}


def _read_eval_curve(path: Path) -> list[dict]:
    """``gpu_eval/eval_curve.jsonl``: uma linha por época e split; linha truncada ou com erro é ignorada.

    Parâmetros:
        path: caminho do ``eval_curve.jsonl`` gravado pelo worker de avaliação na GPU (gpu_eval.py).

    Retorna:
        Lista de pontos ``{"epoch", "split", "map50", "precision", "recall", "seconds",
        "onnx_sha256"}``; lista vazia se o arquivo não existir.

    Chamada por: ``load_run``.
    """
    # Arquivo ausente (treino sem --gpu-eval): sem pontos.
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    points = []
    # Uma linha JSON por avaliação; linhas inválidas são puladas sem erro.
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue  # treino interrompido no meio da escrita
        # Época em que a avaliação falhou ({"epoch": n, "error": "..."}) não é um ponto.
        if (not isinstance(record, dict) or "error" in record or isinstance(record.get("epoch"), bool)
                or not isinstance(record.get("epoch"), int)):
            continue
        # Ponto válido: split ausente vira "test"; NaN/infinito viram None.
        points.append({"epoch": record["epoch"], "split": str(record.get("split") or "test"),
                       "map50": _finite(record.get("map50")), "precision": _finite(record.get("precision")),
                       "recall": _finite(record.get("recall")), "seconds": _finite(record.get("seconds")),
                       "onnx_sha256": record.get("onnx_sha256")})
    return points


def _gpu_split(runs: list[dict]) -> str | None:
    """Split da curva na GPU: teste quando existe, senão validação, senão o primeiro gravado.

    Parâmetros:
        runs: dicts devolvidos por ``load_run`` (usa a chave ``gpu_eval``).

    Retorna:
        Nome do split escolhido ou ``None`` quando nenhum ponto tem mAP50.

    Chamada por: ``build_model_report`` (gráfico ``gpu_eval_curves.png``).
    """
    # Splits de todos os pontos com mAP50, na ordem em que aparecem.
    present = [point["split"] for run in runs for point in run["gpu_eval"] if point["map50"] is not None]
    # Preferência de SPLIT_ORDER (test > val > train); senão, o primeiro split visto.
    for split in SPLIT_ORDER:
        if split in present:
            return split
    return present[0] if present else None


def load_run(directory: Path, catalog: ModelCatalog) -> dict:
    """Pasta ``runs/pilot_train_*``: curvas de validação, parâmetros, estado e avaliação na GPU.

    Parâmetros:
        directory: pasta de um treino piloto.
        catalog: catálogo onde o ``fit/weights/best.pt`` do treino é registrado.

    Retorna:
        Dict com ``id`` (8 hex do nome da pasta, ou o nome inteiro se não casar com
        ``RUN_NAME``), ``directory``, ``model`` (id no catálogo), ``state`` (do summary.json),
        ``parameters`` (epochs/imgsz/batch do execution.json), ``checkpoint`` (peso inicial),
        ``epochs`` (épocas com curva), ``best_val_map50``, ``best_epoch``, ``curve`` (pontos
        ``{"epoch", "map50", "precision", "recall"}``) e ``gpu_eval`` (pontos da GPU).

    Erros:
        ``OSError`` se ``best.pt`` existir mas não puder ser lido (hash). Arquivos JSON/CSV
        ausentes ou corrompidos não derrubam: viram valores vazios.

    Chamada por: ``build_model_report``.
    """
    # 1. Identificação da pasta e leitura dos JSONs opcionais do treino.
    directory = Path(directory)
    match = RUN_NAME.match(directory.name)
    execution = _optional_json(directory / "execution.json") or {}
    summary = _optional_json(directory / "summary.json") or {}
    parameters = execution.get("parameters") or {}
    weights = directory / "fit/weights/best.pt"
    # 2. Registro do modelo do treino no catálogo.
    # ATENCAO: o registro acontece mesmo que o treino depois seja ignorado (sem curva) em
    # build_model_report; nesse caso o modelo ainda ocupa uma cor e uma linha na tabela "Modelos".
    # O treino entra no catálogo pelo seu best.pt: assim a curva usa a mesma cor
    # do modelo quando ele também aparece num gate ou numa comparação.
    model_id = catalog.register(weights, file_hash(weights) if weights.is_file() else None)
    # 3. Curva de validação por época (read_curves, em operations.py, lê fit/results.csv).
    curve = [{"epoch": point["epoch"], "map50": point.get("map50"), "precision": point.get("precision"),
              "recall": point.get("recall")} for point in read_curves(directory / "fit/results.csv")]
    # 4. Melhor época = maior mAP50 de validação (default=None quando não há nenhum mAP50).
    best = max((point for point in curve if point["map50"] is not None), key=lambda point: point["map50"],
               default=None)
    # 5. Resumo do treino. EXTENSAO: outro parâmetro de treino na tabela "Curvas de treino":
    # acrescente a chave na tupla ("epochs", "imgsz", "batch") e a coluna em _render_html.
    return {"id": match["id"] if match else directory.name, "directory": str(directory), "model": model_id,
            "state": summary.get("state"), "parameters": {key: parameters.get(key) for key in ("epochs", "imgsz", "batch")},
            "checkpoint": (execution.get("runtime") or {}).get("checkpoint"), "epochs": len(curve),
            "best_val_map50": best["map50"] if best else None, "best_epoch": best["epoch"] if best else None,
            "curve": curve, "gpu_eval": _read_eval_curve(directory / "gpu_eval/eval_curve.jsonl")}


def discover_runs(root: Path) -> list[Path]:
    """Treinos piloto em ``runs/`` que já têm curva; pastas só preparadas ficam de fora.

    Parâmetros:
        root: pasta raiz do projeto (``config.root``, a que contém ``runs/``).

    Retorna:
        Lista ordenada (por nome, logo por data) das pastas ``runs/pilot_train_*`` com nome no
        padrão ``RUN_NAME`` e ``fit/results.csv``.

    Chamada por: ``build_model_report`` quando ``runs`` é ``None`` (sem ``--run`` na CLI).
    """
    return sorted(directory for directory in (root / "runs").glob("pilot_train_*")
                  if RUN_NAME.match(directory.name) and (directory / "fit/results.csv").is_file())


# ----------------------------------------------------------------------------
# Fontes dos gráficos (gate primeiro; eval_por_classe quando não há gate)
# ----------------------------------------------------------------------------

def _format_values(values) -> str:
    """Junta valores distintos com "/" para o texto de escopo (ex.: ``"0.25/0.3"``).

    Parâmetros:
        values: iterável de valores (números, textos ou None).

    Retorna:
        Os valores únicos não nulos, em ordem, números no formato ``:g``; ``"?"`` se não houver nenhum.
    """
    unique = sorted({value for value in values if value is not None}, key=str)
    return "/".join(f"{value:g}" if isinstance(value, (int, float)) else str(value) for value in unique) or "?"


def _gate_scope(gates: list[dict]) -> str:
    """Vale para gates e medições: ambos trazem ``conf`` e ``images_evaluated``.

    Parâmetros:
        gates: dicts de ``load_gate`` ou ``load_measurement``.

    Retorna:
        Texto ``"conf <valores>, <nº> fotos"`` usado nos títulos dos gráficos e tabelas.
    """
    return (f"conf {_format_values(gate['conf'] for gate in gates)}, "
            f"{_format_values(gate['images_evaluated'] for gate in gates)} fotos")


def wilson_interval(successes: int, total: int, z: float = WILSON_Z) -> tuple[float, float] | None:
    """Intervalo de Wilson para uma proporção: honesto com n pequeno, ao contrário do normal.

    Fórmula (p = sucessos/total, n = total):
        centro = (p + z²/2n) / (1 + z²/n)
        meia-largura = z · sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)

    Parâmetros:
        successes: acertos (ex.: TP de uma classe).
        total: tentativas (ex.: caixas humanas da classe).
        z: quantil da normal (padrão ``WILSON_Z`` = 1,96, intervalo de 95%).

    Retorna:
        ``(inferior, superior)`` arredondados a 4 casas e limitados a [0, 1], ou ``None`` com total 0.

    Chamada por: ``_chart_per_class_recall`` e ``_recall_cell``.
    """
    if not total:
        return None
    p, z2 = successes / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    half = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    # Corta em [0, 1] por segurança contra arredondamento.
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def _eval_split(evals: list[dict]) -> str | None:
    """Split honesto preferido: teste; senão validação; senão o primeiro que existir.

    Parâmetros:
        evals: dicts de ``load_eval``.

    Retorna:
        Nome do split ou ``None`` se nenhuma avaliação tiver split.
    """
    present = [split for item in evals for by_split in item["models"].values() for split in by_split]
    for split in SPLIT_ORDER:
        if split in present:
            return split
    return present[0] if present else None


def _overall_source(measurements: list[dict], gates: list[dict], evals: list[dict]) -> dict | None:
    """Escolhe de onde vêm as métricas gerais (mAP50, precisão, recall) de cada modelo.

    Prioridade: medição > gate > eval_por_classe (no split de ``_eval_split``). Quando um
    modelo aparece em mais de um arquivo, vale o primeiro (``setdefault``).

    Parâmetros:
        measurements: dicts de ``load_measurement``.
        gates: dicts de ``load_gate``.
        evals: dicts de ``load_eval``.

    Retorna:
        ``{"kind": "measurement"|"gate"|"eval:<split>", "scope": texto, "title": texto,
        "values": {modelo: {"map50", "precision", "recall", ...}}}`` ou ``None`` sem dados.

    Chamada por: ``build_model_report`` (fonte principal e, com ``[]`` nas medições, a do arquivo).
    """
    values = {}
    # 1. Medição: todas as séries nas mesmas fotos (comparação principal).
    if measurements:
        for measurement in measurements:
            for series_id, metrics in measurement["overall"].items():
                values.setdefault(series_id, metrics)
        scope = f"{_gate_scope(measurements)}, as mesmas para todas as séries"
        return {"kind": "measurement", "scope": scope,
                "title": f"Métricas gerais nas fotos fora do treino ({scope})", "values": values}
    # 2. Sem medição: os gates (cada um compara só dois modelos).
    if gates:
        for gate in gates:
            for model_id, metrics in gate["overall"].items():
                values.setdefault(model_id, metrics)
        return {"kind": "gate", "scope": _gate_scope(gates),
                "title": f"Métricas gerais nas fotos fora do treino ({_gate_scope(gates)})", "values": values}
    # 3. Sem medição nem gate: eval_por_classe (model.val da Ultralytics) no split preferido.
    # EXTENSAO: nova fonte de métricas gerais: acrescente um bloco "if <dados>:" na posição da
    # prioridade desejada, devolvendo o mesmo formato {"kind", "scope", "title", "values"}.
    split = _eval_split(evals)
    if split is None:
        return None
    for item in evals:
        for model_id, by_split in item["models"].items():
            if split in by_split:
                values.setdefault(model_id, by_split[split]["overall"])
    scope = f"split {split}, model.val da Ultralytics"
    return {"kind": f"eval:{split}", "scope": scope, "title": f"Métricas gerais no {scope}", "values": values}


def _per_class_source(measurements: list[dict], gates: list[dict], evals: list[dict], names: list[str]) -> dict | None:
    """Escolhe de onde vêm as contagens por classe (TP/FP/FN, recall, AP50) e normaliza o formato.

    Prioridade: medição > gate > eval_por_classe. As chaves são uniformizadas para
    ``{"tp", "fp", "fn", "boxes", "precision", "recall", "ap50"}`` (``boxes`` = caixas humanas).

    Parâmetros:
        measurements, gates, evals: dicts das funções ``load_*``.
        names: slugs das classes ativas da taxonomia, na ordem dos IDs (``detection_names``).

    Retorna:
        ``{"kind", "scope", "classes": [ordem], "values": {classe: {modelo: contagens}}}`` ou
        ``None`` quando não há fonte.

    Chamada por: ``build_model_report``.
    """
    classes: dict[str, dict] = {}
    if measurements or gates:
        # Medição e gate têm o mesmo bloco por classe; a medição manda quando existe.
        for item in measurements or gates:
            for name, by_model in item["per_class"].items():
                for model_id, values in by_model.items():
                    classes.setdefault(name, {}).setdefault(model_id, {
                        "tp": values.get("tp") or 0, "fp": values.get("fp") or 0, "fn": values.get("fn") or 0,
                        "boxes": values.get("human_boxes") or 0, "precision": values.get("precision"),
                        "recall": values.get("recall"), "ap50": values.get("ap50")})
        kind = "measurement" if measurements else "gate"
        scope = _gate_scope(measurements or gates)
    else:
        # Sem medição nem gate: contagens na confiança das sugestões do eval_por_classe
        # (sem AP50 por classe nesse formato).
        split = _eval_split(evals)
        if split is None:
            return None
        for item in evals:
            for model_id, by_split in item["models"].items():
                for name, values in (by_split.get(split) or {}).get("per_class_at_conf", {}).items():
                    classes.setdefault(name, {}).setdefault(model_id, {
                        "tp": values["tp"], "fp": values["fp"], "fn": values["fn"], "boxes": values["truth"],
                        "precision": values["precision"], "recall": values["recall"], "ap50": None})
        kind, scope = f"eval:{split}", f"split {split}, conf {_format_values(item['conf'] for item in evals)}"
    # Ordem fixa da taxonomia; classes desconhecidas (taxonomia antiga) vão ao fim.
    order = list(names) + sorted(set(classes) - set(names))
    return {"kind": kind, "scope": scope, "classes": order, "values": classes}


def _class_boxes(source: dict) -> dict[str, int]:
    """Caixas humanas por classe: o maior valor entre os modelos (devem coincidir num gate).

    Parâmetros:
        source: fonte por classe (``_per_class_source``).

    Retorna:
        ``{classe: nº de caixas humanas}`` para todas as classes de ``source["classes"]``.
    """
    return {name: max((values.get("boxes") or 0 for values in source["values"].get(name, {}).values()), default=0)
            for name in source["classes"]}


def _source_models(source: dict, catalog: ModelCatalog) -> list[dict]:
    """Modelos com dados na fonte: geral é ``{modelo: ...}``, por classe é ``{classe: {modelo: ...}}``.

    Parâmetros:
        source: fonte geral (``_overall_source``) ou por classe (``_per_class_source``).
        catalog: catálogo dos modelos.

    Retorna:
        Dicts dos modelos presentes, na ordem do catálogo.
    """
    # A presença da chave "classes" distingue a fonte por classe da fonte geral.
    if "classes" in source:
        ids = {model_id for by_model in source["values"].values() for model_id in by_model}
    else:
        ids = set(source["values"])
    return catalog.ordered(ids)


# ----------------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------------

def _plotting():
    """Backend Agg só na hora de desenhar: importar este módulo continua barato.

    Agg desenha em memória (sem janela), o que funciona em servidor e em testes.

    Retorna:
        O módulo ``matplotlib.pyplot``, passado como ``plt`` para todas as funções ``_chart_*``.
    """
    # ATENCAO: matplotlib.use("Agg") tem de vir antes do import de pyplot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _points(pixels: float) -> float:
    """matplotlib mede linhas em pontos; a especificação dos gráficos é em pixels.

    Parâmetros:
        pixels: tamanho em pixels da imagem final.

    Retorna:
        O tamanho em pontos (1 ponto = 1/72 polegada; ``DPI`` pixels por polegada).
    """
    return pixels * 72 / DPI


def _lighten(color: str, amount: float) -> str:
    """Mistura a cor com branco (0 = igual, 1 = branco): o FP hachurado na cor do modelo.

    Parâmetros:
        color: cor ``"#rrggbb"`` (ATENCAO: não aceita ``"#rgb"`` nem nomes de cor).
        amount: fração de branco, de 0 a 1.

    Retorna:
        Nova cor ``"#rrggbb"``.
    """
    # Lê os três canais em hexadecimal (posições 1-2, 3-4, 5-6) e aproxima cada um de 255.
    channels = (int(color[index:index + 2], 16) for index in (1, 3, 5))
    return "#" + "".join(f"{round(channel + (255 - channel) * amount):02x}" for channel in channels)


def _data_per_pixel(ax, axis: str) -> float:
    """Quantas unidades do eixo cabem em um pixel (para a folga de 2 px entre barras).

    Parâmetros:
        ax: eixo do matplotlib (já com os limites definidos).
        axis: ``"x"`` ou ``"y"``.

    Retorna:
        Unidades de dados por pixel no eixo pedido.
    """
    # Converte os pixels (0,0) e (1,1) da tela para coordenadas de dados e mede a diferença.
    inverse = ax.transData.inverted()
    origin, step = inverse.transform((0.0, 0.0)), inverse.transform((1.0, 1.0))
    return abs(step[0] - origin[0]) if axis == "x" else abs(step[1] - origin[1])


def _save(plt, fig, path: Path) -> None:
    """Grava a figura em PNG (``DPI``, sem margens sobrando) e fecha para liberar memória.

    Parâmetros:
        plt: módulo pyplot (de ``_plotting``).
        fig: figura a gravar.
        path: caminho do PNG.

    Retorna:
        None.
    """
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _fraction_axis(ax, which: str) -> None:
    """Eixo de 0,0 a 1,0 em frações (nunca porcentagem, nunca dois eixos).

    Parâmetros:
        ax: eixo do matplotlib.
        which: ``"y"`` (barras verticais e curvas) ou ``"x"`` (barras horizontais).

    Retorna:
        None.
    """
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    axis = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_ylim if which == "y" else ax.set_xlim)(0, 1.06)  # folga para o rótulo da barra cheia
    axis.set_major_locator(MultipleLocator(0.2))
    axis.set_major_formatter(FormatStrFormatter("%.1f"))


def _legend(ax, **kwargs) -> None:
    """Legenda sem repetição: vários modelos "outros" viram uma entrada só.

    Parâmetros:
        ax: eixo do matplotlib.
        **kwargs: repassados a ``ax.legend`` (ex.: ``loc="upper right"``).

    Retorna:
        None (sem legenda quando nada tem rótulo).
    """
    # dict(zip(...)) mantém um handle por rótulo: rótulos repetidos ("outros") viram um só.
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), **kwargs)


def _grouped_bars(ax, categories: list[str], series: list[tuple], horizontal: bool = False,
                  value_labels: bool = False) -> None:
    """Barras finas lado a lado por categoria; cada série é (rótulo, cor, valores) de um modelo.

    Parâmetros:
        ax: eixo do matplotlib.
        categories: rótulos das categorias (ex.: métricas, classes, splits).
        series: lista de tuplas ``(rótulo, cor, valores)`` ou ``(rótulo, cor, valores, intervalos)``;
            ``valores`` tem um item por categoria (``None`` = sem barra); ``intervalos`` tem
            ``(inferior, superior)`` ou ``None`` por categoria (barra de erro).
        horizontal: ``True`` desenha barras horizontais (categorias no eixo y, primeira no topo).
        value_labels: escreve o valor (2 casas) na ponta de cada barra.

    Retorna:
        None.

    Chamada por: ``_chart_overall``, ``_chart_per_class_recall``, ``_chart_strata``, ``_chart_compare``.
    """
    # 1. Eixo das categorias: uma posição inteira por categoria.
    count = max(len(series), 1)
    positions = list(range(len(categories)))
    if horizontal:
        ax.set_ylim(len(categories) - 0.5, -0.5)  # primeira categoria no topo
        ax.set_yticks(positions, categories)
    else:
        ax.set_xlim(-0.5, len(categories) - 0.5)
        ax.set_xticks(positions, categories)
    # 2. Largura: o grupo ocupa GROUP_WIDTH; cada série recebe um "slot" e a barra perde BAR_GAP_PX
    #    pixels de folga (no mínimo metade do slot).
    slot = GROUP_WIDTH / count
    width = max(slot - BAR_GAP_PX * _data_per_pixel(ax, "y" if horizontal else "x"), slot / 2)
    # 3. Uma série (modelo) por vez, deslocada do centro da categoria: offset vai de
    #    -(n-1)/2 a +(n-1)/2 slots, centralizando o grupo.
    for index, (label, color, values, *rest) in enumerate(series):
        offset = (index - (count - 1) / 2) * slot
        intervals = rest[0] if rest else [None] * len(values)
        # Valor ausente (None) não vira barra zero: zero mede, ausente não.
        placed = [(position + offset, value, interval)
                  for position, value, interval in zip(positions, values, intervals) if value is not None]
        if not placed:
            continue
        at, heights, bounds = zip(*placed)
        errors = None
        if any(bound is not None for bound in bounds):
            # Barra de erro fina (intervalo de Wilson) do valor até cada extremo.
            errors = [[value - (bound[0] if bound else value) for value, bound in zip(heights, bounds)],
                      [(bound[1] if bound else value) - value for value, bound in zip(heights, bounds)]]
        # Barra de erro discreta (cinza escuro, fina, com traço curto nas pontas).
        error_kw = {"ecolor": "#444444", "elinewidth": 0.7, "capsize": 2, "capthick": 0.7}
        if horizontal:
            bars = ax.barh(at, heights, height=width, color=color, label=label, xerr=errors, error_kw=error_kw)
        else:
            bars = ax.bar(at, heights, width=width, color=color, label=label, yerr=errors, error_kw=error_kw)
        if value_labels:
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2, color="#333")


def _chart_overall(plt, path: Path, source: dict, models: list[dict], catalog: ModelCatalog) -> None:
    """Gráfico de barras com mAP50, precisão e recall gerais de cada modelo.

    Parâmetros:
        plt: módulo pyplot.
        path: PNG a gravar (``overall_metrics.png`` ou ``overall_metrics_arquivo.png``).
        source: fonte geral (``_overall_source``).
        models: modelos a desenhar (``_source_models``), na ordem do catálogo.
        catalog: catálogo (rótulos da legenda).

    Retorna:
        None (grava o PNG).

    Chamada por: ``_overall_chart``.
    """
    # EXTENSAO: nova métrica geral no gráfico (ex.: ("map50_95", "mAP50-95")): acrescente o par
    # (chave na fonte, rótulo) aqui; a chave tem de existir em source["values"][modelo].
    metrics = (("map50", "mAP50"), ("precision", "Precisão"), ("recall", "Recall"))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        # Uma série por modelo: os valores das métricas na ordem de "metrics".
        series = [(catalog.legend_label(model), model["color"],
                   [_finite(source["values"][model["id"]].get(key)) for key, _ in metrics]) for model in models]
        _grouped_bars(ax, [label for _, label in metrics], series, value_labels=True)
        _fraction_axis(ax, "y")
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Métrica")
        ax.set_ylabel("Valor (fração de 0,0 a 1,0)")
        ax.set_title(source["title"])
        # Legenda só com dois ou mais modelos (com um só, o título já basta).
        if len(models) >= 2:
            _legend(ax, loc="upper right")
        _save(plt, fig, path)


def _chart_per_class_recall(plt, path: Path, source: dict, models: list[dict], catalog: ModelCatalog) -> None:
    """Barras horizontais de recall por classe, com intervalo de Wilson de 95% em cada barra.

    Classes com menos de ``FEW_BOXES`` caixas humanas não ganham barra, só um texto
    ("não medida" ou "n<3: pouco medida").

    Parâmetros:
        plt: módulo pyplot.
        path: PNG a gravar (``per_class_recall.png`` ou ``per_class_recall_arquivo.png``).
        source: fonte por classe (``_per_class_source``).
        models: modelos a desenhar.
        catalog: catálogo (rótulos da legenda).

    Retorna:
        None (grava o PNG).

    Chamada por: ``_per_class_charts``.
    """
    classes, values, boxes = source["classes"], source["values"], _class_boxes(source)
    with plt.rc_context(STYLE):
        # Altura cresce com o número de classes (0,42 polegada por classe).
        fig, ax = plt.subplots(figsize=(8, 1.6 + 0.42 * len(classes)))
        series = []
        for model in models:
            recalls, intervals = [], []
            for name in classes:
                counts = values.get(name, {}).get(model["id"], {})
                # Com menos de 3 caixas a barra engana mais do que informa: só o texto.
                measured = boxes[name] >= FEW_BOXES and counts.get("recall") is not None
                recalls.append(_finite(counts["recall"]) if measured else None)
                intervals.append(wilson_interval(counts.get("tp") or 0, counts.get("boxes") or 0) if measured else None)
            series.append((catalog.legend_label(model), model["color"], recalls, intervals))
        _grouped_bars(ax, [f"{name} ({boxes[name]})" for name in classes], series, horizontal=True)
        _fraction_axis(ax, "x")
        ax.grid(axis="y", visible=False)
        # Texto cinza em itálico no lugar da barra das classes pouco medidas.
        for index, name in enumerate(classes):
            if boxes[name] < FEW_BOXES:
                text = "não medida" if not boxes[name] else "n<3: pouco medida"
                ax.text(0.01, index, text, va="center", ha="left", fontsize=8, color="#777", style="italic")
        ax.set_xlabel("Recall (caixas humanas reencontradas / caixas humanas; barra fina = IC 95% de Wilson)")
        ax.set_ylabel("Classe (número de caixas humanas)")
        ax.set_title(f"Recall por classe ({source['scope']})")
        if len(models) >= 2:
            _legend(ax, loc="lower right")
        _save(plt, fig, path)


def _chart_per_class_errors(plt, path: Path, source: dict, models: list[dict]) -> None:
    """Um painel por modelo com barras empilhadas TP | FP | FN por classe (contagem de caixas).

    Parâmetros:
        plt: módulo pyplot.
        path: PNG a gravar (``per_class_errors.png`` ou ``per_class_errors_arquivo.png``).
        source: fonte por classe (``_per_class_source``).
        models: modelos a desenhar (um painel cada, lado a lado).

    Retorna:
        None (grava o PNG).

    Chamada por: ``_per_class_charts``.
    """
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    values, boxes = source["values"], _class_boxes(source)
    # Classe sem caixa humana entra só quando algum modelo inventou caixas nela (alucinação).
    classes = [name for name in source["classes"] if boxes[name]
               or any((counts.get("fp") or 0) for counts in values.get(name, {}).values())]
    labels = [name if boxes[name] else f"{name} (0 caixas humanas)" for name in classes]
    # Um painel por modelo, todos com o mesmo eixo y (sharey). squeeze=False garante axes 2D
    # mesmo com um só modelo (axes[0] é a linha de painéis).
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(models), sharey=True, squeeze=False,
                                 figsize=(2.8 * len(models) + 1.6, 0.34 * len(classes) + 2.2))
        positions = list(range(len(classes)))
        for ax, model in zip(axes[0], models):
            counts = [values.get(name, {}).get(model["id"], {}) for name in classes]
            tp = [item.get("tp") or 0 for item in counts]
            fp = [item.get("fp") or 0 for item in counts]
            fn = [item.get("fn") or 0 for item in counts]
            # Empilhamento: TP (cor do modelo) a partir de 0; FP (cor clareada e hachurada) a partir
            # do fim do TP; FN (cinza) a partir de TP + FP.
            ax.barh(positions, tp, height=0.6, color=model["color"])
            ax.barh(positions, fp, left=tp, height=0.6, color=_lighten(model["color"], 0.55),
                    edgecolor=model["color"], hatch="//", linewidth=0.5)
            ax.barh(positions, fn, left=[a + b for a, b in zip(tp, fp)], height=0.6, color=FN_COLOR)
            ax.set_ylim(len(classes) - 0.5, -0.5)
            ax.set_yticks(positions, labels)
            ax.grid(axis="y", visible=False)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Caixas")
            ax.set_title(model["label"])
        # Legenda comum aos painéis, embaixo da figura.
        handles = [Patch(facecolor="#555555", label="TP: acertos (na cor do modelo)"),
                   Patch(facecolor="white", edgecolor="#555555", hatch="//", label="FP: falsos positivos"),
                   Patch(facecolor=FN_COLOR, label="FN: caixas humanas não encontradas")]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Acertos e erros por classe ({source['scope']})")
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        _save(plt, fig, path)


def _chart_strata(plt, path: Path, measurements: list[dict], catalog: ModelCatalog) -> bool:
    """Recall total e mAP50 por estrato (fotos fatiadas do drone × fotos inteiras do laudo).

    Parâmetros:
        plt: módulo pyplot.
        path: PNG a gravar (``strata_recall.png``).
        measurements: dicts de ``load_measurement``.
        catalog: catálogo dos modelos.

    Retorna:
        ``True`` se gravou o gráfico; ``False`` quando nenhuma medição traz estratos.

    Chamada por: ``build_model_report``.
    """
    # Junta {estrato: {série: totais}} de todas as medições (vale a primeira de cada série).
    strata: dict[str, dict] = {}
    for measurement in measurements:
        for stratum, by_series in measurement["strata"].items():
            for series_id, totals in by_series.items():
                strata.setdefault(stratum, {}).setdefault(series_id, totals)
    if not strata:
        return False
    # Ordem dos estratos: a do dict STRATA; desconhecidos depois, em ordem alfabética.
    order = sorted(strata, key=lambda name: (list(STRATA).index(name) if name in STRATA else len(STRATA), name))
    models = catalog.ordered({series_id for by_series in strata.values() for series_id in by_series})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(4.2 * len(order) + 3, 4.4))
        # Dois painéis lado a lado: recall total e mAP50.
        for ax, (key, title) in zip(axes, (("recall", "Recall total"), ("map50", "mAP50"))):
            series = [(catalog.legend_label(model), model["color"],
                       [_finite(strata[name].get(model["id"], {}).get(key)) for name in order]) for model in models]
            _grouped_bars(ax, [STRATA.get(name, name) for name in order], series, value_labels=True)
            _fraction_axis(ax, "y")
            ax.grid(axis="x", visible=False)
            ax.set_xlabel("Estrato de fotos")
            ax.set_title(title)
        axes[0].set_ylabel("Valor (fração de 0,0 a 1,0)")
        if len(models) >= 2:
            _legend(axes[0], loc="upper left")
        fig.suptitle(f"Recall e mAP50 por estrato ({_gate_scope(measurements)})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(plt, fig, path)
    return True


def _curve_points(run: dict, key: str, split: str | None) -> list[tuple[int, float]]:
    """Pontos (época, mAP50) de uma curva do treino, ordenados por época.

    Parâmetros:
        run: dict de ``load_run``.
        key: ``"curve"`` (validação do results.csv) ou ``"gpu_eval"`` (avaliação na GPU).
        split: filtra os pontos por split (só para ``gpu_eval``); ``None`` aceita todos.

    Retorna:
        Lista de tuplas ``(época, mAP50)`` sem pontos de mAP50 ausente.
    """
    return sorted((point["epoch"], point["map50"]) for point in run[key]
                  if point["map50"] is not None and (split is None or point.get("split") == split))


def _chart_curves(plt, path: Path, runs: list[dict], catalog: ModelCatalog, key: str, title: str,
                  ylabel: str, split: str | None = None) -> bool:
    """Uma linha por treino; devolve False (e não grava) quando nenhum treino tem a curva.

    Parâmetros:
        plt: módulo pyplot.
        path: PNG a gravar (``training_curves.png`` ou ``gpu_eval_curves.png``).
        runs: dicts de ``load_run``.
        catalog: catálogo (cor do modelo de cada treino).
        key: ``"curve"`` ou ``"gpu_eval"`` (ver ``_curve_points``).
        title: título do gráfico.
        ylabel: rótulo do eixo y.
        split: split dos pontos de ``gpu_eval`` (``None`` para a curva de validação).

    Retorna:
        ``True`` se gravou; ``False`` se nenhum treino tem pontos.

    Chamada por: ``build_model_report``.
    """
    from matplotlib.ticker import MaxNLocator

    drawable = [run for run in runs if _curve_points(run, key, split)]
    if not drawable:
        return False
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for run in drawable:
            model = catalog.by_id(run["model"])
            points = _curve_points(run, key, split)
            xs, ys = zip(*points)
            # Rótulo = id do treino (8 hex); treinos além da paleta viram "outros".
            # Marcadores só com poucos pontos (MAX_MARKER_POINTS).
            label = run["id"] if model["color"] != OTHERS_COLOR else OTHERS_LABEL
            ax.plot(xs, ys, color=model["color"], linewidth=_points(LINE_PX),
                    marker="o" if len(points) <= MAX_MARKER_POINTS else None, markersize=3.5, label=label)
        _fraction_axis(ax, "y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if len(drawable) >= 2:
            _legend(ax, loc="best")
        _save(plt, fig, path)
    return True


def _split_base(split: str) -> tuple[str, bool]:
    """Separa o nome do split do sufixo ``SEEN_SUFFIX``.

    Parâmetros:
        split: nome gravado no compare (ex.: ``"val (visto no treino)"``).

    Retorna:
        ``(nome sem sufixo, True se tinha o sufixo)``, ex.: ``("val", True)``.
    """
    seen = split.endswith(SEEN_SUFFIX)
    return (split[:-len(SEEN_SUFFIX)] if seen else split), seen


def _split_key(category: str) -> tuple:
    """Chave de ordenação das categorias de split (test, val, train, depois as demais).

    Parâmetros:
        category: nome do split, opcionalmente com ``"\\n<dataset>"`` (ver ``_chart_compare``).

    Retorna:
        ``(posição em SPLIT_ORDER ou fim, categoria)``.
    """
    base = category.split("\n")[0]
    return (SPLIT_ORDER.index(base) if base in SPLIT_ORDER else len(SPLIT_ORDER), category)


def _chart_compare(plt, path: Path, compares: list[dict], catalog: ModelCatalog) -> None:
    """Fração de caixas humanas reencontradas por split, em painéis "fora do treino" e "visto no treino".

    Parâmetros:
        plt: módulo pyplot.
        path: PNG a gravar (``compare_found.png`` ou ``compare_found_arquivo.png``).
        compares: dicts de ``load_compare``.
        catalog: catálogo dos modelos.

    Retorna:
        None (grava o PNG).

    Chamada por: ``build_model_report``.
    """
    # Fotos vistas no treino ficam num painel à parte: lá o modelo pode só ter decorado.
    panels = {False: {}, True: {}}
    datasets = {compare["dataset"] for compare in compares}
    # Monta {visto?: {categoria: {modelo: found_fraction}}}. Com compares de datasets
    # diferentes, a categoria ganha uma 2ª linha com o nome da pasta do dataset.
    for compare in compares:
        suffix = f"\n{PurePosixPath(compare['dataset']).parent.name}" if len(datasets) > 1 else ""
        for split, by_model in compare["splits"].items():
            base, seen = _split_base(split)
            for model_id, counts in by_model.items():
                panels[seen].setdefault(base + suffix, {}).setdefault(model_id, counts["found_fraction"])
    # Só os painéis que têm dados; a largura de cada um é proporcional ao nº de categorias.
    panels = {seen: data for seen, data in panels.items() if data}
    ids = {model_id for data in panels.values() for by_model in data.values() for model_id in by_model}
    models = catalog.ordered(ids)
    titles = {False: "Fotos fora do treino", True: "Fotos vistas no treino (memorização, não generalização)"}
    widths = [max(len(data), 1) for data in panels.values()]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                                 figsize=(2.2 * sum(widths) + 2.5, 4.4), gridspec_kw={"width_ratios": widths})
        for ax, (seen, data) in zip(axes[0], panels.items()):
            categories = sorted(data, key=_split_key)
            series = [(catalog.legend_label(model), model["color"],
                       [_finite(data[category].get(model["id"])) for category in categories]) for model in models]
            _grouped_bars(ax, categories, series, value_labels=True)
            _fraction_axis(ax, "y")
            ax.grid(axis="x", visible=False)
            ax.set_xlabel("Split")
            ax.set_title(titles[seen])
        axes[0][0].set_ylabel("Caixas humanas reencontradas (fração)")
        if len(models) >= 2:
            _legend(axes[0][0], loc="upper left")
        scope = f"conf {_format_values(c['conf'] for c in compares)}, IoU ≥ {_format_values(c['iou_match'] for c in compares)}"
        fig.suptitle(f"Caixas humanas reencontradas por split ({scope})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save(plt, fig, path)


# ----------------------------------------------------------------------------
# Notas e HTML
# ----------------------------------------------------------------------------

def _notes(measurements: list[dict], gates: list[dict], per_class: dict | None, overall: dict | None,
           archive: bool, catalog: ModelCatalog, skipped: list[str]) -> list[str]:
    """Monta as notas de leitura do relatório (seção "Notas" do HTML e ``models.json``).

    Parâmetros:
        measurements, gates: dicts de ``load_measurement`` e ``load_gate``.
        per_class, overall: fontes principais (``_per_class_source`` e ``_overall_source``).
        archive: ``True`` quando existe a seção "medições anteriores".
        catalog: catálogo (rótulos dos modelos do gate).
        skipped: avisos de treinos ignorados (gerados em ``build_model_report``).

    Retorna:
        Lista de frases em português, na ordem em que aparecem.

    Chamada por: ``build_model_report``.
    """
    notes = []
    # 1. Quantas fotos vistas no treino foram excluídas em cada medição/gate.
    for item, kind in [(m, "medição") for m in measurements] + [(g, "gate") for g in gates]:
        excluded = item["images_excluded_seen_in_training"]
        notes.append(f"Fotos vistas no treino de qualquer um dos modelos foram excluídas das métricas do {kind} "
                     f"({excluded if excluded is not None else '?'} excluídas).")
    # 2. Avisos fixos. ATENCAO: o relatório nunca aprova modelo; mantenha esta nota.
    # EXTENSAO: nova nota fixa: acrescente notes.append("...") aqui.
    notes.append("Nenhum resultado aqui aprova um modelo para produção: a avaliação usa uma única edificação.")
    if measurements:
        notes.append("Na medição principal todas as séries foram avaliadas nas mesmas fotos; o mesmo peso em dois "
                     "regimes (janelas × inteira) aparece como duas séries.")
    if archive:
        notes.append(f"{ARCHIVE_TITLE}: cada arquivo usou outro conjunto de fotos; compare só dentro de cada tabela.")
    # 3. Veredito de cada gate, copiado do gate.json (adopt e reasons).
    for gate in gates:
        baseline, candidate = catalog.by_id(gate["baseline"])["label"], catalog.by_id(gate["candidate"])["label"]
        verdict = "sim" if gate["adopt"] else "não"
        reasons = "; ".join(gate["reasons"]) if gate["reasons"] else "sem ressalvas pela regra do gate"
        notes.append(f"Gate {PurePosixPath(normalize_path(gate['file'])).parent.name}: adotar o candidato "
                     f"{candidate} no lugar de {baseline}? {verdict}. Motivos: {reasons}")
    # 4. Avisos sobre a fonte por classe e a fonte geral.
    if per_class:
        boxes = _class_boxes(per_class)
        few = [f"{name} ({boxes[name]})" for name in per_class["classes"] if boxes[name] < FEW_BOXES]
        if few:
            notes.append(f"Classes pouco medidas (menos de {FEW_BOXES} caixas humanas): {', '.join(few)}. "
                         "Os números delas não sustentam conclusão.")
        if per_class["kind"].startswith("eval"):
            notes.append(f"Sem gate.json: métricas por classe vêm de eval_por_classe ({per_class['scope']}), "
                         "que não exclui fotos vistas no treino.")
    if overall and overall["kind"].startswith("eval"):
        notes.append(f"Métricas gerais vêm de eval_por_classe ({overall['scope']}): precisão e recall são os do "
                     "model.val da Ultralytics, não os da confiança das sugestões.")
    # 5. Avisos de treinos ignorados vêm por último.
    notes.extend(skipped)
    return notes


def _esc(value) -> str:
    """Escapa texto para HTML (``&``, ``<``, ``>``, aspas).

    Parâmetros:
        value: qualquer valor (convertido com ``str``).

    Retorna:
        Texto seguro para pôr dentro de tags e atributos.

    ATENCAO: todo texto vindo dos arquivos de entrada (caminhos, nomes de classe, motivos)
    tem de passar por aqui antes de entrar no HTML.
    """
    return html.escape(str(value), quote=True)


def _fmt(value, digits: int = 3) -> str:
    """Formata um valor para exibição (sem escapar).

    Parâmetros:
        value: None, bool, float ou outro.
        digits: casas decimais dos floats (padrão 3).

    Retorna:
        ``"—"`` para None, ``"sim"``/``"não"`` para bool, float com ``digits`` casas, ou ``str(value)``.
    """
    # bool antes de float/int: True/False são ints em Python.
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(value, digits: int = 3) -> str:
    """Valor formatado (``_fmt``) e escapado (``_esc``), pronto para uma célula de tabela.

    Parâmetros:
        value: valor a mostrar.
        digits: casas decimais dos floats.

    Retorna:
        Texto HTML seguro.
    """
    return _esc(_fmt(value, digits))


def _table(headers: list[str], rows: list[list[str]], caption: str | None = None) -> str:
    """Células já escapadas; os cabeçalhos são escapados aqui.

    Parâmetros:
        headers: textos dos cabeçalhos (sem escapar).
        rows: linhas; cada célula já em HTML seguro (``_cell``, ``_esc``, ``_swatch``).
        caption: legenda da tabela (sem escapar) ou ``None``.

    Retorna:
        O HTML de uma ``<table>`` completa.
    """
    head = "".join(f"<th>{_esc(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    title = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return f"<table>{title}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _figure(directory: Path, name: str, alt: str) -> str:
    """PNG embutido em base64: a página funciona copiada sozinha ou aberta do disco.

    Parâmetros:
        directory: pasta do relatório onde o PNG foi gravado.
        name: nome do PNG (ex.: ``overall_metrics.png``).
        alt: texto alternativo da imagem.

    Retorna:
        HTML de uma ``<figure>`` com a imagem embutida e o nome do arquivo como legenda.

    Erros:
        ``OSError`` se o PNG não existir (só chame com nomes presentes em ``charts``).
    """
    encoded = base64.b64encode((directory / name).read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{_esc(alt)}"><figcaption>{_esc(name)}</figcaption></figure>'


def _swatch(model: dict) -> str:
    """Quadradinho com a cor do modelo seguido do rótulo (HTML escapado).

    Parâmetros:
        model: dict do modelo no catálogo.

    Retorna:
        HTML ``<span class="swatch" ...></span>rótulo``.
    """
    return f'<span class="swatch" style="background:{_esc(model["color"])}"></span>{_esc(model["label"])}'


# Folha de estilo da página report.html (embutida em <style>, sem arquivo externo).
# EXTENSAO: estilo de um elemento novo da página: acrescente a regra CSS aqui.
CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif;max-width:1100px;margin:2rem auto;
padding:0 1rem;color:#222;line-height:1.45;background:#fff}
h1{font-size:1.6rem;margin-bottom:.2rem}h2{font-size:1.2rem;margin-top:2.2rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}
h3{font-size:1rem;margin-top:1.4rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;font-size:.92rem}
th,td{border:1px solid #ccc;padding:.35rem .55rem;text-align:left;vertical-align:top}
th{background:#eef1f4}tbody tr:nth-child(even){background:#f6f8fa}caption{text-align:left;font-weight:600;padding:.3rem 0}
figure{margin:1rem 0}img{max-width:100%;height:auto;border:1px solid #e3e3e3;background:#fff}
figcaption{font-size:.8rem;color:#666}code{font-size:.88em;word-break:break-all}
.muted{color:#666}.swatch{display:inline-block;width:.85em;height:.85em;border-radius:2px;vertical-align:-.05em;margin-right:.4em}
.archive{border-left:4px solid #e0a800;padding-left:1rem;margin-top:1rem}ul.notes li{margin:.3rem 0}
"""


def _overall_section(parts: list[str], directory: Path, source: dict | None, chart: str, charts: list[str],
                     gates: list[dict], catalog: ModelCatalog, level: str) -> None:
    """Acrescenta em ``parts`` a seção "Métricas gerais": gráfico (se houver) e tabela por série.

    Parâmetros:
        parts: lista de pedaços de HTML que vira a página (alterada no lugar).
        directory: pasta do relatório (para embutir o PNG).
        source: fonte geral (``_overall_source``) ou ``None``.
        chart: nome do PNG desta seção (ex.: ``overall_metrics.png``).
        charts: PNGs efetivamente gravados.
        gates: gates, para a coluna "Papel" (baseline/candidato); ``[]`` na medição principal.
        catalog: catálogo dos modelos.
        level: tag do título (``"h2"`` na seção principal, ``"h3"`` no arquivo).

    Retorna:
        None.
    """
    parts.append(f"<{level}>Métricas gerais</{level}>")
    if not source:
        parts.append("<p class='muted'>Sem medição, gate.json nem eval_por_classe: nenhuma métrica geral.</p>")
        return
    if chart in charts:
        parts.append(_figure(directory, chart, "Métricas gerais por modelo"))
    # Papel de cada modelo nos gates (o primeiro papel encontrado vale).
    roles = {}
    for gate in gates:
        roles.setdefault(gate["baseline"], "atual (baseline)")
        roles.setdefault(gate["candidate"], "candidato")
    values = source["values"]
    parts.append(_table(["Série", "Papel", "mAP50", "Precisão", "Recall"], [
        [_swatch(model), _cell(roles.get(model["id"])), _cell(values[model["id"]].get("map50")),
         _cell(values[model["id"]].get("precision")), _cell(values[model["id"]].get("recall"))]
        for model in _source_models(source, catalog)], caption=source["title"]))


def _recall_cell(counts: dict, boxes: int) -> str:
    """Célula de recall com intervalo de Wilson, ex.: ``"0.750 [0.41–0.93]"``.

    Parâmetros:
        counts: contagens do modelo na classe (``tp``, ``boxes``, ``recall``).
        boxes: caixas humanas da classe (``_class_boxes``).

    Retorna:
        Texto HTML seguro: "não medida" (0 caixas), "n<3: pouco medida", ``"—"`` sem recall, ou o valor.
    """
    if not boxes:
        return "não medida"
    if boxes < FEW_BOXES:
        return _esc("n<3: pouco medida")
    interval = wilson_interval(counts.get("tp") or 0, counts.get("boxes") or boxes)
    recall = counts.get("recall")
    if recall is None or interval is None:
        return "—"
    return _esc(f"{_fmt(recall)} [{_fmt(interval[0], 2)}–{_fmt(interval[1], 2)}]")


def _per_class_section(parts: list[str], directory: Path, source: dict | None, suffix: str, charts: list[str],
                       catalog: ModelCatalog, level: str) -> None:
    """Acrescenta em ``parts`` a seção "Por classe": gráficos e tabela TP/FP/FN, recall e AP50.

    Parâmetros:
        parts: lista de pedaços de HTML (alterada no lugar).
        directory: pasta do relatório.
        source: fonte por classe ou ``None``.
        suffix: ``""`` (principal) ou ``"_arquivo"``; compõe o nome dos PNGs.
        charts: PNGs efetivamente gravados.
        catalog: catálogo dos modelos.
        level: tag do título (``"h2"`` ou ``"h3"``).

    Retorna:
        None.
    """
    parts.append(f"<{level}>Por classe</{level}>")
    if not source:
        parts.append("<p class='muted'>Sem medição, gate.json nem eval_por_classe: nenhuma métrica por classe.</p>")
        return
    models = _source_models(source, catalog)
    for name, alt in ((f"per_class_recall{suffix}.png", "Recall por classe"),
                      (f"per_class_errors{suffix}.png", "Acertos e erros por classe")):
        if name in charts:
            parts.append(_figure(directory, name, alt))
    # Tabela: 2 colunas fixas + 3 colunas por modelo (TP/FP/FN, recall com IC, AP50).
    # EXTENSAO: nova coluna por modelo na tabela por classe: acrescente o cabeçalho aqui e o
    # valor na mesma posição dentro do laço "for model in models" das linhas abaixo.
    boxes = _class_boxes(source)
    headers = ["Classe", "Caixas humanas"]
    for model in models:
        headers += [f"TP/FP/FN · {model['label']}", f"Recall [IC95] · {model['label']}", f"AP50 · {model['label']}"]
    rows = []
    for name in source["classes"]:
        by_model = source["values"].get(name, {})
        row = [_esc(name), _cell(boxes[name])]
        for model in models:
            counts = by_model.get(model["id"], {})
            row.append(_esc(f"{counts.get('tp') or 0}/{counts.get('fp') or 0}/{counts.get('fn') or 0}"))
            row.append(_recall_cell(counts, boxes[name]))
            # AP50 só aparece com FEW_BOXES ou mais caixas humanas.
            row.append(_cell(counts.get("ap50")) if boxes[name] >= FEW_BOXES else "—")
        rows.append(row)
    parts.append(_table(headers, rows, caption=f"Contagens na confiança das sugestões ({source['scope']}); "
                                               f"AP50 só com {FEW_BOXES} ou mais caixas humanas"))


def _compare_section(parts: list[str], directory: Path, compares: list[dict], chart: str, charts: list[str],
                     catalog: ModelCatalog, level: str) -> None:
    """Acrescenta em ``parts`` a seção "Reencontradas por split": gráfico e uma tabela por compare.

    Parâmetros:
        parts: lista de pedaços de HTML (alterada no lugar).
        directory: pasta do relatório.
        compares: dicts de ``load_compare``.
        chart: nome do PNG (``compare_found.png`` ou ``compare_found_arquivo.png``).
        charts: PNGs efetivamente gravados.
        catalog: catálogo dos modelos.
        level: tag do título.

    Retorna:
        None.
    """
    parts.append(f"<{level}>Reencontradas por split</{level}>")
    if not compares:
        parts.append("<p class='muted'>Nenhum compare_*.json informado.</p>")
        return
    if chart in charts:
        parts.append(_figure(directory, chart, "Caixas humanas reencontradas por split"))
    parts.append("<p class='muted'>Splits marcados como “visto no treino” contêm fotos que o modelo treinou: "
                 "medem memorização, não generalização, e ficam separados.</p>")
    for compare in compares:
        models = catalog.ordered(compare["models"])
        headers = (["Split"] + [f"Reencontradas / caixas · {model['label']}" for model in models]
                   + [f"Corretas / sugestões · {model['label']}" for model in models])
        rows = []
        # Linhas: splits fora do treino primeiro, depois os vistos; dentro de cada grupo,
        # a ordem de SPLIT_ORDER. Cada célula: "achadas / total (fração)" ou "—" sem dados.
        for split in sorted(compare["splits"], key=lambda value: (_split_base(value)[1], _split_key(_split_base(value)[0]))):
            by_model = compare["splits"][split]
            found = [(f"{_cell(c['found'])} / {_cell(c['truth'])} ({_cell(c['found_fraction'], 2)})" if (c := by_model.get(model["id"])) else "—")
                     for model in models]
            correct = [(f"{_cell(c['correct'])} / {_cell(c['suggestions'])} ({_cell(c['correct_fraction'], 2)})" if (c := by_model.get(model["id"])) else "—")
                       for model in models]
            rows.append([_esc(split)] + found + correct)
        caption = f"{PurePosixPath(normalize_path(compare['file'])).name} · dataset {compare['dataset'] or '?'} · conf {_fmt(compare['conf'])} · IoU ≥ {_fmt(compare['iou_match'])}"
        parts.append(_table(headers, rows, caption=caption))


def _render_html(directory: Path, title: str, created_at: str, catalog: ModelCatalog, measurements: list[dict],
                 gates: list[dict], compares: list[dict], runs: list[dict], sources: dict, charts: list[str],
                 notes: list[str]) -> str:
    """Monta o texto completo de ``report.html`` (página autônoma, sem arquivos externos).

    Ordem das seções: Modelos → (com medição) Métricas gerais, Por classe, Estratos e a
    seção "medições anteriores"; (sem medição) Métricas gerais, Por classe, Reencontradas
    por split → Curvas de treino → Notas.

    Parâmetros:
        directory: pasta do relatório (PNGs a embutir; mostrada no topo).
        title: título da página.
        created_at: data ISO 8601 (UTC) da geração.
        catalog: catálogo dos modelos (tabela "Modelos").
        measurements, gates, compares, runs: dicts das funções ``load_*``.
        sources: ``{"overall", "per_class", "archive_overall", "archive_per_class"}`` montado em
            ``build_model_report``.
        charts: PNGs efetivamente gravados.
        notes: frases de ``_notes``.

    Retorna:
        O HTML inteiro (texto).

    Chamada por: ``build_model_report``.
    """
    # Cabeçalho da página com o CSS embutido e o título.
    parts = [f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{_esc(title)}</title>'
             f"<style>{CSS}</style></head><body>",
             f"<h1>{_esc(title)}</h1><p class='muted'>Gerado em {_esc(created_at)} · pasta <code>{_esc(directory)}</code></p>"]

    # Séries: a mesma cor e o mesmo rótulo valem para todos os gráficos abaixo.
    parts.append("<h2>Modelos</h2>")
    parts.append(_table(["Série", "Arquivo", "SHA-256", "Treino", "Regime"], [
        [_swatch(model), f"<code>{_esc(model['weights'])}</code>",
         f"<code>{_esc((model['sha256'] or '')[:12] or '—')}</code>", _cell(model["run_id"]), _cell(model["regime"])]
        for model in catalog.models]))

    # Com medição: ela é a comparação principal; gates/compares/evals vão para a caixa "arquivo".
    if measurements:
        _overall_section(parts, directory, sources["overall"], "overall_metrics.png", charts, [], catalog, "h2")
        _per_class_section(parts, directory, sources["per_class"], "", charts, catalog, "h2")
        parts.append("<h2>Estratos de fotos</h2>")
        if "strata_recall.png" in charts:
            parts.append(_figure(directory, "strata_recall.png", "Recall e mAP50 por estrato"))
            counts = {}
            for measurement in measurements:
                for stratum, number in measurement["strata_images"].items():
                    counts.setdefault(stratum, number)
            parts.append("<p class='muted'>" + _esc("Fotos por estrato: " + ", ".join(
                f"{STRATA.get(name, name)} = {_fmt(number)}" for name, number in counts.items())) + "</p>")
        else:
            parts.append("<p class='muted'>A medição não trouxe métricas por estrato.</p>")
        if gates or compares or sources["archive_overall"] or sources["archive_per_class"]:
            parts.append(f"<h2>{_esc(ARCHIVE_TITLE)}</h2><div class='archive'>")
            _overall_section(parts, directory, sources["archive_overall"], "overall_metrics_arquivo.png", charts,
                             gates, catalog, "h3")
            _per_class_section(parts, directory, sources["archive_per_class"], "_arquivo", charts, catalog, "h3")
            _compare_section(parts, directory, compares, "compare_found_arquivo.png", charts, catalog, "h3")
            parts.append("</div>")
    # Sem medição: gates (ou eval_por_classe) e compares são a fonte principal.
    else:
        _overall_section(parts, directory, sources["overall"], "overall_metrics.png", charts, gates, catalog, "h2")
        _per_class_section(parts, directory, sources["per_class"], "", charts, catalog, "h2")
        _compare_section(parts, directory, compares, "compare_found.png", charts, catalog, "h2")
    # EXTENSAO: seção nova na página: acrescente parts.append(...) aqui (use _figure para um PNG
    # de "charts", _table para tabelas e _esc em todo texto vindo dos arquivos).

    # Curvas de treino: gráfico, tabela de resumo de cada treino e a curva da GPU.

    parts.append("<h2>Curvas de treino</h2>")
    if runs:
        if "training_curves.png" in charts:
            parts.append(_figure(directory, "training_curves.png", "mAP50 de validação por época"))
        else:
            parts.append("<p class='muted'>Nenhum treino com fit/results.csv legível.</p>")
        rows = []
        for run in runs:
            model = catalog.by_id(run["model"])
            parameters = run["parameters"]
            rows.append([_esc(run["id"]), _swatch(model), _cell(run["state"] or "desconhecido"), _cell(run["epochs"]),
                         _cell(run["best_val_map50"]), _cell(run["best_epoch"]),
                         _esc(f"{_fmt(parameters.get('epochs'))} / {_fmt(parameters.get('imgsz'))} / {_fmt(parameters.get('batch'))}"),
                         _cell(len(run["gpu_eval"]))])
        parts.append(_table(["Treino", "Modelo", "Estado", "Épocas com curva", "Melhor mAP50 (val)", "Época",
                             "epochs / imgsz / batch", "Pontos da avaliação GPU"], rows))
        parts.append("<h3>Avaliação no teste durante o treino (GPU)</h3>")
        if "gpu_eval_curves.png" in charts:
            parts.append(_figure(directory, "gpu_eval_curves.png", "mAP50 por época na avaliação em GPU (fotos inteiras)"))
        else:
            parts.append("<p class='muted'>Nenhum treino tem gpu_eval/eval_curve.jsonl com mAP50.</p>")
    else:
        parts.append("<p class='muted'>Nenhum treino piloto informado ou encontrado em runs/.</p>")

    # Notas (cada uma escapada) e fechamento da página.
    parts.append("<h2>Notas</h2><ul class='notes'>" + "".join(f"<li>{_esc(note)}</li>" for note in notes) + "</ul>")
    parts.append("</body></html>")
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Montagem do relatório
# ----------------------------------------------------------------------------

def _per_class_charts(plt, directory: Path, source: dict | None, suffix: str, catalog: ModelCatalog,
                      charts: list[str]) -> None:
    """Grava os dois gráficos por classe (recall e erros) e anota os nomes em ``charts``.

    Parâmetros:
        plt: módulo pyplot.
        directory: pasta do relatório.
        source: fonte por classe ou ``None`` (sem dados não grava nada).
        suffix: ``""`` ou ``"_arquivo"`` (nome dos PNGs).
        catalog: catálogo dos modelos.
        charts: lista dos PNGs gravados (alterada no lugar).

    Retorna:
        None.
    """
    if not (source and source["values"]):
        return
    models = _source_models(source, catalog)
    _chart_per_class_recall(plt, directory / f"per_class_recall{suffix}.png", source, models, catalog)
    charts.append(f"per_class_recall{suffix}.png")
    _chart_per_class_errors(plt, directory / f"per_class_errors{suffix}.png", source, models)
    charts.append(f"per_class_errors{suffix}.png")


def _overall_chart(plt, directory: Path, source: dict | None, name: str, catalog: ModelCatalog,
                   charts: list[str]) -> None:
    """Grava o gráfico de métricas gerais (se houver dados) e anota o nome em ``charts``.

    Parâmetros:
        plt: módulo pyplot.
        directory: pasta do relatório.
        source: fonte geral ou ``None``.
        name: nome do PNG (``overall_metrics.png`` ou ``overall_metrics_arquivo.png``).
        catalog: catálogo dos modelos.
        charts: lista dos PNGs gravados (alterada no lugar).

    Retorna:
        None.
    """
    if source and source["values"]:
        _chart_overall(plt, directory / name, source, _source_models(source, catalog), catalog)
        charts.append(name)


def build_model_report(config: ProjectConfig, output: str | Path | None = None, gates=(), compares=(), evals=(),
                       runs=None, title: str | None = None, measurements=()) -> Path:
    """Grava ``models.json``, os PNGs com dados e ``report.html`` na pasta de saída.

    ``measurements`` (``measurement.json``) é a comparação principal: N séries nas
    mesmas fotos. Com ela presente, gates, comparações e avaliações viram a seção
    "medições anteriores", com gráficos ``*_arquivo.png``. ``runs`` None usa todos
    os ``runs/pilot_train_*`` do projeto que já têm curva; ``output`` None cria
    ``runs/report_<data>_<id>``. Levanta ``ValueError`` quando nenhuma entrada
    traz dados.

    ENTRADA: comando ``report-models`` do cli.py e etapa 6 do ``retrain-cycle`` (cycle.py).

    Parâmetros:
        config: configuração do projeto (``config.root`` e ``config.taxonomy_path``).
        output: pasta nova do relatório (``--output``); não pode existir. ``None`` cria
            ``runs/report_<data>_<id>``.
        gates: caminhos de ``gate.json`` (``--gate``).
        compares: caminhos de ``runs/compare_*.json`` (``--compare``).
        evals: caminhos de ``runs/eval_por_classe_*.json`` (``--eval``).
        runs: pastas ``runs/pilot_train_*`` (``--run``); ``None`` = todas com curva (``discover_runs``).
        title: título da página (padrão ``"Comparativo de modelos de detecção"``).
        measurements: caminhos de ``measurement.json`` (``--measurement``).

    Retorna:
        Path da pasta do relatório (com ``models.json``, os PNGs e ``report.html``).

    Erros:
        ``ValueError`` sem dados ou com medições incomparáveis (``check_measurements_comparable``);
        ``FileExistsError`` se ``output`` já existir; ``OSError``/``ValueError`` de JSON ilegível.
    """
    # 1. Classes ativas da taxonomia (ordem fixa das linhas por classe) e catálogo vazio.
    names = detection_names(load_taxonomy(config.taxonomy_path))
    catalog = ModelCatalog()
    # 2. Leitura das entradas.
    # ATENCAO: não troque a ordem destas leituras: ela define as cores e os rótulos.
    # EXTENSAO: formato de entrada novo: leia aqui com load_<formato>(Path(path), catalog),
    # no ponto da ordem em que os modelos dele devem receber cor.
    # Ordem de registro define cor e posição: medições, gates, comparações, avaliações, treinos.
    measurement_data = [load_measurement(Path(path), catalog) for path in measurements]
    check_measurements_comparable(measurement_data)
    gate_data = [load_gate(Path(path), catalog) for path in gates]
    compare_data = [load_compare(Path(path), catalog) for path in compares]
    eval_data = [load_eval(Path(path), catalog) for path in evals]
    # Treinos: os informados em --run ou todos os runs/pilot_train_* com results.csv.
    run_directories = [Path(path) for path in runs] if runs is not None else discover_runs(Path(config.root))
    run_data, skipped = [], []
    # Treino sem curva de validação nem curva da GPU é ignorado, com um aviso nas notas.
    for directory in run_directories:
        run = load_run(directory, catalog)
        if run["curve"] or run["gpu_eval"]:
            run_data.append(run)
        else:
            skipped.append(f"Treino {run['id']} ignorado: sem fit/results.csv nem gpu_eval/eval_curve.jsonl legíveis.")
    # 3. Sem nenhum dado: erro claro (e nenhuma pasta criada).
    if not (measurement_data or gate_data or compare_data or eval_data or run_data):
        raise ValueError("Nenhum dado para o relatório: informe measurement.json, gate.json, compare_*.json, "
                         "eval_por_classe_*.json ou treinos em runs/pilot_train_* com fit/results.csv.")

    # 4. Pasta de saída. ATENCAO: nunca reaproveita uma pasta existente (exist_ok=False).
    directory = Path(output) if output is not None else make_run_directory(Path(config.root), "report")
    if output is not None:
        directory.mkdir(parents=True, exist_ok=False)  # nunca sobrescreve um relatório anterior
    title = title or "Comparativo de modelos de detecção"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 5. Fontes de cada seção.
    # Com medição, gate/eval vão para o arquivo; sem medição, eles são a fonte principal.
    archive = bool(measurement_data) and bool(gate_data or eval_data or compare_data)
    sources = {"overall": _overall_source(measurement_data, gate_data, eval_data),
               "per_class": _per_class_source(measurement_data, gate_data, eval_data, names),
               "archive_overall": _overall_source([], gate_data, eval_data) if archive else None,
               "archive_per_class": _per_class_source([], gate_data, eval_data, names) if archive else None}
    # 6. Gráficos. Cada função só grava se houver dados; "charts" guarda os PNGs gravados,
    #    e o HTML só embute o que está nessa lista.
    plt = _plotting()
    charts: list[str] = []
    _overall_chart(plt, directory, sources["overall"], "overall_metrics.png", catalog, charts)
    _per_class_charts(plt, directory, sources["per_class"], "", catalog, charts)
    if measurement_data and _chart_strata(plt, directory / "strata_recall.png", measurement_data, catalog):
        charts.append("strata_recall.png")
    _overall_chart(plt, directory, sources["archive_overall"], "overall_metrics_arquivo.png", catalog, charts)
    _per_class_charts(plt, directory, sources["archive_per_class"], "_arquivo", catalog, charts)
    if _chart_curves(plt, directory / "training_curves.png", run_data, catalog, "curve",
                     "mAP50 de validação por época", "mAP50 de validação"):
        charts.append("training_curves.png")
    gpu_split = _gpu_split(run_data)
    if gpu_split and _chart_curves(plt, directory / "gpu_eval_curves.png", run_data, catalog, "gpu_eval",
                                   f"mAP50 no split {gpu_split} durante o treino (avaliação na GPU)",
                                   f"mAP50 no split {gpu_split} (GPU, fotos inteiras)", split=gpu_split):
        charts.append("gpu_eval_curves.png")
    # Gráfico das comparações por split: vai para o "arquivo" quando há medição principal.
    if any(compare["splits"] for compare in compare_data):
        compare_chart = "compare_found_arquivo.png" if measurement_data else "compare_found.png"
        _chart_compare(plt, directory / compare_chart, compare_data, catalog)
        charts.append(compare_chart)

    # EXTENSAO: gráfico novo: chame _chart_<nome>(plt, directory / "<nome>.png", ...) aqui e faça
    # charts.append("<nome>.png"); depois mostre-o em _render_html com _figure.

    # 7. Notas, models.json (NaN → None por _clean) e report.html.
    notes = _notes(measurement_data, gate_data, sources["per_class"], sources["overall"], archive, catalog, skipped)
    # EXTENSAO: novo campo em models.json: acrescente a chave neste dict (e suba SCHEMA_VERSION
    # se mudar o formato de algo existente).
    write_json(directory / "models.json", _clean({
        "schema_version": SCHEMA_VERSION, "created_at": created_at, "title": title,
        "models": [{key: model[key] for key in ("id", "label", "weights", "sha256", "run_id", "regime", "color")}
                   for model in catalog.models],
        "measurements": measurement_data, "gates": gate_data, "compares": compare_data, "evals": eval_data,
        "runs": [{key: run[key] for key in ("id", "directory", "model", "state", "parameters", "checkpoint", "epochs",
                                            "best_val_map50", "best_epoch", "gpu_eval")} for run in run_data],
        "charts": charts, "notes": notes,
    }))
    page = _render_html(directory, title, created_at, catalog, measurement_data, gate_data, compare_data, run_data,
                        sources, charts, notes)
    (directory / "report.html").write_text(page, encoding="utf-8")
    return directory
