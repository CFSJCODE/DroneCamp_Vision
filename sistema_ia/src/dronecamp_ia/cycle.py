"""Ciclo de retreino: dataset → treino (CPU+GPU) → exportação → gate → medição → relatório.

Função no projeto: depois que o revisor exporta as decisões e o ``import-review``
as grava nos registros, o próximo modelo exige uma sequência longa de comandos
na ordem certa. Este módulo encadeia essa sequência (comando ``retrain-cycle``)
e grava ``runs/cycle_*/cycle.json`` com o caminho de cada etapa, para que a
comparação com os modelos anteriores saia sempre nas mesmas fotos.

O que faz:
- ``run_cycle``: executa as etapas abaixo e para na primeira que falhar.
  1. ``build_pilot_dataset`` (pilot.py), com ``sequence_block`` para as fotos de voo.
  2. ``train_pilot`` (training.py) com ``tile`` (janelas das fotos grandes) e o
     worker de avaliação na GPU (gpu_eval.py).
  3. ``export_onnx`` (exporting.py) do ``best.pt`` sem paridade (o gate já roda o ONNX).
  4. ``compare_models`` (model_gate.py) candidato × cada modelo anterior.
  5. ``measure_models`` (model_gate.py): todos os modelos nas mesmas fotos.
  6. ``build_model_report`` (model_report.py): gráficos comparativos.

Nada aqui aprova pesos: ``gate.json`` de cada comparação diz se há ganho medido.

Quando mexer: para acrescentar uma etapa ao ciclo ou mudar a ordem.

Conteúdo do arquivo (funções):
- ``_write(cycle, record)``: grava o estado atual em ``runs/cycle_*/cycle.json``.
- ``run_cycle(...)``: o ciclo inteiro (única função pública).

Quem chama:
- ``cli.py``, comando ``python -m dronecamp_ia retrain-cycle`` (ramo ``elif args.command ==
  "retrain-cycle"``). É o cli.py que monta ``overrides`` (epochs/imgsz/batch e ``tile=True``
  salvo ``--no-tile``) e ``gpu_eval`` (``None`` com ``--no-gpu-eval``).
- ``tests/test_cycle.py`` (troca as etapas pesadas por funções falsas).

O que lê e grava no disco:
- Lê: os ``registry.json`` das revisões (via pilot.py), ``<run>/execution.json`` do treino
  e ``<gate_run>/gate.json`` de cada comparação.
- Grava: ``runs/cycle_<data>_<id>/cycle.json`` (reescrito depois de cada etapa, para
  acompanhar o andamento e saber onde parou se falhar). As demais pastas
  (``data/pilot/<versão>``, ``runs/pilot_train_*``, ``runs/export_*``, ``runs/compare_*``,
  ``runs/measure_*``, ``runs/report_*``) são criadas pelas funções chamadas.

Formato de ``cycle.json`` (chaves principais):
    started_at, finished_at   datas ISO 8601 em UTC
    state                     "running" | "completed" | "failed"
    error                     "TipoDoErro: mensagem" (só quando state == "failed")
    registries, baselines, overrides, sequence_block   o que foi pedido
    steps.dataset / steps.training / steps.export / steps.gates / steps.measurement / steps.report
    adopt_any                 True se algum gate recomendou adotar; None sem baseline
    note                      explicação curta de onde vem ``adopt``

Onde mexer:
- Para acrescentar uma etapa nova ao ciclo (ex.: calibrar o candidato depois do export):
  procure ``EXTENSAO: nova etapa`` dentro de ``run_cycle``; grave o resultado em
  ``record["steps"]["<nome>"]`` e chame ``_write`` logo depois. Acrescente também a opção
  correspondente no subparser ``retrain-cycle`` do cli.py se a etapa precisar de parâmetro,
  e atualize ``tests/test_cycle.py``.
- Para passar um parâmetro novo a uma etapa existente: acrescente o argumento na
  assinatura de ``run_cycle`` (com valor padrão), no ramo ``retrain-cycle`` do cli.py e
  na chamada da etapa aqui.
- Para mudar a ordem das etapas: respeite as dependências (o treino usa o dataset; o
  export usa o candidato do treino; o gate e a medição usam o ONNX; o relatório usa os
  gates e a medição).
- Para mudar o que vai no relatório do ciclo: ver a chamada de ``build_model_report``
  (etapa 6) e o próprio model_report.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .config import ProjectConfig
from .io import make_run_directory, write_json


# ---------------------------------------------------------------------------
# Gravação do estado do ciclo (cycle.json)
# ---------------------------------------------------------------------------

def _write(cycle: Path, record: dict) -> None:
    """Grava (sobrescreve) ``cycle.json`` dentro da pasta do ciclo com o estado atual.

    É chamada depois de cada etapa, para que o arquivo mostre até onde o ciclo chegou
    mesmo se o processo for interrompido.

    Parâmetros:
        cycle: pasta ``runs/cycle_<data>_<id>`` criada por ``make_run_directory``.
        record: dicionário com o estado do ciclo (formato no cabeçalho do módulo).

    Retorna:
        None.

    Erros:
        ``ValueError`` de ``write_json`` se ``record`` tiver NaN/infinito (allow_nan=False).

    Chamada por: ``run_cycle``.
    """
    # ATENCAO: sobrescrever aqui é intencional: cycle.json é o arquivo de estado do PRÓPRIO
    # ciclo (pasta nova e exclusiva). As execuções anteriores ficam em outras pastas.
    write_json(cycle / "cycle.json", record)


# ---------------------------------------------------------------------------
# Ciclo completo de retreino (comando retrain-cycle)
# ---------------------------------------------------------------------------

def run_cycle(config: ProjectConfig, registries: list[Path], output: Path, baselines: list[str] = (),
              weights: str | None = None, overrides: dict | None = None, seed: int = 42,
              sequence_block: int = 0, gpu_eval: dict | None = None, onnx_provider: str = "directml",
              splits: tuple[str, ...] = ("test",), title: str | None = None) -> Path:
    """Executa o ciclo completo e devolve ``runs/cycle_*`` com ``cycle.json``.

    ENTRADA: comando ``retrain-cycle`` do cli.py.

    Roda, nesta ordem, dataset piloto → treino → export ONNX → gate contra cada modelo
    anterior → medição de todos os modelos nas mesmas fotos → relatório com gráficos.
    Para na primeira etapa que falhar, grava ``state="failed"`` e o erro em ``cycle.json``
    e repassa a exceção.

    Parâmetros:
        config: configuração do projeto (``configs/project.yaml``); ``config.root`` é a
            pasta ``sistema_ia`` onde fica ``runs/``.
        registries: caminhos dos ``registry.json`` das revisões
            (ex.: ``data/reviews/ceasa_2025/registry.json``) usados para montar o dataset.
        output: pasta NOVA do dataset piloto (ex.: ``data/pilot/v7``); não pode existir.
        baselines: pesos anteriores (``.pt`` ou ``.onnx``) a comparar com o candidato.
            Vazio = sem gate (o candidato não pode ser adotado só com este ciclo).
        weights: peso inicial do treino (padrão do training.py: ``pilot.model``).
        overrides: parâmetros de treino que substituem o project.yaml
            (ex.: ``{"epochs": 80, "imgsz": 1280, "tile": True}``).
        seed: semente da divisão treino/validação/teste do dataset (padrão 42).
        sequence_block: quantas fotos consecutivas de voo formam uma cena (0 = desligado);
            mantém fotos quase iguais no mesmo split.
        gpu_eval: opções do worker de avaliação na GPU durante o treino
            (``{"enabled", "every", "provider", "split"}``) ou ``None`` para desligar.
        onnx_provider: runtime do gate e da medição: ``"directml"`` (GPU) ou ``"cpu"``.
        splits: partes do dataset usadas no gate e na medição (padrão ``("test",)``).
        title: título do relatório (padrão ``"Ciclo <nome da pasta>"``).

    Retorna:
        Path da pasta ``runs/cycle_<data>_<id>`` (contém ``cycle.json``).

    Erros:
        Qualquer exceção de uma etapa (ValueError, OSError, RuntimeError...) é registrada em
        ``cycle.json`` como ``"Tipo: mensagem"`` e relançada sem alteração.

    Chamada por: ``cli.py`` (``retrain-cycle``) e ``tests/test_cycle.py``.
    """
    # Importações dentro da função: as etapas puxam Ultralytics/ONNX Runtime/matplotlib,
    # que são pesados; importar só quando o ciclo roda deixa a CLI rápida para outros comandos.
    # ATENCAO: tests/test_cycle.py troca estas funções nos módulos de origem (mock); se
    # mudar para "import no topo", os testes deixam de interceptar as etapas.
    from .exporting import export_onnx
    from .model_gate import compare_models, measure_models
    from .model_report import build_model_report
    from .pilot import build_pilot_dataset
    from .training import train_pilot

    # 0. Pasta nova do ciclo (runs/cycle_<data UTC>_<8 hex>) e estado inicial "running".
    #    make_run_directory nunca reaproveita uma pasta existente.
    cycle = make_run_directory(config.root, "cycle")
    # Estado inicial: registra o pedido (registros com caminho absoluto, baselines,
    # overrides e sequence_block) antes de qualquer etapa, para auditoria.
    # EXTENSAO: novo parâmetro de entrada a registrar no cycle.json: acrescente a chave aqui.
    record = {"started_at": datetime.now(timezone.utc).isoformat(), "state": "running", "steps": {},
              "registries": [str(Path(path).resolve()) for path in registries], "baselines": list(baselines),
              "overrides": dict(overrides or {}), "sequence_block": sequence_block}
    _write(cycle, record)
    # O try envolve todas as etapas: qualquer falha cai no except, que marca "failed".
    try:
        # 1. Dataset piloto (fotos inteiras; as janelas são geradas no treino com tile).
        # build_pilot_dataset devolve um dict; "dataset" é o caminho do dataset.yaml.
        dataset = build_pilot_dataset(config, list(registries), Path(output), seed, sequence_block=sequence_block)
        record["steps"]["dataset"] = dataset
        _write(cycle, record)
        data_path = Path(dataset["dataset"])
        # 2. Treino na CPU com avaliação paralela na GPU.
        # train_pilot devolve a pasta runs/pilot_train_*; execution.json traz os parâmetros
        # efetivos, o plano de janelas ("tiling") e o resultado do worker da GPU ("gpu_eval").
        run = train_pilot(config, data_path, weights, overrides, gpu_eval)
        execution = json.loads((run / "execution.json").read_text(encoding="utf-8"))
        # Candidato: best.pt (melhor época na validação) ou, se não existir, last.pt
        # (última época), caminhos padrão da Ultralytics em <run>/fit/weights/.
        best = run / "fit" / "weights" / "best.pt"
        candidate = best if best.is_file() else run / "fit" / "weights" / "last.pt"
        record["steps"]["training"] = {"run": str(run), "candidate": str(candidate), "tiling": execution.get("tiling"),
                                       "gpu_eval": execution.get("gpu_eval")}
        _write(cycle, record)
        # 3. ONNX do candidato (o gate e a medição rodam o ONNX na GPU).
        # parity_images=[] pula a conferência .pt × .onnx (o gate já mede o ONNX de verdade).
        export = export_onnx(config, str(candidate), parity_images=[])
        onnx = export / "model.onnx"
        record["steps"]["export"] = {"run": str(export), "onnx": str(onnx)}
        _write(cycle, record)
        # O gate e a medição avaliam o ONNX recém-exportado (o artefato que roda na GPU),
        # para que a decisão valide o mesmo arquivo usado nas sugestões.
        # Se o export não gerou model.onnx, segue com o .pt (o gate aceita os dois formatos).
        candidate = onnx if onnx.is_file() else candidate
        # EXTENSAO: nova etapa entre o export e o gate (ex.: calibrar o candidato): chame a
        # função aqui, grave record["steps"]["<nome>"] = {...} e chame _write(cycle, record).
        # 4. Gate candidato × cada anterior, nas fotos fora do treino de ambos.
        # Um compare_models por baseline; de cada gate.json guardamos só o resumo
        # (adopt, reasons, images_evaluated) e o caminho da pasta, onde está o detalhe.
        gates = []
        for baseline in baselines:
            gate_run = compare_models(config, data_path, baseline, str(candidate), splits, onnx_provider=onnx_provider)
            gate = json.loads((gate_run / "gate.json").read_text(encoding="utf-8"))
            # EXTENSAO: outro campo do gate.json no resumo do ciclo: acrescente a chave aqui.
            gates.append({"baseline": baseline, "run": str(gate_run), "adopt": gate.get("adopt"),
                          "reasons": gate.get("reasons"), "images_evaluated": gate.get("images_evaluated")})
        record["steps"]["gates"] = gates
        _write(cycle, record)
        # 5. Medição de todos os modelos nas mesmas fotos e 6. relatório.
        # Lista de pesos = baselines + candidato (o candidato fica por último); conf=None
        # usa o limiar do project.yaml.
        measurement = measure_models(config, data_path, [*baselines, str(candidate)], splits, None, onnx_provider)
        record["steps"]["measurement"] = {"run": str(measurement)}
        # Relatório: output=None (pasta nova runs/report_*), os gate.json desta rodada,
        # sem compare/eval avulsos, só a pasta deste treino e o measurement.json da etapa 5.
        # EXTENSAO: mais fontes no relatório do ciclo (ex.: outros treinos para a curva):
        # acrescente aqui nos argumentos posicionais de build_model_report (gates, compares,
        # evals, runs) ou em measurements=.
        report = build_model_report(config, None, [Path(item["run"]) / "gate.json" for item in gates], (), (), [run],
                                    title or f"Ciclo {cycle.name}", measurements=[measurement / "measurement.json"])
        record["steps"]["report"] = {"run": str(report), "html": str(report / "report.html")}
        # EXTENSAO: nova etapa no fim do ciclo (depois do relatório): coloque antes desta
        # linha que marca "completed", gravando em record["steps"].
        record["state"] = "completed"
        # ATENCAO: o ciclo nunca aprova pesos. adopt_any só resume o que os gate.json dizem;
        # sem baseline fica None (não há comparação, logo não há recomendação).
        record["adopt_any"] = any(item["adopt"] for item in gates) if gates else None
        record["note"] = ("Sem baseline não há comparação: o candidato só pode ser adotado após um compare-models."
                          if not gates else "adopt vem de cada gate.json (fotos fora do treino de ambos os modelos).")
    except Exception as error:
        # Falha em qualquer etapa: registra o erro no cycle.json (as etapas já concluídas
        # continuam em record["steps"]) e relança para a CLI mostrar a mensagem.
        record["state"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        _write(cycle, record)
        raise
    # Sucesso: data de término e gravação final.
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write(cycle, record)
    return cycle
