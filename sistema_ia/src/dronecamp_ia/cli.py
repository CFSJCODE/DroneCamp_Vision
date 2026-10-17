"""Linha de comando: ``python -m dronecamp_ia <comando>``.

Função no projeto: é a porta de entrada. Cada comando chama uma função de outro
arquivo; este arquivo só lê os argumentos e mostra o resultado.

Comandos e onde está o código de cada um:
- ``train-pilot`` → ``training.train_pilot``  |  ``build-pilot-data`` → ``pilot.build_pilot_dataset``
- ``train`` / ``evaluate`` / ``tune`` → ``training.py``  |  ``validate-data`` → ``dataset.py``
- ``suggest`` → ``suggestions.py``  |  ``predict`` → ``prediction.py``  |  ``export`` → ``exporting.py``
- ``import-review`` / ``add-ai-proposals`` / ``migrate-review`` / ``prepare-review`` → ``review_data.py``
- ``render-review`` → ``review_render.py``  |  ``build-reviewed-data`` → ``review_dataset.py``
- ``extract-report`` → ``report_images.py``  |  ``categories`` → mostra a taxonomia
- ``refresh-page`` → ``review_render.refresh_review_page``  |  ``platform`` → ``platform_server.py``
- Aprendizado a partir das revisões: ``prioritize-review`` / ``learn-review`` → ``active_learning.py``
  (bandit LinUCB); ``fit-calibrator`` → ``calibration.py`` (scikit-learn);
  ``compare-models`` / ``measure-models`` → ``model_gate.py``; ``tune-pilot-bandit`` → ``hparam_bandit.py``;
  ``report-models`` → ``model_report.py``; ``train-pilot --gpu-eval`` / ``gpu-eval-worker`` → ``gpu_eval.py``;
  ``retrain-cycle`` → ``cycle.py``
- Marcação automática: ``suggest --zero-shot`` e ``evaluate-autolabel`` → ``autolabel.py`` (YOLOE)
- Família/versão do YOLO (26 → 27): ``model-info`` → ``model_family.py``
- VLM gratuito (OpenRouter): ``describe-occurrences`` / ``report-support`` / ``list-free-models`` → ``vlm_assist.py``
- Plataforma comercial: ``import-media`` → ``ingest.py`` (fotos e vídeos enviados viram revisão nova)
- Fatiamento de dataset: ``tile-dataset`` → ``tiling.build_tiled_dataset``
- Retomar treino: ``resume-pilot`` (ou ``train-pilot --resume``) → ``training.resume_pilot``
- Treino em GPU do Kaggle: ``kaggle-package`` (no PC) / ``kaggle-train`` (no Kaggle) → ``kaggle_package.py``
- Aprendizado mútuo YOLO26 + RF-DETR + D-FINE com RL: ``dml-plan`` / ``dml-train`` / ``dml-cache`` /
  ``dml-evaluate`` → ``mutual_learning.py`` (+ ``dml_controller.py``, ``detector_adapters.py``);
  rede local: ``dml-lan-coordinator`` / ``dml-lan-worker`` → ``dml_lan.py``

Quando mexer: para criar um comando novo ou uma opção nova (``--algo``) em um
comando existente. A lógica do comando fica no arquivo indicado acima.

O que o arquivo contém:
- ``build_parser``: declara TODOS os comandos e opções (argparse).
- ``main``: lê a configuração e despacha o comando para o módulo certo
  (um ramo ``if``/``elif`` por comando).

Quem chama: ``__main__.py`` (``python -m dronecamp_ia``) e a plataforma
(``platform_server.JobRunner`` roda este mesmo CLI num processo filho; as opções
montadas pela página ficam em ``platform_server._job_arguments``).

Disco: este arquivo em si só lê ``configs/project.yaml`` (via ``load_config``) e,
em alguns comandos, grava o JSON de ``--output``; o resto é gravado pelos módulos.

Códigos de saída: 0 = sucesso; 2 = erro previsto (mensagem "Etapa não concluída: ...")
ou validação reprovada (validate-data, build-reviewed-data); 1 = só no gpu-eval-worker.

Onde mexer:
- Opção nova num comando existente: (1) ache o ``add_parser`` do comando em
  ``build_parser`` (cada um tem um comentário "Comando <nome>"); (2) acrescente
  ``<variavel>.add_argument("--minha-opcao", ...)``; (3) no ramo do comando em
  ``main`` (comentário "Ramo <nome>"), passe ``args.minha_opcao`` para a função
  do módulo (o argparse troca "-" por "_"); (4) acrescente o parâmetro na função
  do módulo. Se a plataforma também deve usar a opção, veja ``platform_server._job_arguments``.
- Comando novo: veja o EXTENSAO acima de ``build_parser``.
- Mudar a mensagem de erro ou o código de saída: fim de ``main`` (``except``).
"""

import argparse
import json
from pathlib import Path
import sys

# Imports leves usados por vários comandos. Os módulos pesados (Ultralytics,
# scikit-learn, ONNX) são importados DENTRO de cada ramo de main(), para que
# comandos simples abram rápido e não exijam essas bibliotecas instaladas.
from .config import load_config, load_taxonomy, detection_names
from .dataset import validate_dataset
from .io import write_json


# ---------------------------------------------------------------------------
# Definição dos comandos e de suas opções.
# ---------------------------------------------------------------------------

# EXTENSAO: comando novo = (1) commands.add_parser(...) aqui, com as opções; (2) um ramo
# ``elif args.command == "..."`` em main(), com o import dentro do ramo; (3) uma linha na
# lista de comandos do docstring do topo; (4) um teste em tests/. Guia: docs/guia_manutencao.md.
def build_parser() -> argparse.ArgumentParser:
    """Monta o analisador de argumentos com todos os comandos e suas opções.

    Cada ``commands.add_parser("nome", help=...)`` cria um comando; as linhas
    ``<variavel>.add_argument(...)`` logo abaixo são as opções dele. Os comandos
    estão agrupados por assunto (consulta, inferência/treino, revisão, piloto,
    aprendizado, VLM).

    Parâmetros:
        nenhum.

    Retorna:
        ``argparse.ArgumentParser`` pronto; ``parse_args`` devolve um Namespace com
        ``command`` (nome do comando) e um atributo por opção (``--minha-opcao``
        vira ``args.minha_opcao``; opção não informada vale ``None`` ou o ``default``).

    Chamada por: ``main`` (aqui).
    """
    parser = argparse.ArgumentParser(description="DroneCamp: detecção assistida de evidências em telhados.")
    # Opção global (vem ANTES do nome do comando): outro project.yaml.
    # Ex.: python -m dronecamp_ia --config outro.yaml suggest ...
    parser.add_argument("--config", type=Path, help="Arquivo configs/project.yaml alternativo.")
    # Subcomandos; dest="command" guarda o nome escolhido em args.command (usado em main).
    commands = parser.add_subparsers(dest="command", required=True)
    # Consulta e validação (não alteram nada).
    # Comando categories: imprime a taxonomia (configs/taxonomy.json) em JSON.
    # Ramo em main: "categories". Não grava nada.
    commands.add_parser("categories", help="Exibir catálogo e limites das categorias.")
    # Comando validate-data: valida um dataset YOLO (dataset.validate_dataset).
    # Grava só o JSON de --output, se informado. Código 2 se o dataset for inválido.
    validate = commands.add_parser("validate-data", help="Revisar dataset sem treino ou downloads.")
    validate.add_argument("--data", type=Path)
    validate.add_argument("--output", type=Path, help="Salvar relatório JSON opcional.")
    # EXTENSAO: opção nova do validate-data: acrescente aqui e passe no ramo "validate-data" de main.
    # Inferência, treino de produção, avaliação, tuning e exportação.
    # Comando predict: prediction.predict. Grava runs/predict_* (fotos anotadas,
    # JSONL de detecções, execution.json, summary.json).
    inference = commands.add_parser("predict", help="Gerar evidências e candidatos à revisão.")
    inference.add_argument("--source", type=Path, required=True)
    inference.add_argument("--weights", help="Checkpoint local especializado; YOLO26 oficial para demo.")
    inference.add_argument("--demo", action="store_true", help="Autoriza testar classes genéricas; não detecta categorias do projeto.")
    # Comando train: training.train (fine-tuning de produção). Grava runs/train_*.
    training = commands.add_parser("train", help="Fine-tuning somente com dataset aprovado.")
    training.add_argument("--data", type=Path)
    training.add_argument("--weights")
    # Comando evaluate: training.evaluate. Grava runs/evaluate_<split>_*/evaluation.json.
    evaluation = commands.add_parser("evaluate", help="Avaliar detector especializado e métricas por classe.")
    evaluation.add_argument("--data", type=Path)
    evaluation.add_argument("--weights", required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="val")
    evaluation.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml",
                            help="Para --weights .onnx: directml usa a GPU; cpu, o ONNX Runtime na CPU.")
    # Comando tune: training.tune (busca de hiperparâmetros da Ultralytics). Grava runs/tune_*.
    tuning = commands.add_parser("tune", help="Vários treinos para buscar hiperparâmetros.")
    tuning.add_argument("--data", type=Path)
    tuning.add_argument("--weights")
    tuning.add_argument("--iterations", type=int, required=True)
    tuning.add_argument("--epochs", type=int, required=True)
    # Comando export: exporting.export_onnx (ONNX FP32 + paridade com o .pt). Grava runs/export_*.
    # ATENCAO: a opção --parity-source deste comando é declarada mais abaixo (perto
    # de tune-pilot-bandit), com exporting.add_argument; procure lá antes de duplicar.
    exporting = commands.add_parser("export", help="Exportar checkpoint para ONNX FP32.")
    exporting.add_argument("--weights", required=True)
    exporting.add_argument("--demo", action="store_true")
    # Revisão humana: extração do laudo, página de revisão, importação e propostas.
    # Comando extract-report: report_images.extract_report_images. Grava em --output
    # (pasta nova) as fotos do PDF, manifest.json e review_queue.csv.
    # ATENCAO: não tem ramo elif próprio em main; é o "else" final.
    extraction = commands.add_parser("extract-report", help="Extrair imagens do PDF como referência, sem labels.")
    extraction.add_argument("--pdf", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    # Comando prepare-review: review_data.build_review_registry (+ apply_visual_audits
    # com --audits). Grava registry.json (e registry_second_pass.json) em --directory.
    review = commands.add_parser("prepare-review", help="Consolidar propostas e segunda leitura visual, sem aprovar.")
    review.add_argument("--directory", type=Path, required=True)
    review.add_argument("--audits", action="store_true", help="Exigir auditoria visual completa por outro revisor.")
    # Comando render-review: review_render.render_review_package. Grava ao lado do
    # registro: index.html, browser_data.json, review_report.md, images/ e evidence/.
    render = commands.add_parser("render-review", help="Gerar página local para conferir e corrigir caixas.")
    render.add_argument("--registry", type=Path, required=True)
    # Comando refresh-page: review_render.refresh_review_page. Regrava só index.html
    # e browser_data.json (sem recopiar fotos).
    refresh = commands.add_parser("refresh-page", help="Regravar só o index.html da revisão com o modelo e os treinos atuais.")
    refresh.add_argument("--registry", type=Path, required=True)
    # Comando platform: platform_server.serve_platform (servidor 127.0.0.1 até Ctrl+C).
    # Não grava nada por si; o que a página pede (uploads, laudos, tarefas) é gravado pelo servidor.
    platform = commands.add_parser("platform", help="Servidor local (127.0.0.1): inspeções, importação de mídia, laudos, treinos e monitoramento pela página.")
    platform.add_argument("--registry", type=Path, action="append", help="Abrir só esta revisão (repita para várias; a 1ª abre em /). Sem a opção, todas as inspeções de data/reviews ficam disponíveis.")
    platform.add_argument("--port", type=int, default=8765)
    platform.add_argument("--no-browser", action="store_true", help="Não abrir o navegador automaticamente.")
    # Comando import-review: review_data.import_human_feedback. Grava o registro novo
    # em --output (recusa se existir). É o ÚNICO caminho de uma decisão humana para o registro.
    feedback = commands.add_parser("import-review", help="Importar decisões exportadas por revisor humano.")
    feedback.add_argument("--registry", type=Path, required=True)
    feedback.add_argument("--feedback", type=Path, required=True)
    feedback.add_argument("--output", type=Path, required=True)
    # Comando add-ai-proposals: review_data.apply_ai_proposals. Grava registro novo em
    # --output (caixas de IA como candidatas, nunca aprovadas) + resumo da revisão.
    proposals = commands.add_parser("add-ai-proposals", help="Preencher revisão com propostas visuais de IA, sem aprovar.")
    proposals.add_argument("--registry", type=Path, required=True)
    proposals.add_argument("--proposals", type=Path, required=True)
    proposals.add_argument("--output", type=Path, required=True)
    # Comando build-reviewed-data: review_dataset.build_approved_dataset. Grava em
    # --output (pasta nova) o dataset de produção, dataset.yaml, provenance.json e readiness.json.
    approved = commands.add_parser("build-reviewed-data", help="Construir versão de dados com aprovação humana e proveniência.")
    approved.add_argument("--registry", type=Path, required=True)
    approved.add_argument("--assignments", type=Path, required=True)
    approved.add_argument("--output", type=Path, required=True)
    # Comando migrate-review: review_data.migrate_review_taxonomy. Grava registro novo
    # em --output adaptado à taxonomia atual (classes novas exigem nova revisão).
    migration = commands.add_parser("migrate-review", help="Acrescentar ou fundir classes (mapa da taxonomia nova) preservando caixas e exigindo nova revisão.")
    migration.add_argument("--registry", type=Path, required=True)
    migration.add_argument("--previous-taxonomy", type=Path, required=True)
    migration.add_argument("--output", type=Path, required=True)
    # Ciclo piloto: dataset, treino (as opções abaixo sobrescrevem pilot.training) e sugestões.
    # Comando build-pilot-data: pilot.build_pilot_dataset. Grava em --output (pasta
    # nova, ex.: data/pilot/v8) fotos, labels, groups.csv, dataset.yaml e pilot.json.
    pilot_data = commands.add_parser("build-pilot-data", help="Dataset piloto com fotos aprovadas de uma edificação (não é produção).")
    pilot_data.add_argument("--registry", type=Path, action="append", required=True, help="Registro revisado; repita para várias revisões.")
    pilot_data.add_argument("--output", type=Path, required=True)
    pilot_data.add_argument("--seed", type=int, default=42)
    pilot_data.add_argument("--sequence-block", type=int, default=0,
                            help="Agrupar fotos sequenciais de voo (DJI_0194…) em blocos de N no split (ex.: 10).")
    # Comando remap-pilot-data: pilot.remap_pilot_dataset. Copia um piloto da taxonomia
    # anterior para --output com as classes trocadas pelo mapa da migração atual.
    pilot_remap = commands.add_parser("remap-pilot-data", help="Copiar um piloto da taxonomia anterior com as classes da migração (só fusões).")
    pilot_remap.add_argument("--source", type=Path, required=True, help="Pasta do piloto na taxonomia anterior.")
    pilot_remap.add_argument("--output", type=Path, required=True)
    # Fatiamento em alta resolução (Tiling / Patch Slicing) para pequenos defeitos em fotos de drone.
    # Comando tile-dataset: tiling.build_tiled_dataset. Grava em --output o dataset
    # fatiado (janelas + labels), dataset.yaml e tiling_manifest.json.
    tiling = commands.add_parser("tile-dataset", help="Fatiar fotos em alta resolução em patches uniformes (ex.: 1280x1280 com overlap).")
    tiling.add_argument("--input", type=Path, required=True, help="Pasta do dataset original com dataset.yaml.")
    tiling.add_argument("--output", type=Path, required=True, help="Pasta de destino onde o dataset fatiado será criado.")
    tiling.add_argument("--patch-size", type=int, default=1280, help="Tamanho do patch quadrado em pixels (padrão: 1280).")
    tiling.add_argument("--overlap", type=float, default=0.2, help="Fração de sobreposição entre patches vizinhos (padrão: 0.20).")
    tiling.add_argument("--negative-ratio", type=float, default=0.2, help="Proporção máxima de patches sem defeito (padrão: 0.20).")
    tiling.add_argument("--seed", type=int, default=42, help="Semente aleatória para amostragem.")
    # Comando train-pilot: training.train_pilot (ou training.resume_pilot com --resume).
    # Grava runs/pilot_train_<data>_<id>/ (pesos, results.csv, execution.json,
    # dataset_snapshot.json, auditoria; sampling.json com --repeat-factor-threshold).
    # EXTENSAO: opção que sobrescreve um valor de pilot.training do project.yaml:
    # acrescente o add_argument aqui E o nome (com "_") na tupla de "overrides" do
    # ramo "train-pilot" em main; training.train_pilot recebe o dict overrides.
    pilot_train = commands.add_parser("train-pilot", help="Treino real com o dataset piloto; pesos só sugerem caixas.")
    pilot_train.add_argument("--data", type=Path, help="Dataset piloto com dataset.yaml (obrigatório se não usar --resume).")
    pilot_train.add_argument("--weights")
    pilot_train.add_argument("--epochs", type=int)
    pilot_train.add_argument("--imgsz", type=int)
    pilot_train.add_argument("--batch", type=int)
    pilot_train.add_argument("--repeat-factor-threshold", type=float,
                             help="Repete fotos de classes raras na lista de treino (RFS); ex.: 0.3.")
    # ATENCAO: --tile é store_true, então vale False (não None) quando omitido e entra
    # SEMPRE em overrides; ou seja, pela CLI o treino só usa janelas com --tile, e um
    # eventual "tile: true" em pilot.training do project.yaml seria sobrescrito por
    # False (hoje o YAML não tem essa chave; veja "parameters.pop('tile')" em training.train_pilot).
    pilot_train.add_argument("--tile", action="store_true",
                             help="Treinar em janelas das fotos grandes (pilot.tiling; ex.: 1280 px → imgsz 640).")
    pilot_train.add_argument("--resume", type=Path,
                             help="Retoma uma execução piloto anterior a partir de sua pasta (ex.: runs/pilot_train_...).")
    pilot_train.add_argument("--gpu-eval", action="store_true",
                             help="CPU + GPU: um worker avalia last.pt na GPU (DirectML) a cada N épocas (pilot.gpu_eval).")
    pilot_train.add_argument("--gpu-eval-every", type=int, help="Épocas entre avaliações na GPU (padrão: pilot.gpu_eval.every).")
    pilot_train.add_argument("--gpu-eval-provider", choices=["directml", "cpu"], help="directml (GPU) ou cpu (só testes).")
    pilot_train.add_argument("--gpu-eval-split", choices=["val", "test"], help="Split avaliado na GPU (padrão: val).")
    # Comando gpu-eval-worker: gpu_eval.run_worker. Processo filho lançado pelo
    # train-pilot --gpu-eval; grava as avaliações na pasta da fila (--work).
    worker = commands.add_parser("gpu-eval-worker", help="Processo interno do train-pilot --gpu-eval (não use à mão).")
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--work", type=Path, help="Pasta da fila (padrão: <run>/gpu_eval ou a área de gravação).")
    worker.add_argument("--data", type=Path, required=True)
    worker.add_argument("--provider", choices=["directml", "cpu"], default="directml")
    worker.add_argument("--split", choices=["val", "test"], default="val")
    worker.add_argument("--imgsz", type=int, required=True, help="Tamanho de entrada do ONNX (imgsz do treino).")
    worker.add_argument("--poll", type=float, default=5.0, help="Segundos entre consultas à fila.")
    worker.add_argument("--max-idle", type=float, help="Encerrar após N s sem checkpoint novo (worker órfão).")
    worker.add_argument("--watch-stdin", action="store_true", help="Sair quando o processo do treino fechar o stdin.")
    worker.add_argument("--parent-pid", type=int,
                        help="Pid do treino; no Windows o worker sai quando ele termina (substitui o stdin, que travava).")
    # Comando report-models: model_report.build_model_report. Grava runs/report_<data>_<id>/
    # (ou --output) com models.json, gráficos PNG e report.html.
    report = commands.add_parser("report-models", help="Relatório com gráficos comparativos entre modelos (PNG + HTML).")
    report.add_argument("--measurement", type=Path, action="append",
                        help="measurement.json do measure-models (comparação principal); repita para vários.")
    report.add_argument("--gate", type=Path, action="append", help="gate.json do compare-models; repita para vários.")
    report.add_argument("--compare", type=Path, action="append", help="runs/compare_*.json; repita para vários.")
    report.add_argument("--eval", type=Path, action="append", help="runs/eval_por_classe_*.json; repita para vários.")
    report.add_argument("--run", type=Path, action="append", help="runs/pilot_train_* (padrão: todos com results.csv).")
    report.add_argument("--output", type=Path, help="Pasta nova do relatório (padrão: runs/report_<data>_<id>).")
    report.add_argument("--title")
    # Comando retrain-cycle: cycle.run_cycle (dataset → treino → export → gate →
    # medição → relatório). Grava runs/cycle_<data>_<id>/cycle.json e as pastas de cada etapa.
    cycle = commands.add_parser("retrain-cycle", help="Ciclo completo: dataset → treino (CPU+GPU) → export → gate → medição → relatório.")
    cycle.add_argument("--registry", type=Path, action="append", required=True, help="registry.json das revisões; repita.")
    cycle.add_argument("--output", type=Path, required=True, help="Pasta nova do dataset piloto (data/pilot/<versão>).")
    cycle.add_argument("--baseline", action="append", default=[], help="Pesos anteriores a comparar; repita para vários.")
    cycle.add_argument("--weights", help="Peso inicial do treino (padrão: pilot.model).")
    cycle.add_argument("--epochs", type=int)
    cycle.add_argument("--imgsz", type=int)
    cycle.add_argument("--batch", type=int)
    cycle.add_argument("--seed", type=int, default=42)
    cycle.add_argument("--sequence-block", type=int, default=0, help="Fotos consecutivas de voo por cena (ex.: 10).")
    cycle.add_argument("--no-tile", action="store_true", help="Treinar nas fotos inteiras (padrão: --tile).")
    cycle.add_argument("--no-gpu-eval", action="store_true", help="Não avaliar na GPU durante o treino.")
    cycle.add_argument("--gpu-eval-every", type=int)
    cycle.add_argument("--gpu-eval-provider", choices=["directml", "cpu"])
    cycle.add_argument("--onnx-provider", choices=["directml", "cpu"], help="Runtime do gate e da medição (padrão: directml).")
    cycle.add_argument("--split", action="append", help="Splits do gate/medição (padrão: test).")
    cycle.add_argument("--title", help="Título do relatório.")
    # Comando resume-pilot: training.resume_pilot. Continua gravando na mesma pasta runs/pilot_train_*.
    # O argumento "run" é posicional (sem "--").
    resume_pilot_cmd = commands.add_parser("resume-pilot", help="Retoma um treino piloto interrompido.")
    resume_pilot_cmd.add_argument("run", type=Path, help="Pasta da execução em runs/ (ex.: runs/pilot_train_...).")
    # Comando kaggle-package: kaggle_package.build_package. Roda no PC; grava a pasta do pacote e o .zip.
    kaggle_pkg = commands.add_parser("kaggle-package", help="Pacote do dataset em janelas para treinar numa GPU do Kaggle.")
    kaggle_source = kaggle_pkg.add_mutually_exclusive_group(required=True)
    kaggle_source.add_argument("--data", type=Path, help="dataset.yaml do piloto (build-pilot-data) para um treino novo.")
    kaggle_source.add_argument("--resume-run", type=Path, help="runs/pilot_train_* a continuar a partir do last.pt.")
    kaggle_pkg.add_argument("--output", type=Path, required=True, help="Pasta onde o pacote e o .zip são criados.")
    kaggle_pkg.add_argument("--no-tile", action="store_true", help="Treino novo com fotos inteiras (padrão: janelas).")
    kaggle_pkg.add_argument("--no-zip", action="store_true", help="Não gravar o .zip.")
    # Comando kaggle-train: kaggle_package.train_from_package. Roda no Kaggle (ou em qualquer CUDA).
    kaggle_train = commands.add_parser("kaggle-train", help="Treino piloto a partir de um pacote do kaggle-package.")
    kaggle_train.add_argument("--package", type=Path, required=True, help="Pasta do pacote (ou /kaggle/input/<dataset>).")
    kaggle_train.add_argument("--device", default="0", help="0 = primeira GPU; cpu para testar.")
    kaggle_train.add_argument("--fresh", action="store_true", help="Ignora o checkpoint do pacote e treina do zero.")
    kaggle_train.add_argument("--checkpoint", type=Path, help="resume.pt/last.pt de uma sessão anterior para continuar.")
    kaggle_train.add_argument("--weights", help="Peso inicial do treino novo (padrão: o do pacote).")
    kaggle_train.add_argument("--epochs", type=int)
    kaggle_train.add_argument("--batch", type=int)
    kaggle_train.add_argument("--workers", type=int)
    kaggle_train.add_argument("--no-amp", action="store_true", help="Desliga a precisão mista na GPU.")
    kaggle_train.add_argument("--max-hours", type=float, help="Para com segurança após esta duração (sessão do Kaggle: 12 h).")
    # Comando suggest: suggestions.suggest_for_registry (--registry) ou
    # suggestions.create_review_for_new_images (--source/--output/--group).
    # Grava suggestions.json ao lado do registro e regenera a página da revisão;
    # com --source cria a pasta --output (originals/ + registry.json) antes.
    suggest = commands.add_parser("suggest", help="Sugerir caixas na página de revisão com um detector treinado.")
    suggest.add_argument("--weights", required=True)
    suggest.add_argument("--registry", type=Path, help="Revisão existente que receberá sugestões.")
    suggest.add_argument("--source", type=Path, help="Pasta ou foto nova para criar uma revisão.")
    suggest.add_argument("--output", type=Path, help="Pasta nova da revisão das fotos novas.")
    suggest.add_argument("--group", help="Edificação/campanha das fotos novas.")
    suggest.add_argument("--conf", type=float)
    suggest.add_argument("--calibrator", type=Path, help="Pasta de fit-calibrator: ordena por aceitação estimada.")
    suggest.add_argument("--allow-unproven-calibrator", action="store_true",
                         help="Usar calibrador sem ganho medido na validação cruzada (só para testes).")
    suggest.add_argument("--zero-shot", action="store_true",
                         help="Somar a busca aberta (YOLOE + configs/zero_shot_prompts.json) às sugestões do piloto.")
    suggest.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml",
                         help="Para --weights .onnx (comando export): directml usa a GPU; cpu, o ONNX Runtime na CPU.")
    suggest.add_argument("--keep-decided", type=Path, action="append",
                         help="Arquivo de revisão salvo pela página: fotos com decisão mantêm as sugestões atuais (repita).")
    suggest.add_argument("--imgsz", type=int,
                         help="Tamanho de inferência; para pesos piloto fora de runs/ (sem summary.json), use o do treino (ex.: 640).")
    # EXTENSAO: opção nova do suggest: acrescente aqui e passe-a nas DUAS chamadas do
    # ramo "suggest" em main (suggest_for_registry e create_review_for_new_images),
    # que recebem os argumentos por posição.
    # Comando import-media: ingest.ingest_media. Grava em --output (pasta nova) a
    # revisão: midia_enviada/, originals/, quadros dos vídeos, registry.json e a página.
    ingest = commands.add_parser("import-media", help="Importar fotos e vídeos enviados pela plataforma numa revisão nova (sem modelo).")
    ingest.add_argument("--source", type=Path, action="append", required=True, help="Foto, vídeo ou pasta; repita para vários.")
    ingest.add_argument("--output", type=Path, required=True, help="Pasta nova da revisão (ex.: data/reviews/<versão>).")
    ingest.add_argument("--group", required=True, help="Edificação/campanha das fotos (ex.: galpao_b_2026_10).")
    ingest.add_argument("--every-seconds", type=float, default=2.0, help="Segundos entre quadros extraídos de vídeos (padrão 2).")
    ingest.add_argument("--max-frames", type=int, default=200, help="Máximo de quadros por vídeo; o intervalo cresce se preciso (padrão 200).")
    ingest.add_argument("--notes", default="", help="Observações da campanha gravadas no registro.")
    # Comando evaluate-autolabel: autolabel.evaluate_autolabel. Grava
    # runs/autolabel_eval_<data>_<id>/report.json (piloto × busca aberta × as duas).
    autolabel = commands.add_parser("evaluate-autolabel", help="Medir por classe piloto × busca aberta × as duas juntas.")
    autolabel.add_argument("--data", type=Path, required=True)
    autolabel.add_argument("--weights", required=True)
    autolabel.add_argument("--conf", type=float)
    # Aprendizado com as decisões humanas (nada aqui aprova caixas).
    # Comando prioritize-review: active_learning.prioritize_review. Grava
    # review_priority.json e review_priority.csv ao lado do registro (ordem da fila).
    priority = commands.add_parser("prioritize-review", help="Ordenar a fila de revisão com o bandit LinUCB (RL).")
    priority.add_argument("--registry", type=Path, required=True)
    priority.add_argument("--policy", type=Path, help="policy.json aprendido com learn-review (opcional).")
    priority.add_argument("--pilot-manifest", type=Path, help="pilot.json do dataset de treino (raridade das classes).")
    priority.add_argument("--suggestions", type=Path, help="suggestions.json (padrão: ao lado do registro).")
    priority.add_argument("--alpha", type=float, help="Peso da exploração no LinUCB (padrão 1.0).")
    # Comando learn-review: active_learning.update_policy. Regrava o --policy
    # (policy.json) e imprime o resumo. Não tem --alpha (a política mantém o salvo).
    learn = commands.add_parser("learn-review", help="Atualizar a política LinUCB com as decisões humanas importadas.")
    learn.add_argument("--registry", type=Path, required=True, help="Registro já com o feedback importado.")
    learn.add_argument("--policy", type=Path, required=True)
    learn.add_argument("--pilot-manifest", type=Path)
    learn.add_argument("--suggestions", type=Path, help="suggestions.json que o revisor viu.")
    # Comando fit-calibrator: calibration.fit_calibrator. Grava
    # runs/calibrator_<data>_<id>/ (calibrator.joblib + calibrator.json).
    calibrate = commands.add_parser("fit-calibrator", help="Calibrar a confiança das sugestões com scikit-learn.")
    calibrate.add_argument("--data", type=Path, required=True, help="dataset.yaml de um dataset piloto.")
    calibrate.add_argument("--weights", required=True)
    calibrate.add_argument("--conf", type=float, default=0.01)
    calibrate.add_argument("--include-trained", action="store_true",
                           help="Usar também fotos do treino destes pesos (enviesado; só para testes).")
    calibrate.add_argument("--algorithm", choices=["logistic", "lightgbm"], default="logistic",
                           help="logistic (padrão) ou lightgbm (gradient boosting; precisa do pacote lightgbm).")
    # Comando compare-models: model_gate.compare_models. Grava runs/gate_<data>_<id>/gate.json
    # (métricas por classe dos dois modelos e a decisão de adoção).
    # EXTENSAO: para expor os limites da regra de adoção (min_map_gain etc.), crie as
    # opções aqui e passe-as como palavras-chave em compare_models(..., min_map_gain=...).
    gate = commands.add_parser("compare-models", help="Comparar candidato e modelo atual por classe e decidir adoção.")
    gate.add_argument("--data", type=Path, required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--split", action="append", choices=["train", "val", "test"],
                      help="Split avaliado; repita para vários (padrão: test).")
    gate.add_argument("--conf", type=float)
    gate.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml",
                      help="Com pesos .onnx: GPU (directml) ou CPU.")
    # Comando measure-models: model_gate.measure_models. Grava
    # runs/measure_<data>_<id>/measurement.json (ou --output); entrada do report-models.
    measure = commands.add_parser("measure-models", help="Medir N modelos nas mesmas fotos (entrada do report-models).")
    measure.add_argument("--data", type=Path, required=True)
    measure.add_argument("--weights", action="append", required=True, help=".pt ou .onnx; repita para vários modelos.")
    measure.add_argument("--split", action="append", choices=["train", "val", "test"])
    measure.add_argument("--conf", type=float)
    measure.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml")
    measure.add_argument("--regime", action="append", choices=["janelas", "inteira"],
                         help="Regime de inferência por série (padrão: os dois quando pilot.tiling está ligado).")
    measure.add_argument("--output", type=Path, help="Pasta nova (padrão: runs/measure_<data>_<id>).")
    # Comando tune-pilot-bandit: hparam_bandit.successive_halving. Grava
    # runs/bandit_<data>_<id>/bandit.json (e uma pasta de treino por braço/rodada).
    bandit = commands.add_parser("tune-pilot-bandit", help="Hiperparâmetros do piloto por successive halving (bandit).")
    bandit.add_argument("--data", type=Path, required=True)
    bandit.add_argument("--weights")
    bandit.add_argument("--arms", type=int, help="Quantos braços sortear do espaço (padrão: todos).")
    bandit.add_argument("--min-epochs", type=int, default=5)
    bandit.add_argument("--eta", type=int, default=3)
    bandit.add_argument("--rounds", type=int)
    bandit.add_argument("--space", type=Path, help="JSON {parâmetro: [valores]} (padrão: pilot.search_space).")
    bandit.add_argument("--imgsz", type=int)
    bandit.add_argument("--batch", type=int)
    # Opção --parity-source do comando export (declarado lá em cima): pasta de fotos
    # usada para comparar as saídas do .pt e do .onnx.
    exporting.add_argument("--parity-source", type=Path, help="Pasta de fotos para comparar .pt e .onnx.")
    # Família/versão do detector (YOLO26 hoje; YOLO27 quando a Ultralytics publicar).
    # Comando model-info: model_family.model_info + format_model_info. Imprime o
    # relatório; grava o JSON só com --output.
    info = commands.add_parser("model-info", help="Família configurada, alvo (YOLO27) e suporte na Ultralytics instalada.")
    info.add_argument("--model", help="Peso a conferir (padrão: model do project.yaml).")
    info.add_argument("--output", type=Path, help="Salvar o relatório JSON.")
    # Apoio de um VLM gratuito (OpenRouter): descrições e texto de apoio, sempre pendentes.
    # Comando describe-occurrences: vlm_assist.describe_occurrences. Grava ai_notes.json
    # ao lado do registro (descrições pendentes de validação humana).
    describe = commands.add_parser("describe-occurrences", help="VLM gratuito descreve os recortes das caixas (ai_notes.json, pendente).")
    describe.add_argument("--registry", type=Path, required=True)
    describe.add_argument("--include-suggestions", action="store_true", help="Descrever também as sugestões pendentes do modelo.")
    describe.add_argument("--limit", type=int, help="Máximo de chamadas nesta execução (limites dos modelos gratuitos).")
    describe.add_argument("--model", help="Slug do OpenRouter, vários separados por vírgula (reserva), ou free-vision = todos os VLMs gratuitos (padrão: configs/vlm_assist.json ou OPENROUTER_MODEL).")
    describe.add_argument("--no-resume", action="store_true", help="Ignorar descrições já gravadas para este registro.")
    # Comando report-support: vlm_assist.draft_report_support. Grava laudo_apoio.json e
    # laudo_apoio.md na pasta da revisão (ou --output); campos técnicos ficam em branco.
    support = commands.add_parser("report-support", help="Rascunho de apoio ao laudo (laudo_apoio.json/.md) com campos técnicos em branco.")
    support.add_argument("--registry", type=Path, required=True)
    support.add_argument("--output", type=Path, help="Pasta de destino (padrão: a pasta da revisão).")
    # Comando list-free-models: vlm_assist.list_free_vision_models (consulta a rede).
    # Imprime a lista; grava o JSON só com --output.
    free_models = commands.add_parser("list-free-models", help="Modelos gratuitos com entrada de imagem no OpenRouter.")
    free_models.add_argument("--output", type=Path, help="Salvar a lista em JSON.")
    # Aprendizado mútuo (DML) entre YOLO26, RF-DETR e D-FINE + RL do peso de consistência
    # (mutual_learning.py, dml_controller.py) e rodadas na rede local (dml_lan.py).
    # Todos leem o --models (padrão configs/dml.json). Doc: docs/aprendizado_mutuo_dml_rl.md.
    # Comando dml-plan: mutual_learning.plan_dml. Mede o tempo por passo de cada modelo
    # nesta máquina e estima as horas; grava o JSON só com --output.
    dml_plan = commands.add_parser("dml-plan", help="DML: dependências e tempo medido por modelo nesta máquina.")
    dml_plan.add_argument("--data", type=Path, required=True)
    dml_plan.add_argument("--models", type=Path, default=Path("configs/dml.json"))
    dml_plan.add_argument("--batches", type=int, default=3, help="Passos medidos por modelo (o 1.º é aquecimento).")
    dml_plan.add_argument("--output", type=Path)
    dml_plan.add_argument("--threads", type=int, help="Threads de CPU do torch (deixe livre parte da CPU para outro treino).")
    # Comando dml-train: mutual_learning.train_mutual. Grava runs/dml_*/ (pesos por modelo,
    # history.jsonl, controller.json, summary.json).
    dml_train = commands.add_parser("dml-train", help="DML: treino mútuo online, destilação ou com cache dos pares.")
    dml_train.add_argument("--data", type=Path, required=True)
    dml_train.add_argument("--models", type=Path, default=Path("configs/dml.json"))
    dml_train.add_argument("--mode", choices=("online", "distill", "cached", "solo"), default="online")
    dml_train.add_argument("--student", help="Modelo que aprende (modos distill, cached e solo).")
    dml_train.add_argument("--peer-cache", type=Path, help="Pasta com <par>.npz (modo cached).")
    dml_train.add_argument("--epochs", type=int)
    dml_train.add_argument("--batch", type=int)
    dml_train.add_argument("--device", help="cpu ou cuda (padrão: train.device do JSON).")
    dml_train.add_argument("--threads", type=int, help="Threads de CPU do torch.")
    dml_train.add_argument("--no-rl", action="store_true", help="Desliga o controlador RL (usa λ fixo).")
    dml_train.add_argument("--max-batches", type=int, help="Só para teste rápido: lotes por época.")
    # Comando dml-cache: mutual_learning.build_peer_cache. Grava <saída>/<modelo>.npz/.json.
    dml_cache = commands.add_parser("dml-cache", help="DML: gravar as previsões de um modelo nas fotos de treino.")
    dml_cache.add_argument("--data", type=Path, required=True)
    dml_cache.add_argument("--models", type=Path, default=Path("configs/dml.json"))
    dml_cache.add_argument("--model", required=True, help="Nome do modelo no JSON (use weights apontando para o treino).")
    dml_cache.add_argument("--weights", help="Pesos a usar no lugar do weights do JSON.")
    dml_cache.add_argument("--split", default="train")
    dml_cache.add_argument("--output", type=Path, required=True)
    # Comando dml-evaluate: mutual_learning.evaluate_models. Grava runs/dml_eval_*/evaluation.json.
    dml_eval = commands.add_parser("dml-evaluate", help="DML: medir cada modelo, a complementaridade e o ensemble por classe.")
    dml_eval.add_argument("--data", type=Path, required=True)
    dml_eval.add_argument("--models", type=Path, default=Path("configs/dml.json"))
    dml_eval.add_argument("--split", default="test")
    # Comando dml-lan-coordinator: dml_lan.serve_coordinator (servidor até o fim das rodadas).
    lan_coord = commands.add_parser("dml-lan-coordinator", help="DML na rede local: coordenador das rodadas.")
    lan_coord.add_argument("--models", type=Path, default=Path("configs/dml.json"))
    lan_coord.add_argument("--host", required=True, help="IP desta máquina na rede local (ex.: 192.168.0.82).")
    lan_coord.add_argument("--port", type=int, default=8790)
    lan_coord.add_argument("--rounds", type=int, default=4, help="Rodadas (a 0 treina cada modelo sozinho).")
    lan_coord.add_argument("--epochs-per-round", type=int, default=5)
    lan_coord.add_argument("--token", help="Segredo compartilhado (padrão: gerado e impresso).")
    lan_coord.add_argument("--resume", type=Path, help="Pasta runs/dml_lan_* para retomar.")
    # Comando dml-lan-worker: dml_lan.run_worker (treina os --model desta máquina por rodada).
    lan_worker = commands.add_parser("dml-lan-worker", help="DML na rede local: trabalhador que treina um ou mais modelos.")
    lan_worker.add_argument("--data", type=Path, required=True)
    lan_worker.add_argument("--models", type=Path, default=Path("configs/dml.json"))
    lan_worker.add_argument("--model", action="append", required=True, help="Modelo desta máquina (repita para vários).")
    lan_worker.add_argument("--coordinator", required=True, help="http://IP:porta do coordenador.")
    lan_worker.add_argument("--token", required=True)
    lan_worker.add_argument("--name", help="Nome desta máquina no estado do coordenador.")
    lan_worker.add_argument("--device", help="cpu ou cuda.")
    lan_worker.add_argument("--threads", type=int)
    lan_worker.add_argument("--max-batches", type=int, help="Só para o teste de fumaça: lotes por época.")
    lan_worker.add_argument("--workdir", type=Path, help="Pasta runs/dml_lan_worker_* anterior (retomar após reiniciar).")
    # EXTENSAO: comando novo: acrescente o add_parser aqui (antes do return) com um
    # comentário "Comando <nome>" no mesmo estilo, e o ramo elif correspondente em main.
    return parser


# ---------------------------------------------------------------------------
# Execução: lê a configuração e despacha cada comando para seu módulo.
# ---------------------------------------------------------------------------

# ENTRADA: todo ``python -m dronecamp_ia <comando>`` (e toda tarefa iniciada pela plataforma,
# que roda este mesmo CLI num processo filho) começa aqui.
def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada protegido por __main__ para compatibilidade com Windows.

    Lê os argumentos, carrega ``configs/project.yaml`` e chama a função do módulo
    do comando escolhido. A maioria dos ramos guarda em ``output`` o caminho do que
    foi gravado e cai no ``print("Execução salva em: ...")`` do fim; ramos que
    imprimem seu próprio resultado terminam com ``return`` antes.

    Parâmetros:
        argv: lista de argumentos (sem o nome do programa); None = ``sys.argv[1:]``.
            Os testes passam a lista diretamente, ex.: ``main(["categories"])``.

    Retorna:
        Código de saída: 0 sucesso; 2 erro previsto ou validação reprovada.
        (``gpu-eval-worker`` encerra o processo com ``os._exit`` e não retorna.)

    Erros:
        ValueError, OSError, ImportError e RuntimeError dos módulos viram a mensagem
        "Etapa não concluída: <motivo>" em stderr e código 2. Outros erros (bugs)
        saem com traceback completo. Opções inválidas: o argparse imprime o uso e
        encerra com SystemExit(2).
    """
    # 1. Lê a linha de comando (o argparse valida opções obrigatórias e escolhas).
    args = build_parser().parse_args(argv)
    # 2. --config alternativo também vale para processos filhos (worker da GPU).
    if args.config:
        import os

        os.environ["DRONECAMP_CONFIG"] = str(Path(args.config).resolve())  # lido pelo worker do gpu-eval
    try:
        # 3. Configuração do projeto (project.yaml validado, caminhos absolutos).
        config = load_config(args.config)
        # 4. Despacho: um ramo por comando.
        # Os imports ficam dentro de cada ramo: a Ultralytics só carrega quando é usada.
        # Ramo categories: imprime a taxonomia em JSON; não grava nada.
        if args.command == "categories":
            print(json.dumps(load_taxonomy(config.taxonomy_path), ensure_ascii=False, indent=2))
            return 0
        # Ramo validate-data: dataset.validate_dataset no --data (ou config.dataset_path);
        # grava --output se pedido; código 2 se o dataset for inválido.
        if args.command == "validate-data":
            report = validate_dataset(args.data or config.dataset_path, detection_names(load_taxonomy(config.taxonomy_path)))
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["valid"] else 2
        # Ramo predict: prediction.predict; grava runs/predict_* e devolve a pasta.
        if args.command == "predict":
            from .prediction import predict
            output = predict(config, args.source, args.weights, args.demo)
        # Ramo train / evaluate / tune: training.py; sem --data usa config.dataset_path.
        # Gravam runs/train_*, runs/evaluate_<split>_* e runs/tune_*.
        elif args.command in {"train", "evaluate", "tune"}:
            from .training import train, evaluate, tune
            dataset = args.data or config.dataset_path
            if args.command == "train":
                output = train(config, dataset, args.weights)
            elif args.command == "evaluate":
                output = evaluate(config, dataset, args.weights, args.split, args.onnx_provider)
            else:
                output = tune(config, dataset, args.iterations, args.epochs, args.weights)
        # Ramo export: exporting.export_onnx; grava runs/export_*. Com --parity-source,
        # junta as fotos da pasta (extensões de dataset.IMAGE_EXTENSIONS) para a paridade.
        elif args.command == "export":
            from .exporting import export_onnx
            images = None
            if args.parity_source:
                from .dataset import IMAGE_EXTENSIONS
                images = sorted(path for path in args.parity_source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
            output = export_onnx(config, args.weights, args.demo, images)
        # Ramo prepare-review: review_data.build_review_registry (+ apply_visual_audits);
        # grava registry.json (e registry_second_pass.json) em --directory.
        elif args.command == "prepare-review":
            from .review_data import build_review_registry, apply_visual_audits
            build_review_registry(config, args.directory)
            if args.audits:
                apply_visual_audits(config, args.directory)
            output = args.directory
        # Ramo render-review: review_render.render_review_package; devolve o index.html gerado.
        elif args.command == "render-review":
            from .review_render import render_review_package
            output = render_review_package(config, args.registry)
        # Ramo refresh-page: review_render.refresh_review_page; regrava só a página.
        elif args.command == "refresh-page":
            from .review_render import refresh_review_page
            output = refresh_review_page(config, args.registry)
        # Ramo platform: platform_server.serve_platform; bloqueia até Ctrl+C.
        elif args.command == "platform":
            from .platform_server import serve_platform
            serve_platform(config, args.registry, args.port, not args.no_browser)
            return 0
        # Ramo import-review: review_data.import_human_feedback; grava o registro em --output.
        elif args.command == "import-review":
            from .review_data import import_human_feedback
            import_human_feedback(config, args.registry, args.feedback, args.output)
            output = args.output
        # Ramo add-ai-proposals: review_data.apply_ai_proposals; grava o registro em --output.
        elif args.command == "add-ai-proposals":
            from .review_data import apply_ai_proposals
            apply_ai_proposals(config, args.registry, args.proposals, args.output)
            output = args.output
        # Ramo build-reviewed-data: review_dataset.build_approved_dataset; imprime a
        # prontidão e sai com 2 se o dataset ainda não estiver pronto para treino.
        elif args.command == "build-reviewed-data":
            from .review_dataset import build_approved_dataset
            report = build_approved_dataset(config, args.registry, args.assignments, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready_for_training"] else 2
        # Ramo build-pilot-data: pilot.build_pilot_dataset; grava o dataset em --output e imprime o resumo.
        elif args.command == "build-pilot-data":
            from .pilot import build_pilot_dataset
            report = build_pilot_dataset(config, args.registry, args.output, args.seed,
                                         sequence_block=args.sequence_block)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        # Ramo remap-pilot-data: pilot.remap_pilot_dataset; valida a cópia antes de devolver.
        elif args.command == "remap-pilot-data":
            from .pilot import remap_pilot_dataset
            print(json.dumps(remap_pilot_dataset(config, args.source, args.output), ensure_ascii=False, indent=2))
            return 0
        # Ramo tile-dataset: tiling.build_tiled_dataset; grava o dataset fatiado em --output.
        elif args.command == "tile-dataset":
            from .tiling import build_tiled_dataset
            report = build_tiled_dataset(args.input, args.output, args.patch_size, args.overlap, args.negative_ratio, args.seed)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        # Ramo train-pilot: com --resume, training.resume_pilot; senão training.train_pilot
        # (exige --data). Grava runs/pilot_train_*.
        elif args.command == "train-pilot":
            if args.resume:
                from .training import resume_pilot
                output = resume_pilot(config, args.resume)
            else:
                if not args.data:
                    raise ValueError("Informe --data para iniciar um treino novo ou --resume para retomar.")
                from .training import train_pilot
                # overrides: só as opções informadas (valor não None) sobrescrevem pilot.training.
                # O ":=" guarda o valor de getattr em "value" e já testa se não é None.
                # EXTENSAO: nova opção que sobrescreve pilot.training: acrescente o nome aqui.
                overrides = {key: value for key in ("epochs", "imgsz", "batch", "repeat_factor_threshold", "tile")
                             if (value := getattr(args, key)) is not None}
                # gpu_eval: None nos campos = usar pilot.gpu_eval do project.yaml.
                gpu_eval = {"enabled": True if args.gpu_eval else None, "every": args.gpu_eval_every,
                            "provider": args.gpu_eval_provider, "split": args.gpu_eval_split}
                output = train_pilot(config, args.data, args.weights, overrides, gpu_eval)
        # Ramo gpu-eval-worker: gpu_eval.run_worker; imprime o resultado em uma linha
        # JSON (lida pelo treino) e encerra o processo na hora.
        elif args.command == "gpu-eval-worker":
            from .gpu_eval import run_worker
            result = run_worker(config, args.run, args.data, args.provider, args.split, args.imgsz, args.work,
                                args.poll, max_idle_seconds=args.max_idle, watch_stdin=args.watch_stdin,
                                parent_pid=args.parent_pid)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            # Saída imediata: a finalização normal do interpretador com ONNX Runtime + Torch
            # carregados pode abortar (SIGABRT) depois do trabalho feito; a curva já está gravada.
            import os

            # ATENCAO: os._exit não executa "finally" nem esvazia buffers; por isso o
            # print acima usa flush=True. Código 1 = o worker falhou.
            os._exit(0 if result.get("status") in ("done", "idle_timeout") else 1)
        # Ramo report-models: model_report.build_model_report; listas vazias () quando a opção não foi usada.
        elif args.command == "report-models":
            from .model_report import build_model_report
            output = build_model_report(config, args.output, args.gate or (), args.compare or (), args.eval or (),
                                        args.run, args.title, measurements=args.measurement or ())
        # Ramo retrain-cycle: cycle.run_cycle; grava runs/cycle_* e as pastas de cada etapa.
        elif args.command == "retrain-cycle":
            from .cycle import run_cycle
            overrides = {key: value for key in ("epochs", "imgsz", "batch") if (value := getattr(args, key)) is not None}
            # No ciclo o fatiamento é o padrão; --no-tile desliga.
            if not args.no_tile:
                overrides["tile"] = True
            # --no-gpu-eval = None (sem worker); senão liga com os valores informados.
            gpu_eval = None if args.no_gpu_eval else {"enabled": True, "every": args.gpu_eval_every,
                                                       "provider": args.gpu_eval_provider, "split": None}
            # EXTENSAO: opção nova do ciclo: passe-a aqui como palavra-chave de run_cycle.
            output = run_cycle(config, args.registry, args.output, baselines=args.baseline, weights=args.weights,
                               overrides=overrides, seed=args.seed, sequence_block=args.sequence_block,
                               gpu_eval=gpu_eval, onnx_provider=args.onnx_provider or "directml",
                               splits=tuple(args.split or ["test"]), title=args.title)
        # Ramo resume-pilot: training.resume_pilot com a pasta posicional "run".
        elif args.command == "resume-pilot":
            from .training import resume_pilot
            output = resume_pilot(config, args.run)
        # Ramo kaggle-package: kaggle_package.build_package (pasta do pacote + .zip).
        elif args.command == "kaggle-package":
            from .kaggle_package import build_package
            output = build_package(config, args.output, data_path=args.data, resume_run=args.resume_run,
                                   tile=not args.no_tile, make_zip=not args.no_zip)
        # Ramo kaggle-train: kaggle_package.train_from_package; grava runs/pilot_train_*.
        elif args.command == "kaggle-train":
            from .kaggle_package import train_from_package
            overrides = {key: value for key in ("epochs", "batch", "workers") if (value := getattr(args, key)) is not None}
            if args.no_amp:
                overrides["amp"] = False
            output = train_from_package(config, args.package, device=args.device, overrides=overrides, fresh=args.fresh,
                                        weights=args.weights, checkpoint=args.checkpoint, max_hours=args.max_hours)
        # Ramo suggest: exatamente um entre --registry (revisão existente) e --source
        # (fotos novas, exige --output e --group). Grava suggestions.json e a página.
        elif args.command == "suggest":
            from .suggestions import create_review_for_new_images, suggest_for_registry
            # bool(a) == bool(b): os dois informados ou nenhum dos dois = erro.
            if bool(args.registry) == bool(args.source):
                raise ValueError("Use --registry (revisão existente) ou --source com --output e --group (fotos novas).")
            if args.registry:
                # ATENCAO: argumentos por POSIÇÃO; a ordem tem de ser a da assinatura
                # de suggest_for_registry (confira lá ao acrescentar um).
                # --keep-decided: fotos com decisão (status != pending) nos arquivos de revisão salvos.
                decided = {entry["image_sha256"] for path in (args.keep_decided or [])
                           for entry in json.loads(Path(path).read_text(encoding="utf-8")).get("images", [])
                           if entry.get("status", "pending") != "pending"}
                output = suggest_for_registry(config, args.registry, args.weights, args.conf, args.calibrator,
                                              args.allow_unproven_calibrator, args.zero_shot, args.onnx_provider, args.imgsz,
                                              decided)
            else:
                if not args.output or not args.group:
                    raise ValueError("Fotos novas exigem --output e --group.")
                output = create_review_for_new_images(config, args.source, args.output, args.group, args.weights,
                                                      args.conf, args.calibrator, args.allow_unproven_calibrator,
                                                      args.zero_shot, args.onnx_provider, args.imgsz)
        # Ramo import-media: ingest.ingest_media; imprime o resumo e devolve a pasta da revisão nova.
        elif args.command == "import-media":
            from .ingest import ingest_media
            report = ingest_media(config, args.source, args.output, args.group, every_seconds=args.every_seconds,
                                  max_frames=args.max_frames, campaign_notes=args.notes)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            output = Path(report["registry"]).parent
        # Ramo evaluate-autolabel: autolabel.evaluate_autolabel; grava runs/autolabel_eval_*.
        elif args.command == "evaluate-autolabel":
            from .autolabel import evaluate_autolabel
            output = evaluate_autolabel(config, args.data, args.weights, args.conf)
        # Ramo prioritize-review: active_learning.prioritize_review; devolve review_priority.json.
        elif args.command == "prioritize-review":
            from .active_learning import prioritize_review
            output = prioritize_review(config, args.registry, args.policy, args.pilot_manifest, args.suggestions, args.alpha)
        # Ramo learn-review: active_learning.update_policy; regrava policy.json e imprime o resumo.
        elif args.command == "learn-review":
            from .active_learning import update_policy
            print(json.dumps(update_policy(config, args.registry, args.policy, args.pilot_manifest, args.suggestions),
                             ensure_ascii=False, indent=2))
            return 0
        # Ramo fit-calibrator: calibration.fit_calibrator; grava runs/calibrator_* e
        # imprime os primeiros 2000 caracteres do calibrator.json.
        elif args.command == "fit-calibrator":
            from .calibration import fit_calibrator
            output = fit_calibrator(config, args.data, args.weights, args.conf, args.include_trained,
                                    algorithm=args.algorithm)
            print((output / "calibrator.json").read_text(encoding="utf-8")[:2000])
        # Ramo compare-models: model_gate.compare_models; grava runs/gate_*/gate.json e
        # imprime só adopt, reasons e images_evaluated.
        elif args.command == "compare-models":
            from .model_gate import compare_models
            output = compare_models(config, args.data, args.baseline, args.candidate, tuple(args.split or ["test"]), args.conf,
                                    onnx_provider=args.onnx_provider)
            gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
            print(json.dumps({key: gate[key] for key in ("adopt", "reasons", "images_evaluated")}, ensure_ascii=False, indent=2))
        # Ramo measure-models: model_gate.measure_models; grava runs/measure_*/measurement.json.
        elif args.command == "measure-models":
            from .model_gate import measure_models
            output = measure_models(config, args.data, args.weights, tuple(args.split or ["test"]), args.conf,
                                    args.onnx_provider, tuple(args.regime) if args.regime else None, args.output)
        # Ramo tune-pilot-bandit: hparam_bandit.successive_halving; grava runs/bandit_*.
        # fixed = valores fixos (não sorteados) de imgsz/batch, só os informados.
        elif args.command == "tune-pilot-bandit":
            from .hparam_bandit import load_space, successive_halving
            fixed = {key: value for key in ("imgsz", "batch") if (value := getattr(args, key)) is not None}
            output = successive_halving(config, args.data, args.weights, args.arms, args.min_epochs, args.eta,
                                        args.rounds, space=load_space(args.space), fixed=fixed)
        # Ramo model-info: model_family.model_info; grava --output se pedido e imprime em texto.
        elif args.command == "model-info":
            from .model_family import format_model_info, model_info
            report = model_info(config, args.model)
            if args.output:
                write_json(args.output, report)
            print(format_model_info(report))
            return 0
        # Ramo describe-occurrences: vlm_assist.describe_occurrences; grava ai_notes.json e
        # imprime só os contadores principais.
        elif args.command == "describe-occurrences":
            from .vlm_assist import describe_occurrences
            output = describe_occurrences(config, args.registry, args.include_suggestions, args.limit, args.model,
                                          resume=not args.no_resume)
            notes = json.loads(output.read_text(encoding="utf-8"))
            print(json.dumps({key: notes[key] for key in ("described", "total_boxes", "calls_this_run", "errors_this_run", "agreement")},
                             ensure_ascii=False, indent=2))
        # Ramo report-support: vlm_assist.draft_report_support; devolve laudo_apoio.json.
        elif args.command == "report-support":
            from .vlm_assist import draft_report_support
            output = draft_report_support(config, args.registry, args.output)
        # Ramo list-free-models: vlm_assist.list_free_vision_models (usa configs/vlm_assist.json).
        elif args.command == "list-free-models":
            from .vlm_assist import list_free_vision_models, load_vlm_settings
            rows = list_free_vision_models(load_vlm_settings(config))
            if args.output:
                write_json(args.output, rows)
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            return 0
        # Ramo migrate-review: review_data.migrate_review_taxonomy; grava o registro em --output.
        elif args.command == "migrate-review":
            from .review_data import migrate_review_taxonomy
            migrate_review_taxonomy(config, args.registry, args.previous_taxonomy, args.output)
            output = args.output
        # Ramos dml-*: aprendizado mútuo (mutual_learning.py) e rede local (dml_lan.py).
        # _dml_settings aplica as opções da CLI por cima do configs/dml.json.
        elif args.command in {"dml-plan", "dml-train", "dml-cache", "dml-evaluate", "dml-lan-coordinator", "dml-lan-worker"}:
            from .mutual_learning import load_dml_config
            models_path = args.models if args.models.is_absolute() else config.root / args.models
            settings = load_dml_config(models_path)
            for key in ("epochs", "batch", "device", "threads"):
                if getattr(args, key, None) is not None:
                    settings["train"][key] = getattr(args, key)
            if getattr(args, "no_rl", False):
                settings["rl"]["enabled"] = False
            if args.command == "dml-plan":
                from .mutual_learning import plan_dml
                report = plan_dml(args.data, settings, args.batches)
                if args.output:
                    write_json(args.output, report)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0
            if args.command == "dml-train":
                from .mutual_learning import train_mutual
                output = train_mutual(config.root, args.data, settings, args.mode, args.student, args.peer_cache,
                                      max_batches=args.max_batches)
            elif args.command == "dml-cache":
                from .detector_adapters import build_adapter
                from .mutual_learning import TileDataset, build_peer_cache
                spec = next((item for item in settings["models"] if item["name"] == args.model), None)
                if spec is None:
                    raise ValueError(f"Modelo {args.model} não está em {models_path}.")
                if args.weights:
                    spec = {**spec, "weights": args.weights}
                dataset = TileDataset(args.data, args.split, int(settings["imgsz"]))
                adapter = build_adapter(spec, dataset.names, int(settings["imgsz"])).to(settings["train"].get("device") or "cpu")
                output = build_peer_cache(adapter, dataset, args.output, int(settings["train"]["batch"]))
            elif args.command == "dml-evaluate":
                from .mutual_learning import evaluate_models
                output = evaluate_models(config.root, args.data, settings, args.split)
            elif args.command == "dml-lan-coordinator":
                from .dml_lan import serve_coordinator
                output = serve_coordinator(config.root, settings, args.host, args.port, args.rounds,
                                           args.epochs_per_round, args.token, args.resume)
            else:
                from .dml_lan import run_worker
                output = run_worker(config.root, args.data, settings, args.model, args.coordinator, args.token,
                                    host_label=args.name, max_batches=args.max_batches, workdir=args.workdir)
        # EXTENSAO: ramo de comando novo: acrescente aqui, ANTES do else, um
        # ``elif args.command == "<nome>":`` com o import do módulo dentro do ramo;
        # defina ``output`` (Path gravado) ou imprima e faça ``return 0``.
        else:
            # Ramo extract-report (único comando restante): report_images.extract_report_images.
            # ATENCAO: este else pega qualquer comando sem ramo próprio. Comando novo sem o seu elif cai
            # aqui e quebra com AttributeError ('Namespace' sem 'pdf'); confira o elif ao criar um comando.
            from .report_images import extract_report_images
            manifest = extract_report_images(args.pdf, args.output)
            print(json.dumps({"count": manifest["count"], "unique_binary_images": manifest["unique_binary_images"], "output": str(args.output.resolve())}, ensure_ascii=False))
            return 0
        # 5. Ramos que definiram "output" terminam aqui.
        print(f"Execução salva em: {output}")
        return 0
    # Erros previstos (dados inválidos, arquivo ausente) viram mensagem curta e código 2.
    # EXTENSAO: para tratar outro tipo de erro como "previsto", acrescente-o nesta tupla.
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Etapa não concluída: {error}", file=sys.stderr)
        return 2


# Execução direta do arquivo (o caminho normal é __main__.py, que chama main()).
if __name__ == "__main__":
    raise SystemExit(main())
