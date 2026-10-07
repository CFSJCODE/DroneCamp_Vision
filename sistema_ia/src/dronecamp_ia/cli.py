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
  ``compare-models`` → ``model_gate.py``; ``tune-pilot-bandit`` → ``hparam_bandit.py``
- Marcação automática: ``suggest --zero-shot`` e ``evaluate-autolabel`` → ``autolabel.py`` (YOLOE)

Quando mexer: para criar um comando novo ou uma opção nova (``--algo``) em um
comando existente. A lógica do comando fica no arquivo indicado acima.
"""

import argparse
import json
from pathlib import Path
import sys

from .config import load_config, load_taxonomy, detection_names
from .dataset import validate_dataset
from .io import write_json


# ---------------------------------------------------------------------------
# Definição dos comandos e de suas opções.
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DroneCamp: detecção assistida de evidências em telhados.")
    parser.add_argument("--config", type=Path, help="Arquivo configs/project.yaml alternativo.")
    commands = parser.add_subparsers(dest="command", required=True)
    # Consulta e validação (não alteram nada).
    commands.add_parser("categories", help="Exibir catálogo e limites das categorias.")
    validate = commands.add_parser("validate-data", help="Revisar dataset sem treino ou downloads.")
    validate.add_argument("--data", type=Path)
    validate.add_argument("--output", type=Path, help="Salvar relatório JSON opcional.")
    # Inferência, treino de produção, avaliação, tuning e exportação.
    inference = commands.add_parser("predict", help="Gerar evidências e candidatos à revisão.")
    inference.add_argument("--source", type=Path, required=True)
    inference.add_argument("--weights", help="Checkpoint local especializado; YOLO26 oficial para demo.")
    inference.add_argument("--demo", action="store_true", help="Autoriza testar classes genéricas; não detecta categorias do projeto.")
    training = commands.add_parser("train", help="Fine-tuning somente com dataset aprovado.")
    training.add_argument("--data", type=Path)
    training.add_argument("--weights")
    evaluation = commands.add_parser("evaluate", help="Avaliar detector especializado e métricas por classe.")
    evaluation.add_argument("--data", type=Path)
    evaluation.add_argument("--weights", required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="val")
    evaluation.add_argument("--onnx-provider", choices=["directml", "cpu"], default="directml",
                            help="Para --weights .onnx: directml usa a GPU; cpu, o ONNX Runtime na CPU.")
    tuning = commands.add_parser("tune", help="Vários treinos para buscar hiperparâmetros.")
    tuning.add_argument("--data", type=Path)
    tuning.add_argument("--weights")
    tuning.add_argument("--iterations", type=int, required=True)
    tuning.add_argument("--epochs", type=int, required=True)
    exporting = commands.add_parser("export", help="Exportar checkpoint para ONNX FP32.")
    exporting.add_argument("--weights", required=True)
    exporting.add_argument("--demo", action="store_true")
    # Revisão humana: extração do laudo, página de revisão, importação e propostas.
    extraction = commands.add_parser("extract-report", help="Extrair imagens do PDF como referência, sem labels.")
    extraction.add_argument("--pdf", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("prepare-review", help="Consolidar propostas e segunda leitura visual, sem aprovar.")
    review.add_argument("--directory", type=Path, required=True)
    review.add_argument("--audits", action="store_true", help="Exigir auditoria visual completa por outro revisor.")
    render = commands.add_parser("render-review", help="Gerar página local para conferir e corrigir caixas.")
    render.add_argument("--registry", type=Path, required=True)
    refresh = commands.add_parser("refresh-page", help="Regravar só o index.html da revisão com o modelo e os treinos atuais.")
    refresh.add_argument("--registry", type=Path, required=True)
    platform = commands.add_parser("platform", help="Servidor local (127.0.0.1): monitorar e iniciar treinos pela página.")
    platform.add_argument("--registry", type=Path, action="append", required=True, help="Revisão aberta na página; repita para alternar entre várias (a 1ª abre em /).")
    platform.add_argument("--port", type=int, default=8765)
    platform.add_argument("--no-browser", action="store_true", help="Não abrir o navegador automaticamente.")
    feedback = commands.add_parser("import-review", help="Importar decisões exportadas por revisor humano.")
    feedback.add_argument("--registry", type=Path, required=True)
    feedback.add_argument("--feedback", type=Path, required=True)
    feedback.add_argument("--output", type=Path, required=True)
    proposals = commands.add_parser("add-ai-proposals", help="Preencher revisão com propostas visuais de IA, sem aprovar.")
    proposals.add_argument("--registry", type=Path, required=True)
    proposals.add_argument("--proposals", type=Path, required=True)
    proposals.add_argument("--output", type=Path, required=True)
    approved = commands.add_parser("build-reviewed-data", help="Construir versão de dados com aprovação humana e proveniência.")
    approved.add_argument("--registry", type=Path, required=True)
    approved.add_argument("--assignments", type=Path, required=True)
    approved.add_argument("--output", type=Path, required=True)
    migration = commands.add_parser("migrate-review", help="Expandir classes preservando caixas e exigindo nova revisão.")
    migration.add_argument("--registry", type=Path, required=True)
    migration.add_argument("--previous-taxonomy", type=Path, required=True)
    migration.add_argument("--output", type=Path, required=True)
    # Ciclo piloto: dataset, treino (as opções abaixo sobrescrevem pilot.training) e sugestões.
    pilot_data = commands.add_parser("build-pilot-data", help="Dataset piloto com fotos aprovadas de uma edificação (não é produção).")
    pilot_data.add_argument("--registry", type=Path, action="append", required=True, help="Registro revisado; repita para várias revisões.")
    pilot_data.add_argument("--output", type=Path, required=True)
    pilot_data.add_argument("--seed", type=int, default=42)
    pilot_data.add_argument("--sequence-block", type=int, default=0,
                            help="Agrupar fotos sequenciais de voo (DJI_0194…) em blocos de N no split (ex.: 10).")
    # Fatiamento em alta resolução (Tiling / Patch Slicing) para pequenos defeitos em fotos de drone.
    tiling = commands.add_parser("tile-dataset", help="Fatiar fotos em alta resolução em patches uniformes (ex.: 1280x1280 com overlap).")
    tiling.add_argument("--input", type=Path, required=True, help="Pasta do dataset original com dataset.yaml.")
    tiling.add_argument("--output", type=Path, required=True, help="Pasta de destino onde o dataset fatiado será criado.")
    tiling.add_argument("--patch-size", type=int, default=1280, help="Tamanho do patch quadrado em pixels (padrão: 1280).")
    tiling.add_argument("--overlap", type=float, default=0.2, help="Fração de sobreposição entre patches vizinhos (padrão: 0.20).")
    tiling.add_argument("--negative-ratio", type=float, default=0.2, help="Proporção máxima de patches sem defeito (padrão: 0.20).")
    tiling.add_argument("--seed", type=int, default=42, help="Semente aleatória para amostragem.")
    pilot_train = commands.add_parser("train-pilot", help="Treino real com o dataset piloto; pesos só sugerem caixas.")
    pilot_train.add_argument("--data", type=Path, help="Dataset piloto com dataset.yaml (obrigatório se não usar --resume).")
    pilot_train.add_argument("--weights")
    pilot_train.add_argument("--epochs", type=int)
    pilot_train.add_argument("--imgsz", type=int)
    pilot_train.add_argument("--batch", type=int)
    pilot_train.add_argument("--repeat-factor-threshold", type=float,
                             help="Repete fotos de classes raras na lista de treino (RFS); ex.: 0.3.")
    pilot_train.add_argument("--tile", action="store_true",
                             help="Treinar em janelas das fotos grandes (pilot.tiling; ex.: 1280 px → imgsz 640).")
    pilot_train.add_argument("--resume", type=Path,
                             help="Retoma uma execução piloto anterior a partir de sua pasta (ex.: runs/pilot_train_...).")
    resume_pilot_cmd = commands.add_parser("resume-pilot", help="Retoma um treino piloto interrompido.")
    resume_pilot_cmd.add_argument("run", type=Path, help="Pasta da execução em runs/ (ex.: runs/pilot_train_...).")
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
    autolabel = commands.add_parser("evaluate-autolabel", help="Medir por classe piloto × busca aberta × as duas juntas.")
    autolabel.add_argument("--data", type=Path, required=True)
    autolabel.add_argument("--weights", required=True)
    autolabel.add_argument("--conf", type=float)
    # Aprendizado com as decisões humanas (nada aqui aprova caixas).
    priority = commands.add_parser("prioritize-review", help="Ordenar a fila de revisão com o bandit LinUCB (RL).")
    priority.add_argument("--registry", type=Path, required=True)
    priority.add_argument("--policy", type=Path, help="policy.json aprendido com learn-review (opcional).")
    priority.add_argument("--pilot-manifest", type=Path, help="pilot.json do dataset de treino (raridade das classes).")
    priority.add_argument("--suggestions", type=Path, help="suggestions.json (padrão: ao lado do registro).")
    priority.add_argument("--alpha", type=float, help="Peso da exploração no LinUCB (padrão 1.0).")
    learn = commands.add_parser("learn-review", help="Atualizar a política LinUCB com as decisões humanas importadas.")
    learn.add_argument("--registry", type=Path, required=True, help="Registro já com o feedback importado.")
    learn.add_argument("--policy", type=Path, required=True)
    learn.add_argument("--pilot-manifest", type=Path)
    learn.add_argument("--suggestions", type=Path, help="suggestions.json que o revisor viu.")
    calibrate = commands.add_parser("fit-calibrator", help="Calibrar a confiança das sugestões com scikit-learn.")
    calibrate.add_argument("--data", type=Path, required=True, help="dataset.yaml de um dataset piloto.")
    calibrate.add_argument("--weights", required=True)
    calibrate.add_argument("--conf", type=float, default=0.01)
    calibrate.add_argument("--include-trained", action="store_true",
                           help="Usar também fotos do treino destes pesos (enviesado; só para testes).")
    gate = commands.add_parser("compare-models", help="Comparar candidato e modelo atual por classe e decidir adoção.")
    gate.add_argument("--data", type=Path, required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--split", action="append", choices=["train", "val", "test"],
                      help="Split avaliado; repita para vários (padrão: test).")
    gate.add_argument("--conf", type=float)
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
    exporting.add_argument("--parity-source", type=Path, help="Pasta de fotos para comparar .pt e .onnx.")
    return parser


# ---------------------------------------------------------------------------
# Execução: lê a configuração e despacha cada comando para seu módulo.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada protegido por __main__ para compatibilidade com Windows."""
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        # Os imports ficam dentro de cada ramo: a Ultralytics só carrega quando é usada.
        if args.command == "categories":
            print(json.dumps(load_taxonomy(config.taxonomy_path), ensure_ascii=False, indent=2))
            return 0
        if args.command == "validate-data":
            report = validate_dataset(args.data or config.dataset_path, detection_names(load_taxonomy(config.taxonomy_path)))
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["valid"] else 2
        if args.command == "predict":
            from .prediction import predict
            output = predict(config, args.source, args.weights, args.demo)
        elif args.command in {"train", "evaluate", "tune"}:
            from .training import train, evaluate, tune
            dataset = args.data or config.dataset_path
            if args.command == "train":
                output = train(config, dataset, args.weights)
            elif args.command == "evaluate":
                output = evaluate(config, dataset, args.weights, args.split, args.onnx_provider)
            else:
                output = tune(config, dataset, args.iterations, args.epochs, args.weights)
        elif args.command == "export":
            from .exporting import export_onnx
            images = None
            if args.parity_source:
                from .dataset import IMAGE_EXTENSIONS
                images = sorted(path for path in args.parity_source.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
            output = export_onnx(config, args.weights, args.demo, images)
        elif args.command == "prepare-review":
            from .review_data import build_review_registry, apply_visual_audits
            build_review_registry(config, args.directory)
            if args.audits:
                apply_visual_audits(config, args.directory)
            output = args.directory
        elif args.command == "render-review":
            from .review_render import render_review_package
            output = render_review_package(config, args.registry)
        elif args.command == "refresh-page":
            from .review_render import refresh_review_page
            output = refresh_review_page(config, args.registry)
        elif args.command == "platform":
            from .platform_server import serve_platform
            serve_platform(config, args.registry, args.port, not args.no_browser)
            return 0
        elif args.command == "import-review":
            from .review_data import import_human_feedback
            import_human_feedback(config, args.registry, args.feedback, args.output)
            output = args.output
        elif args.command == "add-ai-proposals":
            from .review_data import apply_ai_proposals
            apply_ai_proposals(config, args.registry, args.proposals, args.output)
            output = args.output
        elif args.command == "build-reviewed-data":
            from .review_dataset import build_approved_dataset
            report = build_approved_dataset(config, args.registry, args.assignments, args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready_for_training"] else 2
        elif args.command == "build-pilot-data":
            from .pilot import build_pilot_dataset
            report = build_pilot_dataset(config, args.registry, args.output, args.seed,
                                         sequence_block=args.sequence_block)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        elif args.command == "tile-dataset":
            from .tiling import build_tiled_dataset
            report = build_tiled_dataset(args.input, args.output, args.patch_size, args.overlap, args.negative_ratio, args.seed)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        elif args.command == "train-pilot":
            if args.resume:
                from .training import resume_pilot
                output = resume_pilot(config, args.resume)
            else:
                if not args.data:
                    raise ValueError("Informe --data para iniciar um treino novo ou --resume para retomar.")
                from .training import train_pilot
                overrides = {key: value for key in ("epochs", "imgsz", "batch", "repeat_factor_threshold", "tile")
                             if (value := getattr(args, key)) is not None}
                output = train_pilot(config, args.data, args.weights, overrides)
        elif args.command == "resume-pilot":
            from .training import resume_pilot
            output = resume_pilot(config, args.run)
        elif args.command == "suggest":
            from .suggestions import create_review_for_new_images, suggest_for_registry
            if bool(args.registry) == bool(args.source):
                raise ValueError("Use --registry (revisão existente) ou --source com --output e --group (fotos novas).")
            if args.registry:
                output = suggest_for_registry(config, args.registry, args.weights, args.conf, args.calibrator,
                                              args.allow_unproven_calibrator, args.zero_shot, args.onnx_provider)
            else:
                if not args.output or not args.group:
                    raise ValueError("Fotos novas exigem --output e --group.")
                output = create_review_for_new_images(config, args.source, args.output, args.group, args.weights,
                                                      args.conf, args.calibrator, args.allow_unproven_calibrator,
                                                      args.zero_shot, args.onnx_provider)
        elif args.command == "evaluate-autolabel":
            from .autolabel import evaluate_autolabel
            output = evaluate_autolabel(config, args.data, args.weights, args.conf)
        elif args.command == "prioritize-review":
            from .active_learning import prioritize_review
            output = prioritize_review(config, args.registry, args.policy, args.pilot_manifest, args.suggestions, args.alpha)
        elif args.command == "learn-review":
            from .active_learning import update_policy
            print(json.dumps(update_policy(config, args.registry, args.policy, args.pilot_manifest, args.suggestions),
                             ensure_ascii=False, indent=2))
            return 0
        elif args.command == "fit-calibrator":
            from .calibration import fit_calibrator
            output = fit_calibrator(config, args.data, args.weights, args.conf, args.include_trained)
            print((output / "calibrator.json").read_text(encoding="utf-8")[:2000])
        elif args.command == "compare-models":
            from .model_gate import compare_models
            output = compare_models(config, args.data, args.baseline, args.candidate, tuple(args.split or ["test"]), args.conf)
            gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
            print(json.dumps({key: gate[key] for key in ("adopt", "reasons", "images_evaluated")}, ensure_ascii=False, indent=2))
        elif args.command == "tune-pilot-bandit":
            from .hparam_bandit import load_space, successive_halving
            fixed = {key: value for key in ("imgsz", "batch") if (value := getattr(args, key)) is not None}
            output = successive_halving(config, args.data, args.weights, args.arms, args.min_epochs, args.eta,
                                        args.rounds, space=load_space(args.space), fixed=fixed)
        elif args.command == "migrate-review":
            from .review_data import migrate_review_taxonomy
            migrate_review_taxonomy(config, args.registry, args.previous_taxonomy, args.output)
            output = args.output
        else:
            from .report_images import extract_report_images
            manifest = extract_report_images(args.pdf, args.output)
            print(json.dumps({"count": manifest["count"], "unique_binary_images": manifest["unique_binary_images"], "output": str(args.output.resolve())}, ensure_ascii=False))
            return 0
        print(f"Execução salva em: {output}")
        return 0
    # Erros previstos (dados inválidos, arquivo ausente) viram mensagem curta e código 2.
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Etapa não concluída: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
