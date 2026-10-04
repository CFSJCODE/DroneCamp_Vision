"""Comandos pequenos, para revisar cada etapa antes de consumir treinamento."""

import argparse
import json
from pathlib import Path
import sys

from .config import load_config, load_taxonomy, detection_names
from .dataset import validate_dataset
from .io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DroneCamp: detecção assistida de evidências em telhados.")
    parser.add_argument("--config", type=Path, help="Arquivo configs/project.yaml alternativo.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("categories", help="Exibir catálogo e limites das categorias.")
    validate = commands.add_parser("validate-data", help="Revisar dataset sem treino ou downloads.")
    validate.add_argument("--data", type=Path)
    validate.add_argument("--output", type=Path, help="Salvar relatório JSON opcional.")
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
    tuning = commands.add_parser("tune", help="Vários treinos para buscar hiperparâmetros.")
    tuning.add_argument("--data", type=Path)
    tuning.add_argument("--weights")
    tuning.add_argument("--iterations", type=int, required=True)
    tuning.add_argument("--epochs", type=int, required=True)
    exporting = commands.add_parser("export", help="Exportar checkpoint para ONNX FP32.")
    exporting.add_argument("--weights", required=True)
    exporting.add_argument("--demo", action="store_true")
    extraction = commands.add_parser("extract-report", help="Extrair imagens do PDF como referência, sem labels.")
    extraction.add_argument("--pdf", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("prepare-review", help="Consolidar propostas e segunda leitura visual, sem aprovar.")
    review.add_argument("--directory", type=Path, required=True)
    review.add_argument("--audits", action="store_true", help="Exigir auditoria visual completa por outro revisor.")
    render = commands.add_parser("render-review", help="Gerar página local para conferir e corrigir caixas.")
    render.add_argument("--registry", type=Path, required=True)
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
    pilot_data = commands.add_parser("build-pilot-data", help="Dataset piloto com fotos aprovadas de uma edificação (não é produção).")
    pilot_data.add_argument("--registry", type=Path, action="append", required=True, help="Registro revisado; repita para várias revisões.")
    pilot_data.add_argument("--output", type=Path, required=True)
    pilot_data.add_argument("--seed", type=int, default=42)
    pilot_train = commands.add_parser("train-pilot", help="Treino real com o dataset piloto; pesos só sugerem caixas.")
    pilot_train.add_argument("--data", type=Path, required=True)
    pilot_train.add_argument("--weights")
    pilot_train.add_argument("--epochs", type=int)
    pilot_train.add_argument("--imgsz", type=int)
    pilot_train.add_argument("--batch", type=int)
    suggest = commands.add_parser("suggest", help="Sugerir caixas na página de revisão com um detector treinado.")
    suggest.add_argument("--weights", required=True)
    suggest.add_argument("--registry", type=Path, help="Revisão existente que receberá sugestões.")
    suggest.add_argument("--source", type=Path, help="Pasta ou foto nova para criar uma revisão.")
    suggest.add_argument("--output", type=Path, help="Pasta nova da revisão das fotos novas.")
    suggest.add_argument("--group", help="Edificação/campanha das fotos novas.")
    suggest.add_argument("--conf", type=float)
    exporting.add_argument("--parity-source", type=Path, help="Pasta de fotos para comparar .pt e .onnx.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada protegido por __main__ para compatibilidade com Windows."""
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
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
                output = evaluate(config, dataset, args.weights, args.split)
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
            report = build_pilot_dataset(config, args.registry, args.output, args.seed)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        elif args.command == "train-pilot":
            from .training import train_pilot
            overrides = {key: value for key in ("epochs", "imgsz", "batch") if (value := getattr(args, key)) is not None}
            output = train_pilot(config, args.data, args.weights, overrides)
        elif args.command == "suggest":
            from .suggestions import create_review_for_new_images, suggest_for_registry
            if bool(args.registry) == bool(args.source):
                raise ValueError("Use --registry (revisão existente) ou --source com --output e --group (fotos novas).")
            if args.registry:
                output = suggest_for_registry(config, args.registry, args.weights, args.conf)
            else:
                if not args.output or not args.group:
                    raise ValueError("Fotos novas exigem --output e --group.")
                output = create_review_for_new_images(config, args.source, args.output, args.group, args.weights, args.conf)
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
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Etapa não concluída: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
