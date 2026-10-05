"""Reamostragem por fator de repetição (RFS) para classes raras no treino.

Função no projeto: classes com poucas caixas (ex.: ``fixador_telha_frouxo`` com 1
caixa, ``rufo_deslocado`` com 1) quase não aparecem nos lotes de treino e o
detector aprende a ignorá-las. A RFS repete, só na lista de treino, as fotos que
contêm essas classes, sem copiar arquivo nem alterar label.

Regra (Gupta et al., LVIS, CVPR 2019):
    f_c = fração das fotos de treino com a classe c
    r_c = max(1, sqrt(t / f_c))          t = limiar (``repeat_factor_threshold``)
    r_i = max(r_c das classes da foto i)  fotos negativas: r_i = 1
A parte fracionária de r_i vira repetição extra com probabilidade igual a ela,
sorteada com semente fixa (mesmo seed, mesma lista).

Limite honesto: repetir a mesma foto não cria variedade nova. Com 1 exemplo o
modelo continua sem aprender a classe; a RFS só evita que exemplos raros sumam
entre os comuns. A solução de fundo continua sendo anotar mais fotos.

Quem usa: ``training.train_pilot`` quando ``pilot.sampling.repeat_factor_threshold``
(``configs/project.yaml``) ou ``train-pilot --repeat-factor-threshold`` é informado.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random


def image_classes(label_path: Path) -> set[int]:
    """Classes presentes numa label YOLO (vazia = foto negativa)."""
    classes = set()
    for line in Path(label_path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            classes.add(int(line.split()[0]))
    return classes


def repeat_factors(classes_per_image: list[set[int]], threshold: float) -> tuple[list[float], dict[int, float]]:
    """Fator de repetição de cada foto e de cada classe."""
    if not 0 < threshold <= 1:
        raise ValueError("repeat_factor_threshold deve estar no intervalo (0, 1].")
    total = len(classes_per_image)
    if total == 0:
        raise ValueError("Sem fotos de treino para reamostrar.")
    frequency = Counter(class_id for classes in classes_per_image for class_id in classes)
    by_class = {class_id: max(1.0, math.sqrt(threshold / (count / total))) for class_id, count in frequency.items()}
    per_image = [max((by_class[class_id] for class_id in classes), default=1.0) for classes in classes_per_image]
    return per_image, by_class


def write_repeat_factor_list(image_dir: Path, label_dir: Path, output: Path, threshold: float, seed: int = 0) -> dict:
    """Grava a lista de treino com repetições e devolve o resumo para auditoria.

    ``output`` é um .txt com um caminho absoluto de foto por linha; a Ultralytics
    aceita esse formato em ``train:`` e encontra a label trocando ``images`` por ``labels``.
    """
    images = sorted(path for path in Path(image_dir).rglob("*") if path.is_file())
    classes = [image_classes(Path(label_dir) / path.relative_to(image_dir).with_suffix(".txt")) for path in images]
    per_image, by_class = repeat_factors(classes, threshold)
    generator = random.Random(seed)
    lines, repeats = [], Counter()
    for path, factor in zip(images, per_image):
        # Parte inteira sempre; parte fracionária sorteada (arredondamento estocástico).
        copies = int(factor) + (1 if generator.random() < factor - int(factor) else 0)
        repeats[copies] += 1
        lines.extend([path.resolve().as_posix()] * copies)
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "method": "repeat_factor_sampling_lvis", "threshold": threshold, "seed": seed,
        "images": len(images), "entries": len(lines),
        "class_repeat_factor": {str(key): round(value, 3) for key, value in sorted(by_class.items())},
        "images_by_copies": {str(key): value for key, value in sorted(repeats.items())},
        "list": str(Path(output).resolve()),
    }
