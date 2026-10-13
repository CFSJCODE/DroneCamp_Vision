"""Reamostragem por fator de repetição (RFS) para classes raras no treino.

Função no projeto: classes com poucas caixas (ex.: ``fixador_irregular`` com 1
caixa na v7) quase não aparecem nos lotes de treino e o
detector aprende a ignorá-las. A RFS repete, só na lista de treino, as fotos que
contêm essas classes, sem copiar arquivo nem alterar label.

Regra (Gupta et al., LVIS, CVPR 2019):
    f_c = fração das fotos de treino com a classe c
    r_c = max(1, sqrt(t / f_c))          t = limiar (``repeat_factor_threshold``)
    r_i = max(r_c das classes da foto i)  fotos negativas: r_i = 1
A parte fracionária de r_i vira repetição extra com probabilidade igual a ela,
sorteada com semente fixa (mesmo seed, mesma lista).

Exemplo numérico: 200 fotos de treino, t = 0,1. Uma classe presente em 2 fotos
tem f_c = 0,01 e r_c = sqrt(0,1 / 0,01) = 3,16: cada uma dessas fotos entra 3
vezes, e uma quarta vez com 16% de chance. Uma classe em 40 fotos (f_c = 0,2)
tem r_c = max(1, sqrt(0,5)) = 1: não repete.

Limite honesto: repetir a mesma foto não cria variedade nova. Com 1 exemplo o
modelo continua sem aprender a classe; a RFS só evita que exemplos raros sumam
entre os comuns. A solução de fundo continua sendo anotar mais fotos.

Quem usa: ``training.train_pilot`` quando ``pilot.sampling.repeat_factor_threshold``
(``configs/project.yaml``) ou ``train-pilot --repeat-factor-threshold`` é informado.
A chamada concreta está em ``training._apply_repeat_factor_sampling``, que troca
``train:`` do ``dataset_resolved.yaml`` pela lista gerada aqui e grava o resumo em
``runs/<treino>/sampling.json``.

O que o arquivo contém:
- ``image_classes``: conjunto de classes de uma label YOLO.
- ``repeat_factors``: aplica a regra acima (r_c e r_i), sem tocar no disco.
- ``write_repeat_factor_list``: lê as labels, sorteia as repetições e grava o .txt.

Disco: LÊ as labels do split de treino; GRAVA um .txt com um caminho de foto por
linha (``runs/<treino>/train_repeat_factor.txt``). Nunca copia nem altera fotos/labels.

Onde mexer:
- Para mudar quanto as classes raras se repetem: ajuste o limiar t no
  ``project.yaml`` (``pilot.sampling.repeat_factor_threshold``) ou na CLI; não
  precisa mexer no código. t maior = mais classes contam como raras e repetem mais.
- Para limitar o número máximo de cópias de uma foto: veja o EXTENSAO em
  ``repeat_factors``.
- Para outra estratégia (ex.: por quantidade de caixas, não de fotos): troque o
  cálculo de ``frequency`` em ``repeat_factors``.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import random


# ---------------------------------------------------------------------------
# Leitura das classes de cada foto
# ---------------------------------------------------------------------------

def image_classes(label_path: Path) -> set[int]:
    """Classes presentes numa label YOLO (vazia = foto negativa).

    Parâmetros:
        label_path: caminho do .txt YOLO da foto (``class_id cx cy w h`` por linha).

    Retorna:
        Conjunto de class_id (int) presentes; conjunto vazio para foto sem caixa.
        Várias caixas da mesma classe contam uma vez só (a regra é por foto).

    Erros:
        FileNotFoundError se a label não existir (toda foto de treino precisa ter
        um .txt, mesmo vazio); ValueError se o primeiro campo não for inteiro.
    """
    classes = set()
    # "utf-8-sig" ignora o BOM que alguns editores do Windows gravam no início.
    for line in Path(label_path).read_text(encoding="utf-8-sig").splitlines():
        # Só o primeiro campo (class_id) interessa; linhas vazias são ignoradas.
        if line.strip():
            classes.add(int(line.split()[0]))
    return classes


# ---------------------------------------------------------------------------
# Regra RFS (cálculo puro, sem disco)
# ---------------------------------------------------------------------------

def repeat_factors(classes_per_image: list[set[int]], threshold: float) -> tuple[list[float], dict[int, float]]:
    """Fator de repetição de cada foto e de cada classe.

    Aplica a regra LVIS descrita no cabeçalho:
    ``r_c = max(1, sqrt(t / f_c))`` e ``r_i = max(r_c das classes da foto)``.

    Parâmetros:
        classes_per_image: uma entrada por foto de treino, com o conjunto de
            classes daquela foto (saída de ``image_classes``).
        threshold: limiar t, em (0, 1]. É uma FRAÇÃO de fotos: classes presentes
            em menos que t das fotos ganham fator > 1.

    Retorna:
        ``(per_image, by_class)``: ``per_image[i]`` é o fator (float >= 1) da foto i,
        na mesma ordem da entrada; ``by_class`` é ``{class_id: r_c}``.

    Erros:
        ValueError "repeat_factor_threshold deve estar no intervalo (0, 1]." se o
        limiar for inválido; ValueError "Sem fotos de treino para reamostrar." se a
        lista estiver vazia.
    """
    # Validação do limiar: fora de (0, 1] a fórmula não faz sentido (t=0 nunca
    # repete; t>1 repetiria até classes presentes em todas as fotos).
    if not 0 < threshold <= 1:
        raise ValueError("repeat_factor_threshold deve estar no intervalo (0, 1].")
    total = len(classes_per_image)
    # Sem fotos não há fração f_c para calcular (evita divisão por zero).
    if total == 0:
        raise ValueError("Sem fotos de treino para reamostrar.")
    # frequency[c] = número de FOTOS que contêm a classe c (não de caixas).
    frequency = Counter(class_id for classes in classes_per_image for class_id in classes)
    # r_c = max(1, sqrt(t / f_c)), com f_c = count / total.
    # EXTENSAO: para limitar as cópias (ex.: no máximo 4x), troque por
    # min(4.0, max(1.0, ...)) aqui e documente o teto no sampling.json
    # (write_repeat_factor_list).
    by_class = {class_id: max(1.0, math.sqrt(threshold / (count / total))) for class_id, count in frequency.items()}
    # r_i = maior r_c entre as classes da foto; foto negativa (sem classe) = 1.0.
    per_image = [max((by_class[class_id] for class_id in classes), default=1.0) for classes in classes_per_image]
    return per_image, by_class


# ---------------------------------------------------------------------------
# Lista de treino com repetições (grava em disco)
# ---------------------------------------------------------------------------

def write_repeat_factor_list(image_dir: Path, label_dir: Path, output: Path, threshold: float, seed: int = 0) -> dict:
    """Grava a lista de treino com repetições e devolve o resumo para auditoria.

    ``output`` é um .txt com um caminho absoluto de foto por linha; a Ultralytics
    aceita esse formato em ``train:`` e encontra a label trocando ``images`` por ``labels``.

    Parâmetros:
        image_dir: pasta das fotos de treino (ex.: ``data/pilot/<v>/images/train``).
            Todos os arquivos dentro dela (recursivo) são considerados fotos.
        label_dir: pasta das labels correspondentes; a label de
            ``image_dir/a/b.jpg`` é ``label_dir/a/b.txt``.
        output: caminho do .txt a gravar (é sobrescrito se existir; quem chama usa
            uma pasta nova em ``runs/``).
        threshold: limiar t da regra RFS, em (0, 1].
        seed: semente do sorteio da parte fracionária (mesma semente = mesma lista).

    Retorna:
        Resumo para o ``sampling.json``: ``method``, ``threshold``, ``seed``,
        ``images`` (fotos distintas), ``entries`` (linhas da lista, com repetições),
        ``class_repeat_factor`` (``{"id": r_c}``), ``images_by_copies``
        (``{"nº de cópias": nº de fotos}``) e ``list`` (caminho absoluto do .txt).

    Erros:
        Os de ``repeat_factors`` e ``image_classes`` (limiar inválido, pasta vazia,
        label ausente).

    Chamada por: training._apply_repeat_factor_sampling.
    """
    # 1. Fotos em ordem alfabética (ordem estável = sorteio reproduzível).
    # ATENCAO: qualquer arquivo da pasta conta como foto (inclusive um .txt ou
    # Thumbs.db perdido); mantenha image_dir só com imagens.
    images = sorted(path for path in Path(image_dir).rglob("*") if path.is_file())
    # 2. Label de cada foto: mesmo caminho relativo, extensão .txt.
    classes = [image_classes(Path(label_dir) / path.relative_to(image_dir).with_suffix(".txt")) for path in images]
    # 3. Fatores r_i por foto e r_c por classe.
    per_image, by_class = repeat_factors(classes, threshold)
    # 4. Sorteio com gerador próprio (não mexe no random global do processo).
    generator = random.Random(seed)
    lines, repeats = [], Counter()
    for path, factor in zip(images, per_image):
        # Parte inteira sempre; parte fracionária sorteada (arredondamento estocástico).
        # Ex.: factor 2,3 -> 2 cópias sempre e uma 3ª com 30% de chance.
        copies = int(factor) + (1 if generator.random() < factor - int(factor) else 0)
        # repeats conta quantas fotos ficaram com 1, 2, 3... cópias (para auditoria).
        repeats[copies] += 1
        # Caminho absoluto com "/" (as_posix) repetido "copies" vezes.
        lines.extend([path.resolve().as_posix()] * copies)
    # 5. Grava a lista (uma foto por linha, termina com quebra de linha).
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 6. Resumo gravado por quem chama em runs/<treino>/sampling.json.
    # EXTENSAO: novo campo de auditoria (ex.: teto de cópias): acrescente a chave
    # aqui; o sampling.json é lido pela auditoria do treino (training.py).
    return {
        "method": "repeat_factor_sampling_lvis", "threshold": threshold, "seed": seed,
        "images": len(images), "entries": len(lines),
        "class_repeat_factor": {str(key): round(value, 3) for key, value in sorted(by_class.items())},
        "images_by_copies": {str(key): value for key, value in sorted(repeats.items())},
        "list": str(Path(output).resolve()),
    }
