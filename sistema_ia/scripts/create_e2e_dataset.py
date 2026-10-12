"""Dataset mínimo de integração: só prova que treino/predição/exportação rodam.

Usa o acervo humano histórico (v3, dez classes, IDs 0–9 iguais aos atuais) e
repete o treino como validação. Métricas daqui não medem nada; não use estes
pesos para sugerir caixas. O dataset piloto (build-pilot-data) é o caminho certo.

Função no projeto:
    Script avulso de "fumaça" (smoke test). Monta um dataset YOLO pequeno, com as fotos
    e labels do acervo histórico, só para conferir de ponta a ponta que o treino, a
    predição e a exportação ONNX não quebram numa máquina nova. Foi usado no roteiro
    manual de integração (ver docs/treino_piloto.md).

O que contém:
    Não tem funções: é um script de cima para baixo, em 4 etapas (copiar pastas,
    copiar fotos e labels, validar que copiou algo, gravar o data.yaml).

Quem chama:
    Ninguém no código. Rode à mão, a partir da pasta ``sistema_ia`` (os caminhos são
    relativos à pasta atual):
        python scripts/create_e2e_dataset.py
    Depois aponte um treino de teste para ``data/datasets/e2e_test_v1/data.yaml``.

O que lê e grava no disco:
    Lê:    data/annotation_corpus/ceasa_v*/images/*.jpg e .../labels/*.txt
    Grava: data/datasets/e2e_test_v1/images/train/, labels/train/ e data.yaml
    ATENCAO: arquivos com o mesmo nome são SOBRESCRITOS (shutil.copy) e a pasta não
    é limpa antes; rodar de novo só acrescenta/atualiza arquivos.

Onde mexer:
    - Para usar outro acervo: mude ``SRC`` (e o padrão ``ceasa_v*`` do laço).
    - Para gravar em outra pasta: mude ``OUT``.
    - Para aceitar fotos .png/.jpeg: acrescente outro ``glob`` no laço das imagens.
    - Para mudar as classes: edite ``nc`` e ``names`` do data.yaml (etapa 4), na MESMA
      ordem dos IDs de configs/taxonomy.json.
"""

import pathlib
import shutil

# ---------------------------------------------------------------------------
# Configuração (caminhos relativos à pasta de onde o script é rodado: sistema_ia)
# ---------------------------------------------------------------------------

# Pasta do acervo histórico: cada subpasta ceasa_v* tem images/ e labels/ no formato YOLO.
# EXTENSAO: outro acervo de origem = troque este caminho (e o padrão do glob abaixo).
SRC = pathlib.Path("data/annotation_corpus")
# Pasta do dataset de teste gerado. Mudar o nome cria outro dataset sem tocar no antigo.
OUT = pathlib.Path("data/datasets/e2e_test_v1")

# ---------------------------------------------------------------------------
# 1. Cria a estrutura YOLO (images/train e labels/train). Não há val/test separados.
# ---------------------------------------------------------------------------
for split in ("images/train", "labels/train"):
    # exist_ok=True: rodar de novo não falha se a pasta já existe.
    (OUT / split).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Copia fotos e labels de todas as versões do acervo (ceasa_v3_..., etc.)
# ---------------------------------------------------------------------------
# Contador de fotos copiadas, usado para detectar acervo vazio (etapa 3).
copied = 0
for corpus_dir in SRC.glob("ceasa_v*"):
    # O acervo guarda fotos e labels em subpastas images/ e labels/.
    # EXTENSAO: outro formato de foto = repita este laço com glob("*.png") etc.
    for f in (corpus_dir / "images").glob("*.jpg"):
        # ATENCAO: o nome do arquivo é a ligação foto <-> label (mesmo nome, extensão .txt);
        # duas versões do acervo com o mesmo nome de arquivo se sobrescrevem aqui.
        shutil.copy(f, OUT / "images/train" / f.name)
        copied += 1
    # Labels YOLO (uma linha por caixa: classe cx cy w h, normalizados 0..1). Não conta no total.
    for f in (corpus_dir / "labels").glob("*.txt"):
        shutil.copy(f, OUT / "labels/train" / f.name)

# ---------------------------------------------------------------------------
# 3. Validação: sem nenhuma foto o treino falharia mais adiante com erro confuso.
# ---------------------------------------------------------------------------
if copied == 0:
    raise SystemExit(f"Nenhuma foto encontrada em {SRC}/ceasa_v*/images.")

# ---------------------------------------------------------------------------
# 4. Grava o data.yaml lido pela Ultralytics.
# ---------------------------------------------------------------------------
# path: caminho absoluto do dataset (com barras "/" mesmo no Windows, por as_posix()).
# train e val apontam para a MESMA pasta: por isso as métricas deste dataset não valem nada.
# nc/names: número e nomes das classes, na ordem dos IDs (posição na lista = ID na label).
# ATENCAO: estes nomes são uma cópia fixa da taxonomia da época; se configs/taxonomy.json
# mudar, este data.yaml não acompanha. As labels do acervo v3 só usam os IDs 0 a 9.
# EXTENSAO: classe nova = acrescente o nome no fim de names e some 1 em nc.
(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/train\nnc: 14\n"
    "names: [telha_quebrada,telha_ausente,residuos_telha,reparo_telha,rufo_ausente,"
    "rufo_quebrado,residuos_calha,vegetacao_calha,pedaco_telha,rufo_deslocado,"
    "pedaco_telha_sobreposto,fixador_telha_frouxo,reparo_rufo,pedaco_calha_sobre_telha]\n",
    encoding="utf-8",
)
# Resumo final: pasta gerada e quantas fotos entraram.
print("OK:", OUT, f"({copied} fotos)")
