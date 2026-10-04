"""Dataset mínimo de integração: só prova que treino/predição/exportação rodam.

Usa o acervo humano histórico (v3, dez classes, IDs 0–9 iguais aos atuais) e
repete o treino como validação. Métricas daqui não medem nada; não use estes
pesos para sugerir caixas. O dataset piloto (build-pilot-data) é o caminho certo.
"""

import pathlib
import shutil

SRC = pathlib.Path("data/annotation_corpus")
OUT = pathlib.Path("data/datasets/e2e_test_v1")
for split in ("images/train", "labels/train"):
    (OUT / split).mkdir(parents=True, exist_ok=True)
copied = 0
for corpus_dir in SRC.glob("ceasa_v*"):
    # O acervo guarda fotos e labels em subpastas images/ e labels/.
    for f in (corpus_dir / "images").glob("*.jpg"):
        shutil.copy(f, OUT / "images/train" / f.name)
        copied += 1
    for f in (corpus_dir / "labels").glob("*.txt"):
        shutil.copy(f, OUT / "labels/train" / f.name)
if copied == 0:
    raise SystemExit(f"Nenhuma foto encontrada em {SRC}/ceasa_v*/images.")
(OUT / "data.yaml").write_text(
    f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/train\nnc: 13\n"
    "names: [telha_quebrada,telha_ausente,residuos_telha,reparo_telha,rufo_ausente,"
    "rufo_quebrado,residuos_calha,vegetacao_calha,pedaco_telha,rufo_deslocado,"
    "pedaco_telha_sobreposto,fixador_telha_frouxo,reparo_rufo]\n",
    encoding="utf-8",
)
print("OK:", OUT, f"({copied} fotos)")
