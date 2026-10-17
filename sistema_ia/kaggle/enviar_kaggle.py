"""Envia um pacote ao Kaggle e dispara o notebook de treino na T4, a partir do PC.

Função no projeto: junta num comando os passos manuais de ``docs/treino_kaggle.md``:
monta o pacote (``kaggle-package``), sobe o ``.zip`` como Dataset PRIVADO (cria na
primeira vez, nova versão nas seguintes), espera o Kaggle processar e envia o notebook
``treino_t4.ipynb`` com as opções desta execução, já ligado a uma T4 com internet.

Uso (na pasta ``sistema_ia``, com o cliente ``kaggle`` instalado e o token em
``%USERPROFILE%\\.kaggle\\access_token`` ou na variável ``KAGGLE_API_TOKEN``)::

    # teste: continua o treino do PC na T4 e para depois de ~15 min (mede min/época)
    .\\.venv\\Scripts\\python.exe kaggle\\enviar_kaggle.py --usuario SEU_USUARIO ^
        --run "runs\\pilot_train_..._614d070a" --max-horas 0.25

    # acompanhar e baixar o resultado
    kaggle kernels status SEU_USUARIO/dronecamp-treino-t4
    kaggle kernels output SEU_USUARIO/dronecamp-treino-t4 -p C:\\DroneCamp_kaggle\\resultado

O token nunca é lido nem gravado por este script; o cliente ``kaggle`` o encontra sozinho.
O treino do PC não é tocado: o pacote só lê o run.

ATENCAO: o pacote contém recortes das fotos do cliente; só envie com autorização.
"""

from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SISTEMA_IA = HERE.parent
NOTEBOOK = HERE / "treino_t4.ipynb"


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Roda o cliente ``kaggle`` (``python -m kaggle``) e devolve o resultado com a saída em texto."""
    result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if check and result.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} falhou:\n{result.stdout}\n{result.stderr}")
    return result


def patch_notebook(text: str, settings: dict) -> str:
    """Troca os valores da célula de configuração do notebook (``NOME = valor``).

    Parâmetros:
        text: conteúdo JSON do ``treino_t4.ipynb``.
        settings: ``{"BRANCH": "main", "MAX_HOURS": 0.25, ...}``; valores viram ``repr`` Python.

    Retorna:
        str: o JSON do notebook com as linhas trocadas.

    Erros:
        ValueError: alguma chave não existe na célula de configuração.
    """
    notebook = json.loads(text)
    missing = set(settings)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = []
        for line in cell["source"]:
            match = re.match(r"^([A-Z_]+) = .*?(\s*#.*)?(\n?)$", line)
            if match and match.group(1) in settings:
                comment = match.group(2) or ""
                line = f"{match.group(1)} = {settings[match.group(1)]!r}{comment}{match.group(3)}"
                missing.discard(match.group(1))
            lines.append(line)
        cell["source"] = lines
    if missing:
        raise ValueError(f"Configurações ausentes no notebook: {sorted(missing)}.")
    return json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"


def dataset_metadata(user: str, slug: str) -> dict:
    """``dataset-metadata.json`` de um Dataset privado (privado é o padrão do ``datasets create``)."""
    return {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}


def kernel_metadata(user: str, kernel: str, dataset: str) -> dict:
    """``kernel-metadata.json``: notebook privado, GPU T4, internet ligada e o Dataset como entrada."""
    return {"id": f"{user}/{kernel}", "title": kernel, "code_file": "treino_t4.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{user}/{dataset}"],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}


def build_package(args) -> Path:
    """Chama ``kaggle-package`` no projeto e devolve a pasta do pacote criada."""
    command = [sys.executable, "-m", "dronecamp_ia", "kaggle-package", "--output", str(args.saida)]
    command += ["--resume-run", str(args.run)] if args.run else ["--data", str(args.data)]
    print("Montando o pacote:", " ".join(command))
    result = subprocess.run(command, cwd=SISTEMA_IA, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(result.stdout[-2000:], result.stderr[-2000:], sep="\n")
    match = re.search(r"Execução salva em: (.+)", result.stdout)
    if result.returncode or not match:
        raise SystemExit("kaggle-package não concluiu; veja a mensagem acima.")
    return Path(match.group(1).strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--usuario", required=True, help="Seu nome de usuário no Kaggle.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="runs/pilot_train_* a continuar na GPU.")
    source.add_argument("--data", type=Path, help="dataset.yaml do piloto para um treino novo.")
    source.add_argument("--pacote", type=Path, help="Pasta de um pacote já montado (pula o kaggle-package).")
    parser.add_argument("--saida", type=Path, default=Path(r"C:\DroneCamp_kaggle"), help="Pasta local de trabalho.")
    parser.add_argument("--dataset", default="dronecamp-pacote", help="Nome do Dataset privado no Kaggle.")
    parser.add_argument("--kernel", default="dronecamp-treino-t4", help="Nome do notebook no Kaggle.")
    parser.add_argument("--branch", default="main", help="Branch do repositório clonado pelo notebook.")
    parser.add_argument("--max-horas", type=float, default=11.0, help="Para com segurança após este tempo.")
    parser.add_argument("--extra", default="", help='Opções extras do kaggle-train, ex.: "--batch 8".')
    parser.add_argument("--so-preparar", action="store_true", help="Só prepara as pastas; não envia nada.")
    args = parser.parse_args(argv)
    args.saida = args.saida.expanduser().resolve()
    args.saida.mkdir(parents=True, exist_ok=True)
    user, dataset = args.usuario.lower(), args.dataset.lower()

    # 1. Pacote (só leitura no run/dataset de origem).
    package = args.pacote.resolve() if args.pacote else build_package(args)
    archive = package.with_suffix(".zip")
    if not archive.is_file():
        shutil.make_archive(str(package), "zip", root_dir=package)
    # 2. Pasta de envio do Dataset: só o .zip e o metadata (o Kaggle descompacta o zip).
    upload = args.saida / f"envio_{package.name}"
    upload.mkdir(exist_ok=True)
    shutil.copy2(archive, upload / archive.name)
    (upload / "dataset-metadata.json").write_text(json.dumps(dataset_metadata(user, dataset), indent=1), encoding="utf-8")
    # 3. Pasta do notebook com as opções desta execução.
    kernel_dir = args.saida / "notebook"
    kernel_dir.mkdir(exist_ok=True)
    settings = {"BRANCH": args.branch, "MAX_HOURS": args.max_horas, "PACKAGE_DIR": "/kaggle/input",
                "CONTINUE_FROM": None, "EXTRA_ARGS": args.extra.split()}
    (kernel_dir / "treino_t4.ipynb").write_text(patch_notebook(NOTEBOOK.read_text(encoding="utf-8"), settings),
                                                encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(user, args.kernel, dataset), indent=1), encoding="utf-8")
    print(f"Pacote: {archive} ({archive.stat().st_size / 1e6:.0f} MB)\nEnvio: {upload}\nNotebook: {kernel_dir}")
    if args.so_preparar:
        return 0
    # 4. Dataset privado: cria na primeira vez; depois, nova versão com o pacote novo.
    exists = kaggle("datasets", "status", f"{user}/{dataset}", check=False).returncode == 0
    if exists:
        print("Enviando nova versão do Dataset...")
        kaggle("datasets", "version", "-p", str(upload), "-m", package.name)
    else:
        print("Criando o Dataset privado...")
        kaggle("datasets", "create", "-p", str(upload))
    # 5. Espera o Kaggle processar (o notebook só monta Datasets prontos).
    for _ in range(90):
        status = kaggle("datasets", "status", f"{user}/{dataset}", check=False).stdout.strip().lower()
        if "ready" in status:
            break
        if "error" in status:
            raise SystemExit(f"O Kaggle recusou o Dataset: {status}")
        time.sleep(20)
    else:
        raise SystemExit("O Dataset não ficou pronto em 30 min; rode de novo com --pacote para só reenviar.")
    # 6. Notebook: roda em segundo plano na T4; a saída fica na versão do notebook.
    print("Enviando o notebook para rodar na T4...")
    print(kaggle("kernels", "push", "-p", str(kernel_dir), "--accelerator", "NvidiaTeslaT4").stdout)
    print(f"Acompanhar: kaggle kernels status {user}/{args.kernel}\n"
          f"Resultado:  kaggle kernels output {user}/{args.kernel} -p {args.saida / 'resultado'}\n"
          f"Página:     https://www.kaggle.com/code/{user}/{args.kernel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
