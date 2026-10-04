"""Permite rodar ``python -m dronecamp_ia <comando>``; a lógica está em ``cli.py``."""

from .cli import main

# A guarda __main__ é necessária no Windows (a Ultralytics cria subprocessos).
if __name__ == "__main__":
    raise SystemExit(main())
