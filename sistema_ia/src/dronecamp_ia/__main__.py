"""Permite rodar ``python -m dronecamp_ia <comando>``; a lógica está em ``cli.py``.

Função no projeto: é a porta de entrada do pacote quando ele é executado como
programa. Não contém lógica própria: só repassa para ``cli.main``, que lê os
argumentos (``sys.argv``) e chama o módulo certo.

O que contém:
- nenhuma função pública; apenas a chamada de ``cli.main`` sob a guarda ``__main__``.

Quem chama: o Python, quando se digita ``python -m dronecamp_ia <comando>`` na
pasta ``sistema_ia`` (ou com o pacote instalado). A plataforma
(``platform_server.py``) também inicia tarefas assim, em subprocessos.

O que lê e grava no disco: nada diretamente; tudo depende do comando executado.

Onde mexer:
- Para acrescentar um comando novo da CLI, NÃO mexa aqui: vá em ``cli.py``
  (criar o subparser em ``build_parser`` e o ``elif args.command == ...`` em ``main``).
- Este arquivo só muda se a forma de iniciar o programa mudar (ex.: configurar
  logs ou o método de criação de subprocessos antes de chamar ``main``).
"""

# ENTRADA: todo comando da CLI começa aqui (python -m dronecamp_ia <comando>) e segue para cli.main.
from .cli import main

# A guarda __main__ é necessária no Windows (a Ultralytics cria subprocessos).
# ATENCAO: sem esta guarda, no Windows cada subprocesso da Ultralytics (leitura de dados,
# workers) reimportaria este arquivo e executaria o comando de novo, em cascata.
# O código de saída de main() (0 = sucesso, 2 = dataset não pronto etc.) vira o código
# de saída do processo, que a plataforma e scripts usam para saber se deu certo.
if __name__ == "__main__":
    raise SystemExit(main())
