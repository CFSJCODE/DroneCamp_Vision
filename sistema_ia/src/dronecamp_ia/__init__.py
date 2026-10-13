"""Pacote dronecamp_ia: núcleo de visão do DroneCamp.

Mapa rápido (o que cada arquivo faz):
- Treino: ``training.py`` (treina e audita), ``pilot.py`` (dataset do piloto),
  ``backend.py`` (carrega o YOLO), ``dataset.py`` (valida o dataset).
- Parâmetros: ``config.py`` lê ``configs/project.yaml`` (épocas, imgsz, batch...).
- Uso do modelo: ``suggestions.py`` (sugestões na revisão), ``prediction.py``
  (inferência com evidências), ``exporting.py`` (ONNX e paridade).
- Revisão humana: ``review_data.py`` (registros e decisões), ``review_render.py``
  (página), ``review_dataset.py`` e ``review_provenance.py`` (dataset de produção).
- Plataforma: ``operations.py`` (treinos, curvas, comparações) e
  ``platform_server.py`` (servidor local: monitoramento ao vivo e tarefas).
- Entrada: ``cli.py`` (comandos) e ``__main__.py`` (``python -m dronecamp_ia``).

A revisão técnica permanece humana: o modelo só propõe caixas.

Função deste arquivo no projeto: marca a pasta ``src/dronecamp_ia`` como pacote
Python e guarda a versão do pacote. Ele é executado sempre que qualquer módulo
do pacote é importado, por isso NÃO importa nada pesado (Ultralytics, torch):
cada módulo importa o que precisa só quando é usado.

O que contém:
- ``__version__``: versão do pacote (texto ``"MAIOR.MENOR.CORREÇÃO"``).
- ``YOLO_AUTOINSTALL=false`` no ambiente (se ainda não definido), antes de qualquer
  import da Ultralytics em qualquer comando.

Quem usa: o Python, ao importar ``dronecamp_ia``; ``__version__`` pode ser lido
por quem quiser registrar a versão do código.

O que lê e grava no disco: nada (só define uma variável de ambiente do processo).

Onde mexer:
- Para registrar uma versão nova do pacote, troque ``__version__`` abaixo.
- Para acrescentar um módulo novo, crie o arquivo nesta pasta e inclua uma linha
  no "Mapa rápido" acima e no mapa de ``docs/guia_manutencao.md``.
- ATENCAO: não acrescente importações pesadas aqui; elas deixariam todos os
  comandos (até ``--help``) lentos e exigiriam a Ultralytics instalada.
"""

import os

# ATENCAO: a Ultralytics lê YOLO_AUTOINSTALL no próprio import e, ligada, tenta
# "uv/pip install" de pacotes que acha faltando. Em 13/10 o gpu-eval-worker importou a
# Ultralytics antes de backend.py desligar isso, não reconheceu o onnxruntime-directml
# como "onnxruntime" e sobrescreveu parte dele com o onnxruntime 1.31 (pacote misturado,
# session.run quebrado). Definir aqui vale para todo comando do pacote.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# Versão do pacote (versionamento semântico: maior.menor.correção).
# EXTENSAO: suba este número quando publicar uma versão nova do código.
__version__ = "0.1.0"
