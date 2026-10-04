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
"""

__version__ = "0.1.0"
