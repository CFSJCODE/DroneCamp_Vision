@echo off
setlocal
rem Plataforma de operacoes do DroneCamp (servidor local em 127.0.0.1): revisao, treinos ao vivo,
rem comparacao de modelos e tarefas (train-pilot, suggest). Feche esta janela para encerrar.
rem Sem o servidor, a mesma pagina abre direto do disco pelo "Abrir revisao DroneCamp.cmd".
set "dronecamp_registry=data\reviews\ceasa_v8_revisao002_ff34e226416c\registry.json"
cd /d "%~dp0sistema_ia"
if not exist ".venv\Scripts\python.exe" (
  echo O ambiente .venv nao foi encontrado em sistema_ia.
  echo Crie o ambiente conforme o README antes de abrir a plataforma.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m dronecamp_ia platform --registry "%dronecamp_registry%"
pause
endlocal
