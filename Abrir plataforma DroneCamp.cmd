@echo off
setlocal
rem Plataforma DroneCamp Vision (servidor local em 127.0.0.1): inspecoes, importacao de fotos e videos,
rem revisao, laudos e downloads, treinos ao vivo e comparacao de modelos. Feche esta janela para encerrar.
rem Sem o servidor, a pagina de uma revisao abre direto do disco pelo "Abrir revisao DroneCamp.cmd".
rem Todas as inspecoes de sistema_ia\data\reviews ficam disponiveis; a mais recente abre primeiro.
rem Para abrir so uma, acrescente: --registry "data\reviews\<pasta>\registry.json"
cd /d "%~dp0sistema_ia"
if not exist ".venv\Scripts\python.exe" (
  echo O ambiente .venv nao foi encontrado em sistema_ia.
  echo Crie o ambiente conforme o README antes de abrir a plataforma.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m dronecamp_ia platform
pause
endlocal
