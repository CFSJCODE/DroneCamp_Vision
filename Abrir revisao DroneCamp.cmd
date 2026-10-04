@echo off
setlocal
rem Abre a revisao v8 (2a rodada da revisao002) com sugestoes do modelo piloto v8; o template interno nao contem fotos.
set "dronecamp_review=%~dp0sistema_ia\data\reviews\ceasa_v8_revisao002_ff34e226416c\index.html"
if not exist "%dronecamp_review%" (
  echo A pagina de revisao nao foi encontrada na pasta esperada.
  echo Mantenha este atalho na pasta principal do projeto.
  pause
  exit /b 1
)
start "" "%dronecamp_review%"
endlocal
