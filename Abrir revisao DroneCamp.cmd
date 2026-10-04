@echo off
setlocal
rem Abre a revisao v7 (sua revisao002) com sugestoes do modelo piloto; o template interno nao contem fotos.
set "dronecamp_review=%~dp0sistema_ia\data\reviews\ceasa_v7_revisao002_ba8cc323c8a1\index.html"
if not exist "%dronecamp_review%" (
  echo A pagina de revisao nao foi encontrada na pasta esperada.
  echo Mantenha este atalho na pasta principal do projeto.
  pause
  exit /b 1
)
start "" "%dronecamp_review%"
endlocal
