@echo off
setlocal
rem Abre a pagina gerada com as correcoes humanas; o template interno nao contem fotos.
set "dronecamp_review=%~dp0sistema_ia\data\reviews\ceasa_v6_corrigida_a36ea7f67d08\index.html"
if not exist "%dronecamp_review%" (
  echo A pagina de revisao nao foi encontrada na pasta esperada.
  echo Mantenha este atalho na pasta principal do projeto.
  pause
  exit /b 1
)
start "" "%dronecamp_review%"
endlocal
