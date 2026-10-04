# Publicação do DroneCamp Vision

Data: 04/10/2026. Destino: [CFSJCODE/DroneCamp_Vision](https://github.com/CFSJCODE/DroneCamp_Vision), branch `main`.

## Escopo

A publicação contém o código, testes, configurações, documentação técnica, apresentação institucional, fotos de referência, revisões humanas, datasets, evidências, resultados, pesos e exportações do projeto. Backups e registros auxiliares úteis de `.runtime` também são preservados.

O [inventário](inventario-publicacao.json) registra 1221 arquivos de conteúdo, 914,840,555 bytes e um SHA-256 por arquivo. Este relatório e o próprio inventário são incluídos no repositório, mas não entram nos hashes autorreferentes do inventário.

Há 12 arquivos de modelos/exportações armazenados no Git LFS. Para recebê-los completos, instale Git LFS e execute `git lfs pull` após a clonagem.

## Exclusões da publicação pública

- Oito documentos pessoais de `Documentos PUCTEC`: identificação/CPF, currículo, histórico escolar, formulários bancários e termos de bolsista. A apresentação institucional está incluída.
- Ambiente Python `.venv`, caches `__pycache__`, bytecode, metadados `*.egg-info` e configurações/cache locais do Ultralytics.
- Logs temporários e estado do servidor de revisão em `sistema_ia/.runtime`.
- Arquivos de credenciais e configuração privada, como `.env`, caso existam. A varredura desta publicação não encontrou credenciais reais no conteúdo incluído.

Os arquivos excluídos permanecem na pasta original. Nenhum arquivo original foi apagado.

## Preparação e validação

- Repositório de destino conferido antes do envio: vazio e público, com acesso de escrita disponível.
- Varredura textual de credenciais e extração textual dos documentos técnicos: sem credenciais reais ou identificadores pessoais encontrados no escopo incluído. Imagens não passaram por OCR.
- Navegação acrescentada no README da raiz; atualização documental do estado piloto no README do sistema e correção do caminho do PDF no exemplo de extração.
- Pesos e ONNX encaminhados ao Git LFS; imagens e documentação mantêm sua estrutura de pastas.
- `.gitattributes` preserva os bytes e quebras de linha originais. O comando de clonagem documentado ativa caminhos longos somente no clone, evitando o limite de nomes do Git no Windows.
- A integridade da publicação pode ser conferida em um clone completo comparando o SHA-256 e o tamanho de cada arquivo com `inventario-publicacao.json`.

## Limites e pendências do projeto

Esta entrega publica arquivos; não executa um novo treinamento, não confirma desempenho do detector nem aprova uso operacional. Os artefatos preservam `production_ready: false`.

Registros, manifestos e YAMLs históricos mantêm caminhos absolutos da máquina original. O fluxo completo de revisão e reconstrução de dataset em outro diretório precisa de migração explícita para uma nova versão, com conferência de proveniência e hashes. A página v7 e o atalho da raiz utilizam caminhos relativos para abrir a revisão local.

O acervo de terceiros mantém suas referências originais; esta publicação não representa auditoria de direitos de redistribuição nem atribuição de nova licença. Consulte a documentação técnica para o estado da coleta, avaliação por classe e evolução do modelo.
