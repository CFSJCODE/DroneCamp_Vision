# Hooks locais

`commit-msg` verifica títulos no formato `tipo(escopo opcional)!: descrição`, conforme [Conventional Commits 1.0.0](https://www.conventionalcommits.org/pt-br/v1.0.0/). Tipos são aceitos sem distinguir maiúsculas/minúsculas; prefira minúsculas. O escopo e o `!` são opcionais, e a descrição precisa conter texto. O hook não impõe um limite de caracteres e não valida corpo ou rodapés.

Títulos automáticos iniciados por `Merge ` são uma exceção somente quando o Git registra um merge em andamento (`MERGE_HEAD`). Squash e revert devem usar um título convencional. O hook ignora linhas vazias e comentários de modelo iniciados por `#` antes do título e aceita arquivos de mensagem com finais de linha LF ou CRLF.

Os quatro hooks do Git LFS foram copiados sem alteração dos hooks ativos do projeto: `pre-push`, `post-checkout`, `post-commit` e `post-merge`. Eles delegam para o Git LFS e precisam continuar presentes se o diretório for usado como `core.hooksPath`.

## Ativação por clone

```powershell
git lfs install --local
git config --local core.hooksPath .githooks
git config --local commit.template .gitmessage
```

O clone não ativa essas configurações sozinho. Os scripts são executados pelo shell do Git; no Windows, use Git for Windows. Em sistemas Unix, os cinco scripts precisam estar com permissão de execução. Confira eventuais hooks próprios antes de trocar o diretório ativo. Hooks locais não constituem validação obrigatória no GitHub e podem ser ignorados por opções do Git; esta adoção não configura CI ou proteção de branch.

## Verificação manual

Salve uma mensagem em um arquivo temporário e execute, no Git Bash e na raiz do clone:

```sh
sh .githooks/commit-msg /caminho/para/mensagem.txt
```

Código de saída `0` indica aceitação; `1`, título inválido; `2`, arquivo de mensagem ausente ou ilegível. Não é necessário criar um commit para testar o formato.

Em 04/10/2026, o shell do Git for Windows passou em 31 verificações: 28 títulos (com e sem escopo, `!`, tipos em diferentes caixas, tipo adicional, mensagem com CRLF, comentários de modelo, título longo e entradas inválidas), arquivo/argumento ausentes e um merge real pendente em repositório temporário. Os quatro hooks do LFS foram comparados por SHA-256 com seus originais e são idênticos. Isso valida o formato local; não constitui teste de CI ou da aplicação de IA.
