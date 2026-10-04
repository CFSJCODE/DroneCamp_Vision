# Contribuindo com o DroneCamp Vision

O projeto adota [Conventional Commits 1.0.0](https://www.conventionalcommits.org/pt-br/v1.0.0/) para novos commits. O histórico já publicado é preservado; a adoção não exige reescrever commits anteriores.

## Mensagens de commit

```text
tipo(escopo opcional)!: descrição da mudança

Corpo opcional: explique o motivo e o efeito da mudança.

BREAKING CHANGE: impacto incompatível e orientação de migração, quando houver.
```

O escopo e o `!` são opcionais. Use uma descrição específica e não vazia, com um espaço depois de `:`. Prefira tipos e escopos em minúsculas e títulos concisos; não há limite obrigatório de caracteres. O corpo e os rodapés, quando presentes, são separados por uma linha em branco.

| Tipo recomendado | Quando usar |
| --- | --- |
| `feat` | Adicionar uma funcionalidade |
| `fix` | Corrigir um defeito |
| `docs` | Melhorar documentação, instruções ou navegação |
| `refactor` | Reorganizar código sem alterar seu comportamento |
| `test` | Adicionar ou corrigir testes |
| `perf` | Melhorar desempenho |
| `build` | Alterar dependências, empacotamento ou exportação de artefatos |
| `ci` | Alterar automações de integração, quando existirem |
| `chore` | Manutenção do repositório e configuração de ferramentas |
| `style` | Formatação de código, sem mudança de comportamento |
| `revert` | Reverter uma mudança, indicando o commit no corpo ou em `Refs:` |

Escopos úteis neste projeto: `cli`, `dataset`, `review`, `taxonomy`, `training`, `prediction`, `export`, `docs` e `repo`. Eles correspondem à CLI, aos dados, à revisão humana, às classes, ao treinamento, à inferência, às exportações, à documentação e à configuração do repositório. São sugestões, não uma lista fechada.

```text
docs: melhorar descrição e mapa de pastas do projeto
fix(review): corrigir caminhos relativos das imagens
test(dataset): verificar rastreabilidade das anotações
chore(repo): adotar Conventional Commits com hooks locais
```

Uma mudança incompatível usa `!` antes dos dois-pontos ou o rodapé `BREAKING CHANGE:`. Prefira explicar também a migração:

```text
feat(taxonomy)!: substituir identificadores de classes

BREAKING CHANGE: datasets anteriores precisam remapear os identificadores
das classes antes de um novo treinamento.
```

Esse exemplo ilustra o formato; não anuncia uma mudança já realizada. A convenção, por si só, não gera versões, releases ou changelog automaticamente.

## Ativar a verificação neste clone

Depois de instalar o Git LFS, execute na raiz do repositório:

```powershell
git lfs install --local
git config --local core.hooksPath .githooks
git config --local commit.template .gitmessage
```

As configurações são locais ao clone. Clonar o repositório não ativa os hooks nem o modelo de mensagem automaticamente. Se você já usa hooks próprios, confira-os antes de trocar `core.hooksPath`: apenas um diretório de hooks fica ativo por vez.

O hook `commit-msg` valida o título e aceita variações de maiúsculas/minúsculas nos tipos, embora o padrão recomendado seja minúsculas. Tipos adicionais também são permitidos. Ele não valida o sentido da mensagem, o corpo, os rodapés ou o impacto real de uma mudança. Títulos automáticos começando com `Merge ` são aceitos apenas quando existe uma operação de merge em andamento; commits de squash e reversão devem usar o formato convencional.

Os hooks `pre-push`, `post-checkout`, `post-commit` e `post-merge` do Git LFS estão preservados em `.githooks`, para que a ativação continue carregando e enviando os modelos rastreados pelo LFS. Consulte [os detalhes dos hooks](.githooks/README.md).

## Preparar uma contribuição

1. Mantenha a mudança focada e confira quais arquivos serão incluídos no commit.
2. Atualize a documentação quando comandos, caminhos ou formatos mudarem.
3. Execute as verificações adequadas à mudança; para código Python, consulte [o README do sistema](sistema_ia/README.md).
4. Preserve a proveniência, os hashes, os estados de revisão e as evidências dos dados. Uma mudança em dados ou modelo precisa de avaliação documentada antes de alegar melhoria.
5. Evite incluir ambientes locais, credenciais ou documentos pessoais. Confira [o escopo de publicação](PUBLICACAO.md).
