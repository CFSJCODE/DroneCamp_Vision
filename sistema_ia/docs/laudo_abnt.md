# Laudo técnico no formato ABNT

O botão **Gerar laudo técnico** (seção *Laudos e downloads*) e `exports.build_report` produzem o laudo de inspeção da cobertura a partir das decisões da revisão. O texto fixo e a formatação ficam em `src/dronecamp_ia/laudo_abnt.py`; os dados (`laudo.json`) são montados em `exports.py`.

## Estrutura

A apresentação segue a ABNT NBR 10719 (relatório técnico), com seções numeradas pela NBR 6024, sumário pela NBR 6027 e referências pela NBR 6023. O conteúdo segue a ABNT NBR 16747 (inspeção predial) e a lista de verificação do laudo de referência do projeto (`CEASA Minas - Pavilão A - Laudo.pdf`).

| Parte | Conteúdo |
|---|---|
| Capa e folha de rosto | Empresa, solicitante, título, local, natureza do trabalho, número do laudo, versão da revisão, data, situação "preliminar" |
| Sumário | Com links (HTML) e páginas (PDF) |
| 1 Identificação do laudo | Solicitante, local, endereço, data da inspeção e da emissão, revisores, responsável técnico, registro profissional, ART |
| 2 Objetivo | Texto do laudo CEASA, com a ressalva da validação técnica |
| 3 Normas e documentos aplicáveis | Tabela com documento, título e aplicação |
| 4 Metodologia | 4.1 levantamento de campo; 4.2 processamento com IA (hash, detector, revisão humana, VLM); 4.3 escala de conformidade do laudo CEASA e prioridades da NBR 16747; 4.4 limitações |
| 5 Descrição do imóvel | Tipo de cobertura, inclinação, estrutura de apoio, calha, área: só o que foi informado |
| 6 Registro fotográfico e análise de conformidades | Um item por categoria com ocorrência confirmada: pergunta do laudo CEASA, constatação factual (demarcações e fotos), figuras numeradas com legenda acima e fonte abaixo, subsídio de IA e a tabela Conformidade / Risco / Diagnóstico / Recomendação / Prioridade, sempre pendente. Depois: categorias sem ocorrência confirmada e itens da lista de verificação que a detecção não cobre (inclinação, tirantes, chaminés, entorno...) |
| 7 Quadro-resumo | Item, categoria, demarcações, fotos, figuras e os campos técnicos pendentes |
| 8 Recomendações gerais | Só orientações de procedimento (prioridades pela NBR 16747, reparos pela NBR 16280, manutenção pela NBR 5674) |
| 9 Considerações finais | Resumo factual da revisão; conclusão técnica pendente |
| 10 Responsabilidade técnica | Declaração do laudo CEASA e bloco de assinatura com registro e ART |
| Referências e Anexos A–E | Fotos sem ocorrência, sugestões pendentes do detector, subsídios de IA (concordância do VLM por classe), demarcações por foto e apoio ao responsável técnico com rastreabilidade |

## Normas citadas

ABNT NBR 16747:2020, NBR 5674, NBR 7196, NBR 15575-5, NBR 10844:1989, NBR 16280, NBR 13752, NBR 10719:2015, NBR 6023:2018, NBR 6024:2012, NBR 6027:2012, ANAC RBAC-E nº 94 e Lei nº 6.496/1977 (ART). A lista e o uso de cada uma estão em `NORMATIVE_REFERENCES`. Normas sem ano no código tiveram edições recentes (a NBR 7196 tem edições de 2020 e 2026 no catálogo e a NBR 13752 foi revisada em 2024): confirme a edição vigente no Catálogo ABNT antes de emitir, e preencha o ano em `NORMATIVE_REFERENCES` quando o responsável técnico definir. NBR 13531 e 13532 tratam de elaboração de projetos e não se aplicam a laudo de inspeção.

## O que nunca é preenchido pelo sistema

- Conformidade, risco/gravidade, diagnóstico, recomendação específica, prioridade e conclusão técnica: "pendente de validação técnica".
- A gravidade de referência da taxonomia e o texto que o laudo CEASA deu a cada categoria aparecem só no Anexo E, marcados como "não aplicados".
- Área, inclinação, extensões em metros e quantidades físicas: só se informadas no formulário.

## Uso dos modelos de IA

- Detector YOLO: as sugestões sem decisão humana ficam no Anexo B e não entram em contagens nem figuras.
- Modelo de visão e linguagem: o comando `describe-occurrences` que grava o `ai_notes.json` está no PR #11, ainda não mesclado nesta branch. Quando existe `ai_notes.json` da mesma versão do registro, a descrição de cada caixa humana entra no item como "Subsídio de IA" e a concordância por classe vai ao Anexo C. Notas de outra versão são ignoradas.

## Dados técnicos opcionais

`build_report(..., details={...})` e o bloco "Dados técnicos do laudo" na página aceitam: `report_number`, `inspection_date`, `requester`, `address`, `roof_type`, `slope`, `support_structure`, `gutter_type`, `area`, `equipment`, `flight_description`, `company`, `responsible_name`, `responsible_title`, `responsible_registry`, `art_number`. Sem `requester`, vale o cliente; sem `responsible_name`, o campo "Responsável técnico (validação)".
