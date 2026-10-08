# Pacote de códigos — Conformidade Visual

Este ZIP organiza todos os arquivos Python que estavam disponíveis no ambiente
desta conversa para o pipeline de conformidade visual da dissertação.

## Estrutura

- `01_preparacao_especificacao`: separação/organização de prompts.
- `02_tecnica_grafica/01_final`: classificador exclusivo de produção.
- `02_tecnica_grafica/02_validacao_auditoria`: validações cegas e auditorias.
- `02_tecnica_grafica/03_utilitarios`: inventários/exportação/localização de fontes.
- `03_conteudo_ticks/01_numerico_temporal`: ferramentas de auditoria B-A V7.
- `03_conteudo_ticks/02_categorico/01_final`: B-A-CAT V9, holdout e produção.
- `03_conteudo_ticks/02_categorico/99_historico`: versões V1–V8 preservadas.
- `04_presenca_ticklabels`: adjudicação manual dos casos de presença.
- `05_integracao`: integração por imagem.
- `06_configuracoes_e_metadados`: especificação F, normalização B-B e preflight.
- `00_documentacao`: manifesto, hashes, inventário e dependências.

## Regra metodológica central

B-A observa/extrai o que está visível na imagem e não deve receber os valores
esperados de F. B-B compara posteriormente o observado com a especificação F.

## Completude

O pacote inclui todos os fontes que estavam montados no ambiente no momento da
criação. Alguns scripts centrais são conhecidos por nome/versão/hash, mas o
arquivo-fonte não estava montado. Eles não foram recriados por aproximação.

Veja `ARQUIVOS_FONTE_NAO_MONTADOS.txt`.

No computador original, use `LOCALIZAR_DEPENDENCIAS_FALTANTES.cmd` para procurar
essas dependências em `C:\Users\Labvis\Downloads\imagens3120`.

O arquivo `Manifesto_Reprodutibilidade_Conformidade_Visual_Camile.xlsx` é o
metadado principal do pipeline.


## Atualização V2

Foram acrescentados e verificados por SHA-256:

- `axis_conformity_calibration_v3.py`
- `mark_presence_validation_v3.py`
- `ticklabel_content_extraction_calibration_ba_v7.py`

Os três arquivos coincidem exatamente com os hashes previamente registrados no
manifesto metodológico. Portanto, não são reconstruções: são as versões
congeladas utilizadas no pipeline.


## Atualização V3

Foram incorporados os novos fontes de área de plotagem, presença de tick labels,
classificação/holdout de técnica, auditoria de MARK_ABSENT, orquestração de cor,
módulos de cor de barras e dispersão, holdout B-A V7 e runner de produção B-A V7.

O arquivo `validate_color_lines_v3.py` foi mantido como histórico/não confirmado,
pois o orquestrador final exige `validate_color_lines_v4.py`.

Consulte `ARQUIVOS_FONTE_AINDA_FALTANTES_V3.txt` para a lista atualizada.


## Atualização V5

A V5 incorpora os wrappers finais de eixos e presença de tick labels, o
pós-processamento cromático com sua configuração, o comparador B-B unificado de
conteúdo de ticks, os códigos finais de produção/auditoria da técnica e os
principais resultados finais necessários para rastreabilidade.

A única pendência de fonte original remanescente é o script que gerou
`INTEGRACAO_CONFORMIDADE_VISUAL_MASTER_V4_FINAL.csv`. O CSV final está
preservado; não foi inventado um fonte original substituto.
