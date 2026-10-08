B-A-CAT V5 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
=================================================

Diagnóstico da V4
-----------------
A V4 manteve:
- 24/24 imagens processadas;
- 0 erros;
- 24/24 orientações VERTICAL;
- plot_bbox_source = AXIS_V3;
- 23/24 casos usando BAR_PRIMARY.

Automaticamente:
- 18 CATEGORICAL_COMPLETE;
- 5 CATEGORICAL_AMBIGUOUS;
- 1 CATEGORICAL_PARTIAL.

Mas a inspeção visual mostrou que a geometria das barras não pode ser a única
fonte de slots. Há três situações importantes:

1. PLOT BBOX TRUNCADO À ESQUERDA
   Em vários BI, plot_x_left começa depois da primeira barra/categoria.
   Ex.: BI_007_R02, BI_009_R10, BI_016_R06, BI_021_R01 e BI_050_R09.
   A barra/categoria Infra-Estrutura ficou fora da busca geométrica.

2. BARRAS ZERO OU MUITO PEQUENAS
   Uma categoria pode existir visualmente no eixo mesmo quando a barra tem
   altura zero ou quase zero. Logo, "sem retângulo detectável" não significa
   "sem categoria".

3. AGRUPAMENTO GEOMÉTRICO IMPERFEITO
   Em BC_046_R04, por exemplo, a geometria produziu quatro grupos para um eixo
   cujo texto está organizado em menos blocos, fragmentando categorias.

Mudança central da V5
---------------------
A V5 passa de BAR-PRIMARY para ROW-FIRST + BAR SUPPORT.

1. A CATEGORICAL LANE X É AMPLIADA HORIZONTALMENTE
   - preserva a faixa vertical de ticks;
   - busca labels quase em toda a largura da figura;
   - não depende de plot_x_left para encontrar a primeira categoria.

2. OCR WORD-LEVEL DA LINHA DE TICKS
   - usa image_to_data;
   - separa linhas de texto;
   - privilegia a linha mais próxima do eixo;
   - agrupa palavras adjacentes em frases por gaps horizontais;
   - não usa dicionário de categorias.

3. FRASES DA LINHA COMO ANCHORS
   Exemplos observados, quando o OCR consegue:
   AGROPECUARIA E PESCA
   COMERCIO E SERVICOS
   INDUSTRIA DE TRANSFORMACAO

   Essas frases definem slots diretamente.

4. BARRAS COMO SUPORTE
   - a busca de barras X é expandida horizontalmente;
   - centros de barras podem adicionar um slot quando existe uma lacuna
     clara sem frase textual correspondente;
   - barras não criam duplicatas perto de uma frase já detectada.

5. CATEGORIAS COM VALOR ZERO
   - continuam recuperáveis pela linha textual mesmo sem barra visível.

6. EVIDÊNCIA ROW_PHRASE
   - a frase detectada na linha completa entra como candidato independente;
   - recebe bônus por coerência espacial da própria imagem;
   - não há comparação com F.

7. FALLBACK
   - se a linha textual não for confiável, usa barras;
   - se ambos falharem, volta ao mecanismo multi-evidência anterior.

Independência
-------------
A V5 NÃO recebe:
- categorias esperadas;
- número esperado de categorias;
- ordem esperada;
- especificação F;
- conteúdo da figura de referência como vocabulário.

Amostra
-------
As mesmas 24 imagens de calibração:
- 12 BI
- 12 BC

O holdout independente anterior continua fora do desenvolvimento.

SHA-256 desta versão de calibração
----------------------------------
7b33c1ef12cc0d089c8d31104a03f0a031c2692962905d7db3a4b7672c08c78a

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v5") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v5" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v5.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v5") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v5" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v5.py

ARQUIVOS A ENVIAR
-----------------
1. ba_cat_v5_summary.txt
2. ba_cat_v5_images.csv
3. ba_cat_v5_slots.csv
4. ba_cat_v5_priority_review.csv
5. ba_cat_v5_errors.csv
6. contact_BA_CAT_V5_BI.png
7. contact_BA_CAT_V5_BC.png

Se houver problemas:
8. ba_cat_v5_candidates.csv

Campos novos importantes
------------------------
row_groups_detected
row_groups_text
row_group_quality
row_group_psm
categorical_slot_strategy

Estratégias possíveis
---------------------
ROW_PRIMARY
ROW_PLUS_BAR_EXTRAS
BAR_PRIMARY
MULTI_EVIDENCE_FALLBACK

O que esperamos observar
------------------------
- recuperação da categoria à esquerda nos BI onde plot_bbox a omitia;
- recuperação de categorias com barra zero/pequena;
- redução da fragmentação observada em BC_046;
- menos falsos CATEGORICAL_COMPLETE;
- manutenção da orientação correta;
- menos contaminação por título do eixo;
- redução real, e não apenas nominal, de PARTIAL/AMBIGUOUS/WRONG.

Ainda é calibração. Somente após auditoria manual das 24 imagens devemos
considerar congelamento e novo holdout independente.
