B-A-CAT V3 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
=================================================

Diagnóstico da V2
-----------------
A V2 resolveu a orientação na amostra de desenvolvimento:
- 24/24 imagens = VERTICAL;
- 0 erros de execução.

Mas o status categórico ainda não era confiável:
- 11 CATEGORICAL_COMPLETE;
- 12 CATEGORICAL_AMBIGUOUS;
- 1 CATEGORICAL_PARTIAL;
- 24/24 casos ainda em revisão prioritária.

A inspeção mostrou três causas:
1. microcaixas espúrias gerando labels como IM, VV, AN, CC e VL;
2. axis label "SETOR CNAE" incorporado a categorias centrais;
3. caixas amplas/concatenadas fundindo categorias vizinhas ou omitindo
   categorias congestionadas.

Correções da V3
---------------
1. PLOT BBOX REAL
   A V3 lê diretamente os campos do detector congelado:
   plot_x_left
   plot_x_right
   plot_y_top
   plot_y_bottom

   O FALLBACK_GEOMETRIC passa a ser apenas último recurso.

2. CATEGORICAL LANE DA V4
   Prioriza:
   _x_layer_rect / _y_layer_rect
   que representam a camada textual próxima ao eixo.

3. MÁSCARA DE AXIS LABEL
   Usa:
   _x_axislabel / _y_axislabel
   para mascarar prováveis títulos de eixo antes do OCR.

4. PRUNING DE MICROCAIXAS
   Caixas claramente minúsculas em relação às demais são descartadas
   geometricamente, sem usar palavras esperadas.

5. ANCHORS MULTI-EVIDÊNCIA
   O número e a posição de slots são inferidos por consenso entre:
   - centros/grupos observados de barras;
   - grupos coarse da camada textual V4;
   - caixas segmentadas de tick labels.

   Não há quantidade esperada de categorias.

6. SLOTS VORONOI RESTRITOS À LANE
   Limites horizontais/verticais são definidos entre anchors vizinhos,
   evitando que o crop de uma categoria incorpore a categoria seguinte.

7. OCR APÓS MÁSCARA
   O OCR multi-rotação e multi-preprocessamento continua, mas é aplicado
   em regiões mais limpas.

Independência
-------------
A V3 continua NÃO recebendo:
- categorias esperadas;
- número esperado de categorias;
- ordem esperada;
- especificação F;
- figura de referência como vocabulário.

Amostra
-------
Somente as mesmas 24 imagens de calibração:
- 12 BI
- 12 BC

Nenhuma imagem do holdout independente anterior é usada para ajuste.

SHA-256 desta versão de calibração
----------------------------------
32bc3db93da2ee06332185fd8b0b4310fe3de9c2e2cb2e30072774c7291949c8

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v3") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v3" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v3.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v3") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v3" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v3.py

ARQUIVOS A ENVIAR
-----------------
ba_cat_v3_summary.txt
ba_cat_v3_images.csv
ba_cat_v3_slots.csv
ba_cat_v3_priority_review.csv
ba_cat_v3_errors.csv
contact_BA_CAT_V3_BI.png
contact_BA_CAT_V3_BC.png

Se houver ambiguidade:
ba_cat_v3_candidates.csv

Critério para avaliar a V3
--------------------------
Ainda é calibração.

Queremos observar:
- plot_bbox_source deixar de ser FALLBACK_GEOMETRIC na grande maioria;
- manutenção da orientação correta;
- desaparecimento dos micro-labels espúrios;
- menor contaminação por "SETOR CNAE";
- recuperação mais completa de categorias congestionadas;
- redução substancial de CATEGORICAL_AMBIGUOUS/PARTIAL;
- CATEGORICAL_COMPLETE apenas quando a estrutura observada estiver de fato
  integralmente coberta.

Somente após auditoria visual das 24 imagens será considerada a possibilidade
de congelar o módulo categórico e iniciar novo holdout independente BI/BC.
