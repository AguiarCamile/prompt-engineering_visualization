B-A-CAT V2 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
=================================================

Motivo da V2
------------
A B-A-CAT V1 processou 24/24 imagens sem erros, mas revelou falha estrutural:
- 20/24 orientações ficaram AMBIGUOUS;
- 24/24 casos foram para revisão prioritária;
- nenhuma imagem ficou CATEGORICAL_COMPLETE;
- todas as 24 usaram FALLBACK_GEOMETRIC para a área de plotagem.

A inspeção dos resultados mostrou que a categorical lane ampla misturava:
- tick labels;
- valores escritos acima das barras;
- título do eixo;
- outros textos próximos ao plot.

Isso elevava simultaneamente as evidências alfabética e numérica dos dois eixos.

Mudança central da V2
---------------------
A orientação não é mais inferida pelo OCR de faixas amplas.

A V2 executa primeiro a segmentação congelada da B-A V7/V4 e caracteriza
as caixas de tick labels efetivamente detectadas em X e Y.

Hipóteses concorrentes:
VERTICAL:
    X categórico + Y numérico

HORIZONTAL:
    X numérico + Y categórico

A decisão usa apenas evidência da própria imagem:
- OCR leve das caixas de tick labels;
- coerência numérica já produzida pela V7;
- evidência da banda categórica;
- geometria auxiliar das barras.

Nenhum conteúdo de F é carregado.

Melhoria dos slots
------------------
A V1 criava slots largos a partir de centros e isso podia incluir valores
sobre as barras ou o título do eixo.

Na V2:
- cada slot nasce de uma caixa real de tick label;
- o crop é apenas ligeiramente expandido;
- para X categórico é priorizada a category_band_bbox da V7;
- para Y categórico é usada a união estreita das caixas observadas;
- o limiar de agrupamento de variantes OCR do mesmo slot foi reduzido
  de 0,80 para 0,72 para unir pequenas variações da mesma leitura.

Amostra
-------
Continua usando exclusivamente:
- 12 BI;
- 12 BC;
já pertencentes à calibração B-A V7.

O holdout anterior NÃO é usado para ajuste.

SHA-256 desta versão de calibração
----------------------------------
0c4a809dfc5513453250df147576b13abcfbaf3fc18fedd170d18b8eb2b68076

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v2") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v2" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v2.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v2") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v2" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v2.py

ARQUIVOS A ENVIAR
-----------------
1. ba_cat_v2_summary.txt
2. ba_cat_v2_images.csv
3. ba_cat_v2_slots.csv
4. ba_cat_v2_priority_review.csv
5. ba_cat_v2_errors.csv
6. contact_BA_CAT_V2_BI.png
7. contact_BA_CAT_V2_BC.png

Se houver ambiguidades:
8. ba_cat_v2_candidates.csv

Critério desta etapa
--------------------
Ainda é calibração, não holdout.

Primeiro queremos observar:
- forte redução de ORIENTATION_AMBIGUOUS;
- redução de CATEGORICAL_AMBIGUOUS;
- aparecimento de CATEGORICAL_COMPLETE;
- ausência de falso rótulo sistemático.

Somente depois de auditar as 24 imagens a V2 poderá ser ajustada/congelada.
