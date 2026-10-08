B-A-CAT V4 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
=================================================

Diagnóstico da V3
-----------------
A V3 melhorou a infraestrutura:
- 24/24 imagens processadas;
- 0 erros;
- 24/24 orientações VERTICAL;
- plot_bbox_source = AXIS_V3;
- 18 CATEGORICAL_COMPLETE;
- 6 CATEGORICAL_AMBIGUOUS.

Entretanto, a auditoria visual mostrou que o status COMPLETE ainda podia ser
falso. Os problemas residuais foram:
- slots duplicados criados por fragmentos da mesma categoria;
- omissão de categorias quando o OCR não gerava uma caixa independente;
- contaminação pelo título "SETOR CNAE";
- truncamento de frases como "AGROPECUARIA E PESCA";
- fragmentos como "ERVICOS", "SS", "D/SERVICOS" tratados como categorias.

Mudança central da V4
---------------------
Os slots passam a ser definidos prioritariamente pela GEOMETRIA DAS BARRAS.

1. DETECÇÃO DE BARRAS POR HSV
   - usa saturação/cor no interior do plot;
   - remove fundo, grade e texto preto/cinza;
   - procura componentes preenchidos;
   - exige alinhamento com linha de base comum;
   - não conhece número esperado de barras/categorias.

2. AGRUPAMENTO DE BARRAS
   - barras individuais são agrupadas pela escala dos gaps;
   - gaps pequenos indicam barras de uma mesma categoria;
   - gaps homogêneos preservam barras como categorias distintas;
   - nenhum número esperado de categorias é fornecido.

3. BAR-PRIMARY SLOTS
   - quando a geometria é confiável, somente os centros dos grupos de barras
     definem os slots;
   - caixas OCR e layer groups deixam de criar slots extras;
   - evidência textual continua sendo usada dentro do slot.

4. REFINAMENTO DA CATEGORICAL LANE
   - OCR de palavras identifica a linha textual mais próxima do eixo;
   - a linha de tick labels é separada de títulos inferiores;
   - "SETOR CNAE" tende a ficar fora do crop final.

5. OCR MULTI-EVIDÊNCIA
   - slot completo continua sendo lido em múltiplos preprocessamentos,
     rotações e PSMs;
   - caixas V7 são somente evidência auxiliar;
   - pequenas variações da mesma leitura são agrupadas.

6. COMPLETUDE DE FRASE
   - dentro do mesmo slot limpo, frases mais completas recebem pequeno bônus;
   - isso favorece "AGROPECUARIA E PESCA" sobre "AGROPECUARIA";
   - não há comparação com F.

7. FALLBACK
   Se a geometria das barras não for confiável, a V4 retorna ao mecanismo
   multi-evidência da V3. O fallback continua cego à especificação.

Independência
-------------
A V4 NÃO recebe:
- categorias esperadas;
- número esperado de categorias;
- ordem esperada;
- especificação F;
- conteúdo da imagem de referência como dicionário.

Amostra de desenvolvimento
--------------------------
Somente as mesmas 24 imagens:
- 12 BI
- 12 BC

O holdout independente anterior não é usado para ajuste.

SHA-256 desta versão de calibração
----------------------------------
2d524ab8315ca61360cea17c4f6240b26d8d6d72dd28e795d4fa5d23e2aad905

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v4") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v4" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v4.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v4") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v4" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v4.py

ARQUIVOS A ENVIAR
-----------------
1. ba_cat_v4_summary.txt
2. ba_cat_v4_images.csv
3. ba_cat_v4_slots.csv
4. ba_cat_v4_priority_review.csv
5. ba_cat_v4_errors.csv
6. contact_BA_CAT_V4_BI.png
7. contact_BA_CAT_V4_BC.png

Se ainda houver casos problemáticos:
8. ba_cat_v4_candidates.csv

O que avaliar na V4
-------------------
- quantidade de grupos de barras detectados;
- uso de BAR_PRIMARY na maioria dos casos;
- redução de slots duplicados;
- recuperação de Infra-Estrutura nos BI;
- recuperação de "Agropecuária e Pesca" nos BC;
- desaparecimento de "SETOR CNAE" do conteúdo categórico;
- desaparecimento de fragmentos tratados como categorias autônomas;
- redução de CATEGORICAL_AMBIGUOUS sem aumento de falsos COMPLETE.

Ainda é calibração. Só depois da auditoria manual das 24 imagens será possível
decidir pelo congelamento e pelo novo holdout independente BI/BC.
