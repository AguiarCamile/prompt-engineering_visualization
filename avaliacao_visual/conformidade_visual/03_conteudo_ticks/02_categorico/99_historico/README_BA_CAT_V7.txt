B-A-CAT V7 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
=================================================

Decisão após a V6
-----------------
A V6 não deve ser congelada.

Resultado automático:
- 24/24 imagens processadas;
- 0 erros;
- 24/24 orientações VERTICAL;
- 19 CATEGORICAL_COMPLETE;
- 5 CATEGORICAL_AMBIGUOUS.

A inspeção mostrou regressão importante:
a expansão vertical global da lane recuperou segundas linhas de categorias,
mas também reintroduziu o título do eixo "SETOR CNAE" em muitos slots BC e
alguns BI.

Exemplos observados:
- BC_006: COMERCIO E SERVICOS SETOR CNAE
- BC_008: COMERCIO E SERVICOS SETOR CNAE
- BC_013: COMERCIO E SERVICOS SETOR CNAE
- BC_017: COMERCIO E SERVICOS SETOR CNAE
- BC_021: COMERCIO E SERVICOS SETOR CNAE
- BC_037: COMERCIO E SERVICOS SETOR CNAE
- BC_050: COMERCIO E SERVICOS SETOR CNAE
- BI_036 / BI_050: contaminação semelhante no segundo slot.

Portanto, a V7 RETORNA À V5 como baseline estrutural.

O que a V7 preserva da V5
-------------------------
- orientação por evidência observada;
- ROW_FIRST + BAR_SUPPORT;
- slots estruturais da V5;
- lane estrutural estreita;
- ausência de expansão vertical global;
- mesma amostra de 24 imagens de calibração.

Melhorias direcionadas da V7
----------------------------

1. CONTINUATION LINES TARGETED
   A V7 procura linhas imediatamente inferiores aos ticks, mas NÃO amplia
   todos os slots.

2. GATE CONTRA TÍTULO DO EIXO
   Uma linha inferior com um único grupo central é tratada como provável
   axis title e não é incorporada automaticamente.

   Linhas inferiores com múltiplos grupos alinhados a slots diferentes são
   interpretadas como forte evidência de continuação de labels multilinha.

3. CONTINUAÇÃO EM SLOT EXTREMO
   Um único grupo inferior afastado do centro global pode ser aceito como
   continuação, pois é improvável que seja título central do eixo.

4. COMPOSIÇÃO ESPACIAL
   Quando a linha principal e a continuação pertencem ao mesmo slot:
       AGROPECUARIA + E PESCA
   pode gerar:
       AGROPECUARIA E PESCA

   Isso é composição de duas evidências observadas da MESMA imagem.
   Não há dicionário externo.

5. EXPANSÃO VERTICAL SOMENTE QUANDO NECESSÁRIA
   O crop do slot permanece estreito.
   Só um slot com continuação aprovada recebe crop vertical composto.

6. FOLGA HORIZONTAL PEQUENA
   Cada slot recebe pequena margem lateral para tentar corrigir:
       NFRA-ESTRUTURA
       OMERCIO/SERVICO
   sem reabrir a lane vertical.

7. ROTAÇÕES ±15°
   Mantidas para labels inclinados como BC_048.

Independência
-------------
A V7 NÃO recebe:
- categorias esperadas;
- número esperado de categorias;
- ordem esperada;
- especificação F;
- dicionário derivado da referência.

Amostra
-------
Somente as mesmas 24 imagens de calibração:
- 12 BI
- 12 BC

O holdout independente anterior permanece totalmente fora do desenvolvimento.

SHA-256 desta versão de calibração
----------------------------------
48d83885d64e7b9e45381060728f8319e0118de37231f0a83c5e72e219031753

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v7") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v7" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v7.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v7") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v7" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v7.py

ARQUIVOS A ENVIAR
-----------------
1. ba_cat_v7_summary.txt
2. ba_cat_v7_images.csv
3. ba_cat_v7_slots.csv
4. ba_cat_v7_priority_review.csv
5. ba_cat_v7_errors.csv
6. contact_BA_CAT_V7_BI.png
7. contact_BA_CAT_V7_BC.png

Se ainda houver casos problemáticos:
8. ba_cat_v7_candidates.csv

Campos novos importantes
------------------------
continuation_lines_considered
continuation_groups_accepted
probable_axis_title_groups_rejected
continuation_text
ocr_slot_bbox
structural_slot_bbox

O que queremos observar
-----------------------
BC:
- recuperar E PESCA / DE TRANSFORMACAO quando realmente estão em segunda linha;
- NÃO incorporar SETOR CNAE ao slot central;
- melhorar BC_006, BC_015, BC_046 e BC_048.

BI:
- preservar os casos que já estavam corretos na V5;
- melhorar cortes de borda em BI_031;
- tentar recuperar primeiro slot de BI_027/BI_040 sem degradar os demais;
- BI_006 pode permanecer explicitamente ambíguo se a estrutura observada não
  fornecer evidência suficiente.

A V7 ainda é CALIBRAÇÃO.
Somente após auditoria manual das 24 imagens devemos decidir pelo congelamento
e novo holdout independente BI/BC.
