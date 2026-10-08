B-A-CAT V9 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
=================================================

Diagnóstico manual da V8
------------------------
A V8 processou 24/24 imagens, sem erros, com:
- 20 CATEGORICAL_COMPLETE
- 4 CATEGORICAL_AMBIGUOUS

A auditoria visual mostrou que a automação ficou mais conservadora, mas ainda
existem falsos COMPLETE principalmente por incorporação do título do eixo ao
slot central.

Padrões residuais:
- BC_006: COMERCIO E SERVICOS SETOR CNAE
- BC_015: COMERCIO E SERVICOS (SETOR CNAE)
- BC_046: COMERCIO E SERVICOS SETOR CNAE
- BC_048: truncamento de início dos labels inclinados
- BI_006: estrutura incompleta, corretamente enviada a revisão
- BI_014: leitura INFRA-ESTRUTURA aparece repetidamente nos candidatos, mas o
          score ainda favorece um cluster ruidoso
- BI_031: artefatos de borda, corretamente enviados a revisão

Mudanças da V9
--------------
1. CONSENSO MAIS FORTE
   Na V8, o bonus de consenso saturava cedo demais.
   Uma leitura repetida em 18 variantes podia receber praticamente o mesmo
   bonus de uma leitura ruidosa presente em apenas 5 variantes.

   Na V9:
   - recorrência independente recebe peso maior;
   - o teto de consenso foi aumentado;
   - a extensão da frase deixa de dominar evidência repetida.

2. HIGH-CONSENSUS OVERRIDE
   Uma leitura com:
   - suporte >= 8 variantes independentes;
   - confiança >= 90%;
   - suporte claramente maior que o segundo cluster
   pode ser aceita como CONSENSUS.

   Objetivo: resolver casos como BI_014 sem usar F.

3. GATE CENTRAL POR GRUPO
   Na V8, quando uma linha inferior continha vários grupos, todos podiam ser
   aceitos. Isso permitia:
       E PESCA | SETOR CNAE | TRANSFORMACAO
   e o grupo central acabava anexado ao label do meio.

   Na V9, o gate é aplicado a cada grupo:
   - grupos laterais podem ser continuações;
   - grupo secundário central é bloqueado como provável título do eixo.

   A regra é deliberadamente conservadora: é preferível deixar uma possível
   continuação central para revisão do que incorporar silenciosamente um
   título do eixo à categoria.

4. GATES DA V8 PRESERVADOS
   - MULTI_EVIDENCE_FALLBACK não declara COMPLETE sozinho;
   - conflito BAR_PRIMARY x evidência textual gera AMBIGUOUS;
   - artefatos de borda geram revisão.

Independência
-------------
A V9 NÃO recebe:
- categorias esperadas;
- número esperado de categorias;
- ordem esperada;
- especificação F;
- dicionário derivado da referência.

Amostra
-------
As mesmas 24 imagens de calibração:
- 12 BI
- 12 BC

Nenhuma imagem do holdout independente é usada no desenvolvimento.

SHA-256 desta versão de calibração
----------------------------------
65d29b4c1aaad4e054922e37b0cb5496084520fd31b1977bc1d003d9fd2fc5e9

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v9") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v9" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v9.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v9") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v9" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v9.py

ARQUIVOS A ENVIAR
-----------------
1. ba_cat_v9_summary.txt
2. ba_cat_v9_images.csv
3. ba_cat_v9_slots.csv
4. ba_cat_v9_priority_review.csv
5. ba_cat_v9_errors.csv
6. contact_BA_CAT_V9_BI.png
7. contact_BA_CAT_V9_BC.png

Se ainda houver casos problemáticos:
8. ba_cat_v9_candidates.csv

O que queremos observar
-----------------------
BC:
- BC_006, BC_015 e BC_046 sem "SETOR CNAE" incorporado;
- BC_048 permanecendo AMBIGUOUS se o OCR inclinado continuar incompleto.

BI:
- BI_014 deve selecionar INFRA-ESTRUTURA pelo forte consenso;
- BI_006 deve continuar AMBIGUOUS se a estrutura não estiver coberta;
- BI_031 deve continuar em revisão se os artefatos de borda persistirem;
- preservar os demais BI já corretos.

A V9 ainda é CALIBRAÇÃO.
Somente após nova auditoria manual das 24 imagens devemos decidir se ela pode
ser congelada para o novo holdout independente.
