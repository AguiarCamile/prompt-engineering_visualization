B-A-CAT V8 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
=================================================

Diagnóstico da V7
-----------------
A V7 produziu o melhor diagnóstico automático até aqui:

- 24/24 imagens processadas;
- 0 erros;
- 24/24 orientações VERTICAL;
- 23 CATEGORICAL_COMPLETE;
- 1 CATEGORICAL_AMBIGUOUS.

Porém, a auditoria visual mostra que 23/24 COMPLETE NÃO equivale a 23/24
extrações corretas.

Casos residuais importantes observados:
- BC_006: "AGROPECUARIA PESCA", "COMERCIO E SERVICOS SETOR CNAE",
          "INDUSTRIA TRANSFORMACAO";
- BC_015: "COMERCIO E SERVICOS (SETOR CNAE)";
- BC_046: "COMERCIO E SERVICOS SETOR CNAE";
- BC_048: segundo/terceiro labels ainda fundidos ou truncados;
- BI_006: apenas 2 slots em uma estrutura visual maior;
- BI_014: primeiro slot correto aparece repetidamente nos candidatos,
          mas o eixo ficou AMBIGUOUS;
- BI_031: artefatos de borda, como "INFRA-ESTRUTURA C" e
          "-OMERCIO/SERVICOS".

Portanto, a V7 ainda NÃO deve ser congelada.

Mudanças da V8
--------------
1. CONTINUATION LINE MAIS ESTRITA
   A tolerância vertical para agrupar palavras em uma mesma linha foi reduzida.
   Objetivo: separar melhor segunda linha de categoria e título do eixo.

2. ZONA CENTRAL DE PROVÁVEL AXIS TITLE MAIS CONSERVADORA
   Uma continuação inferior central isolada é bloqueada numa região central
   maior.

3. RE-OCR DA CONTINUAÇÃO
   Depois que uma linha inferior é aprovada geometricamente, a V8 relê
   somente a estreita faixa vertical da continuação dentro do slot.

   Isso tenta recuperar, por exemplo:
       E PESCA
       DE TRANSFORMACAO

   sem abrir verticalmente todos os slots.

4. HIGH-CONSENSUS OVERRIDE
   Se uma leitura tem muitas repetições independentes, confiança alta e o
   segundo cluster tem suporte muito menor, ela pode ser aceita mesmo com
   margem de score pequena.

   Objetivo principal: casos como BI_014.

5. GATE MULTI_EVIDENCE_FALLBACK
   Um eixo baseado somente no fallback não pode ser automaticamente COMPLETE.
   Evita falso COMPLETE como BI_006.

6. GATE BAR_PRIMARY × ROW
   Se BAR_PRIMARY produz mais slots do que uma evidência textual razoavelmente
   forte consegue sustentar, o eixo vira AMBIGUOUS em vez de COMPLETE.
   Objetivo: casos como BC_048.

7. GATE DE ARTEFATO DE BORDA
   Strings com sinais típicos de corte/bleed, como pontuação inicial ou uma
   letra isolada no fim de frase longa, são encaminhadas à revisão.

   A string NÃO é corrigida automaticamente.

Independência
-------------
A V8 NÃO recebe:
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

O holdout independente anterior continua fora do desenvolvimento.

SHA-256 desta versão de calibração
----------------------------------
d4a21417df72e65b4a2d7b6cf9cec6871e0a769a76ab86fec230fa1c29f49949

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v8") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v8" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v8.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v8") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v8" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v8.py

ARQUIVOS A ENVIAR
-----------------
1. ba_cat_v8_summary.txt
2. ba_cat_v8_images.csv
3. ba_cat_v8_slots.csv
4. ba_cat_v8_priority_review.csv
5. ba_cat_v8_errors.csv
6. contact_BA_CAT_V8_BI.png
7. contact_BA_CAT_V8_BC.png

Se ainda houver casos problemáticos:
8. ba_cat_v8_candidates.csv

O que avaliar
-------------
BC:
- BC_006/015/046: eliminar contaminação por axis title sem perder
  E PESCA / DE TRANSFORMACAO;
- BC_048: deve melhorar ou, no mínimo, deixar de ser falso COMPLETE.

BI:
- BI_006: deve deixar de ser falso COMPLETE;
- BI_014: verificar se o forte consenso em INFRA-ESTRUTURA resolve a ambiguidade;
- BI_031: deve ir para revisão se o artefato de borda persistir;
- preservar os demais BI que já estavam corretos.

A V8 ainda é CALIBRAÇÃO.
Somente depois da auditoria manual das 24 imagens devemos decidir se há base
para congelamento e novo holdout independente.
