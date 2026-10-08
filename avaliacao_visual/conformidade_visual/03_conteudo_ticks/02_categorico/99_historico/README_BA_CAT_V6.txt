B-A-CAT V6 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
=================================================

Diagnóstico da V5
-----------------
A V5 foi a primeira versão em que a estrutura categórica ficou amplamente
estável:

- 24/24 imagens processadas;
- 0 erros;
- 24/24 orientações VERTICAL;
- 20 CATEGORICAL_COMPLETE;
- 4 CATEGORICAL_AMBIGUOUS;
- 4 casos em revisão prioritária.

A inspeção visual mostra, porém, que o problema residual é principalmente
OCR DE CONTEÚDO, e não mais orientação/estrutura.

Exemplos:
- BC_006: "AGROPECUARIA" em vez de "AGROPECUARIA E PESCA";
- BC_015: "INDUSTRIA E" em vez do label multilinha completo;
- BC_046: "INDUSTRIA DE" truncado;
- BC_048: labels inclinados ainda difíceis;
- BI_031: primeira/última letra cortada em alguns slots;
- BI_014/BI_027: slots estruturalmente encontrados, mas OCR muito fraco.

Mudança central da V6
---------------------
A V6 preserva a estrutura estabilizada da V5 e atua especificamente na leitura.

1. LANE ESTRUTURAL ≠ LANE OCR
   A lane estreita continua definindo posição dos slots.
   Uma segunda lane, mais alta, é usada somente para OCR.

2. LABELS MULTILINHA
   A lane OCR inclui espaço vertical adicional para recuperar:
       AGROPECUARIA
       E PESCA
   ou:
       INDUSTRIA DE
       TRANSFORMACAO

3. FOLGA LATERAL
   O crop OCR de cada slot recebe pequena expansão lateral para reduzir
   cortes como:
       NFRA-ESTRUTURA
       OMERCIO/SERVICOS
       GROPECUARIA

4. ROW_PHRASE DEIXA DE DOMINAR
   A frase de uma única linha continua como evidência auxiliar, mas com
   pontuação menor. Ela não deve vencer automaticamente uma leitura
   multilinha mais completa.

5. ÂNGULOS ±15°
   As rotações OCR passam a incluir -15° e +15°, além das já existentes,
   para casos como BC_048 com labels inclinados.

6. GATE CONTRA FALSO COMPLETE
   Se um slot selecionado for lexicalmente muito fraco, um eixo
   estruturalmente COMPLETE é rebaixado para CATEGORICAL_AMBIGUOUS.
   O gate não conhece F; apenas evita aceitar silenciosamente strings
   extremamente curtas.

Independência
-------------
A V6 NÃO recebe:
- categorias esperadas;
- número esperado de categorias;
- ordem esperada;
- especificação F;
- dicionário derivado da figura de referência.

Amostra
-------
As mesmas 24 imagens de calibração:
- 12 BI;
- 12 BC.

Nenhuma imagem do holdout independente anterior participa do ajuste.

SHA-256 desta versão de calibração
----------------------------------
7eec3dd65e68fe2a8ed6b09a370cd46bb4119b589e7a8061c3207cc2d9cb1e81

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v6") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v6" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v6.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v6") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v6" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v6.py

ARQUIVOS A ENVIAR
-----------------
1. ba_cat_v6_summary.txt
2. ba_cat_v6_images.csv
3. ba_cat_v6_slots.csv
4. ba_cat_v6_priority_review.csv
5. ba_cat_v6_errors.csv
6. contact_BA_CAT_V6_BI.png
7. contact_BA_CAT_V6_BC.png

Se ainda houver casos problemáticos:
8. ba_cat_v6_candidates.csv

O que avaliar
-------------
- BC_006: recuperação de "E PESCA" e "DE TRANSFORMACAO";
- BC_015 e BC_046: redução dos truncamentos de segunda linha;
- BC_048: melhoria com ±15°;
- BI_031: desaparecimento dos cortes de primeira/última letra;
- BI_014 e BI_027: melhoria do OCR sem alterar os quatro slots;
- BI_006: pode continuar sendo caso estrutural difícil e deve permanecer
  explicitamente AMBIGUOUS/PARTIAL se não houver evidência suficiente.

A V6 continua sendo CALIBRAÇÃO. Só deve ser congelada após auditoria manual
das 24 imagens.
