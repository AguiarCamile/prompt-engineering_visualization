B-B TICK CONTENT — PRODUÇÃO 3.120
=================================

Arquivos
--------
1. preflight_producao_ticks_3120.py
   Verifica o conjunto de 3.120 imagens na RAIZ e cria um manifesto local.

2. bb_tick_content_production_v1.py
   Executa a consolidação B-B usando:
   - F_MASTER_V4_TICK_SCOPE_ADJUDICATED.xlsx
   - B_B_NORMALIZATION_V1.json
   - um CSV final do B-A categórico em produção
   - um CSV final do B-A V7 em produção
   - o CSV de eixos do B-A V7

Pré-requisitos
--------------
No diretório:
C:\Users\Labvis\Downloads\imagens3120

Devem estar presentes:
- imagens\*.png  (3.120 arquivos na RAIZ)
- prompts_manifest.csv
- F_MASTER_V4_TICK_SCOPE_ADJUDICATED.xlsx
- B_B_NORMALIZATION_V1.json

Além disso, antes da B-B, devem existir os resultados B-A em produção:
- BA_CATEGORICAL_PRODUCTION_CONSOLIDATED.csv
- ticklabel_ba_v7_production_final.csv
- ticklabel_ba_v7_production_axes.csv

Comandos
--------
No Prompt de Comando:

cd /d C:\Users\Labvis\Downloads\imagens3120
.venv\Scripts\activate.bat

1) Prefight
-----------
python preflight_producao_ticks_3120.py

2) Consolidação B-B
-------------------
python bb_tick_content_production_v1.py ^
  --f-master F_MASTER_V4_TICK_SCOPE_ADJUDICATED.xlsx ^
  --normalization B_B_NORMALIZATION_V1.json ^
  --ba-cat BA_CATEGORICAL_PRODUCTION_CONSOLIDATED.csv ^
  --ba-v7-final ticklabel_ba_v7_production_final.csv ^
  --ba-v7-axes ticklabel_ba_v7_production_axes.csv

Saídas esperadas
----------------
Na pasta:
C:\Users\Labvis\Downloads\imagens3120\_producao_ticks_3120_v1\bb_tick_content_production_v1

Arquivos:
- bb_tick_content_production_v1.csv
- bb_tick_content_production_summary_by_kind.csv
- bb_tick_content_production_summary_by_profile_axis.csv
- bb_tick_content_production_cases_review.csv
- bb_tick_content_production_summary.txt
- bb_tick_content_production_v1.xlsx  (se openpyxl estiver disponível)

Observações metodológicas
-------------------------
- Não usar imagens para adjudicar F.
- Não alterar B_B_NORMALIZATION_V1.
- EXAMPLES_ONLY e NOT_SPECIFIED ficam fora do denominador de completude.
- O denominador de B-B exige F EXHAUSTIVE pronta e B-A utilizável.
- Rótulos de pontos não substituem ticks dos eixos.
- Diferenças de unidade só são equivalentes quando a unidade está visualmente
  presente e a normalização congelada as permite.
