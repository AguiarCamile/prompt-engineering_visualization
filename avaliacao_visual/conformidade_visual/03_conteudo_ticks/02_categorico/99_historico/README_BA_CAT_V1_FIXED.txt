B-A-CAT V1 — BUILD CORRIGIDO
================================

Correções após a primeira execução
----------------------------------
1. Saneamento de bounding boxes:
   - corrige ordem invertida;
   - limita coordenadas à imagem;
   - remove NaN/inf;
   - impede largura/altura <= 0.

2. Slots:
   - descarta centros fora da categorical lane;
   - deduplica centros muito próximos;
   - não cria slots inválidos.

3. PIL:
   - converte coordenadas para int Python antes de crop/draw.

4. Contact sheet:
   - usa renderizador próprio da B-A-CAT V1;
   - não chama make_contact_sheet da B-A V7, que exige x_kind/y_kind.

Essas correções são de implementação. Elas não alteram:
- a B-A V7 congelada;
- a V4 congelada;
- a amostra de desenvolvimento;
- a independência em relação à especificação F.

SHA-256 do build corrigido
--------------------------
b9e82f88d2f6de30004333b191819fd744a73e0c649139152ea9d03a09edf53f

COMANDO PARA RODAR
------------------
PowerShell:

cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v1") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v1" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v1_fixed.py

Se o .venv já estiver ativo:

if (Test-Path ".\imagens\_ticklabel_categorical_extraction_ba_cat_v1") { Remove-Item ".\imagens\_ticklabel_categorical_extraction_ba_cat_v1" -Recurse -Force }
python ticklabel_categorical_extraction_calibration_ba_cat_v1_fixed.py
