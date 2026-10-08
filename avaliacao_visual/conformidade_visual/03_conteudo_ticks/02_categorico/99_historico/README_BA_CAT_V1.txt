B-A-CAT V1 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
================================================

Objetivo
--------
Melhorar especificamente a extração de tick labels categóricos em BI/BC,
sem alterar a B-A V7 congelada e sem usar a especificação F.

O que esta versão tenta corrigir
--------------------------------
1. Barras horizontais, nas quais o eixo categórico observado pode ser Y.
2. Omissão de categorias quando uma caixa OCR individual não é detectada.
3. Corrupção de rótulos longos/inclinados.
4. Uso insuficiente da evidência da faixa completa.
5. Escolha prematura de um OCR categórico apenas por confiança local.

Princípio de independência
--------------------------
A B-A-CAT V1 NÃO recebe:
- categorias esperadas;
- quantidade esperada de categorias;
- ordem esperada;
- valores da especificação F;
- texto da figura de referência como vocabulário;
- qualquer dicionário construído a partir do resultado esperado.

A similaridade textual é usada apenas para agrupar leituras OCR independentes
produzidas pela MESMA imagem e pelo MESMO slot.

Amostra de desenvolvimento
--------------------------
O programa usa exclusivamente as 24 imagens BI/BC que já pertencem ao
manifesto de calibração da B-A V7:

- 12 BI
- 12 BC

O programa NÃO seleciona novas imagens do holdout independente.

Dependências congeladas
-----------------------
B-A V7:
SHA-256
21423b59ce40eb351693f00b6a197936d936fe53131467f47c1031da864fd688

V4 presença:
SHA-256
d15c325290c7596632f9f4c62d907e7a78d0a794862b690c4c4691ed87ea86b2

A execução é interrompida se esses hashes não coincidirem.

Arquivos necessários na mesma pasta do programa
------------------------------------------------
ticklabel_categorical_extraction_calibration_ba_cat_v1.py
ticklabel_content_extraction_calibration_ba_v7.py
ticklabel_presence_calibration_v4.py
axis_conformity_calibration_v3.py
visual_features_v4_plot_area_v6.py

Também devem continuar disponíveis os arquivos e diretórios usados pela B-A V7.

Ambiente
--------
Ative o mesmo .venv utilizado na B-A V7.

Bibliotecas usadas diretamente:
- numpy
- pandas
- Pillow
- opencv-python
- pytesseract

O Tesseract deve continuar configurado como na B-A V7.

Execução
--------
Com o .venv ativo:

python ticklabel_categorical_extraction_calibration_ba_cat_v1.py

Raiz padrão
-----------
C:\Users\Labvis\Downloads\imagens3120\imagens

Para outra raiz:

python ticklabel_categorical_extraction_calibration_ba_cat_v1.py --root "C:\caminho\imagens"

Para informar explicitamente o Tesseract:

python ticklabel_categorical_extraction_calibration_ba_cat_v1.py --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"

Saída
-----
Por padrão:

C:\Users\Labvis\Downloads\imagens3120\imagens\
_ticklabel_categorical_extraction_ba_cat_v1

Arquivos principais
-------------------
ba_cat_v1_manifest.csv
    As 24 imagens BI/BC de desenvolvimento.

ba_cat_v1_images.csv
    Resultado agregado por imagem:
    - orientação observada;
    - eixo categórico observado;
    - escores de evidência;
    - status categórico;
    - sequência final dos slots.

ba_cat_v1_slots.csv
    Resultado por slot categórico:
    - posição;
    - bbox;
    - estado;
    - texto selecionado;
    - confiança;
    - suporte do consenso.

ba_cat_v1_candidates.csv
    Todas as leituras OCR candidatas por slot.
    Este arquivo é importante para depuração e calibração.

ba_cat_v1_audit_all.csv
    Planilha para auditoria manual das 24 imagens.

ba_cat_v1_priority_review.csv
    Casos automáticos que merecem revisão prioritária.

ba_cat_v1_errors.csv
    Erros de execução.

ba_cat_v1_summary.txt
    Resumo automático da execução.

contact_BA_CAT_V1_BI.png
contact_BA_CAT_V1_BC.png
    Pranchas de contato.

overlays\
    Overlays individuais mostrando plot, categorical lane e slots.

Como interpretar os estados
----------------------------
Orientação:

VERTICAL
    Evidência favorece barras verticais.
    Eixo categórico observado = X.

HORIZONTAL
    Evidência favorece barras horizontais.
    Eixo categórico observado = Y.

AMBIGUOUS
    Evidência insuficiente para decidir.
    O programa não força X ou Y.

Slot categórico:

CONSENSUS
    Múltiplas leituras independentes convergiram.

SINGLE_HIGH_CONF
    Uma leitura isolada tem evidência forte suficiente.

AMBIGUOUS
    Há duas ou mais interpretações concorrentes sem margem suficiente.

UNREADABLE
    Não há evidência adequada para selecionar texto.

Eixo categórico:

CATEGORICAL_COMPLETE
    Todos os slots foram selecionados.

CATEGORICAL_PARTIAL
    Pelo menos um slot foi recuperado e pelo menos um ficou UNREADABLE.

CATEGORICAL_AMBIGUOUS
    Existe pelo menos um slot ambíguo.

CATEGORICAL_FAILED
    Não houve recuperação categórica suficiente.

ORIENTATION_AMBIGUOUS
    O próprio eixo categórico observado não pôde ser definido.

Auditoria
---------
Nesta etapa, os estados automáticos NÃO são acurácia.

Auditar as 24 imagens e registrar, no mínimo:

manual_visible_categories
manual_orientation_result
manual_extraction_result
manual_missing_labels
manual_false_labels
manual_notes

Categorias sugeridas para manual_extraction_result:

EXACT
    Conteúdo visível recuperado integralmente.

EQUIVALENT
    Somente normalização benigna, sem mudança de conteúdo.

PARTIAL
    Parte das categorias visíveis foi recuperada e houve omissão.

WRONG
    Houve conteúdo categórico incorreto ou falso label.

UNEVALUABLE
    Não é possível adjudicar visualmente com segurança.

Regra de desenvolvimento
------------------------
Esta é uma versão de CALIBRAÇÃO.

Os parâmetros podem ser modificados depois de examinar estas 24 imagens,
porque elas já pertencem à amostra de desenvolvimento da B-A V7.

NÃO usar o holdout independente anterior para ajustar parâmetros.

Depois de estabilizar a CAT V1:
1. congelar o arquivo;
2. calcular e registrar o SHA-256;
3. definir os critérios de sucesso antes do novo teste;
4. selecionar novas unidades BI/BC nunca usadas;
5. executar novo holdout independente;
6. não reajustar a versão depois de observar esse holdout.

Arquivos a enviar após a execução
---------------------------------
Prioridade:

1. ba_cat_v1_summary.txt
2. ba_cat_v1_audit_all.csv
3. ba_cat_v1_priority_review.csv
4. ba_cat_v1_slots.csv
5. ba_cat_v1_errors.csv
6. contact_BA_CAT_V1_BI.png
7. contact_BA_CAT_V1_BC.png

Se houver casos duvidosos, enviar também:
- ba_cat_v1_candidates.csv
- overlays individuais correspondentes.
