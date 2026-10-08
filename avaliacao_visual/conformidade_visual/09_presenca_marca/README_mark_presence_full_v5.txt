PROCESSAMENTO FINAL — PRESENÇA DA ESTRUTURA GRÁFICA PRINCIPAL

Arquivos necessários na mesma pasta:
1. mark_presence_full_v5.py
2. mark_presence_validation_v3.py

IMPORTANTE
----------
O programa verifica automaticamente se mark_presence_validation_v3.py é
exatamente a versão congelada validada.

SHA-256 esperado:
8f33f4daa7e8f1cc8feba58e5ea5f80bc75639334f9e64cf6339408152bc78a2

Coloque os dois arquivos em:
C:\Users\Labvis\Downloads\imagens3120

Execute:
python mark_presence_full_v5.py

Pasta de imagens:
C:\Users\Labvis\Downloads\imagens3120\imagens

Crosswalk esperado:
C:\Users\Labvis\Downloads\imagens3120\imagens\_experimental_crosswalk_v3

Saída:
C:\Users\Labvis\Downloads\imagens3120\imagens\_mark_presence_full_v5

Arquivos para enviar depois:
- mark_presence_summary_v5.txt
- mark_presence_images_v5.csv
- mark_presence_units_v5.csv
- mark_presence_by_profile_v5.csv
- mark_presence_by_condition_v5.csv
- mark_presence_errors_v5.csv

Opcional:
python mark_presence_full_v5.py --save-absent-overlays

Essa opção salva overlays apenas dos casos classificados como MARK_ABSENT,
úteis para auditoria visual posterior.
