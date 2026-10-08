SEPARAR PROMPTS — imagens3120
================================

Programa:
separar_prompts_imagens3120.py

SHA-256:
07e4112fc45c680ccd551e359a56ee5e32ef4ae8427a868a001624893bb43662

Entrada padrão:
C:\Users\Labvis\Downloads\imagens3120\prompts_imagens.jsonl

Saída padrão:
C:\Users\Labvis\Downloads\imagens3120\prompts

Saídas:
- 312 arquivos TXT: BI_001.txt ... SC_052.txt
- prompts_manifest.csv
- resumo_prompts.txt

Cada TXT contém SOMENTE o prompt_text original.

PowerShell:
cd C:\Users\Labvis\Downloads\imagens3120
.\.venv\Scripts\Activate.ps1
python separar_prompts_imagens3120.py

Para sobrescrever arquivos existentes com conteúdo diferente:
python separar_prompts_imagens3120.py --overwrite
