from pathlib import Path
import shutil
import csv
import sys

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES = ROOT / "imagens"
OUT = ROOT / "revisao_bb_ticks_7"

CASES = [
    ("BI_047_R09.png", "categorical", "X"),
    ("BI_051_R07.png", "categorical", "X"),
    ("LC_032_R03.png", "temporal", "X"),
    ("LI_045_R05.png", "numeric", "Y"),
    ("SI_032_R03.png", "numeric", "Y"),
    ("SI_006_R01.png", "numeric", "X"),
    ("SI_019_R05.png", "numeric", "Y"),
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    missing = []
    copied = []

    for filename, kind, axis in CASES:
        src = IMAGES / filename
        dst = OUT / filename
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, dst)
        copied.append((filename, kind, axis, str(src), str(dst)))

    manifest = OUT / "manifest_revisao_bb_ticks_7.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "kind", "axis", "source", "copied_to"])
        w.writerows(copied)

    print("=" * 72)
    print("REVISÃO B-B — 7 CASOS NÃO CONFORMES DO PILOTO")
    print("=" * 72)
    print(f"Pasta de origem: {IMAGES}")
    print(f"Pasta de revisão: {OUT}")
    print(f"Arquivos copiados: {len(copied)}/7")
    print(f"Manifesto: {manifest}")

    if missing:
        print("\nARQUIVOS NÃO ENCONTRADOS:")
        for p in missing:
            print(" -", p)
        sys.exit(1)

    print("\nTodos os 7 arquivos foram copiados.")
    print("Envie os 7 PNG da pasta de revisão para fazermos a adjudicação visual.")

if __name__ == "__main__":
    main()
