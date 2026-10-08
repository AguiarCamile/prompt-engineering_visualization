from pathlib import Path
import csv
import shutil
import sys

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES = ROOT / "imagens"
QUEUE = ROOT / "FILA_REVISAO_BA_V7_PRODUCAO_88.csv"
OUT = ROOT / "revisao_ba_v7_producao_88"

def main():
    if not QUEUE.exists():
        print("ERRO: fila não encontrada:", QUEUE)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)

    with QUEUE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    filenames = sorted({r["filename"] for r in rows if r.get("filename")})

    copied = 0
    missing = []

    for fn in filenames:
        src = IMAGES / fn
        dst = OUT / fn
        if not src.exists():
            missing.append(str(src))
            continue
        shutil.copy2(src, dst)
        copied += 1

    print("=" * 72)
    print("REVISÃO B-A V7 — PRODUÇÃO")
    print("=" * 72)
    print("Eixos na fila:", len(rows))
    print("Imagens únicas:", len(filenames))
    print("Imagens copiadas:", copied)
    print("Pasta:", OUT)

    if missing:
        print("\nARQUIVOS NÃO ENCONTRADOS:")
        for p in missing:
            print(" -", p)
        return 1

    print("\nOK — pacote visual preparado.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
