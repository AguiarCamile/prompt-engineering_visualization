from pathlib import Path
import csv
import shutil

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES = ROOT / "imagens"
QUEUE = ROOT / "FILA_REVISAO_BA_CAT_PRODUCAO_280.csv"
OUT = ROOT / "revisao_ba_cat_producao_280"

def safe(s):
    return str(s).replace("/", "_").replace("\\", "_").strip()

def main():
    if not QUEUE.exists():
        print("ERRO: fila não encontrada:", QUEUE)
        return 1

    with QUEUE.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    copied = 0
    missing = []

    for r in rows:
        fn = r["filename"]
        profile = safe(r["profile"])
        status = safe(r["categorical_axis_status"])

        src = IMAGES / fn
        dst_dir = OUT / profile / status
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / fn

        if not src.exists():
            missing.append(str(src))
            continue

        shutil.copy2(src, dst)
        copied += 1

    print("=" * 72)
    print("REVISÃO B-A CATEGÓRICA — PRODUÇÃO")
    print("=" * 72)
    print("Linhas da fila:", len(rows))
    print("Imagens copiadas:", copied)
    print("Pasta:", OUT)

    if missing:
        print("\nARQUIVOS NÃO ENCONTRADOS:")
        for p in missing:
            print(" -", p)
        return 1

    print("\nOK — imagens separadas por perfil e status automático.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
