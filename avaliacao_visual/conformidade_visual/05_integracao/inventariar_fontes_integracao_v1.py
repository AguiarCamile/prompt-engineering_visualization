from pathlib import Path
import csv

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
OUT = ROOT / "INVENTARIO_FONTES_INTEGRACAO_V1.csv"

def classify(path, header, samples):
    h = {x.strip().lower() for x in header if x}
    vals = " ".join(" ".join(str(v) for v in row.values()) for row in samples[:50]).upper()
    name = path.name.lower()

    kinds = []
    if "filename" in h and "color_status_final" in h:
        kinds.append("COLOR_FINAL")
    if "filename" in h and (
        any(x in h for x in ["mark_status","mark_presence_status","mark_present","main_mark_status"])
        or "MARK_PRESENT" in vals or "MARK_ABSENT" in vals
    ):
        kinds.append("MARK_PRESENCE")
    if "filename" in h and (
        any(x in h for x in ["axis_presence_status","axis_status_final","x_present_final","y_present_final"])
        or "BOTH_PRESENT" in vals or "X_ONLY" in vals or "Y_ONLY" in vals
    ):
        kinds.append("AXIS_PRESENCE")
    if "filename" in h and (
        "manual_x_present_final" in h or "manual_y_present_final" in h
        or ("ticklabel" in name and "presence" in name)
    ):
        kinds.append("TICKLABEL_PRESENCE")
    if "filename" in h and (
        "layout_error_mean_vs_f" in h or "plot_area_status" in h
        or ("plot_area" in name and "visual_features" in name)
    ):
        kinds.append("PLOT_AREA")
    if "filename" in h and (
        any(x in h for x in ["technique_status","technique_final","chart_type_final","observed_technique"])
        or ("technique" in name and "summary" not in name)
    ):
        kinds.append("TECHNIQUE")
    if "filename" in h and "bb_collapsed_status" in h and ("axis" in h or "source_family" in h):
        kinds.append("TICK_CONTENT")
    return kinds

rows_out = []
for p in ROOT.rglob("*.csv"):
    try:
        with p.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            samples = []
            for i, row in enumerate(reader):
                samples.append(row)
                if i >= 49:
                    break
        kinds = classify(p, header, samples)
        if kinds:
            rows_out.append({
                "source_type":"|".join(kinds),
                "path":str(p),
                "filename":p.name,
                "columns":"|".join(header),
            })
    except Exception as e:
        pass

rows_out.sort(key=lambda r: (r["source_type"], r["path"]))

with OUT.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["source_type","path","filename","columns"])
    w.writeheader()
    w.writerows(rows_out)

print("="*80)
print("INVENTÁRIO DAS FONTES PARA INTEGRAÇÃO")
print("="*80)
print("Candidatos encontrados:", len(rows_out))
for r in rows_out:
    print(f'{r["source_type"]:28s}  {r["path"]}')
print()
print("Saída:", OUT)
