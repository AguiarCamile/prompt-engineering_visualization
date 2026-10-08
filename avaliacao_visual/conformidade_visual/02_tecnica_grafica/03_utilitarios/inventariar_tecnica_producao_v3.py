# -*- coding: utf-8 -*-
"""
INVENTÁRIO DIRECIONADO DA TÉCNICA GRÁFICA — V3
===============================================

Procura, em todos os CSVs do projeto, arquivos que possam conter classificação
de técnica gráfica por imagem. Diferentemente do inventário anterior, esta
versão avalia:
- número de linhas;
- número de filenames únicos;
- colunas típicas de classificação de técnica;
- valores BAR / LINE / SCATTER / STEM / OTHER;
- presença de campos expected/observed/candidate/manual.

Saídas:
  CANDIDATOS_TECNICA_PRODUCAO_V3.csv
  CANDIDATOS_TECNICA_PRODUCAO_V3_RESUMO.txt
"""

from __future__ import annotations
import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
OUT = ROOT / "CANDIDATOS_TECNICA_PRODUCAO_V3.csv"
SUMMARY = ROOT / "CANDIDATOS_TECNICA_PRODUCAO_V3_RESUMO.txt"

STRONG_COLS = {
    "observed_type", "candidate_type", "manual_type", "expected_type",
    "expected_technique", "observed_technique", "technique_final",
    "technique_status", "chart_type_final", "predicted_technique",
    "detected_technique", "classification_final", "matches_expected_type",
    "expected_structure_present", "exact_expected_only",
    "bar_present", "line_present", "scatter_present", "stem_present",
    "observed_signature", "expected_signature",
}
TECH_TOKENS = {"BAR","BARS","LINE","LINES","SCATTER","STEM","OTHER"}

def analyze_csv(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            low = [str(x).strip().lower() for x in header]
            strong_hits = sorted(STRONG_COLS.intersection(low))

            n_rows = 0
            filenames = set()
            profiles = set()
            token_counts = Counter()
            sample_class_values = set()

            class_cols = [
                c for c in header
                if str(c).strip().lower() in STRONG_COLS
                or any(k in str(c).strip().lower()
                       for k in ("techni","type","signature","bar_present",
                                 "line_present","scatter_present","stem"))
            ]

            for row in reader:
                n_rows += 1
                fn = str(row.get("filename","")).strip()
                if fn:
                    filenames.add(Path(fn).name)
                pr = str(row.get("profile","")).strip()
                if pr:
                    profiles.add(pr)

                for c in class_cols:
                    v = str(row.get(c,"")).strip()
                    if not v:
                        continue
                    up = v.upper()
                    for tok in TECH_TOKENS:
                        if re.search(rf"\b{tok}\b", up):
                            token_counts[tok] += 1
                    if len(sample_class_values) < 20:
                        sample_class_values.add(f"{c}={v}")

            score = 0
            reasons = []

            if strong_hits:
                score += 10 + len(strong_hits)
                reasons.append("strong_columns=" + ",".join(strong_hits))

            if n_rows >= 3000:
                score += 12
                reasons.append("rows>=3000")
            elif n_rows >= 1000:
                score += 8
                reasons.append("rows>=1000")
            elif n_rows >= 500:
                score += 5
                reasons.append("rows>=500")

            if len(filenames) >= 3000:
                score += 15
                reasons.append("unique_filename>=3000")
            elif len(filenames) >= 1000:
                score += 8
                reasons.append("unique_filename>=1000")
            elif len(filenames) >= 500:
                score += 4
                reasons.append("unique_filename>=500")

            if token_counts:
                score += 5
                reasons.append("technique_tokens")

            name = path.name.lower()
            if re.search(r"chart.?type|techni|classif|multiclass|exclusive|stem", name):
                score += 3
                reasons.append("filename_hint")

            return {
                "score": score,
                "path": str(path),
                "filename": path.name,
                "n_rows": n_rows,
                "n_unique_filename": len(filenames),
                "profiles": "|".join(sorted(profiles)),
                "strong_columns": "|".join(strong_hits),
                "technique_tokens": "|".join(
                    f"{k}:{v}" for k,v in sorted(token_counts.items())
                ),
                "sample_class_values": " || ".join(sorted(sample_class_values)),
                "columns": "|".join(header),
                "reason": "|".join(reasons),
            }
    except Exception:
        return None

def main():
    candidates = []
    for p in ROOT.rglob("*.csv"):
        r = analyze_csv(p)
        if not r:
            continue

        # Mantém resultados com algum sinal real de técnica.
        if (
            int(r["score"]) >= 8
            or r["strong_columns"]
            or r["technique_tokens"]
        ):
            candidates.append(r)

    candidates.sort(
        key=lambda x: (
            -int(x["score"]),
            -int(x["n_unique_filename"]),
            -int(x["n_rows"]),
            x["path"]
        )
    )

    fields = [
        "score","path","filename","n_rows","n_unique_filename","profiles",
        "strong_columns","technique_tokens","sample_class_values",
        "reason","columns"
    ]
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(candidates)

    lines = []
    lines.append("="*80)
    lines.append("INVENTÁRIO DIRECIONADO DA TÉCNICA — V3")
    lines.append("="*80)
    lines.append(f"Candidatos encontrados: {len(candidates)}")
    lines.append("")
    for i,r in enumerate(candidates[:30], 1):
        lines.append(
            f'{i:02d}. score={r["score"]} '
            f'rows={r["n_rows"]} unique={r["n_unique_filename"]} '
            f'{r["path"]}'
        )
        if r["strong_columns"]:
            lines.append("    colunas: " + r["strong_columns"])
        if r["technique_tokens"]:
            lines.append("    tokens : " + r["technique_tokens"])
    lines.append("")
    lines.append(f"CSV: {OUT}")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
