# -*- coding: utf-8 -*-
r"""
INVENTÁRIO DE SCRIPTS DA TÉCNICA GRÁFICA — V1
==============================================

Procura scripts Python relacionados ao classificador de técnica gráfica,
especialmente os que geraram:
- chart_type_exclusive_holdout_v2.csv
- chart_type_multiclass_validation_v1.csv
- absent_chart_type_candidates_v1.csv

Saídas:
  SCRIPTS_TECNICA_PRODUCAO_V1.csv
  SCRIPTS_TECNICA_PRODUCAO_V1_RESUMO.txt

O script NÃO executa classificadores e NÃO modifica resultados.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
OUT = ROOT / "SCRIPTS_TECNICA_PRODUCAO_V1.csv"
SUMMARY = ROOT / "SCRIPTS_TECNICA_PRODUCAO_V1_RESUMO.txt"

TERMS = [
    "chart_type_exclusive_holdout",
    "chart_type_multiclass_validation",
    "absent_chart_type_candidates",
    "observed_type",
    "expected_type",
    "expected_structure_present",
    "exact_expected_only",
    "observed_signature",
    "bar_present",
    "line_present",
    "scatter_present",
    "stem",
    "classify",
    "classification",
    "technique",
]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def snippets(text, terms, radius=180):
    found = []
    low = text.lower()
    for term in terms:
        start = 0
        while True:
            i = low.find(term.lower(), start)
            if i < 0:
                break
            a = max(0, i-radius)
            b = min(len(text), i+len(term)+radius)
            sn = re.sub(r"\s+", " ", text[a:b]).strip()
            found.append(f"{term}: {sn}")
            start = i + len(term)
            if len(found) >= 8:
                return found
    return found

def main():
    rows = []

    for p in ROOT.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        low = text.lower()
        hits = [t for t in TERMS if t.lower() in low]
        name_hint = any(
            k in p.name.lower()
            for k in ("chart_type","techni","classif","multiclass","exclusive","absent")
        )

        if not hits and not name_hint:
            continue

        score = len(hits) * 2 + (5 if name_hint else 0)

        # Bônus para sinais fortes de que o script produz classificação.
        strong = []
        for t in [
            "observed_type",
            "expected_type",
            "observed_signature",
            "expected_structure_present",
            "bar_present",
            "line_present",
            "scatter_present",
        ]:
            if t in low:
                strong.append(t)
                score += 3

        rows.append({
            "score": score,
            "path": str(p),
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "sha256": sha256(p),
            "hits": "|".join(hits),
            "strong_signals": "|".join(strong),
            "snippets": " || ".join(snippets(text, hits)),
        })

    rows.sort(key=lambda r: (-int(r["score"]), r["path"]))

    fields = [
        "score","path","filename","size_bytes","sha256",
        "hits","strong_signals","snippets"
    ]
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "="*80,
        "INVENTÁRIO DE SCRIPTS DA TÉCNICA GRÁFICA — V1",
        "="*80,
        f"Scripts candidatos: {len(rows)}",
        "",
    ]
    for i,r in enumerate(rows[:25],1):
        lines.append(
            f'{i:02d}. score={r["score"]} {r["path"]}'
        )
        lines.append(
            f'    sinais fortes: {r["strong_signals"] or "(nenhum)"}'
        )
    lines += ["", f"CSV: {OUT}"]

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
