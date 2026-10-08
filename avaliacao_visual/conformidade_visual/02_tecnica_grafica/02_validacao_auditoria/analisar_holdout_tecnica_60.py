# -*- coding: utf-8 -*-
r"""
ANALISAR REVISÃO CEGA DO HOLDOUT DE TÉCNICA — V1
=================================================

Compara a classificação visual manual com observed_type produzido pelo
classificador exclusivo V2.

Não usa expected_type para calcular a qualidade do classificador.

Mapeamento para o classificador de 4 classes:
- BAR -> BAR
- LINE -> LINE
- SCATTER -> SCATTER
- OTHER -> OTHER
- EMPTY_NO_DATA -> OTHER
- UNCERTAIN -> UNEVALUABLE

Saídas:
- VALIDACAO_HOLDOUT_TECNICA_60_FINAL.csv
- VALIDACAO_HOLDOUT_TECNICA_60_MATRIZ.csv
- VALIDACAO_HOLDOUT_TECNICA_60_RESUMO.txt
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
FINAL = ROOT / "REVISAO_MANUAL_HOLDOUT_TECNICA_60_FINAL.csv"
OUT = ROOT / "VALIDACAO_HOLDOUT_TECNICA_60_FINAL.csv"
MATRIX = ROOT / "VALIDACAO_HOLDOUT_TECNICA_60_MATRIZ.csv"
SUMMARY = ROOT / "VALIDACAO_HOLDOUT_TECNICA_60_RESUMO.txt"

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def manual_to_four_class(v):
    v = str(v or "").strip().upper()
    if v in {"BAR", "LINE", "SCATTER", "OTHER"}:
        return v
    if v == "EMPTY_NO_DATA":
        return "OTHER"
    if v == "UNCERTAIN":
        return "UNEVALUABLE"
    return "UNEVALUABLE"

def main():
    if not FINAL.exists():
        print("ERRO: arquivo final não encontrado:")
        print(FINAL)
        return 1

    rows = read_csv(FINAL)

    out_rows = []
    matrix = defaultdict(Counter)

    for r in rows:
        manual_raw = r.get("manual_visible_type", "")
        manual4 = manual_to_four_class(manual_raw)
        auto = str(r.get("observed_type", "")).strip().upper()

        if manual4 == "UNEVALUABLE":
            status = "UNEVALUABLE"
        else:
            status = "MATCH" if auto == manual4 else "MISMATCH"
            matrix[manual4][auto] += 1

        out = dict(r)
        out["manual_four_class"] = manual4
        out["classifier_validation_status"] = status
        out_rows.append(out)

    fields = list(out_rows[0].keys())
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    classes = ["BAR", "LINE", "SCATTER", "OTHER"]
    with MATRIX.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["manual\\auto"] + classes + ["TOTAL"])
        for m in classes:
            vals = [matrix[m][a] for a in classes]
            w.writerow([m] + vals + [sum(vals)])

    counts = Counter(
        r["classifier_validation_status"]
        for r in out_rows
    )

    evaluable = counts["MATCH"] + counts["MISMATCH"]
    accuracy = counts["MATCH"] / evaluable if evaluable else 0.0

    lines = [
        "VALIDAÇÃO CEGA — CLASSIFICADOR EXCLUSIVO DE TÉCNICA",
        "=" * 72,
        f"Casos totais: {len(out_rows)}",
        f"Avaliáveis: {evaluable}",
        f"MATCH: {counts['MATCH']}",
        f"MISMATCH: {counts['MISMATCH']}",
        f"UNEVALUABLE: {counts['UNEVALUABLE']}",
        f"Acurácia visual entre avaliáveis: {accuracy:.2%}",
        "",
        "IMPORTANTE:",
        "- esta acurácia compara observed_type com julgamento visual manual;",
        "- expected_type NÃO entra nessa medida;",
        "- o resultado deve ser usado para decidir se as regras V2 podem ser",
        "  congeladas para produção sem recalibração.",
        "",
        f"Detalhado: {OUT}",
        f"Matriz: {MATRIX}",
    ]

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
