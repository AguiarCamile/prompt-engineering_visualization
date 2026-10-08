# -*- coding: utf-8 -*-
r"""
INVENTÁRIO DE IMPLEMENTAÇÕES DA TÉCNICA GRÁFICA — V2
=====================================================

Procura explicitamente:
1. em todo o projeto:
   C:\Users\Labvis\Downloads\imagens3120

2. com prioridade na subpasta:
   C:\Users\Labvis\Downloads\imagens3120\implementações

Extensões verificadas:
- .py
- .ipynb
- .qmd
- .md
- .txt

Busca sinais relacionados ao classificador de técnica gráfica:
BAR / LINE / SCATTER / STEM, observed_type, expected_type,
observed_signature, chart_type, technique, classification etc.

Saídas:
- SCRIPTS_TECNICA_PRODUCAO_V2.csv
- SCRIPTS_TECNICA_PRODUCAO_V2_RESUMO.txt
- IMPLEMENTACOES_TECNICA_V2.csv

O script NÃO executa nenhum classificador.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
OUT = ROOT / "SCRIPTS_TECNICA_PRODUCAO_V2.csv"
SUMMARY = ROOT / "SCRIPTS_TECNICA_PRODUCAO_V2_RESUMO.txt"
IMPL_OUT = ROOT / "IMPLEMENTACOES_TECNICA_V2.csv"

EXTS = {".py", ".ipynb", ".qmd", ".md", ".txt"}

TERMS = [
    "chart_type_exclusive_holdout",
    "chart_type_multiclass_validation",
    "absent_chart_type_candidates",
    "observed_type",
    "expected_type",
    "candidate_type",
    "manual_type",
    "expected_technique",
    "observed_technique",
    "technique_final",
    "technique_status",
    "matches_expected_type",
    "expected_structure_present",
    "exact_expected_only",
    "observed_signature",
    "expected_signature",
    "bar_present",
    "line_present",
    "scatter_present",
    "stem_present",
    "bar",
    "line",
    "scatter",
    "stem",
    "classify",
    "classification",
    "classifier",
    "technique",
    "chart_type",
]

STRONG_TERMS = [
    "observed_type",
    "expected_type",
    "candidate_type",
    "manual_type",
    "observed_signature",
    "expected_signature",
    "expected_structure_present",
    "exact_expected_only",
    "bar_present",
    "line_present",
    "scatter_present",
    "stem_present",
    "chart_type_exclusive_holdout",
    "chart_type_multiclass_validation",
    "absent_chart_type_candidates",
]

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def is_implementacoes(path: Path) -> bool:
    parts = [strip_accents(p).lower() for p in path.parts]
    return "implementacoes" in parts

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_text(path: Path) -> str:
    if path.suffix.lower() == ".ipynb":
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            chunks = []
            for cell in obj.get("cells", []):
                chunks.extend(cell.get("source", []))
            return "\n".join(chunks)
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")

def snippets(text: str, terms, radius=180, max_snippets=10):
    found = []
    low = text.lower()
    for term in terms:
        start = 0
        while True:
            i = low.find(term.lower(), start)
            if i < 0:
                break
            a = max(0, i - radius)
            b = min(len(text), i + len(term) + radius)
            sn = re.sub(r"\s+", " ", text[a:b]).strip()
            found.append(f"{term}: {sn}")
            start = i + len(term)
            if len(found) >= max_snippets:
                return found
    return found

def analyze_file(path: Path):
    try:
        text = read_text(path)
    except Exception:
        return None

    low = text.lower()
    hits = [t for t in TERMS if t.lower() in low]
    strong = [t for t in STRONG_TERMS if t.lower() in low]

    name_low = path.name.lower()
    name_hint = any(
        k in name_low
        for k in (
            "chart_type", "techni", "classif", "multiclass",
            "exclusive", "absent", "stem", "scatter", "bar_line"
        )
    )

    impl = is_implementacoes(path)

    if not hits and not name_hint:
        return None

    score = len(hits)
    score += len(strong) * 4
    if name_hint:
        score += 5
    if impl:
        score += 12

    # Bônus por indícios de produção em lote.
    production_signals = []
    for sig in [
        "3120",
        "rglob",
        "glob(",
        "filename",
        "profile",
        "unit_id",
        "to_csv",
        "csv.dictwriter",
    ]:
        if sig in low:
            production_signals.append(sig)
            score += 2

    # Bônus se menciona as três famílias esperadas.
    family_count = sum(
        1 for sig in ("bar", "line", "scatter")
        if re.search(rf"\b{sig}\b", low)
    )
    if family_count == 3:
        score += 8

    if re.search(r"\bstem\b", low):
        score += 2

    return {
        "score": score,
        "inside_implementacoes": "YES" if impl else "NO",
        "path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "hits": "|".join(hits),
        "strong_signals": "|".join(strong),
        "production_signals": "|".join(production_signals),
        "snippets": " || ".join(snippets(text, strong if strong else hits)),
    }

def main():
    impl_dirs = [
        p for p in ROOT.rglob("*")
        if p.is_dir() and strip_accents(p.name).lower() == "implementacoes"
    ]

    rows = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        r = analyze_file(p)
        if r:
            rows.append(r)

    rows.sort(
        key=lambda r: (
            r["inside_implementacoes"] != "YES",
            -int(r["score"]),
            r["path"]
        )
    )

    fields = [
        "score", "inside_implementacoes", "path", "filename",
        "extension", "size_bytes", "sha256",
        "hits", "strong_signals", "production_signals", "snippets"
    ]

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    impl_rows = [r for r in rows if r["inside_implementacoes"] == "YES"]
    with IMPL_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(impl_rows)

    lines = [
        "=" * 84,
        "INVENTÁRIO DE IMPLEMENTAÇÕES DA TÉCNICA GRÁFICA — V2",
        "=" * 84,
        f"Pasta raiz: {ROOT}",
        f"Pastas 'implementações' localizadas: {len(impl_dirs)}",
    ]

    for d in impl_dirs:
        lines.append(f"  - {d}")

    lines += [
        "",
        f"Candidatos totais: {len(rows)}",
        f"Candidatos dentro de 'implementações': {len(impl_rows)}",
        "",
        "TOP 30 — priorizando a pasta implementações",
        "-" * 84,
    ]

    for i, r in enumerate(rows[:30], 1):
        lines.append(
            f'{i:02d}. score={r["score"]:>3} '
            f'impl={r["inside_implementacoes"]} '
            f'{r["path"]}'
        )
        if r["strong_signals"]:
            lines.append("    sinais fortes: " + r["strong_signals"])
        if r["production_signals"]:
            lines.append("    sinais produção: " + r["production_signals"])

    lines += [
        "",
        f"Inventário geral: {OUT}",
        f"Somente implementações: {IMPL_OUT}",
    ]

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
