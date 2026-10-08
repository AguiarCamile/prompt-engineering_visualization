# -*- coding: utf-8 -*-
r"""
CLASSIFICAÇÃO EXCLUSIVA DE TÉCNICA — PRODUÇÃO 3.120 — V2 FROZEN
================================================================

Objetivo
--------
Aplicar às 3.120 imagens exatamente as regras do classificador exclusivo V2
validadas no holdout cego, SEM usar a técnica esperada/F para escolher a
classe observada.

Classes observadas:
    BAR
    LINE
    SCATTER
    OTHER

Regras congeladas
-----------------
1. BAR:
   MAX_H_FRAC >= 0.10

2. LINE:
   se não BAR, XBIN_SUPPORT >= 0.50

3. SCATTER:
   se não BAR nem LINE,
   N_MARKERS >= 2
   e (XDISP >= 0.03 ou YDISP >= 0.03)

4. OTHER:
   nenhum critério anterior.

IMPORTANTE
----------
- NÃO lê F.
- NÃO usa profile/expected_type para decidir a classe.
- O profile extraído do nome do arquivo é apenas metadado.
- NÃO recalibra limiares.
- Processa apenas as imagens canônicas na raiz:
      C:\Users\Labvis\Downloads\imagens3120\imagens
  e ignora cópias em subpastas.
- É retomável: grava progresso periodicamente.

Dependência estrutural
----------------------
Usa exatamente os descritores de mark_presence_validation_v3.py:
- detect_bars
- detect_lines
- detect_scatter
- infer_plot_roi

O programa procura o módulo nesta ordem:
1. C:\Users\Labvis\Downloads\imagens3120\mark_presence_validation_v3.py
2. C:\Users\Labvis\Downloads\imagens3120\mark_presence_full_v5_package\
   mark_presence_validation_v3.py
3. C:\Users\Labvis\Downloads\imagens3120\implementações\
   mark_presence_validation_v3.py

Saídas
------
C:\Users\Labvis\Downloads\imagens3120\
  CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_PROGRESS.csv
  CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_FINAL.csv
  CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_ERRORS.csv
  CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_SUMMARY.txt
  CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_MANIFEST.txt
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES = ROOT / "imagens"

PROGRESS = ROOT / "CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_PROGRESS.csv"
FINAL = ROOT / "CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_FINAL.csv"
ERRORS = ROOT / "CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_ERRORS.csv"
SUMMARY = ROOT / "CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_SUMMARY.txt"
MANIFEST = ROOT / "CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_MANIFEST.txt"

NAME_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.I,
)

MODULE_CANDIDATES = [
    ROOT / "mark_presence_validation_v3.py",
    ROOT / "mark_presence_full_v5_package" / "mark_presence_validation_v3.py",
    ROOT / "implementações" / "mark_presence_validation_v3.py",
]

FIELDNAMES = [
    "filename",
    "profile",
    "unit_id",
    "repeat",
    "observed_type",
    "decision_reason",
    "bar_max_h_frac",
    "line_xbin_support",
    "scatter_n_markers",
    "scatter_xdisp",
    "scatter_ydisp",
    "roi_method",
    "classifier_version",
]

ERROR_FIELDS = [
    "filename",
    "profile",
    "unit_id",
    "repeat",
    "error_type",
    "error_message",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_v3_module():
    module_path = None
    for p in MODULE_CANDIDATES:
        if p.exists():
            module_path = p
            break

    if module_path is None:
        raise FileNotFoundError(
            "mark_presence_validation_v3.py não foi encontrado.\n"
            "Locais testados:\n- "
            + "\n- ".join(str(p) for p in MODULE_CANDIDATES)
        )

    spec = importlib.util.spec_from_file_location(
        "mark_presence_validation_v3_frozen_dependency",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "infer_plot_roi",
        "detect_bars",
        "detect_lines",
        "detect_scatter",
    ]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise RuntimeError(
            "Dependência incompatível. Funções ausentes: "
            + ", ".join(missing)
        )

    return module, module_path


def inventory():
    rows = []
    for p in sorted(IMAGES.iterdir()):
        if not p.is_file():
            continue
        m = NAME_RE.match(p.name)
        if not m:
            continue

        profile = m.group(1).upper()
        unit_number = int(m.group(2))
        repeat = int(m.group(3))

        rows.append({
            "path": p,
            "filename": p.name,
            "profile": profile,
            "unit_id": f"{profile}_{unit_number:03d}",
            "repeat": repeat,
        })

    return rows


def parse_evidence(s):
    out = {}
    for part in str(s).split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def classify_exclusive(db, dl, ds):
    """
    REGRAS V2 CONGELADAS.
    Não usa técnica esperada.
    """
    eb = parse_evidence(db["evidence"])
    el = parse_evidence(dl["evidence"])
    es = parse_evidence(ds["evidence"])

    bar_max_h_frac = float(eb.get("MAX_H_FRAC", 0.0))
    line_xbin_support = float(el.get("XBIN_SUPPORT", 0.0))

    scatter_n_markers = int(es.get("N_MARKERS", 0.0))
    scatter_xdisp = float(es.get("XDISP", 0.0))
    scatter_ydisp = float(es.get("YDISP", 0.0))

    if bar_max_h_frac >= 0.10:
        observed = "BAR"
        reason = "BAR_MAX_H_FRAC>=0.10"

    elif line_xbin_support >= 0.50:
        observed = "LINE"
        reason = "LINE_XBIN_SUPPORT>=0.50"

    elif (
        scatter_n_markers >= 2
        and (
            scatter_xdisp >= 0.03
            or scatter_ydisp >= 0.03
        )
    ):
        observed = "SCATTER"
        reason = "SCATTER_MARKERS_AND_DISPERSION"

    else:
        observed = "OTHER"
        reason = "NO_EXCLUSIVE_RULE"

    return {
        "observed_type": observed,
        "decision_reason": reason,
        "bar_max_h_frac": bar_max_h_frac,
        "line_xbin_support": line_xbin_support,
        "scatter_n_markers": scatter_n_markers,
        "scatter_xdisp": scatter_xdisp,
        "scatter_ydisp": scatter_ydisp,
    }


def analyze_one_frozen(path: Path, v3):
    """
    Carrega a imagem uma única vez e aplica os três descritores V3
    no mesmo ROI, preservando as funções originais do detector.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    roi, roi_method = v3.infer_plot_roi(arr)

    db = v3.detect_bars(arr, roi)
    dl = v3.detect_lines(arr, roi)
    ds = v3.detect_scatter(arr, roi)

    cls = classify_exclusive(db, dl, ds)
    cls["roi_method"] = roi_method
    return cls


def read_existing(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    if not IMAGES.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {IMAGES}")

    v3, module_path = load_v3_module()
    module_hash = sha256(module_path)

    inv = inventory()

    if len(inv) != 3120:
        raise RuntimeError(
            f"Inventário canônico esperado: 3120 imagens; encontrado: {len(inv)}."
        )

    existing = read_existing(PROGRESS)
    done_map = {
        r["filename"]: r
        for r in existing
        if str(r.get("observed_type", "")).strip()
    }

    error_rows = read_existing(ERRORS)
    error_map = {r["filename"]: r for r in error_rows}

    rows = []
    errors = []

    # Preserva progresso válido anterior.
    for item in inv:
        fn = item["filename"]
        if fn in done_map:
            rows.append(done_map[fn])

    processed_now = 0
    t0 = time.time()

    print("=" * 84)
    print("CLASSIFICAÇÃO EXCLUSIVA DE TÉCNICA — PRODUÇÃO 3.120 — V2 FROZEN")
    print("=" * 84)
    print(f"Imagens canônicas: {len(inv)}")
    print(f"Dependência V3: {module_path}")
    print(f"SHA-256 dependência: {module_hash}")
    print(f"Já concluídas no progresso: {len(done_map)}")
    print()
    print("IMPORTANTE: F/expected_type NÃO são carregados.")
    print()

    completed = set(done_map)

    for item in inv:
        fn = item["filename"]
        if fn in completed:
            continue

        try:
            cls = analyze_one_frozen(item["path"], v3)

            row = {
                "filename": fn,
                "profile": item["profile"],
                "unit_id": item["unit_id"],
                "repeat": item["repeat"],
                **cls,
                "classifier_version": "CHART_TYPE_EXCLUSIVE_V2_FROZEN",
            }
            rows.append(row)
            completed.add(fn)
            error_map.pop(fn, None)

        except Exception as exc:
            error_map[fn] = {
                "filename": fn,
                "profile": item["profile"],
                "unit_id": item["unit_id"],
                "repeat": item["repeat"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

        processed_now += 1
        done = len(completed) + len(error_map)

        if processed_now % 50 == 0 or done >= len(inv):
            rows.sort(key=lambda r: r["filename"])
            write_rows(PROGRESS, rows, FIELDNAMES)

            errors = sorted(
                error_map.values(),
                key=lambda r: r["filename"]
            )
            write_rows(ERRORS, errors, ERROR_FIELDS)

            elapsed = time.time() - t0
            print(
                f"Estado: {len(completed)}/{len(inv)} concluídas | "
                f"erros atuais={len(errors)} | "
                f"tempo={elapsed/60:.1f} min"
            )

    rows.sort(key=lambda r: r["filename"])
    errors = sorted(error_map.values(), key=lambda r: r["filename"])

    write_rows(PROGRESS, rows, FIELDNAMES)
    write_rows(ERRORS, errors, ERROR_FIELDS)

    if len(rows) != 3120 or errors:
        print()
        print("ATENÇÃO:")
        print(f"Classificadas: {len(rows)}/3120")
        print(f"Erros: {len(errors)}")
        print("O arquivo FINAL não será criado enquanto houver erro.")
        return 2

    write_rows(FINAL, rows, FIELDNAMES)

    counts = Counter(r["observed_type"] for r in rows)
    by_profile = {}
    for profile in ["BI", "BC", "LI", "LC", "SI", "SC"]:
        g = [r for r in rows if r["profile"] == profile]
        by_profile[profile] = Counter(r["observed_type"] for r in g)

    lines = [
        "CLASSIFICAÇÃO EXCLUSIVA DE TÉCNICA — PRODUÇÃO V2 FROZEN",
        "=" * 78,
        f"Imagens classificadas: {len(rows)}",
        f"Erros: {len(errors)}",
        "",
        "REGRAS CONGELADAS:",
        "BAR: MAX_H_FRAC >= 0.10",
        "LINE: se não BAR, XBIN_SUPPORT >= 0.50",
        "SCATTER: se não BAR/LINE, N_MARKERS >= 2 e "
        "(XDISP >= 0.03 ou YDISP >= 0.03)",
        "OTHER: nenhum critério anterior",
        "",
        "DISTRIBUIÇÃO OBSERVADA:",
    ]

    for k in ["BAR", "LINE", "SCATTER", "OTHER"]:
        lines.append(f"{k}: {counts.get(k, 0)}")

    lines += ["", "POR PERFIL (apenas resumo pós-extração):"]
    for profile in ["BI", "BC", "LI", "LC", "SI", "SC"]:
        c = by_profile[profile]
        lines.append(
            f"{profile}: "
            + ", ".join(
                f"{k}={c.get(k,0)}"
                for k in ["BAR","LINE","SCATTER","OTHER"]
            )
        )

    lines += [
        "",
        "INDEPENDÊNCIA B-A:",
        "- nenhuma especificação F foi carregada;",
        "- profile não participou de classify_exclusive;",
        "- expected_type não foi fornecido ao classificador.",
        "",
        f"Dependência V3: {module_path}",
        f"SHA-256 dependência V3: {module_hash}",
        f"Saída final: {FINAL}",
    ]

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    manifest_lines = [
        "CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN",
        "=" * 60,
        f"root={ROOT}",
        f"images={IMAGES}",
        f"n_images={len(rows)}",
        f"dependency={module_path}",
        f"dependency_sha256={module_hash}",
        "bar_threshold_max_h_frac=0.10",
        "line_threshold_xbin_support=0.50",
        "scatter_min_markers=2",
        "scatter_min_xdisp_or_ydisp=0.03",
        "uses_F=NO",
        "uses_expected_type_for_classification=NO",
    ]
    MANIFEST.write_text("\n".join(manifest_lines), encoding="utf-8")

    print()
    print("CONCLUÍDO.")
    print("Abra primeiro:")
    print(SUMMARY)
    print()
    print("CSV final:")
    print(FINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
