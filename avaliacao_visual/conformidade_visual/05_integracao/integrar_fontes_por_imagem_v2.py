# -*- coding: utf-8 -*-
"""
INTEGRAÇÃO DAS FONTES POR IMAGEM — V2
=====================================

Objetivo
--------
Ler as fontes finais já localizadas no inventário e produzir um único CSV
por imagem (3.120 linhas), sem alterar os resultados originais.

Fontes usadas
-------------
1) Presença da marca:
   imagens\_mark_presence_full_v5\mark_presence_images_v5.csv

2) Presença dos eixos:
   imagens\_axis_presence_full_v5_1\axis_presence_images_v5_1.csv

3) Conformidade de cor:
   imagens\_color_conformity_final_v1\color_conformity_final_images.csv

4) Presença dos tick labels:
   imagens\_ticklabel_presence_full_v5\ticklabel_presence_images_v5.csv
   imagens\_ticklabel_presence_full_v5\ticklabel_presence_priority_review_v5.csv

5) Área de plotagem (diagnóstica):
   imagens\_visual_processing_v4\visual_features_v4_plot_area.csv

Saídas
------
INTEGRACAO_FONTES_POR_IMAGEM_V2.csv
INTEGRACAO_FONTES_POR_IMAGEM_V2_RESUMO.txt
CANDIDATOS_TECNICA_INTEGRACAO_V2.csv

Importante
----------
- O script NÃO usa F para alterar B-A.
- Área de plotagem permanece DIAGNOSTIC, sem converter erro geométrico em
  conformidade.
- Presença dos tick labels permanece PENDING_FINAL_ADJUDICATION quando
  houver caso marcado para revisão e não houver adjudicação manual preenchida.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES = ROOT / "imagens"

MARK = IMAGES / "_mark_presence_full_v5" / "mark_presence_images_v5.csv"
AXIS = IMAGES / "_axis_presence_full_v5_1" / "axis_presence_images_v5_1.csv"
COLOR = IMAGES / "_color_conformity_final_v1" / "color_conformity_final_images.csv"
TLP = IMAGES / "_ticklabel_presence_full_v5" / "ticklabel_presence_images_v5.csv"
TLP_REVIEW = IMAGES / "_ticklabel_presence_full_v5" / "ticklabel_presence_priority_review_v5.csv"
PLOT = IMAGES / "_visual_processing_v4" / "visual_features_v4_plot_area.csv"

OUT = ROOT / "INTEGRACAO_FONTES_POR_IMAGEM_V2.csv"
SUMMARY = ROOT / "INTEGRACAO_FONTES_POR_IMAGEM_V2_RESUMO.txt"
TECH = ROOT / "CANDIDATOS_TECNICA_INTEGRACAO_V2.csv"

def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        return list(csv.DictReader(f))

def truthy(v):
    s = str(v or "").strip().lower()
    return s in {"1","true","t","yes","y","sim","s","present","presente","both_present","conforming","ok"}

def falsey(v):
    s = str(v or "").strip().lower()
    return s in {"0","false","f","no","n","nao","não","absent","ausente","nonconforming"}

def normalized_filename(v):
    return Path(str(v or "").strip()).name

def index_by_filename(rows, label):
    d = {}
    dup = []
    for r in rows:
        fn = normalized_filename(r.get("filename"))
        if not fn:
            continue
        if fn in d:
            dup.append(fn)
        d[fn] = r
    if dup:
        raise RuntimeError(f"{label}: filenames duplicados: {dup[:10]}")
    return d

def map_mark(r):
    raw = str(r.get("mark_conformity_status","") or r.get("presence_status_raw","")).strip()
    present = r.get("mark_present","")
    absent = r.get("mark_absent","")

    up = raw.upper()
    if "ABSENT" in up or truthy(absent):
        return raw, "NONCONFORMING"
    if "PRESENT" in up or truthy(present):
        return raw, "CONFORMING"
    return raw, "UNEVALUABLE"

def map_axis(r):
    raw = str(r.get("axes_status","")).strip()
    up = raw.upper()

    if up == "BOTH_PRESENT":
        return raw, "CONFORMING"
    if up in {"UNEVALUABLE","NOT_EVALUABLE"}:
        return raw, "UNEVALUABLE"

    # Se o arquivo já codifica explicitamente presença X/Y, usa como apoio.
    xe = truthy(r.get("axis_x_evaluable",""))
    ye = truthy(r.get("axis_y_evaluable",""))
    xp = truthy(r.get("axis_x_present",""))
    yp = truthy(r.get("axis_y_present",""))

    if xe and ye:
        return raw, "CONFORMING" if (xp and yp) else "NONCONFORMING"

    if up:
        return raw, "NONCONFORMING"

    return raw, "UNEVALUABLE"

def map_color(r):
    raw = str(r.get("color_status_final","")).strip()
    evaluable = truthy(r.get("color_evaluable",""))
    no_mark = truthy(r.get("no_mark",""))
    strict_match = truthy(r.get("strict_color_match",""))

    if no_mark or not evaluable:
        return raw, "UNEVALUABLE"

    if strict_match or raw.upper() == "MATCH":
        return raw, "CONFORMING"

    return raw, "NONCONFORMING"

def manual_ticklabel_final(review_row):
    if not review_row:
        return None, "", "", "", ""

    mx = str(review_row.get("manual_x_present_final","")).strip()
    my = str(review_row.get("manual_y_present_final","")).strip()
    mc = str(review_row.get("manual_classification","")).strip()
    notes = str(review_row.get("manual_notes","")).strip()

    # Primeiro tenta x/y manuais.
    if mx or my:
        if truthy(mx) and truthy(my):
            return "CONFORMING", mx, my, mc, notes
        if (truthy(mx) or falsey(mx)) and (truthy(my) or falsey(my)):
            return "NONCONFORMING", mx, my, mc, notes

    # Depois, classificação manual textual.
    up = mc.upper()
    if up in {"BOTH_PRESENT","CONFORMING","OK","PRESENT"}:
        return "CONFORMING", mx, my, mc, notes
    if up in {"X_ONLY","Y_ONLY","NONE","ABSENT","NONCONFORMING"}:
        return "NONCONFORMING", mx, my, mc, notes
    if up in {"UNEVALUABLE","NOT_EVALUABLE"}:
        return "UNEVALUABLE", mx, my, mc, notes

    return None, mx, my, mc, notes

def map_ticklabel_presence(r, review_row):
    raw = str(r.get("presence_status_auto","")).strip()
    needs_review = truthy(r.get("needs_visual_review",""))
    both_auto = truthy(r.get("both_present_auto",""))

    manual_status, mx, my, mc, notes = manual_ticklabel_final(review_row)

    if needs_review:
        if manual_status:
            return raw, manual_status, "MANUAL_REVIEW", mx, my, mc, notes
        return raw, "PENDING_FINAL_ADJUDICATION", "REVIEW_REQUIRED", mx, my, mc, notes

    # Para os casos não enviados à revisão, usa o resultado automático congelado.
    if both_auto:
        return raw, "CONFORMING", "AUTO_FROZEN", mx, my, mc, notes

    # Conservador: se não for BOTH_PRESENT e não houver revisão preenchida,
    # não transforma automaticamente em não conformidade.
    if manual_status:
        return raw, manual_status, "MANUAL_REVIEW", mx, my, mc, notes

    return raw, "PENDING_FINAL_ADJUDICATION", "AUTO_EXCEPTION", mx, my, mc, notes

def plot_diag(r):
    return {
        "plot_detection_status": str(r.get("plot_detection_status","")).strip(),
        "plot_detection_confidence": str(r.get("plot_detection_confidence","")).strip(),
        "layout_error_mean_vs_F": str(r.get("layout_error_mean_vs_F","")).strip(),
        "plot_left_px": str(r.get("plot_left_px","")).strip(),
        "plot_right_px": str(r.get("plot_right_px","")).strip(),
        "plot_top_px": str(r.get("plot_top_px","")).strip(),
        "plot_bottom_px": str(r.get("plot_bottom_px","")).strip(),
    }

def scan_technique_candidates():
    rows_out = []
    name_pat = re.compile(
        r"(techni|technique|classific|chart[_-]?type|tipo[_-]?graf|bar[_-]?line|stem|scatter)",
        re.I
    )
    col_tokens = {
        "observed_technique","technique_final","technique_status","chart_type_final",
        "predicted_technique","detected_technique","classification_final",
        "bar_detected","line_detected","scatter_detected","stem_detected"
    }

    for p in ROOT.rglob("*.csv"):
        try:
            with p.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            hlow = {str(x).strip().lower() for x in header}
            score = 0
            reasons = []

            if name_pat.search(p.name):
                score += 2
                reasons.append("filename")
            hits = sorted(col_tokens.intersection(hlow))
            if hits:
                score += 5
                reasons.append("columns:" + ",".join(hits))
            if "technique" in hlow and any(
                k in hlow for k in {"prediction","classification","detected","observed"}
            ):
                score += 3
                reasons.append("technique+classification")

            if score >= 2:
                rows_out.append({
                    "score": score,
                    "path": str(p),
                    "filename": p.name,
                    "reason": "|".join(reasons),
                    "columns": "|".join(header),
                })
        except Exception:
            continue

    rows_out.sort(key=lambda x: (-x["score"], x["path"]))
    with TECH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["score","path","filename","reason","columns"])
        w.writeheader()
        w.writerows(rows_out)
    return rows_out

def main():
    mark_rows = read_csv(MARK)
    axis_rows = read_csv(AXIS)
    color_rows = read_csv(COLOR)
    tlp_rows = read_csv(TLP)
    review_rows = read_csv(TLP_REVIEW)
    plot_rows = read_csv(PLOT)

    mark = index_by_filename(mark_rows, "MARK")
    axis = index_by_filename(axis_rows, "AXIS")
    color = index_by_filename(color_rows, "COLOR")
    tlp = index_by_filename(tlp_rows, "TICKLABEL_PRESENCE")
    review = index_by_filename(review_rows, "TICKLABEL_REVIEW")
    plot = index_by_filename(plot_rows, "PLOT")

    # Universo: presença da marca deve cobrir as 3.120 imagens.
    universe = sorted(mark.keys())

    if len(universe) != 3120:
        raise RuntimeError(
            f"Presença da marca deveria conter 3120 imagens, mas contém {len(universe)}."
        )

    missing = {}
    for label, d in [
        ("AXIS", axis), ("COLOR", color), ("TICKLABEL_PRESENCE", tlp), ("PLOT", plot)
    ]:
        miss = [fn for fn in universe if fn not in d]
        if miss:
            missing[label] = miss

    if missing:
        msg = "\n".join(
            f"{k}: {len(v)} ausentes. Ex.: {v[:5]}" for k,v in missing.items()
        )
        raise RuntimeError("Há imagens faltantes nas fontes:\n" + msg)

    out_rows = []

    for fn in universe:
        mr = mark[fn]
        ar = axis[fn]
        cr = color[fn]
        tr = tlp[fn]
        rr = review.get(fn)
        pr = plot[fn]

        mark_raw, mark_final = map_mark(mr)
        axis_raw, axis_final = map_axis(ar)
        color_raw, color_final = map_color(cr)
        (
            tl_raw, tl_final, tl_source, mx, my, mc, mn
        ) = map_ticklabel_presence(tr, rr)

        pd = plot_diag(pr)

        out_rows.append({
            "filename": fn,
            "profile": mr.get("profile",""),
            "unit_id": mr.get("unit_id",""),
            "repeat": mr.get("repeat",""),
            "technique": mr.get("technique",""),
            "task": mr.get("task",""),

            "mark_presence_raw": mark_raw,
            "mark_presence_status_final": mark_final,

            "axis_presence_raw": axis_raw,
            "axis_presence_status_final": axis_final,

            "color_status_raw": color_raw,
            "color_evaluable_raw": cr.get("color_evaluable",""),
            "color_status_final_integrated": color_final,

            "ticklabel_presence_raw": tl_raw,
            "ticklabel_needs_visual_review": tr.get("needs_visual_review",""),
            "ticklabel_presence_status_final": tl_final,
            "ticklabel_presence_source": tl_source,
            "manual_x_present_final": mx,
            "manual_y_present_final": my,
            "manual_classification": mc,
            "manual_ticklabel_notes": mn,

            "plot_area_status_diagnostic": pd["plot_detection_status"],
            "plot_area_confidence": pd["plot_detection_confidence"],
            "layout_error_mean_vs_F": pd["layout_error_mean_vs_F"],
            "plot_left_px": pd["plot_left_px"],
            "plot_right_px": pd["plot_right_px"],
            "plot_top_px": pd["plot_top_px"],
            "plot_bottom_px": pd["plot_bottom_px"],
        })

    fields = list(out_rows[0].keys())
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    tech = scan_technique_candidates()

    # Resumos
    mark_c = Counter(r["mark_presence_status_final"] for r in out_rows)
    axis_c = Counter(r["axis_presence_status_final"] for r in out_rows)
    color_c = Counter(r["color_status_final_integrated"] for r in out_rows)
    tl_c = Counter(r["ticklabel_presence_status_final"] for r in out_rows)
    tl_source_c = Counter(r["ticklabel_presence_source"] for r in out_rows)
    plot_c = Counter(r["plot_area_status_diagnostic"] for r in out_rows)

    # Validações contra os resultados finais já registrados.
    checks = []
    checks.append(("MARK CONFORMING = 3072", mark_c.get("CONFORMING",0) == 3072))
    checks.append(("MARK NONCONFORMING = 48", mark_c.get("NONCONFORMING",0) == 48))
    checks.append(("AXIS CONFORMING = 3091", axis_c.get("CONFORMING",0) == 3091))
    checks.append(("AXIS NONCONFORMING = 20", axis_c.get("NONCONFORMING",0) == 20))
    checks.append(("AXIS UNEVALUABLE = 9", axis_c.get("UNEVALUABLE",0) == 9))
    checks.append(("COLOR CONFORMING = 2816", color_c.get("CONFORMING",0) == 2816))
    checks.append(("COLOR NONCONFORMING = 289", color_c.get("NONCONFORMING",0) == 289))
    checks.append(("COLOR UNEVALUABLE = 15", color_c.get("UNEVALUABLE",0) == 15))
    checks.append(("TICKLABEL PRIORITY REVIEW = 36", len(review_rows) == 36))

    lines = []
    lines.append("="*78)
    lines.append("INTEGRAÇÃO DAS FONTES POR IMAGEM — V2")
    lines.append("="*78)
    lines.append("")
    lines.append(f"Imagens integradas: {len(out_rows)}")
    lines.append("")
    lines.append("PRESENÇA DA MARCA")
    lines.append(str(dict(mark_c)))
    lines.append("")
    lines.append("PRESENÇA DOS EIXOS")
    lines.append(str(dict(axis_c)))
    lines.append("")
    lines.append("CONFORMIDADE DE COR")
    lines.append(str(dict(color_c)))
    lines.append("")
    lines.append("PRESENÇA DOS TICK LABELS")
    lines.append(str(dict(tl_c)))
    lines.append("Fonte da decisão: " + str(dict(tl_source_c)))
    lines.append("")
    lines.append("ÁREA DE PLOTAGEM — DIAGNÓSTICO")
    lines.append(str(dict(plot_c)))
    lines.append("")
    lines.append("VALIDAÇÕES")
    for label, ok in checks:
        lines.append(f"[{'OK' if ok else 'FALHOU'}] {label}")
    lines.append("")
    lines.append(f"Candidatos de técnica encontrados: {len(tech)}")
    lines.append(f"Arquivo de candidatos: {TECH}")
    lines.append("")
    lines.append(f"CSV integrado: {OUT}")
    lines.append("")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))

    if not all(ok for _,ok in checks):
        print("ATENÇÃO: pelo menos uma validação não coincidiu com o resumo final.")
        print("Não use o CSV como fonte final antes de revisar o RESUMO.")
        return 2

    print("OK — fontes principais integradas e validadas.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
