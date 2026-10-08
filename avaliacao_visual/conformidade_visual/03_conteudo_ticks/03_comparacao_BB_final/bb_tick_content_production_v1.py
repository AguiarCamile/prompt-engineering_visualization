import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

try:
    import pandas as pd
except Exception as e:
    print("ERRO: este script requer pandas e openpyxl para ler a F em XLSX.")
    print("Instale no .venv, se necessário:")
    print("  python -m pip install pandas openpyxl")
    raise

def strip_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(c)
    )

def norm(s):
    return strip_accents(s).lower()

def normalize_cat_label(s):
    s = norm(s).strip()
    s = s.replace("infra-estrutura", "infraestrutura")
    s = re.sub(r"\s*/\s*", " e ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def unit_multiplier(unit):
    u = norm(unit).strip()
    if u in ("k", "mil"):
        return 1e3
    if u in ("m", "milhao", "milhoes"):
        return 1e6
    if u in ("b", "bilhao", "bilhoes"):
        return 1e9
    if u == "%":
        return None
    return 1.0

def canonicalize_pairs(pairs):
    out = []
    for v, u in pairs:
        mult = unit_multiplier(u)
        if mult is None:
            out.append(("PCT", round(float(v), 9)))
        else:
            out.append(("NUM", round(float(v) * mult, 6)))
    return out

def canonical_numeric_text(values):
    parts = []
    for typ, v in values:
        if typ == "PCT":
            parts.append(f"{v:g}%")
        elif abs(v - round(v)) < 1e-9:
            parts.append(str(int(round(v))))
        else:
            parts.append(f"{v:g}")
    return " | ".join(parts)

def parse_f_numeric_canonical(text):
    vals = []
    for part in str(text or "").split("|"):
        p = part.strip()
        if not p:
            continue
        if p.endswith("%"):
            vals.append(("PCT", float(p[:-1].strip())))
        else:
            vals.append(("NUM", round(float(p), 6)))
    return vals

def compare_lists(expected, observed):
    ce = Counter(expected)
    co = Counter(observed)
    if ce == co:
        return "NORMALIZED_EQUIVALENT", "CONFORMING", [], []
    missing = list((ce - co).elements())
    extra = list((co - ce).elements())
    if missing and not extra:
        return "NONCONFORMING_MISSING", "NONCONFORMING", missing, extra
    if extra and not missing:
        return "NONCONFORMING_EXTRA", "NONCONFORMING", missing, extra
    return "NONCONFORMING_VALUE", "NONCONFORMING", missing, extra

def parse_manual_numeric_visible(text, raw_ocr):
    pairs = []
    for part in str(text or "").split("|"):
        p = part.strip()
        if not p:
            continue
        m = re.search(r"(-?\d+(?:[\.,]\d+)?)\s*(milhoes?|milhao|bilhoes?|bilhao|mil|[kmb%])?", norm(p))
        if not m:
            continue
        v = float(m.group(1).replace(",", "."))
        u = (m.group(2) or "")
        pairs.append((v, u))

    if not pairs:
        return []

    if all(not u for v, u in pairs if v != 0):
        raw_units = re.findall(r"\d+(?:[\.,]\d+)?\s*(milhoes?|milhao|bilhoes?|bilhao|mil|[kmb%])\b", norm(raw_ocr))
        if raw_units:
            dom = Counter(raw_units).most_common(1)[0][0]
            pairs = [(v, dom if v != 0 else "") for v, u in pairs]
    return canonicalize_pairs(pairs)

def observed_numeric(v7row, axisrow):
    manual = str(v7row.get("manual_visible_ticklabels", "") or "").strip()
    raw = str(axisrow.get("numeric_lane_ocr_raw_tokens", "") or "")

    if manual:
        can = parse_manual_numeric_visible(manual, raw)
        if can:
            return "MANUAL_ADJUDICATED", "OBSERVED_COMPLETE", manual, can

    effective = str(axisrow.get("numeric_effective_values_spatial", "") or "").strip()
    if effective:
        can = []
        for token in effective.split("|"):
            token = token.strip()
            if not token:
                continue
            try:
                can.append(("NUM", round(float(token), 6)))
            except Exception:
                pass
        if can:
            return "AUTO_VALIDATED", "OBSERVED_COMPLETE", str(v7row.get("numeric_selected_values_spatial", "")), can

    selected = str(v7row.get("numeric_selected_values_spatial", "") or "").strip()
    if selected:
        raw_units = re.findall(r"\d+(?:[\.,]\d+)?\s*(milhoes?|milhao|bilhoes?|bilhao|mil|[kmb%])\b", norm(raw))
        dom = Counter(raw_units).most_common(1)[0][0] if raw_units else ""
        pairs = []
        for token in selected.split("|"):
            token = token.strip()
            if not token:
                continue
            try:
                v = float(token)
                pairs.append((v, dom if v != 0 else ""))
            except Exception:
                pass
        if pairs:
            return "AUTO_FALLBACK", "OBSERVED_COMPLETE", selected, canonicalize_pairs(pairs)

    return "NONE", "UNEVALUABLE", "", []

def load_f_master(path):
    df = pd.read_excel(path, sheet_name="F_MASTER_312")
    fmap = {}
    for _, row in df.iterrows():
        uid = row["unit_id"]
        fmap[uid] = {
            "condition": row.get("condition", ""),
            "profile": str(uid)[:2],
            "x_order": row.get("x_tick_order_required", "NOT_SPECIFIED"),
            "X": {
                "scope": row.get("x_tick_list_scope_bb", ""),
                "status": row.get("x_tick_target_status_bb", ""),
                "display": row.get("x_tick_expected_bb", ""),
                "canonical": row.get("x_tick_expected_canonical_bb", ""),
            },
            "Y": {
                "scope": row.get("y_tick_list_scope_bb", ""),
                "status": row.get("y_tick_target_status_bb", ""),
                "display": row.get("y_tick_expected_bb", ""),
                "canonical": row.get("y_tick_expected_canonical_bb", ""),
            },
        }
    return fmap

def new_base(source, filename, profile, uid, condition, axis, kind, ft, x_order):
    return {
        "source_batch": source,
        "filename": filename,
        "profile": profile,
        "unit_id": uid,
        "condition": condition,
        "axis": axis,
        "kind": kind,
        "f_tick_list_scope": ft["scope"],
        "f_target_status": ft["status"],
        "f_expected_display": ft["display"],
        "f_expected_canonical": ft["canonical"],
        "f_order_required": x_order if axis == "X" else "NOT_SPECIFIED",
        "ba_source": "",
        "ba_final_status": "",
        "observed_display": "",
        "observed_canonical": "",
        "bb_detailed_status": "",
        "bb_collapsed_status": "",
        "missing_items": "",
        "extra_items": "",
        "bb_reason": "",
        "denominator_eligible": "NO",
        "normalization_version": "B_B_NORMALIZATION_V1",
    }

def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("B-B TICK CONTENT — PRODUÇÃO 3.120")
    print("=" * 80)
    print("F:", args.f_master)
    print("Normalização:", args.normalization)
    print("BA cat:", args.ba_cat)
    print("BA v7 final:", args.ba_v7_final)
    print("BA v7 axes:", args.ba_v7_axes)
    print("Saída:", outdir)

    fmap = load_f_master(args.f_master)

    ba_cat = list(csv.DictReader(open(args.ba_cat, "r", encoding="utf-8-sig", newline="")))
    ba_v7_final = list(csv.DictReader(open(args.ba_v7_final, "r", encoding="utf-8-sig", newline="")))
    ba_v7_axes = list(csv.DictReader(open(args.ba_v7_axes, "r", encoding="utf-8-sig", newline="")))
    axes_lookup = {(r["filename"], r["axis"]): r for r in ba_v7_axes}

    results = []

    # Categorical (usually X)
    for r in ba_cat:
        uid = r["unit_id"]
        if uid not in fmap:
            continue
        ft = fmap[uid]["X"]
        out = new_base(
            "BA_CAT_PRODUCTION",
            r["filename"], r["profile"], uid, r.get("condition_label", r.get("condition", "")),
            "X", "categorical", ft, fmap[uid]["x_order"]
        )
        out["ba_source"] = r.get("extraction_source", "")
        out["ba_final_status"] = r.get("ba_final_status", "")
        out["observed_display"] = r.get("observed_content_final", "")

        if ft["scope"] != "EXHAUSTIVE":
            out["bb_detailed_status"] = "NOT_EVALUATED"
            out["bb_collapsed_status"] = "NOT_SPECIFIED"
            out["bb_reason"] = "F não define lista categórica exaustiva."
        elif str(ft["status"]).startswith("REVIEW_"):
            out["bb_detailed_status"] = "UNEVALUABLE_SPEC_REVIEW"
            out["bb_collapsed_status"] = "UNEVALUABLE"
            out["bb_reason"] = "F requer adjudicação."
        elif r.get("eligible_for_bb", "") != "YES":
            out["bb_detailed_status"] = "UNEVALUABLE"
            out["bb_collapsed_status"] = "UNEVALUABLE"
            out["bb_reason"] = "B-A não fornece conteúdo completo adjudicável."
        else:
            exp = [x.strip() for x in str(ft["canonical"]).split("|") if x.strip()]
            obs = [normalize_cat_label(x) for x in str(r.get("observed_content_final", "")).split("|") if x.strip()]
            out["observed_canonical"] = " | ".join(obs)
            detailed, collapsed, missing, extra = compare_lists(exp, obs)
            out["bb_detailed_status"] = detailed
            out["bb_collapsed_status"] = collapsed
            out["missing_items"] = " | ".join(str(x) for x in missing)
            out["extra_items"] = " | ".join(str(x) for x in extra)
            out["denominator_eligible"] = "YES"
            out["bb_reason"] = (
                "Conteúdo categórico coincide após normalização congelada."
                if collapsed == "CONFORMING"
                else "Diferença lexical/conteúdo não autorizada pela normalização."
            )
        results.append(out)

    # Temporal and numeric from V7 final
    for r in ba_v7_final:
        kind = str(r.get("kind", ""))
        if kind not in ("temporal", "numeric"):
            continue
        uid = r["unit_id"]
        if uid not in fmap:
            continue
        axis = str(r["axis"]).upper()
        ft = fmap[uid][axis]
        out = new_base(
            "BA_V7_PRODUCTION",
            r["filename"], r["profile"], uid, r.get("condition_label", r.get("condition", "")),
            axis, kind, ft, fmap[uid]["x_order"]
        )

        if ft["scope"] in ("NOT_SPECIFIED", "EXAMPLES_ONLY"):
            out["bb_detailed_status"] = "NOT_EVALUATED"
            out["bb_collapsed_status"] = "NOT_SPECIFIED"
            out["bb_reason"] = (
                "F fornece apenas exemplos; completude não é pontuada."
                if ft["scope"] == "EXAMPLES_ONLY"
                else "F não define lista exaustiva."
            )
            results.append(out)
            continue

        if str(ft["status"]).startswith("REVIEW_"):
            out["bb_detailed_status"] = "UNEVALUABLE_SPEC_REVIEW"
            out["bb_collapsed_status"] = "UNEVALUABLE"
            out["bb_reason"] = "A própria F requer adjudicação antes da B-B."
            results.append(out)
            continue

        if kind == "temporal" and ft["status"] == "READY_TEMPORAL":
            exp = [x.strip() for x in str(ft["canonical"]).split("|") if x.strip()]
            obs = [x.strip() for x in str(r.get("temporal_years_spatial", "") or "").split("|") if x.strip()]
            out["ba_source"] = "AUTO_OR_MANUAL"
            out["ba_final_status"] = r.get("manual_ticklabel_extraction_result", "")
            out["observed_display"] = " | ".join(obs)
            out["observed_canonical"] = " | ".join(obs)

            if not obs:
                out["bb_detailed_status"] = "UNEVALUABLE"
                out["bb_collapsed_status"] = "UNEVALUABLE"
                out["bb_reason"] = "B-A temporal sem lista utilizável."
            else:
                detailed, collapsed, missing, extra = compare_lists(exp, obs)
                out["bb_detailed_status"] = detailed
                out["bb_collapsed_status"] = collapsed
                out["missing_items"] = " | ".join(str(x) for x in missing)
                out["extra_items"] = " | ".join(str(x) for x in extra)
                out["denominator_eligible"] = "YES"
                out["bb_reason"] = (
                    "Sequência temporal corresponde à F."
                    if collapsed == "CONFORMING"
                    else "Ano(s) ausente(s)/adicional(is) em relação à lista exaustiva."
                )
            results.append(out)
            continue

        if kind == "numeric" and str(ft["status"]).startswith("READY_"):
            exp = parse_f_numeric_canonical(ft["canonical"])
            source, ba_status, obs_display, obs = observed_numeric(
                r, axes_lookup.get((r["filename"], axis), {})
            )
            out["ba_source"] = source
            out["ba_final_status"] = ba_status
            out["observed_display"] = obs_display
            out["observed_canonical"] = canonical_numeric_text(obs)

            if not obs:
                out["bb_detailed_status"] = "UNEVALUABLE"
                out["bb_collapsed_status"] = "UNEVALUABLE"
                out["bb_reason"] = "B-A numérica sem conteúdo completo utilizável."
            else:
                detailed, collapsed, missing, extra = compare_lists(exp, obs)
                out["bb_detailed_status"] = detailed
                out["bb_collapsed_status"] = collapsed
                out["missing_items"] = canonical_numeric_text(missing)
                out["extra_items"] = canonical_numeric_text(extra)
                out["denominator_eligible"] = "YES"
                out["bb_reason"] = (
                    "Conteúdo numérico coincide após normalização determinística."
                    if collapsed == "CONFORMING"
                    else "Conteúdo numérico difere de F; nenhuma tolerância fuzzy foi aplicada."
                )
            results.append(out)
            continue

        out["bb_detailed_status"] = "NOT_EVALUATED"
        out["bb_collapsed_status"] = "NOT_SPECIFIED"
        out["bb_reason"] = "Combinação ainda não operacionalizada."
        results.append(out)

    if not results:
        print("Nenhum resultado foi gerado. Verifique as entradas.")
        return 1

    # Export detailed CSV
    detailed_csv = outdir / "bb_tick_content_production_v1.csv"
    headers = list(results[0].keys())
    with detailed_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(results)

    # Summary by kind
    sum_rows = []
    for kind in ("categorical", "temporal", "numeric"):
        sub = [r for r in results if r["kind"] == kind]
        exhaustive = sum(1 for r in sub if r["f_tick_list_scope"] == "EXHAUSTIVE")
        examples = sum(1 for r in sub if r["f_tick_list_scope"] == "EXAMPLES_ONLY")
        not_spec = sum(1 for r in sub if r["f_tick_list_scope"] == "NOT_SPECIFIED")
        spec_review = sum(1 for r in sub if r["bb_detailed_status"] == "UNEVALUABLE_SPEC_REVIEW")
        eligible = sum(1 for r in sub if r["denominator_eligible"] == "YES")
        conform = sum(1 for r in sub if r["bb_collapsed_status"] == "CONFORMING")
        nonconf = sum(1 for r in sub if r["bb_collapsed_status"] == "NONCONFORMING")
        uneval = sum(1 for r in sub if r["bb_collapsed_status"] == "UNEVALUABLE")
        rate = "" if eligible == 0 else round(conform / eligible, 6)
        sum_rows.append({
            "kind": kind,
            "n_rows": len(sub),
            "f_exhaustive": exhaustive,
            "f_examples_only": examples,
            "f_not_specified": not_spec,
            "spec_review": spec_review,
            "eligible": eligible,
            "conforming": conform,
            "nonconforming": nonconf,
            "unevaluable_ba": uneval,
            "conformity_rate": rate,
        })

    summary_csv = outdir / "bb_tick_content_production_summary_by_kind.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
        w.writeheader()
        w.writerows(sum_rows)

    # Summary by profile + axis
    pa = {}
    for r in results:
        key = (r["profile"], r["axis"], r["kind"])
        pa.setdefault(key, {"profile": r["profile"], "axis": r["axis"], "kind": r["kind"], "n_rows": 0, "eligible": 0, "conforming": 0, "nonconforming": 0, "unevaluable": 0})
        pa[key]["n_rows"] += 1
        if r["denominator_eligible"] == "YES":
            pa[key]["eligible"] += 1
        if r["bb_collapsed_status"] == "CONFORMING":
            pa[key]["conforming"] += 1
        elif r["bb_collapsed_status"] == "NONCONFORMING":
            pa[key]["nonconforming"] += 1
        elif r["bb_collapsed_status"] == "UNEVALUABLE":
            pa[key]["unevaluable"] += 1

    summary_pa_csv = outdir / "bb_tick_content_production_summary_by_profile_axis.csv"
    pa_rows = []
    for k in sorted(pa):
        rr = pa[k]
        rr["conformity_rate"] = "" if rr["eligible"] == 0 else round(rr["conforming"] / rr["eligible"], 6)
        pa_rows.append(rr)
    with summary_pa_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pa_rows[0].keys()))
        w.writeheader()
        w.writerows(pa_rows)

    # Review cases
    review_rows = [r for r in results if r["bb_collapsed_status"] in ("NONCONFORMING", "UNEVALUABLE")]
    review_csv = outdir / "bb_tick_content_production_cases_review.csv"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(review_rows)

    # Text summary
    summary_txt = outdir / "bb_tick_content_production_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("B-B TICK CONTENT — PRODUÇÃO 3.120\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"F master: {args.f_master}\n")
        f.write(f"Normalização: {args.normalization}\n")
        f.write(f"BA categórico: {args.ba_cat}\n")
        f.write(f"BA V7 final: {args.ba_v7_final}\n")
        f.write(f"BA V7 axes: {args.ba_v7_axes}\n\n")
        for r in sum_rows:
            f.write(f"{r['kind']}: eligible={r['eligible']} conforming={r['conforming']} nonconforming={r['nonconforming']} rate={r['conformity_rate']}\n")

    # XLSX bundle if possible
    xlsx_path = outdir / "bb_tick_content_production_v1.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(results).to_excel(writer, sheet_name="BB_Production", index=False)
            pd.DataFrame(sum_rows).to_excel(writer, sheet_name="Summary_By_Kind", index=False)
            pd.DataFrame(pa_rows).to_excel(writer, sheet_name="Summary_Profile_Axis", index=False)
            pd.DataFrame(review_rows).to_excel(writer, sheet_name="Cases_Review", index=False)
    except Exception as e:
        print("Aviso: não foi possível gerar XLSX:", e)

    print("\nArquivos gerados:")
    print(" -", detailed_csv)
    print(" -", summary_csv)
    print(" -", summary_pa_csv)
    print(" -", review_csv)
    print(" -", summary_txt)
    if xlsx_path.exists():
        print(" -", xlsx_path)

    print("\nResumo por tipo:")
    for r in sum_rows:
        print(
            f" - {r['kind']}: eligible={r['eligible']} conforming={r['conforming']} "
            f"nonconforming={r['nonconforming']} rate={r['conformity_rate']}"
        )
    return 0

def build_argparser():
    p = argparse.ArgumentParser(
        description="B-B production comparator for tick-label content on 3,120 images."
    )
    p.add_argument("--f-master", required=True, help="Path to F_MASTER_V4_TICK_SCOPE_ADJUDICATED.xlsx")
    p.add_argument("--normalization", required=True, help="Path to B_B_NORMALIZATION_V1.json")
    p.add_argument("--ba-cat", required=True, help="Path to final consolidated categorical B-A CSV for production")
    p.add_argument("--ba-v7-final", required=True, help="Path to final B-A V7 CSV (temporal/numeric) for production")
    p.add_argument("--ba-v7-axes", required=True, help="Path to B-A V7 axes CSV for production")
    p.add_argument("--outdir", default=r"C:\Users\Labvis\Downloads\imagens3120\_producao_ticks_3120_v1\bb_tick_content_production_v1", help="Output directory")
    return p

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    raise SystemExit(run(args))
