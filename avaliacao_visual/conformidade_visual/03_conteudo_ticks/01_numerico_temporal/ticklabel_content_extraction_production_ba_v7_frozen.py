# -*- coding: utf-8 -*-
r"""
B-A V7 — PRODUÇÃO CONGELADA NAS 3.120 IMAGENS
==============================================

Este programa executa a B-A V7 EXATAMENTE como congelada, em todas as
3.120 imagens canônicas da RAIZ. Não carrega F e não julga conformidade.

Dependências congeladas verificadas por SHA-256:
- ticklabel_content_extraction_calibration_ba_v7.py
- ticklabel_presence_calibration_v4.py

A saída desta etapa é observacional B-A.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

import ticklabel_content_extraction_calibration_ba_v7 as ba_v7


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
DEFAULT_PROJECT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
OUT_DIRNAME = "_ticklabel_content_extraction_production_ba_v7_frozen"

EXPECTED_V7_SHA256 = "21423b59ce40eb351693f00b6a197936d936fe53131467f47c1031da864fd688"
EXPECTED_PRESENCE_V4_SHA256 = "d15c325290c7596632f9f4c62d907e7a78d0a794862b690c4c4691ed87ea86b2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None,
                   help="prompts_manifest.csv; usado apenas para metadados experimentais.")
    p.add_argument("--tesseract", type=Path, default=None)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_v7():
    v7_path = Path(ba_v7.__file__).resolve()
    v7_hash = sha256(v7_path)
    if v7_hash != EXPECTED_V7_SHA256:
        raise RuntimeError(
            "B-A V7 diferente da versão congelada.\n"
            f"Arquivo: {v7_path}\n"
            f"SHA esperado: {EXPECTED_V7_SHA256}\n"
            f"SHA encontrado: {v7_hash}"
        )

    presence_path = Path(ba_v7.v4.__file__).resolve()
    presence_hash = sha256(presence_path)
    if presence_hash != EXPECTED_PRESENCE_V4_SHA256:
        raise RuntimeError(
            "V4 de presença diferente da versão congelada.\n"
            f"Arquivo: {presence_path}\n"
            f"SHA esperado: {EXPECTED_PRESENCE_V4_SHA256}\n"
            f"SHA encontrado: {presence_hash}"
        )

    ba_v7.check_v4_hash()
    return v7_path, v7_hash, presence_path, presence_hash


def load_metadata(project_root: Path, explicit=None):
    p = explicit.expanduser().resolve() if explicit else project_root / "prompts_manifest.csv"
    if not p.exists():
        return p, pd.DataFrame()

    df = pd.read_csv(p, dtype=str)
    keep = [
        c for c in [
            "unit_id", "participant_id", "condition", "type", "task",
            "prompt_file", "prompt_sha256"
        ] if c in df.columns
    ]
    if "unit_id" not in keep:
        raise RuntimeError(f"Manifesto sem unit_id: {p}")
    return p, df[keep].drop_duplicates("unit_id")


def process_all(inv, root, outdir, lang):
    crop_dir = outdir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    image_rows = []
    axis_rows = []
    label_rows = []
    numeric_candidate_rows = []
    scale_vote_rows = []
    errors = []

    total = len(inv)

    for i, r in inv.reset_index(drop=True).iterrows():
        path = root / r["filename"]

        print(
            f"[{i+1:04d}/{total}] {r['filename']} ...",
            end=" ", flush=True
        )

        try:
            full_img = Image.open(path).convert("RGB")
            pres = ba_v7.v4.analyze(path)
            axes = ba_v7.v4.axis_v3.analyze(path)

            common_image = {
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "unit_number": r.get("unit_number", ""),
                "repeat": r.get("repeat", ""),
            }

            for c in [
                "participant_id", "condition", "type", "task",
                "prompt_file", "prompt_sha256"
            ]:
                if c in r.index:
                    common_image[c] = r.get(c, "")

            image_row = dict(common_image)

            for axis in ("X", "Y"):
                ar, lrs, svrs, ncrs = ba_v7.axis_extraction(
                    full_img,
                    pres,
                    axes,
                    r["profile"],
                    axis,
                    lang,
                    crop_dir,
                    r["filename"],
                )

                prefix = "x_" if axis == "X" else "y_"
                for k, v in ar.items():
                    if k != "axis":
                        image_row[prefix + k] = v

                common = {
                    "filename": r["filename"],
                    "profile": r["profile"],
                    "unit_id": r["unit_id"],
                    "repeat": r.get("repeat", ""),
                }
                for c in ["participant_id", "condition", "type", "task"]:
                    if c in r.index:
                        common[c] = r.get(c, "")

                axis_rows.append({**common, **ar})

                for lr in lrs:
                    label_rows.append({**common, **lr})

                for nc in ncrs:
                    numeric_candidate_rows.append({**common, **nc})

                for sv in svrs:
                    scale_vote_rows.append({**common, **sv})

            image_rows.append(image_row)
            print("OK", flush=True)

        except Exception as exc:
            errors.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "repeat": r.get("repeat", ""),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            print(f"ERRO: {type(exc).__name__}: {exc}", flush=True)

    return (
        pd.DataFrame(image_rows),
        pd.DataFrame(axis_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(numeric_candidate_rows),
        pd.DataFrame(scale_vote_rows),
        pd.DataFrame(errors),
    )


def make_audit_all_axes(axes_df):
    if axes_df.empty:
        return pd.DataFrame()

    preferred = [
        "filename", "profile", "unit_id", "repeat",
        "participant_id", "condition", "axis", "kind",
        "presence_candidate", "segmentation_ok_v4",
        "n_tick_boxes_after_layer_clip",
        "n_usable_individual", "individual_usable_frac",
        "individual_text_candidate",
        "band_ocr_normalized",
        "temporal_years_spatial",
        "numeric_selection_status",
        "numeric_selected_values_spatial",
        "numeric_effective_values_spatial",
        "numeric_selection_sources",
        "numeric_selected_candidate_sources",
        "extraction_mode_candidate",
        "extraction_support_candidate",
        "scale_interpretation_status",
        "scale_text_normalized",
        "scale_factor",
        "numeric_lane_ocr_raw_tokens",
    ]
    cols = [c for c in preferred if c in axes_df.columns]
    audit = axes_df[cols].copy()

    audit["manual_ticklabel_extraction_result"] = ""
    audit["manual_visible_ticklabels"] = ""
    audit["manual_missing_labels"] = ""
    audit["manual_false_labels"] = ""
    audit["manual_notes"] = ""
    return audit


def make_priority_review(axes_df):
    if axes_df.empty:
        return pd.DataFrame()

    support = axes_df.get(
        "extraction_support_candidate",
        pd.Series("", index=axes_df.index)
    ).astype(str)

    selection = axes_df.get(
        "numeric_selection_status",
        pd.Series("", index=axes_df.index)
    ).astype(str)

    mask = (
        support.isin(["PARTIAL_CANDIDATE", "FAIL_CANDIDATE"])
        |
        selection.isin([
            "SELECTED_BY_COHERENCE",
            "PARTIAL_UNREADABLE",
            "AMBIGUOUS",
            "INSUFFICIENT",
        ])
    )

    review = axes_df[mask].copy()
    review["manual_ticklabel_extraction_result"] = ""
    review["manual_visible_ticklabels"] = ""
    review["manual_missing_labels"] = ""
    review["manual_false_labels"] = ""
    review["manual_notes"] = ""
    return review


def summarize(axes_df):
    rows = []
    if axes_df.empty:
        return pd.DataFrame(rows)

    for (profile, axis, kind), g in axes_df.groupby(
        ["profile", "axis", "kind"], dropna=False
    ):
        support = g.get(
            "extraction_support_candidate",
            pd.Series("", index=g.index)
        ).value_counts().to_dict()

        selection = g.get(
            "numeric_selection_status",
            pd.Series("", index=g.index)
        ).value_counts().to_dict()

        rows.append({
            "profile": profile,
            "axis": axis,
            "kind": kind,
            "n": len(g),
            "multi_evidence": int(support.get("MULTI_EVIDENCE", 0)),
            "strong_candidate": int(support.get("STRONG_CANDIDATE", 0)),
            "single_evidence": int(support.get("SINGLE_EVIDENCE", 0)),
            "partial_candidate": int(support.get("PARTIAL_CANDIDATE", 0)),
            "fail_candidate": int(support.get("FAIL_CANDIDATE", 0)),
            "select_coherent_direct": int(selection.get("COHERENT_DIRECT", 0)),
            "select_by_coherence": int(selection.get("SELECTED_BY_COHERENCE", 0)),
            "select_partial_unreadable": int(selection.get("PARTIAL_UNREADABLE", 0)),
            "select_ambiguous": int(selection.get("AMBIGUOUS", 0)),
            "select_insufficient": int(selection.get("INSUFFICIENT", 0)),
        })

    return pd.DataFrame(rows)


def write_empty_csv(path, columns):
    pd.DataFrame(columns=list(columns)).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def main():
    args = parse_args()

    root = args.root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    outdir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / OUT_DIRNAME
    )
    outdir.mkdir(parents=True, exist_ok=True)

    v7_path, v7_hash, presence_path, presence_hash = verify_frozen_v7()
    t_version, lang, langs = ba_v7.configure_tesseract(args.tesseract)

    inv = ba_v7.inventory(root)

    if len(inv) != 3120:
        raise RuntimeError(
            f"Inventário canônico na RAIZ={len(inv)}; esperado=3120."
        )
    if inv["unit_id"].nunique() != 312:
        raise RuntimeError(
            f"Unidades canônicas={inv['unit_id'].nunique()}; esperado=312."
        )

    manifest_path, meta = load_metadata(project_root, args.manifest)
    if not meta.empty:
        inv = inv.merge(meta, on="unit_id", how="left", validate="many_to_one")

    inv = inv.sort_values(
        ["profile", "unit_id", "repeat"]
    ).reset_index(drop=True)

    inv.to_csv(
        outdir / "ticklabel_ba_v7_production_manifest.csv",
        index=False, encoding="utf-8-sig"
    )

    print("=" * 80)
    print("B-A V7 — PRODUÇÃO CONGELADA — 3.120 IMAGENS")
    print("=" * 80)
    print(f"B-A V7 SHA-256: {v7_hash}")
    print(f"V4 presença SHA-256: {presence_hash}")
    print(f"Tesseract: {t_version}")
    print(f"Idioma OCR: {lang}")
    print(f"Imagens: {len(inv)}")
    print(f"Unidades: {inv['unit_id'].nunique()}")
    print(f"Manifesto experimental: {manifest_path} ({'OK' if manifest_path.exists() else 'AUSENTE'})")
    print("F NÃO é carregada por este programa.")
    print()

    (
        images_df,
        axes_df,
        labels_df,
        numeric_candidates_df,
        scale_votes_df,
        err_df,
    ) = process_all(inv, root, outdir, lang)

    images_df.to_csv(
        outdir / "ticklabel_ba_v7_production_images.csv",
        index=False, encoding="utf-8-sig"
    )
    axes_df.to_csv(
        outdir / "ticklabel_ba_v7_production_axes.csv",
        index=False, encoding="utf-8-sig"
    )
    labels_df.to_csv(
        outdir / "ticklabel_ba_v7_production_labels.csv",
        index=False, encoding="utf-8-sig"
    )
    numeric_candidates_df.to_csv(
        outdir / "ticklabel_ba_v7_production_numeric_candidates.csv",
        index=False, encoding="utf-8-sig"
    )
    scale_votes_df.to_csv(
        outdir / "ticklabel_ba_v7_production_scale_votes.csv",
        index=False, encoding="utf-8-sig"
    )

    if err_df.empty:
        write_empty_csv(
            outdir / "ticklabel_ba_v7_production_errors.csv",
            ["filename", "profile", "unit_id", "repeat", "error_type", "error_message"]
        )
    else:
        err_df.to_csv(
            outdir / "ticklabel_ba_v7_production_errors.csv",
            index=False, encoding="utf-8-sig"
        )

    audit_all = make_audit_all_axes(axes_df)
    audit_all.to_csv(
        outdir / "ticklabel_ba_v7_production_audit_all_axes.csv",
        index=False, encoding="utf-8-sig"
    )

    priority = make_priority_review(axes_df)
    priority.to_csv(
        outdir / "ticklabel_ba_v7_production_priority_review.csv",
        index=False, encoding="utf-8-sig"
    )

    automatic = summarize(axes_df)
    automatic.to_csv(
        outdir / "ticklabel_ba_v7_production_summary_by_profile_axis.csv",
        index=False, encoding="utf-8-sig"
    )

    lines = [
        "B-A V7 — PRODUÇÃO CONGELADA — 3.120 IMAGENS",
        "=" * 78,
        f"Arquivo V7: {v7_path}",
        f"B-A V7 SHA-256: {v7_hash}",
        f"V4 presença SHA-256: {presence_hash}",
        f"Tesseract: {t_version}",
        f"Idioma OCR: {lang}",
        "",
        "ESCOPO:",
        f"- imagens canônicas processadas com sucesso: {len(images_df)}/{len(inv)}",
        f"- unidades canônicas: {inv['unit_id'].nunique()}",
        f"- eixos produzidos: {len(axes_df)}",
        f"- erros de execução: {len(err_df)}",
        f"- eixos em revisão prioritária automática: {len(priority)}",
        "",
        "INDEPENDÊNCIA:",
        "- F não foi carregada;",
        "- categorias/valores esperados não foram fornecidos;",
        "- a saída é observacional B-A;",
        "- nenhum parâmetro foi recalibrado;",
        "",
        "RESUMO AUTOMÁTICO POR PERFIL × EIXO × TIPO:",
    ]

    for _, r in automatic.iterrows():
        lines.append(
            f"{r['profile']} {r['axis']} ({r['kind']}): "
            f"n={int(r['n'])}; "
            f"MULTI={int(r['multi_evidence'])}; "
            f"STRONG={int(r['strong_candidate'])}; "
            f"SINGLE={int(r['single_evidence'])}; "
            f"PARTIAL={int(r['partial_candidate'])}; "
            f"FAIL={int(r['fail_candidate'])}; "
            f"SEL_DIRECT={int(r['select_coherent_direct'])}; "
            f"SEL_COH={int(r['select_by_coherence'])}; "
            f"SEL_PART={int(r['select_partial_unreadable'])}; "
            f"SEL_AMB={int(r['select_ambiguous'])}; "
            f"SEL_INSUF={int(r['select_insufficient'])}"
        )

    summary_path = outdir / "ticklabel_ba_v7_production_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Concluído.")
    print("Resumo:", summary_path)
    print("Eixos:", outdir / "ticklabel_ba_v7_production_axes.csv")
    print("Auditoria:", outdir / "ticklabel_ba_v7_production_audit_all_axes.csv")
    print("Prioridade:", outdir / "ticklabel_ba_v7_production_priority_review.csv")
    print()
    print("NÃO execute B-B final ainda. Primeiro revisar o resumo e a fila de B-A.")


if __name__ == "__main__":
    main()
