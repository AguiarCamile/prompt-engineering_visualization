# -*- coding: utf-8 -*-
r"""
axis_presence_full_v5_1.py

PROCESSAMENTO FINAL V5.1 — PRESENÇA DOS EIXOS X/Y
================================================

Detector congelado
------------------
axis_conformity_calibration_v3.py
SHA-256 esperado:
1582555adf23d8ed2a6b23e89eaefdcd317d0e11a2f7d361fbd69c6258866bfe

Validação independente anterior
-------------------------------
Holdout V4:
- 120 imagens novas;
- 20 por perfil;
- 0 sobreposição com calibração;
- X detectado em 120/120;
- Y detectado em 120/120;
- auditoria visual aprovada.

Escopo desta V5
---------------
Processar as 3.120 imagens canônicas da RAIZ, sem recursão, e registrar:

- presença do eixo X;
- presença do eixo Y;
- presença simultânea de X e Y;
- status estrutural:
    BOTH_PRESENT
    X_ONLY
    Y_ONLY
    NONE
    UNEVALUABLE

A cor dos eixos NÃO faz parte desta etapa.

Princípio metodológico
----------------------
A detecção é geométrica e independente da especificação cromática.
Como todos os seis perfis exigem eixo X e eixo Y, a comparação estrutural
é posterior à extração.

Saída padrão
------------
C:\Users\Labvis\Downloads\imagens3120\imagens\_axis_presence_full_v5_1

Arquivos:
- axis_presence_images_v5_1.csv
- axis_presence_units_v5_1.csv
- axis_presence_by_profile_v5_1.csv
- axis_presence_by_condition_v5_1.csv   (se crosswalk encontrado)
- axis_presence_by_task_v5_1.csv
- axis_presence_by_technique_v5_1.csv
- axis_presence_errors_v5_1.csv
- axis_presence_summary_v5_1.txt
- review_overlays/*.png  (somente casos != BOTH_PRESENT ou UNEVALUABLE)
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

import axis_conformity_calibration_v3 as v3


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
OUT_DIRNAME = "_axis_presence_full_v5_1"

EXPECTED_V3_SHA256 = "1582555adf23d8ed2a6b23e89eaefdcd317d0e11a2f7d361fbd69c6258866bfe"

PROFILES = ("BI", "BC", "LI", "LC", "SI", "SC")

NAME_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--crosswalk", type=Path, default=None)
    p.add_argument(
        "--save-review-overlays",
        action="store_true",
        help="Salvar overlays somente para casos não BOTH_PRESENT/UNEVALUABLE."
    )
    return p.parse_args()


def check_v3_hash():
    path = Path(v3.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_V3_SHA256:
        raise RuntimeError(
            "axis_conformity_calibration_v3.py foi alterado.\n"
            f"SHA esperado: {EXPECTED_V3_SHA256}\n"
            f"SHA encontrado: {actual}"
        )
    return actual


def inventory(root):
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        m = NAME_RE.match(p.name)
        if not m:
            continue

        profile = m.group(1).upper()
        unit = int(m.group(2))
        repeat = int(m.group(3))

        rows.append({
            "filename": p.name,
            "path": str(p),
            "profile": profile,
            "unit_number": unit,
            "repeat": repeat,
            "unit_id": f"{profile}_{unit:03d}",
            "technique": profile[0],
            "task": profile[1],
        })

    return pd.DataFrame(rows)


def normalize_crosswalk_columns(cw):
    """
    Tenta normalizar nomes já usados no projeto.
    O processamento estrutural NÃO depende do crosswalk.
    """
    rename = {}

    candidates = {
        "participant_id": [
            "participant_id", "participant", "participant_number",
            "participante", "p"
        ],
        "condition_code": [
            "condition_code", "condition", "prompt_condition_code",
            "condicao_codigo", "condition_id"
        ],
        "condition_label": [
            "condition_label", "condition_name", "prompt_condition",
            "condicao", "condition"
        ],
        "examples": [
            "examples", "with_examples", "has_examples", "example_factor"
        ],
        "formal_structure": [
            "formal_structure", "with_structure", "structure", "has_structure",
            "formal_structure_factor"
        ],
    }

    lowmap = {str(c).lower(): c for c in cw.columns}

    for target, opts in candidates.items():
        if target in cw.columns:
            continue
        for opt in opts:
            if opt.lower() in lowmap:
                src = lowmap[opt.lower()]
                if src not in rename:
                    rename[src] = target
                    break

    return cw.rename(columns=rename)


def find_crosswalk(root, explicit=None):
    """
    V5.1: usa SOMENTE o crosswalk experimental V3 final.
    Não faz fallback para arquivos antigos de cor/crosswalk.
    """
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Crosswalk informado não encontrado: {p}")
        return p

    candidates = [
        root / "_experimental_crosswalk_v3" / "experimental_unit_crosswalk_v3.csv",
        root / "_experimental_crosswalk_v3" / "experimental_image_crosswalk_v3.csv",
        root.parent / "_experimental_crosswalk_v3" / "experimental_unit_crosswalk_v3.csv",
        root.parent / "_experimental_crosswalk_v3" / "experimental_image_crosswalk_v3.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    expected = root / "_experimental_crosswalk_v3" / "experimental_unit_crosswalk_v3.csv"
    raise FileNotFoundError(
        "Crosswalk V3 final não encontrado automaticamente.\\n"
        "Esperado, preferencialmente:\\n"
        f"{expected}\\n"
        "Gere-o com build_experimental_crosswalk_v3.py ou informe --crosswalk."
    )


def attach_crosswalk(df, crosswalk_path):
    if crosswalk_path is None:
        return df, None

    cw = pd.read_csv(crosswalk_path, dtype=str)
    cw = normalize_crosswalk_columns(cw)

    if "unit_id" not in cw.columns:
        raise RuntimeError(f"Crosswalk V3 sem unit_id: {crosswalk_path}")

    n_units = cw["unit_id"].astype(str).nunique()
    if n_units != 312:
        raise RuntimeError(
            f"Crosswalk V3 inválido/incompleto: {n_units} unidades únicas; esperado=312."
        )

    required_meta = {"participant_id", "condition_code", "condition_label"}
    missing_meta = required_meta - set(cw.columns)
    if missing_meta:
        raise RuntimeError(
            "Crosswalk V3 sem metadados experimentais obrigatórios: "
            + ", ".join(sorted(missing_meta))
        )

    # Uma linha por unidade.
    keep = ["unit_id"]
    for c in [
        "participant_id", "condition_code", "condition_label",
        "examples", "formal_structure"
    ]:
        if c in cw.columns:
            keep.append(c)

    cwu = cw[keep].drop_duplicates("unit_id").copy()

    out = df.merge(cwu, on="unit_id", how="left")

    if "condition_code" in out.columns:
        out["condition_code"] = (
            out["condition_code"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(2)
        )

    return out, None


def status_from(res):
    xe = int(res.get("axis_x_evaluable", 0) or 0)
    ye = int(res.get("axis_y_evaluable", 0) or 0)

    if not (xe and ye):
        return "UNEVALUABLE", None, None, None

    xp = int(res.get("axis_x_present_candidate", 0) or 0)
    yp = int(res.get("axis_y_present_candidate", 0) or 0)

    if xp and yp:
        status = "BOTH_PRESENT"
    elif xp:
        status = "X_ONLY"
    elif yp:
        status = "Y_ONLY"
    else:
        status = "NONE"

    both = int(xp and yp)

    return status, xp, yp, both


def make_overlay(path, row, out_path):
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)

    xl = row.get("plot_x_left")
    xr = row.get("plot_x_right")
    yt = row.get("plot_y_top")
    yb = row.get("plot_y_bottom")

    def ok(v):
        return v not in ("", None) and not pd.isna(v)

    if all(ok(v) for v in (xl, xr, yt, yb)):
        d.rectangle(
            [int(xl), int(yt), int(xr), int(yb)],
            outline=(120,120,120),
            width=2
        )

    xs = row.get("axis_x_segment_start")
    xe = row.get("axis_x_segment_end")
    yc = row.get("axis_x_coord")

    if all(ok(v) for v in (xs, xe, yc)):
        d.line(
            [(int(xs), int(yc)), (int(xe), int(yc))],
            fill=(220,0,0),
            width=3
        )

    xc = row.get("axis_y_coord")
    if all(ok(v) for v in (xc, yt, yb)):
        d.line(
            [(int(xc), int(yt)), (int(xc), int(yb))],
            fill=(0,80,220),
            width=3
        )

    im.save(out_path)


def aggregate_units(df):
    rows = []

    for unit_id, g in df.groupby("unit_id", sort=True):
        base = {
            "unit_id": unit_id,
            "profile": g["profile"].iloc[0],
            "technique": g["technique"].iloc[0],
            "task": g["task"].iloc[0],
            "n_images": len(g),
        }

        for c in [
            "participant_id", "condition_code", "condition_label",
            "examples", "formal_structure"
        ]:
            if c in g.columns:
                vals = g[c].dropna()
                base[c] = vals.iloc[0] if len(vals) else ""

        eval_x = g["axis_x_evaluable"] == 1
        eval_y = g["axis_y_evaluable"] == 1
        eval_both = eval_x & eval_y

        base["n_axis_x_evaluable"] = int(eval_x.sum())
        base["n_axis_y_evaluable"] = int(eval_y.sum())
        base["n_both_evaluable"] = int(eval_both.sum())

        base["n_axis_x_present"] = int(
            g.loc[eval_x, "axis_x_present"].fillna(0).sum()
        )
        base["n_axis_y_present"] = int(
            g.loc[eval_y, "axis_y_present"].fillna(0).sum()
        )
        base["n_both_axes_present"] = int(
            g.loc[eval_both, "both_axes_present"].fillna(0).sum()
        )

        base["p_axis_x_present"] = (
            base["n_axis_x_present"] / base["n_axis_x_evaluable"]
            if base["n_axis_x_evaluable"] else float("nan")
        )
        base["p_axis_y_present"] = (
            base["n_axis_y_present"] / base["n_axis_y_evaluable"]
            if base["n_axis_y_evaluable"] else float("nan")
        )
        base["p_both_axes_present"] = (
            base["n_both_axes_present"] / base["n_both_evaluable"]
            if base["n_both_evaluable"] else float("nan")
        )

        rows.append(base)

    return pd.DataFrame(rows)


def grouped_summary(df, group_col):
    if group_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for key, g in df.groupby(group_col, dropna=False):
        eval_x = g["axis_x_evaluable"] == 1
        eval_y = g["axis_y_evaluable"] == 1
        eval_b = eval_x & eval_y

        nx = int(eval_x.sum())
        ny = int(eval_y.sum())
        nb = int(eval_b.sum())

        xpres = int(g.loc[eval_x, "axis_x_present"].fillna(0).sum())
        ypres = int(g.loc[eval_y, "axis_y_present"].fillna(0).sum())
        bpres = int(g.loc[eval_b, "both_axes_present"].fillna(0).sum())

        rows.append({
            group_col: key,
            "n_images": len(g),
            "n_x_evaluable": nx,
            "n_y_evaluable": ny,
            "n_both_evaluable": nb,
            "n_x_present": xpres,
            "p_x_present": xpres/nx if nx else float("nan"),
            "n_y_present": ypres,
            "p_y_present": ypres/ny if ny else float("nan"),
            "n_both_present": bpres,
            "p_both_present": bpres/nb if nb else float("nan"),
            "n_unevaluable": int((g["axes_status"] == "UNEVALUABLE").sum()),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    outdir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / OUT_DIRNAME
    )
    outdir.mkdir(parents=True, exist_ok=True)

    review_dir = outdir / "review_overlays"
    if args.save_review_overlays:
        review_dir.mkdir(parents=True, exist_ok=True)

    actual_hash = check_v3_hash()

    inv = inventory(root)

    if len(inv) != 3120:
        print(
            f"AVISO: foram encontradas {len(inv)} imagens canônicas na raiz; "
            "o conjunto final esperado contém 3120."
        )

    crosswalk_path = find_crosswalk(root, args.crosswalk)

    rows = []
    errors = []

    print("="*80)
    print("PROCESSAMENTO FINAL V5.1 — PRESENÇA DOS EIXOS X/Y")
    print("="*80)
    print(f"Detector V3 SHA-256: {actual_hash}")
    print(f"Imagens canônicas na raiz: {len(inv)}")
    print(f"Crosswalk: {crosswalk_path if crosswalk_path else 'não encontrado'}")
    print()

    for _, r in inv.iterrows():
        path = Path(r["path"])

        try:
            res = v3.analyze(path)
            status, xp, yp, both = status_from(res)

            row = {
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "unit_number": r["unit_number"],
                "repeat": r["repeat"],
                "technique": r["technique"],
                "task": r["task"],

                "plot_status": res.get("plot_status", ""),
                "plot_confidence": res.get("plot_confidence", ""),

                "plot_x_left": res.get("plot_x_left", ""),
                "plot_x_right": res.get("plot_x_right", ""),
                "plot_y_top": res.get("plot_y_top", ""),
                "plot_y_bottom": res.get("plot_y_bottom", ""),

                "axis_x_evaluable": int(res.get("axis_x_evaluable", 0) or 0),
                "axis_y_evaluable": int(res.get("axis_y_evaluable", 0) or 0),

                "axis_x_present": xp,
                "axis_y_present": yp,
                "both_axes_present": both,
                "axes_status": status,

                "axis_x_coord": res.get("axis_x_coord", ""),
                "axis_y_coord": res.get("axis_y_coord", ""),

                "axis_x_ridge_support": res.get("axis_x_ridge_support", ""),
                "axis_y_ridge_support": res.get("axis_y_ridge_support", ""),

                "axis_x_ridge_run_frac": res.get("axis_x_ridge_run_frac", ""),
                "axis_y_ridge_run_frac": res.get("axis_y_ridge_run_frac", ""),

                "axis_x_segment_start": res.get("axis_x_segment_start", ""),
                "axis_x_segment_end": res.get("axis_x_segment_end", ""),
                "axis_x_segment_frac": res.get("axis_x_segment_frac", ""),

                "axis_y_anchor_source": res.get("axis_y_anchor_source", ""),
            }

            rows.append(row)

            if args.save_review_overlays and status != "BOTH_PRESENT":
                make_overlay(path, row, review_dir / r["filename"])

        except Exception as exc:
            errors.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

        done = len(rows) + len(errors)
        if done % 250 == 0 or done == len(inv):
            print(f"Processadas: {done}/{len(inv)}")

    df = pd.DataFrame(rows)
    err = pd.DataFrame(errors)

    df, cw_warning = attach_crosswalk(df, crosswalk_path)

    images_path = outdir / "axis_presence_images_v5_1.csv"
    errors_path = outdir / "axis_presence_errors_v5_1.csv"

    df.to_csv(images_path, index=False, encoding="utf-8-sig")
    err.to_csv(errors_path, index=False, encoding="utf-8-sig")

    units = aggregate_units(df)
    units.to_csv(
        outdir / "axis_presence_units_v5_1.csv",
        index=False,
        encoding="utf-8-sig"
    )

    grouped_summary(df, "profile").to_csv(
        outdir / "axis_presence_by_profile_v5_1.csv",
        index=False,
        encoding="utf-8-sig"
    )
    grouped_summary(df, "task").to_csv(
        outdir / "axis_presence_by_task_v5_1.csv",
        index=False,
        encoding="utf-8-sig"
    )
    grouped_summary(df, "technique").to_csv(
        outdir / "axis_presence_by_technique_v5_1.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if "condition_label" in df.columns:
        grouped_summary(df, "condition_label").to_csv(
            outdir / "axis_presence_by_condition_v5_1.csv",
            index=False,
            encoding="utf-8-sig"
        )
    elif "condition_code" in df.columns:
        grouped_summary(df, "condition_code").to_csv(
            outdir / "axis_presence_by_condition_v5_1.csv",
            index=False,
            encoding="utf-8-sig"
        )
    else:
        pd.DataFrame().to_csv(
            outdir / "axis_presence_by_condition_v5_1.csv",
            index=False,
            encoding="utf-8-sig"
        )

    n = len(df)
    nerr = len(err)

    status_counts = df["axes_status"].value_counts()

    x_eval = df["axis_x_evaluable"] == 1
    y_eval = df["axis_y_evaluable"] == 1
    b_eval = x_eval & y_eval

    nx = int(x_eval.sum())
    ny = int(y_eval.sum())
    nb = int(b_eval.sum())

    xpres = int(df.loc[x_eval, "axis_x_present"].fillna(0).sum())
    ypres = int(df.loc[y_eval, "axis_y_present"].fillna(0).sum())
    bpres = int(df.loc[b_eval, "both_axes_present"].fillna(0).sum())

    lines = [
        "PROCESSAMENTO FINAL V5.1 — PRESENÇA DOS EIXOS X/Y",
        "="*78,
        f"Detector V3 SHA-256: {actual_hash}",
        f"Imagens processadas: {n}",
        f"Erros: {nerr}",
        f"Unidades: {df['unit_id'].nunique() if len(df) else 0}",
        "",
        "RESULTADO GLOBAL:",
        f"X avaliável: {nx}",
        f"X presente: {xpres}/{nx} "
        f"({(xpres/nx if nx else float('nan')):.4%})",
        f"Y avaliável: {ny}",
        f"Y presente: {ypres}/{ny} "
        f"({(ypres/ny if ny else float('nan')):.4%})",
        f"Ambos avaliáveis: {nb}",
        f"Ambos presentes: {bpres}/{nb} "
        f"({(bpres/nb if nb else float('nan')):.4%})",
        "",
        "STATUS:",
    ]

    for status in ["BOTH_PRESENT", "X_ONLY", "Y_ONLY", "NONE", "UNEVALUABLE"]:
        lines.append(
            f"{status}: {int(status_counts.get(status, 0))}"
        )

    lines.extend([
        "",
        "UNIDADES:",
        f"Unidades com 10/10 BOTH_PRESENT: "
        f"{int((units['n_both_axes_present'] == 10).sum())}/"
        f"{len(units)}",
        f"Unidades com pelo menos 1 não-BOTH/unevaluable: "
        f"{int((units['n_both_axes_present'] < units['n_images']).sum())}/"
        f"{len(units)}",
    ])

    if crosswalk_path:
        lines.append("")
        lines.append(f"Crosswalk utilizado: {crosswalk_path}")
    if cw_warning:
        lines.append(f"AVISO crosswalk: {cw_warning}")

    if args.save_review_overlays:
        lines.extend([
            "",
            f"Overlays para revisão: {review_dir}",
        ])

    summary = outdir / "axis_presence_summary_v5_1.txt"
    summary.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Concluído.")
    print("Resumo:", summary)
    print("CSV por imagem:", images_path)


if __name__ == "__main__":
    main()
