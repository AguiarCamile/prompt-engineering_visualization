# -*- coding: utf-8 -*-
r"""
chart_type_multiclass_validation_v1.py

VALIDAÇÃO MULTICLASSE — TIPO DE GRÁFICO EFETIVAMENTE GERADO

Aplica os três detectores estruturais V3 (barras, linhas e dispersão)
a cada imagem do holdout V4, sem usar o perfil esperado para forçar
a classificação.

Assinaturas:
100 -> BAR_ONLY
010 -> LINE_ONLY
001 -> SCATTER_ONLY
110 -> BAR_LINE
101 -> BAR_SCATTER
011 -> LINE_SCATTER
111 -> BAR_LINE_SCATTER
000 -> NONE_OR_OTHER
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
DEFAULT_HOLDOUT_DIRNAME = "_mark_presence_holdout_v4"
DEFAULT_OUTPUT_DIRNAME = "_chart_type_multiclass_validation_v1"
PROFILES = ["BI", "BC", "LI", "LC", "SI", "SC"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--holdout-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_v3():
    try:
        import mark_presence_validation_v3 as v3
    except Exception as exc:
        raise RuntimeError(
            "Coloque mark_presence_validation_v3.py na mesma pasta deste programa."
        ) from exc
    return v3


def signature_label(b, l, s):
    mapping = {
        (1,0,0): "BAR_ONLY",
        (0,1,0): "LINE_ONLY",
        (0,0,1): "SCATTER_ONLY",
        (1,1,0): "BAR_LINE",
        (1,0,1): "BAR_SCATTER",
        (0,1,1): "LINE_SCATTER",
        (1,1,1): "BAR_LINE_SCATTER",
        (0,0,0): "NONE_OR_OTHER",
    }
    return mapping[(int(bool(b)), int(bool(l)), int(bool(s)))]


def expected_label(technique):
    return {"B": "BAR_ONLY", "L": "LINE_ONLY", "S": "SCATTER_ONLY"}[technique]


def font_default(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def contact_sheet(profile_df, root, output_path):
    if profile_df.empty:
        return

    cols = 3
    panel_w = 500
    panel_h = 390
    img_h = 300
    rows = math.ceil(len(profile_df) / cols)

    sheet = Image.new("RGB", (cols*panel_w, rows*panel_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = font_default(16)

    for idx, row in profile_df.reset_index(drop=True).iterrows():
        path = root / row["filename"]
        if not path.exists():
            continue

        im = Image.open(path).convert("RGB")
        im.thumbnail((panel_w-20, img_h-10))

        col = idx % cols
        rr = idx // cols
        px = col*panel_w + (panel_w-im.width)//2
        py = rr*panel_h + 5
        sheet.paste(im, (px, py))

        txt = (
            f"{row['filename']} | {row['observed_signature']}\n"
            f"B={int(row['bar_present'])} "
            f"L={int(row['line_present'])} "
            f"S={int(row['scatter_present'])} | "
            f"esperado={row['expected_technique']}"
        )
        draw.multiline_text(
            (col*panel_w+10, rr*panel_h+img_h+4),
            txt,
            fill="black",
            font=font,
            spacing=3
        )

    sheet.save(output_path)


def summary_matrix(df, row_col):
    tab = pd.crosstab(df[row_col], df["observed_signature"])
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    pct.columns = [f"{c}_pct" for c in pct.columns]
    return tab.join(pct)


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    holdout_dir = (
        args.holdout_dir.expanduser().resolve()
        if args.holdout_dir is not None
        else root / DEFAULT_HOLDOUT_DIRNAME
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_DIRNAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout_csv = holdout_dir / "mark_presence_holdout_v4.csv"
    if not holdout_csv.exists():
        raise FileNotFoundError(f"Holdout não encontrado: {holdout_csv}")

    holdout = pd.read_csv(holdout_csv)
    v3 = load_v3()

    rows = []
    errors = []

    print("="*80)
    print("VALIDAÇÃO MULTICLASSE — TIPO DE GRÁFICO EFETIVAMENTE GERADO")
    print("="*80)
    print(f"Holdout: {len(holdout)} imagens")

    for _, r in holdout.iterrows():
        path = root / r["filename"]
        try:
            db = v3.analyze_one(path, "B")["det"]
            dl = v3.analyze_one(path, "L")["det"]
            ds = v3.analyze_one(path, "S")["det"]

            b = bool(db["present"])
            l = bool(dl["present"])
            s = bool(ds["present"])
            sig = signature_label(b,l,s)

            expected_tech = str(r["technique"]).upper()
            expected_sig = expected_label(expected_tech)

            expected_present = {"B":b, "L":l, "S":s}[expected_tech]

            rows.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "expected_technique": expected_tech,
                "expected_signature": expected_sig,
                "bar_present": int(b),
                "line_present": int(l),
                "scatter_present": int(s),
                "observed_signature": sig,
                "expected_structure_present": int(expected_present),
                "exact_expected_only": int(sig == expected_sig),
                "bar_confidence": float(db["confidence"]),
                "line_confidence": float(dl["confidence"]),
                "scatter_confidence": float(ds["confidence"]),
                "bar_evidence": db["evidence"],
                "line_evidence": dl["evidence"],
                "scatter_evidence": ds["evidence"],
            })

        except Exception as exc:
            errors.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

        done = len(rows) + len(errors)
        if done % 20 == 0 or done == len(holdout):
            print(f"Processadas: {done}/{len(holdout)}")

    df = pd.DataFrame(rows)
    err = pd.DataFrame(errors)

    df.to_csv(output_dir/"chart_type_multiclass_validation_v1.csv",
              index=False, encoding="utf-8-sig")
    err.to_csv(output_dir/"chart_type_multiclass_validation_errors_v1.csv",
               index=False, encoding="utf-8-sig")

    summary_matrix(df, "profile").to_csv(
        output_dir/"chart_type_signature_by_expected_profile_v1.csv",
        encoding="utf-8-sig"
    )
    summary_matrix(df, "expected_technique").to_csv(
        output_dir/"chart_type_signature_by_expected_technique_v1.csv",
        encoding="utf-8-sig"
    )

    for profile in PROFILES:
        contact_sheet(df[df["profile"] == profile], root,
                      output_dir/f"contact_{profile}.png")

    n = len(df)
    lines = [
        "VALIDAÇÃO MULTICLASSE — TIPO DE GRÁFICO EFETIVAMENTE GERADO",
        "="*78,
        f"Imagens processadas: {n}",
        f"Erros: {len(err)}",
        f"Estrutura esperada presente: {int(df['expected_structure_present'].sum())}/{n}",
        f"Somente a estrutura esperada detectada: {int(df['exact_expected_only'].sum())}/{n}",
        "",
        "ASSINATURAS OBSERVADAS:",
    ]

    for sig, count in df["observed_signature"].value_counts().items():
        lines.append(f"{sig}: {count}/{n} ({count/n:.2%})")

    lines += ["", "POR PERFIL:"]
    for profile in PROFILES:
        g = df[df["profile"] == profile]
        if len(g):
            lines.append(
                f"{profile}: exact_expected_only="
                f"{int(g['exact_expected_only'].sum())}/{len(g)} "
                f"({g['exact_expected_only'].mean():.2%})"
            )

    lines += [
        "",
        "PRÓXIMA ETAPA:",
        "- revisar visualmente os seis contact sheets;",
        "- verificar especialmente assinaturas mistas;",
        "- confirmar que BAR_ONLY, LINE_ONLY e SCATTER_ONLY correspondem ao tipo visível;",
        "- se a validação cruzada for adequada, aplicar às 3.120 imagens.",
    ]

    summary_path = output_dir/"chart_type_multiclass_validation_summary_v1.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Abra primeiro:")
    print(summary_path)


if __name__ == "__main__":
    main()
