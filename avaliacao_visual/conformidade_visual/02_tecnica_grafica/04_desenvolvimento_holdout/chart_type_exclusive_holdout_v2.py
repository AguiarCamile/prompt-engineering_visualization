# -*- coding: utf-8 -*-
r"""
chart_type_exclusive_holdout_v2.py

VALIDAÇÃO INDEPENDENTE — CLASSIFICADOR EXCLUSIVO DO TIPO DE GRÁFICO
===================================================================

Objetivo
--------
Classificar cada imagem em UMA única classe estrutural:

    BAR
    LINE
    SCATTER
    OTHER

A V1 multiclasses mostrou que simplesmente aplicar os três detectores binários
e interpretar BAR_SCATTER, LINE_SCATTER etc. como tipos reais não é adequado:
os detectores foram construídos para detectar presença da estrutura esperada,
não para serem mutuamente exclusivos.

A V2 usa descritores internos dos detectores V3 que, na amostra de 120 imagens
da V1, apresentaram separação estrutural clara:

1. BAR:
   MAX_H_FRAC >= 0.10
   (altura relativa máxima do núcleo espesso de uma barra)

2. LINE:
   se não BAR, XBIN_SUPPORT >= 0.50
   (cobertura horizontal por segmentos oblíquos da trajetória)

3. SCATTER:
   se não BAR nem LINE, >=2 marcadores compactos e dispersão espacial
   X >= 0.03 ou Y >= 0.03

4. OTHER:
   nenhum dos critérios anteriores.

IMPORTANTE
----------
- As regras acima foram definidas com a V1 multiclasses e agora são
  verificadas em unidades NOVAS.
- Este programa exclui todas as unidades usadas na calibração V3 e no
  holdout V4.
- O perfil esperado é usado apenas para AUDITORIA/VALIDAÇÃO, nunca para
  escolher a classe observada.

Entrada padrão
--------------
C:\Users\Labvis\Downloads\imagens3120\imagens

Arquivos necessários:
- mark_presence_validation_v3.py
- calibração V3:
  imagens\_mark_presence_validation_v3\mark_presence_validation_v3.csv
- holdout V4:
  imagens\_mark_presence_holdout_v4\mark_presence_holdout_v4.csv

Saída:
imagens\_chart_type_exclusive_holdout_v2

Por padrão:
10 unidades novas por perfil = 60 imagens.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
DEFAULT_OUTPUT_DIRNAME = "_chart_type_exclusive_holdout_v2"
DEFAULT_CALIB_DIRNAME = "_mark_presence_validation_v3"
DEFAULT_HOLDOUT_DIRNAME = "_mark_presence_holdout_v4"

PROFILES = ["BI", "BC", "LI", "LC", "SI", "SC"]
N_UNITS_PER_PROFILE = 10
SEED = 20260829

NAME_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.I
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--units-per-profile", type=int, default=N_UNITS_PER_PROFILE)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def load_v3():
    try:
        import mark_presence_validation_v3 as v3
    except Exception as exc:
        raise RuntimeError(
            "Coloque mark_presence_validation_v3.py na mesma pasta deste programa."
        ) from exc
    return v3


def inventory(root: Path):
    rows = []
    for p in sorted(root.iterdir()):
        if not (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ):
            continue
        m = NAME_RE.match(p.name)
        if not m:
            continue
        profile = m.group(1).upper()
        unit_number = int(m.group(2))
        repeat = int(m.group(3))
        rows.append({
            "filename": p.name,
            "path": str(p),
            "profile": profile,
            "expected_technique": profile[0],
            "unit_number": unit_number,
            "unit_id": f"{profile}_{unit_number:03d}",
            "repeat": repeat,
        })
    return pd.DataFrame(rows)


def read_used_units(root: Path):
    calib = (
        root / DEFAULT_CALIB_DIRNAME / "mark_presence_validation_v3.csv"
    )
    holdout = (
        root / DEFAULT_HOLDOUT_DIRNAME / "mark_presence_holdout_v4.csv"
    )

    if not calib.exists():
        raise FileNotFoundError(f"Calibração V3 não encontrada: {calib}")
    if not holdout.exists():
        raise FileNotFoundError(f"Holdout V4 não encontrado: {holdout}")

    c = pd.read_csv(calib)
    h = pd.read_csv(holdout)

    return set(c["unit_id"].astype(str)) | set(h["unit_id"].astype(str))


def sample_new_units(inv, used_units, n_per_profile, seed):
    rng = random.Random(seed)
    selected = []

    for profile in PROFILES:
        g = inv[
            (inv["profile"] == profile)
            & (~inv["unit_id"].isin(used_units))
        ].copy()

        units = sorted(g["unit_id"].unique().tolist())

        if len(units) < n_per_profile:
            raise RuntimeError(
                f"{profile}: só há {len(units)} unidades novas; "
                f"foram solicitadas {n_per_profile}."
            )

        rng.shuffle(units)
        chosen = units[:n_per_profile]

        for unit_id in chosen:
            records = g[g["unit_id"] == unit_id].to_dict("records")
            rng.shuffle(records)
            selected.append(records[0])

    return pd.DataFrame(selected).sort_values(
        ["profile", "unit_number", "repeat"]
    ).reset_index(drop=True)


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
    Classificação mutuamente exclusiva.
    Não usa perfil esperado.
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
        if path.exists():
            im = Image.open(path).convert("RGB")
            im.thumbnail((panel_w-20, img_h-10))

            col = idx % cols
            rr = idx // cols
            px = col*panel_w + (panel_w-im.width)//2
            py = rr*panel_h + 5
            sheet.paste(im, (px, py))

            text = (
                f"{row['filename']} | observado={row['observed_type']}\n"
                f"esperado={row['expected_technique']} | "
                f"B_h={row['bar_max_h_frac']:.3f} "
                f"L_x={row['line_xbin_support']:.3f} "
                f"S_n={int(row['scatter_n_markers'])}"
            )
            draw.multiline_text(
                (col*panel_w+10, rr*panel_h+img_h+4),
                text,
                fill="black",
                font=font,
                spacing=3
            )

    sheet.save(output_path)


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_DIRNAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inv = inventory(root)
    used_units = read_used_units(root)

    sample = sample_new_units(
        inv,
        used_units,
        n_per_profile=int(args.units_per_profile),
        seed=int(args.seed)
    )

    if set(sample["unit_id"]) & used_units:
        raise RuntimeError("Sobreposição detectada com calibração/holdout anteriores.")

    v3 = load_v3()

    print("="*80)
    print("VALIDAÇÃO INDEPENDENTE — CLASSIFICADOR EXCLUSIVO DE TIPO")
    print("="*80)
    print(f"Inventário: {len(inv)}")
    print(f"Unidades previamente usadas/excluídas: {len(used_units)}")
    print(f"Nova amostra: {len(sample)}")
    print()

    rows = []
    errors = []

    for _, r in sample.iterrows():
        path = Path(r["path"])
        try:
            db = v3.analyze_one(path, "B")["det"]
            dl = v3.analyze_one(path, "L")["det"]
            ds = v3.analyze_one(path, "S")["det"]

            cls = classify_exclusive(db, dl, ds)

            expected_map = {"B":"BAR", "L":"LINE", "S":"SCATTER"}
            expected_type = expected_map[r["expected_technique"]]

            rows.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "repeat": r["repeat"],
                "expected_technique": r["expected_technique"],
                "expected_type": expected_type,
                **cls,
                "matches_expected_type": int(
                    cls["observed_type"] == expected_type
                ),
            })

        except Exception as exc:
            errors.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

        done = len(rows) + len(errors)
        if done % 20 == 0 or done == len(sample):
            print(f"Processadas: {done}/{len(sample)}")

    df = pd.DataFrame(rows)
    err = pd.DataFrame(errors)

    df.to_csv(
        output_dir/"chart_type_exclusive_holdout_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )
    err.to_csv(
        output_dir/"chart_type_exclusive_holdout_errors_v2.csv",
        index=False,
        encoding="utf-8-sig"
    )

    matrix = pd.crosstab(
        df["profile"],
        df["observed_type"]
    )

    for c in ["BAR","LINE","SCATTER","OTHER"]:
        if c not in matrix.columns:
            matrix[c] = 0

    matrix = matrix[["BAR","LINE","SCATTER","OTHER"]]
    matrix_pct = matrix.div(matrix.sum(axis=1), axis=0)*100

    out = matrix.copy()
    for c in matrix.columns:
        out[f"{c}_pct"] = matrix_pct[c]

    out.to_csv(
        output_dir/"chart_type_exclusive_holdout_matrix_v2.csv",
        encoding="utf-8-sig"
    )

    for profile in PROFILES:
        contact_sheet(
            df[df["profile"] == profile],
            root,
            output_dir/f"contact_{profile}.png"
        )

    n = len(df)
    matches = int(df["matches_expected_type"].sum())

    lines = [
        "VALIDAÇÃO INDEPENDENTE — CLASSIFICADOR EXCLUSIVO DO TIPO DE GRÁFICO",
        "="*80,
        f"Imagens processadas: {n}",
        f"Erros: {len(err)}",
        f"Sobreposição com unidades anteriores: 0",
        f"Tipo observado = tipo esperado: {matches}/{n} ({matches/n:.2%})",
        "",
        "POR PERFIL:",
    ]

    for profile in PROFILES:
        g = df[df["profile"] == profile]
        vc = g["observed_type"].value_counts()
        parts = [
            f"{k}={int(v)}"
            for k,v in vc.items()
        ]
        lines.append(
            f"{profile}: " + ", ".join(parts)
        )

    lines.extend([
        "",
        "CRITÉRIOS EXCLUSIVOS:",
        "BAR: bar_max_h_frac >= 0.10",
        "LINE: se não BAR, line_xbin_support >= 0.50",
        "SCATTER: se não BAR/LINE, >=2 marcadores e dispersão X ou Y >=0.03",
        "OTHER: nenhum critério satisfeito",
        "",
        "PRÓXIMO PASSO:",
        "- revisar visualmente os seis contact sheets;",
        "- discrepâncias entre esperado e observado NÃO são automaticamente erros;",
        "  podem ser imagens em que o modelo realmente gerou outra técnica;",
        "- se a inspeção confirmar a classificação, congelar regras e processar 3.120.",
    ])

    summary_path = output_dir/"chart_type_exclusive_holdout_summary_v2.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Abra primeiro:")
    print(summary_path)


if __name__ == "__main__":
    main()
