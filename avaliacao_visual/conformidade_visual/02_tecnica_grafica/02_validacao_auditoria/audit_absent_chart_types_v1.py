# -*- coding: utf-8 -*-
r"""
audit_absent_chart_types_v1.py

AUDITORIA DOS TIPOS ALTERNATIVOS NOS CASOS MARK_ABSENT
=======================================================

Objetivo
--------
Em vez de tentar reclassificar as 3.120 imagens com três detectores
mutuamente concorrentes, este programa usa a decisão estrutural já validada
(V5) como primeira etapa:

- se MARK_PRESENT -> a técnica solicitada está presente;
- se MARK_ABSENT -> somente então investigamos qual estrutura alternativa
  aparece na imagem.

Isso evita falsos positivos cruzados como os observados na validação
multiclasses, em que barras podiam também ativar o detector de dispersão.

Entrada padrão
--------------
Imagens:
C:\Users\Labvis\Downloads\imagens3120\imagens

Resultado final V5:
imagens\_mark_presence_full_v5\mark_presence_images_v5.csv

Detector:
mark_presence_validation_v3.py
(deve estar na mesma pasta deste programa)

Saída
-----
imagens\_audit_absent_chart_types_v1

Arquivos:
- absent_chart_type_candidates_v1.csv
- absent_chart_type_summary_v1.txt
- contact_BI_01.png ...
- contact_BC_01.png ...
- contact_LI_01.png ...
- contact_LC_01.png ...
- contact_SC_01.png ...

O campo manual_type fica vazio para posterior auditoria visual.
Classes sugeridas para preenchimento manual:
BAR, LINE, SCATTER, EMPTY_NO_DATA, OTHER, UNCERTAIN.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
DEFAULT_RESULTS_DIRNAME = "_mark_presence_full_v5"
DEFAULT_OUTPUT_DIRNAME = "_audit_absent_chart_types_v1"

PER_SHEET = 12
COLS = 3
ROWS = 4
PANEL_W = 500
PANEL_H = 390
IMG_H = 300


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--results-dir", type=Path, default=None)
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


def alternate_candidates(expected, db, dl, ds):
    """
    Gera apenas candidatos para inspeção visual.
    Não substitui a auditoria humana dos 48 casos.
    """
    eb = parse_evidence(db["evidence"])
    el = parse_evidence(dl["evidence"])
    es = parse_evidence(ds["evidence"])

    b_h = float(eb.get("MAX_H_FRAC", 0.0))
    l_x = float(el.get("XBIN_SUPPORT", 0.0))
    s_n = int(es.get("N_MARKERS", 0.0))
    s_x = float(es.get("XDISP", 0.0))
    s_y = float(es.get("YDISP", 0.0))

    hits = []

    # Critérios conservadores para tipos ALTERNATIVOS.
    if expected != "B" and b_h >= 0.10:
        hits.append("BAR")

    if expected != "L" and l_x >= 0.50:
        hits.append("LINE")

    # Exige dispersão nos dois eixos para reduzir textos/legendas.
    if (
        expected != "S"
        and s_n >= 2
        and s_x >= 0.03
        and s_y >= 0.03
    ):
        hits.append("SCATTER")

    if len(hits) == 0:
        candidate = "OTHER_OR_EMPTY"
    elif len(hits) == 1:
        candidate = hits[0]
    else:
        candidate = "MIXED_CANDIDATE:" + "+".join(hits)

    return {
        "candidate_type": candidate,
        "bar_max_h_frac": b_h,
        "line_xbin_support": l_x,
        "scatter_n_markers": s_n,
        "scatter_xdisp": s_x,
        "scatter_ydisp": s_y,
        "bar_evidence": db["evidence"],
        "line_evidence": dl["evidence"],
        "scatter_evidence": ds["evidence"],
    }


def font_default(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_sheets(df, root, output_dir):
    font = font_default(15)

    for profile, g in df.groupby("profile", sort=True):
        g = g.reset_index(drop=True)

        for sheet_idx in range(math.ceil(len(g) / PER_SHEET)):
            chunk = g.iloc[
                sheet_idx*PER_SHEET:
                (sheet_idx+1)*PER_SHEET
            ]

            sheet = Image.new(
                "RGB",
                (COLS*PANEL_W, ROWS*PANEL_H),
                "white"
            )
            draw = ImageDraw.Draw(sheet)

            for j, (_, row) in enumerate(chunk.iterrows()):
                col = j % COLS
                rr = j // COLS
                x0 = col*PANEL_W
                y0 = rr*PANEL_H

                path = root / row["filename"]

                if path.exists():
                    im = Image.open(path).convert("RGB")
                    im.thumbnail((PANEL_W-20, IMG_H-10))

                    px = x0 + (PANEL_W-im.width)//2
                    py = y0 + 5
                    sheet.paste(im, (px, py))
                else:
                    draw.text(
                        (x0+20, y0+80),
                        "ARQUIVO NAO ENCONTRADO",
                        fill="black",
                        font=font
                    )

                txt = (
                    f"{row['filename']} | esperado={row['expected_type']}\n"
                    f"candidato={row['candidate_type']}\n"
                    f"B_h={row['bar_max_h_frac']:.3f} "
                    f"L_x={row['line_xbin_support']:.3f} "
                    f"S_n={int(row['scatter_n_markers'])} "
                    f"Sx={row['scatter_xdisp']:.2f} "
                    f"Sy={row['scatter_ydisp']:.2f}"
                )

                draw.multiline_text(
                    (x0+10, y0+IMG_H+4),
                    txt,
                    fill="black",
                    font=font,
                    spacing=2
                )

            sheet.save(
                output_dir /
                f"contact_{profile}_{sheet_idx+1:02d}.png"
            )


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else root / DEFAULT_RESULTS_DIRNAME
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_DIRNAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = results_dir / "mark_presence_images_v5.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Resultado V5 não encontrado: {results_csv}"
        )

    df = pd.read_csv(
        results_csv,
        dtype={"condition_code": str}
    )
    df["condition_code"] = (
        df["condition_code"]
        .astype(str)
        .str.zfill(2)
    )

    absent = df[df["mark_absent"] == 1].copy()
    absent = absent.sort_values(
        ["profile", "unit_id", "repeat"]
    ).reset_index(drop=True)

    if len(absent) != 48:
        print(
            f"AVISO: foram encontrados {len(absent)} MARK_ABSENT; "
            "o processamento anterior indicava 48."
        )

    v3 = load_v3()

    rows = []
    errors = []

    expected_map = {
        "B": "BAR",
        "L": "LINE",
        "S": "SCATTER",
    }

    print("="*80)
    print("AUDITORIA DE TIPOS ALTERNATIVOS — CASOS MARK_ABSENT")
    print("="*80)
    print(f"Casos a revisar: {len(absent)}")
    print()

    for _, r in absent.iterrows():
        path = root / r["filename"]

        try:
            db = v3.analyze_one(path, "B")["det"]
            dl = v3.analyze_one(path, "L")["det"]
            ds = v3.analyze_one(path, "S")["det"]

            cand = alternate_candidates(
                str(r["technique"]).upper(),
                db, dl, ds
            )

            rows.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "repeat": r["repeat"],
                "participant_id": r.get("participant_id", ""),
                "condition_code": r.get("condition_code", ""),
                "condition_label": r.get("condition_label", ""),
                "expected_technique": r["technique"],
                "expected_type": expected_map[
                    str(r["technique"]).upper()
                ],
                **cand,
                "manual_type": "",
                "manual_note": "",
            })

        except Exception as exc:
            errors.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

        done = len(rows) + len(errors)
        if done % 10 == 0 or done == len(absent):
            print(f"Processadas: {done}/{len(absent)}")

    out = pd.DataFrame(rows)
    err = pd.DataFrame(errors)

    out_path = output_dir / "absent_chart_type_candidates_v1.csv"
    out.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig"
    )

    err.to_csv(
        output_dir / "absent_chart_type_errors_v1.csv",
        index=False,
        encoding="utf-8-sig"
    )

    make_sheets(out, root, output_dir)

    lines = [
        "AUDITORIA DE TIPOS ALTERNATIVOS — CASOS MARK_ABSENT",
        "="*78,
        f"Casos processados: {len(out)}",
        f"Erros: {len(err)}",
        "",
        "DISTRIBUICAO DOS 48 MARK_ABSENT POR PERFIL:",
    ]

    counts = out.groupby("profile").size()
    for profile, n in counts.items():
        lines.append(f"{profile}: {n}")

    lines.extend([
        "",
        "CANDIDATOS AUTOMATICOS:",
    ])

    for typ, n in out["candidate_type"].value_counts().items():
        lines.append(f"{typ}: {n}")

    lines.extend([
        "",
        "AUDITORIA VISUAL:",
        "- revisar todas as pranchas geradas;",
        "- preencher mentalmente/registrar como BAR, LINE, SCATTER,",
        "  EMPTY_NO_DATA, OTHER ou UNCERTAIN;",
        "- a classificacao final dos 48 deve ser visualmente confirmada;",
        "- nenhum parametro do detector MARK_PRESENT/MARK_ABSENT sera alterado.",
    ])

    summary_path = output_dir / "absent_chart_type_summary_v1.txt"
    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print()
    print("Saida:")
    print(output_dir)
    print("Abra primeiro:")
    print(summary_path)


if __name__ == "__main__":
    main()
