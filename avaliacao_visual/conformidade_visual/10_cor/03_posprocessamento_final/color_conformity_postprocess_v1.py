# -*- coding: utf-8 -*-
"""
color_conformity_postprocess_v1.py

PÓS-PROCESSAMENTO FINAL DA CONFORMIDADE DE COR
===============================================

Usa SOMENTE os CSVs produzidos por color_conformity_v4.py.
Não reabre nem reprocessa as 3.120 imagens.

Objetivos:
1. preservar Dmean/Dmax contínuos;
2. aplicar um limiar operacional configurável de equivalência cromática;
3. separar EXTRA bruto em:
      - NEAR_DUPLICATE
      - SEMANTIC_EXTRA_CANDIDATE
4. manter separadamente:
      - cor desviada;
      - cor extra;
      - cor esperada ausente;
      - NO_MARK;
      - variação cromática interna;
      - OVERLAP_DERIVED;
5. gerar uma síntese categórica por imagem;
6. agregar as 10 repetições nas 312 unidades experimentais.

Regra padrão:
    ΔE00 <= 2.0 -> equivalência cromática operacional / near-duplicate.

IMPORTANTE:
- O limiar fica em arquivo JSON e pode ser alterado sem mexer no algoritmo.
- NEAR_DUPLICATE não conta como cor semântica extra.
- OVERLAP_DERIVED continua diagnóstico, não cor extra.
- NO_MARK não recebe penalidade artificial em ΔE00.
- O status categórico NÃO substitui as métricas contínuas.

Execução:
    python color_conformity_postprocess_v1.py
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.color import rgb2lab, deltaE_ciede2000


DEFAULT_ROOT = Path("C:/Users/Labvis/Downloads/imagens3120/imagens")
DEFAULT_INPUT_DIRNAME = "_color_conformity_v2"
DEFAULT_OUTPUT_DIRNAME = "_color_conformity_final_v1"
DEFAULT_CONFIG = Path(__file__).with_name("color_postprocess_config.json")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pós-processamento dos resultados de conformidade de cor."
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Pasta com os CSVs do color_conformity_v4.py."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return p.parse_args()


def read_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    de = float(cfg["deltaE00_equivalence_threshold"])
    if de < 0:
        raise ValueError("deltaE00_equivalence_threshold deve ser >= 0.")
    return cfg


def hex_to_rgb(hx: str):
    hx = str(hx).strip().lstrip("#")
    if len(hx) != 6:
        raise ValueError(f"HEX inválido: {hx}")
    return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_lab_one(rgb):
    arr = np.array(
        [[[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]]],
        dtype=np.float64
    )
    return rgb2lab(arr)[0, 0]


def de00_hex(hx1: str, hx2: str):
    a = rgb_to_lab_one(hex_to_rgb(hx1))
    b = rgb_to_lab_one(hex_to_rgb(hx2))
    return float(
        deltaE_ciede2000(
            np.array([[a]], dtype=np.float64),
            np.array([[b]], dtype=np.float64)
        )[0, 0]
    )


def safe_float(v):
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return None if math.isnan(x) else x


def safe_mean(vals):
    x = [safe_float(v) for v in vals]
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def safe_median(vals):
    x = [safe_float(v) for v in vals]
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def safe_min(vals):
    x = [safe_float(v) for v in vals]
    x = [v for v in x if v is not None]
    return float(np.min(x)) if x else None


def safe_max(vals):
    x = [safe_float(v) for v in vals]
    x = [v for v in x if v is not None]
    return float(np.max(x)) if x else None


def fmt(v, digits=6):
    x = safe_float(v)
    if x is None:
        return ""
    return round(x, digits)


def load_inputs(input_dir: Path):
    required = {
        "images": input_dir / "color_conformity_images.csv",
        "pairs": input_dir / "color_conformity_pairs.csv",
        "units": input_dir / "color_conformity_units.csv",
    }

    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo obrigatório não encontrado ({name}): {path}"
            )

    images = pd.read_csv(required["images"])
    pairs = pd.read_csv(required["pairs"])
    units = pd.read_csv(required["units"])

    scatter_path = input_dir / "color_conformity_scatter_points.csv"
    scatter = pd.read_csv(scatter_path) if scatter_path.exists() else pd.DataFrame()

    return images, pairs, units, scatter


def classify_extra_rows(pairs: pd.DataFrame, threshold: float):
    """
    Para cada EXTRA, calcula ΔE00 em relação às cores OBSERVADAS já
    pareadas na mesma imagem.

    <= threshold -> NEAR_DUPLICATE
    > threshold  -> SEMANTIC_EXTRA_CANDIDATE
    """
    extra_rows = []

    for filename, g in pairs.groupby("filename", sort=False):
        matched = g[
            g["pair_status"].astype(str).str.upper() == "MATCHED"
        ].copy()

        extras = g[
            g["pair_status"].astype(str).str.upper() == "EXTRA"
        ].copy()

        matched_hex = [
            str(h)
            for h in matched["observed_hex"].dropna().tolist()
            if str(h).startswith("#")
        ]

        for _, row in extras.iterrows():
            hx = str(row.get("observed_hex", ""))

            distances = []
            for base_hx in matched_hex:
                try:
                    distances.append(
                        (base_hx, de00_hex(hx, base_hx))
                    )
                except Exception:
                    pass

            if distances:
                nearest_hex, nearest_de = min(
                    distances,
                    key=lambda z: z[1]
                )
            else:
                nearest_hex, nearest_de = "", None

            extra_class = (
                "NEAR_DUPLICATE"
                if nearest_de is not None and nearest_de <= threshold
                else "SEMANTIC_EXTRA_CANDIDATE"
            )

            extra_rows.append({
                "filename": filename,
                "profile": row.get("profile", ""),
                "technique": row.get("technique", ""),
                "task": row.get("task", ""),
                "unit_id": row.get("unit_id", ""),
                "unit_number": row.get("unit_number", ""),
                "repeat": row.get("repeat", ""),
                "observed_hex_extra": hx,
                "nearest_matched_observed_hex": nearest_hex,
                "deltaE00_extra_to_nearest_matched": fmt(nearest_de),
                "extra_class_final": extra_class,
                "support_mean_extra": fmt(row.get("support_mean")),
                "support_min_extra": fmt(row.get("support_min")),
                "family_type": row.get("family_type", ""),
                "encoding_candidate": row.get("encoding_candidate", ""),
            })

    return pd.DataFrame(extra_rows)


def build_image_level(
    images: pd.DataFrame,
    extras: pd.DataFrame,
    threshold: float
):
    out = images.copy()

    if extras.empty:
        extra_summary = pd.DataFrame(
            columns=[
                "filename",
                "n_near_duplicate",
                "n_extra_semantic_candidate",
                "near_duplicate_hex",
                "semantic_extra_hex",
                "semantic_extra_support_min",
            ]
        )
    else:
        records = []

        for filename, g in extras.groupby("filename", sort=False):
            near = g[g["extra_class_final"] == "NEAR_DUPLICATE"]
            semantic = g[
                g["extra_class_final"] == "SEMANTIC_EXTRA_CANDIDATE"
            ]

            records.append({
                "filename": filename,
                "n_near_duplicate": len(near),
                "n_extra_semantic_candidate": len(semantic),
                "near_duplicate_hex": "|".join(
                    near["observed_hex_extra"].astype(str).tolist()
                ),
                "semantic_extra_hex": "|".join(
                    semantic["observed_hex_extra"].astype(str).tolist()
                ),
                "semantic_extra_support_min": (
                    safe_min(semantic["support_min_extra"].tolist())
                    if len(semantic)
                    else None
                ),
            })

        extra_summary = pd.DataFrame(records)

    out = out.merge(extra_summary, on="filename", how="left")

    for col in ["n_near_duplicate", "n_extra_semantic_candidate"]:
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    for col in ["near_duplicate_hex", "semantic_extra_hex"]:
        out[col] = out[col].fillna("")

    out["semantic_extra_support_min"] = pd.to_numeric(
        out["semantic_extra_support_min"],
        errors="coerce"
    )

    dmax = pd.to_numeric(out["Dmax_color"], errors="coerce")

    out["color_evaluable"] = (
        out["extraction_status"].astype(str).str.upper() != "NO_MARK"
    ).astype(int)

    out["color_match_de_threshold"] = np.where(
        dmax.notna(),
        (dmax <= threshold).astype(int),
        np.nan
    )

    out["color_deviation"] = np.where(
        dmax.notna(),
        (dmax > threshold).astype(int),
        0
    ).astype(int)

    out["missing_color"] = (
        pd.to_numeric(out["n_missing_colors"], errors="coerce")
        .fillna(0)
        .gt(0)
    ).astype(int)

    out["no_mark"] = (
        out["extraction_status"]
        .astype(str)
        .str.upper()
        .eq("NO_MARK")
    ).astype(int)

    out["semantic_extra_color"] = (
        out["n_extra_semantic_candidate"] > 0
    ).astype(int)

    out["near_duplicate_present"] = (
        out["n_near_duplicate"] > 0
    ).astype(int)

    out["internal_variation_final"] = (
        pd.to_numeric(out["internal_variation"], errors="coerce")
        .fillna(0)
        .gt(0)
    ).astype(int)

    out["overlap_derived_present"] = (
        pd.to_numeric(
            out["n_overlap_derived_families"],
            errors="coerce"
        )
        .fillna(0)
        .gt(0)
    ).astype(int)

    def status_row(r):
        labels = []

        if int(r["no_mark"]) == 1:
            labels.append("NO_MARK")
        else:
            if int(r["missing_color"]) == 1:
                labels.append("MISSING_COLOR")

            if int(r["semantic_extra_color"]) == 1:
                labels.append("EXTRA_COLOR")

            if int(r["internal_variation_final"]) == 1:
                labels.append("INTERNAL_VARIATION")

            if int(r["color_deviation"]) == 1:
                labels.append("COLOR_DEVIATION")

        if not labels:
            labels.append("MATCH")

        return "+".join(labels)

    out["color_status_final"] = out.apply(status_row, axis=1)

    out["strict_color_match"] = (
        out["color_status_final"] == "MATCH"
    ).astype(int)

    out["deltaE00_equivalence_threshold"] = threshold

    return out


def aggregate_units(final_images: pd.DataFrame):
    rows = []

    for unit_id, g in final_images.groupby("unit_id", sort=True):
        first = g.iloc[0]

        dmax = pd.to_numeric(g["Dmax_color"], errors="coerce")
        dmean = pd.to_numeric(g["Dmean_color"], errors="coerce")

        status_counter = Counter(
            g["color_status_final"].astype(str)
        )

        rows.append({
            "unit_id": unit_id,
            "profile": first["profile"],
            "technique": first["technique"],
            "task": first["task"],
            "n_repeats_found": len(g),
            "n_color_evaluable": int(g["color_evaluable"].sum()),

            "Dmax_mean": fmt(dmax.mean()),
            "Dmax_median": fmt(dmax.median()),
            "Dmax_IQR": fmt(
                dmax.quantile(0.75) - dmax.quantile(0.25)
                if dmax.notna().any()
                else None
            ),
            "Dmax_max": fmt(dmax.max()),

            "Dmean_mean": fmt(dmean.mean()),
            "Dmean_median": fmt(dmean.median()),
            "Dmean_IQR": fmt(
                dmean.quantile(0.75) - dmean.quantile(0.25)
                if dmean.notna().any()
                else None
            ),

            "n_strict_match": int(g["strict_color_match"].sum()),
            "p_strict_color_match": fmt(g["strict_color_match"].mean()),
            "p_color_deviation": fmt(g["color_deviation"].mean()),
            "p_missing_color": fmt(g["missing_color"].mean()),
            "p_semantic_extra_color": fmt(
                g["semantic_extra_color"].mean()
            ),
            "p_near_duplicate": fmt(
                g["near_duplicate_present"].mean()
            ),
            "p_no_mark": fmt(g["no_mark"].mean()),
            "p_internal_variation": fmt(
                g["internal_variation_final"].mean()
            ),
            "p_overlap_derived": fmt(
                g["overlap_derived_present"].mean()
            ),

            "near_duplicate_count_total": int(
                g["n_near_duplicate"].sum()
            ),
            "semantic_extra_count_total": int(
                g["n_extra_semantic_candidate"].sum()
            ),
            "missing_color_count_total": int(
                pd.to_numeric(
                    g["n_missing_colors"],
                    errors="coerce"
                ).fillna(0).sum()
            ),

            "extraction_support_mean": fmt(
                pd.to_numeric(
                    g["extraction_support_mean"],
                    errors="coerce"
                ).mean()
            ),
            "extraction_support_min": fmt(
                pd.to_numeric(
                    g["extraction_support_min"],
                    errors="coerce"
                ).min()
            ),

            "status_distribution": "|".join(
                f"{k}:{v}"
                for k, v in sorted(status_counter.items())
            ),
        })

    return pd.DataFrame(rows)


def profile_summary(final_images: pd.DataFrame):
    records = []

    for profile, g in final_images.groupby("profile", sort=True):
        evaluable = g[g["color_evaluable"] == 1]

        records.append({
            "profile": profile,
            "n_images": len(g),
            "n_evaluable": len(evaluable),
            "n_strict_match": int(g["strict_color_match"].sum()),
            "p_strict_match_all": fmt(g["strict_color_match"].mean()),
            "p_strict_match_evaluable": fmt(
                evaluable["strict_color_match"].mean()
                if len(evaluable)
                else None
            ),
            "n_color_deviation": int(g["color_deviation"].sum()),
            "n_missing_color": int(g["missing_color"].sum()),
            "n_semantic_extra_color": int(
                g["semantic_extra_color"].sum()
            ),
            "n_near_duplicate_images": int(
                g["near_duplicate_present"].sum()
            ),
            "n_no_mark": int(g["no_mark"].sum()),
            "n_internal_variation": int(
                g["internal_variation_final"].sum()
            ),
            "n_overlap_derived": int(
                g["overlap_derived_present"].sum()
            ),
            "Dmax_mean_evaluable": fmt(
                pd.to_numeric(
                    evaluable["Dmax_color"],
                    errors="coerce"
                ).mean()
            ),
            "Dmax_median_evaluable": fmt(
                pd.to_numeric(
                    evaluable["Dmax_color"],
                    errors="coerce"
                ).median()
            ),
            "Dmax_max_evaluable": fmt(
                pd.to_numeric(
                    evaluable["Dmax_color"],
                    errors="coerce"
                ).max()
            ),
        })

    return pd.DataFrame(records)


def main():
    args = parse_args()
    cfg = read_config(args.config)

    threshold = float(cfg["deltaE00_equivalence_threshold"])

    root = args.root.expanduser().resolve()

    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else root / DEFAULT_INPUT_DIRNAME
    )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Pasta de entrada não encontrada: {input_dir}"
        )

    output_dir = root / DEFAULT_OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("PÓS-PROCESSAMENTO FINAL — CONFORMIDADE DE COR")
    print("=" * 84)
    print(f"Entrada: {input_dir}")
    print(f"Saída:   {output_dir}")
    print(f"Limiar operacional ΔE00: {threshold}")
    print()

    images, pairs, _units_old, _scatter = load_inputs(input_dir)

    extras = classify_extra_rows(pairs, threshold)
    final_images = build_image_level(images, extras, threshold)
    final_units = aggregate_units(final_images)
    profiles = profile_summary(final_images)

    images_path = output_dir / "color_conformity_final_images.csv"
    extras_path = output_dir / "color_conformity_final_extras.csv"
    units_path = output_dir / "color_conformity_final_units.csv"
    profiles_path = output_dir / "color_conformity_final_profiles.csv"
    summary_path = output_dir / "color_conformity_final_summary.txt"
    config_path = output_dir / "color_postprocess_config_used.json"

    final_images.to_csv(
        images_path, index=False, encoding="utf-8-sig"
    )
    extras.to_csv(
        extras_path, index=False, encoding="utf-8-sig"
    )
    final_units.to_csv(
        units_path, index=False, encoding="utf-8-sig"
    )
    profiles.to_csv(
        profiles_path, index=False, encoding="utf-8-sig"
    )

    config_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    n = len(final_images)
    n_eval = int(final_images["color_evaluable"].sum())
    n_match = int(final_images["strict_color_match"].sum())

    status_counts = Counter(
        final_images["color_status_final"].astype(str)
    )

    summary = [
        "PÓS-PROCESSAMENTO FINAL — CONFORMIDADE DE COR",
        "=" * 76,
        f"Imagens: {n}",
        f"Unidades: {len(final_units)}",
        f"Imagens cromaticamente avaliáveis: {n_eval}",
        f"Limiar operacional de equivalência: ΔE00 <= {threshold}",
        "",
        "RESULTADOS GLOBAIS:",
        f"strict_color_match: {n_match}/{n} ({n_match/n:.4%})",
        f"color_deviation: {int(final_images['color_deviation'].sum())}",
        f"missing_color: {int(final_images['missing_color'].sum())}",
        f"semantic_extra_color: {int(final_images['semantic_extra_color'].sum())}",
        f"near_duplicate_present: {int(final_images['near_duplicate_present'].sum())}",
        f"no_mark: {int(final_images['no_mark'].sum())}",
        f"internal_variation: {int(final_images['internal_variation_final'].sum())}",
        f"overlap_derived_present: {int(final_images['overlap_derived_present'].sum())}",
        "",
        "DISTRIBUIÇÃO color_status_final:",
    ]

    for status, count in sorted(status_counts.items()):
        summary.append(f"{status}: {count}")

    summary.extend([
        "",
        "INTERPRETAÇÃO:",
        "- MATCH: nenhuma não conformidade cromática operacional detectada;",
        "- COLOR_DEVIATION: Dmax acima do limiar;",
        "- EXTRA_COLOR: família extra não explicada como near-duplicate;",
        "- MISSING_COLOR: cor esperada ausente;",
        "- INTERNAL_VARIATION: família scatter com variação cromática interna;",
        "- NO_MARK: marca cromática não avaliável;",
        "- NEAR_DUPLICATE e OVERLAP_DERIVED são diagnósticos auxiliares,",
        "  não são contados como cores semânticas extras.",
        "",
        "IMPORTANTE:",
        "- as métricas contínuas Dmean/Dmax são preservadas;",
        "- o limiar pode ser alterado no JSON e o pós-processamento refeito;",
        "- nenhum arquivo de imagem é reprocessado.",
        "",
        f"CSV imagens: {images_path}",
        f"CSV extras: {extras_path}",
        f"CSV unidades: {units_path}",
        f"CSV perfis: {profiles_path}",
    ])

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print(f"Imagens: {n}")
    print(f"Unidades: {len(final_units)}")
    print(f"Strict matches: {n_match}")
    print(f"Saída: {output_dir}")
    print()
    print("Abra primeiro:")
    print(summary_path)


if __name__ == "__main__":
    main()
