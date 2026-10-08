# -*- coding: utf-8 -*-
"""
color_conformity_v4.py

AVALIAÇÃO UNIFICADA DE CONFORMIDADE DE COR
==========================================

Integra os três extratores validados/calibrados:

    B-Cor -> validate_color_bars_core_v5.py
    L-Cor -> validate_color_lines_v4.py
    S-Cor -> validate_color_scatter_v7_markercore_final.py

Fluxo:
    imagem
      -> extrator específico da técnica
      -> famílias/cores observadas
      -> pareamento 1:1 com cores esperadas do perfil
      -> CIEDE2000
      -> métricas por imagem
      -> agregação das 10 repetições por unidade experimental

IMPORTANTE
----------
1. A extração NÃO utiliza as cores esperadas para descobrir as marcas.
2. As cores esperadas entram somente depois da extração.
3. Não há limiar final de "conforme/não conforme" por ΔE00 nesta versão.
4. Cor ausente não recebe distância artificial: Dmean/Dmax ficam vazios
   quando não há par cromático avaliável.
5. Cores extras e cores esperadas ausentes são registradas separadamente.
6. Para S-Cor, a classificação brightness/saturation/mixed é preservada
   como diagnóstico; a variável principal é SOLID versus VARIATION.
7. O script processa somente imagens da RAIZ de ROOT_DIR. Subpastas são
   ignoradas, evitando reprocessar as cópias duplicadas já identificadas.

Arquivos de saída
-----------------
_color_conformity_v1/
    color_conformity_images.csv
    color_conformity_pairs.csv
    color_conformity_scatter_points.csv
    color_conformity_units.csv
    color_conformity_errors.csv
    color_conformity_summary.txt
    color_conformity_config_used.json

Execução
--------
    python color_conformity.py

Opcional:
    python color_conformity.py --root "C:/caminho/imagens"
    python color_conformity.py --limit 100
    python color_conformity.py --profiles BI BC
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import math
import random
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000


# =====================================================================
# PADRÕES / CAMINHOS
# =====================================================================

DEFAULT_ROOT = Path("C:/Users/Labvis/Downloads/imagens3120/imagens")
DEFAULT_CONFIG = Path(__file__).with_name("color_conformity_config.json")
OUTPUT_DIRNAME = "_color_conformity_v2"

IMAGE_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.I
)

TECHNIQUE_MAP = {
    "B": "bar",
    "L": "line",
    "S": "scatter",
}

TASK_MAP = {
    "I": "identification",
    "C": "comparison",
}


# =====================================================================
# UTILITÁRIOS
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Avaliação unificada B-Cor/L-Cor/S-Cor."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Pasta raiz que contém as 3120 imagens."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Arquivo JSON com paleta F e cores esperadas."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Processa somente N imagens para teste."
    )
    p.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="Perfis opcionais, ex.: --profiles BI BC SI"
    )
    p.add_argument(
        "--sample-per-profile",
        type=int,
        default=None,
        help=(
            "Seleciona aleatoriamente N imagens de cada perfil "
            "(BI, BC, LI, LC, SI, SC)."
        )
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=20260827,
        help="Semente da amostragem estratificada."
    )
    p.add_argument(
        "--units-per-profile",
        type=int,
        default=None,
        help=(
            "Seleciona N unidades experimentais COMPLETAS por perfil. "
            "Cada unidade precisa conter todas as repetições R01-R10."
        )
    )
    return p.parse_args()


def read_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Configuração não encontrada: {path}"
        )

    cfg = json.loads(
        path.read_text(encoding="utf-8")
    )

    palette = cfg.get("palette_F", {})
    expected = cfg.get(
        "expected_colors_by_profile",
        {}
    )

    required_profiles = {
        "BI", "BC", "LI", "LC", "SI", "SC"
    }

    missing_profiles = (
        required_profiles - set(expected)
    )

    if missing_profiles:
        raise ValueError(
            "Perfis ausentes na configuração: "
            + ", ".join(sorted(missing_profiles))
        )

    for profile, names in expected.items():
        for name in names:
            if name not in palette:
                raise ValueError(
                    f"Cor '{name}' usada em {profile} "
                    "não existe em palette_F."
                )

    return cfg


def hex_to_rgb(hx: str):
    hx = hx.strip().lstrip("#")
    return tuple(
        int(hx[i:i+2], 16)
        for i in (0, 2, 4)
    )


def rgb_to_lab_one(rgb):
    arr = np.array(
        [[[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]]],
        dtype=np.float64
    )
    return rgb2lab(arr)[0, 0]


def de00_hex(hx1: str, hx2: str):
    a = rgb_to_lab_one(
        hex_to_rgb(hx1)
    )
    b = rgb_to_lab_one(
        hex_to_rgb(hx2)
    )

    return float(
        deltaE_ciede2000(
            np.array([[a]], dtype=np.float64),
            np.array([[b]], dtype=np.float64)
        )[0, 0]
    )


def safe_mean(values):
    vals = [
        float(v)
        for v in values
        if v is not None
        and not (
            isinstance(v, float)
            and math.isnan(v)
        )
    ]
    return (
        float(np.mean(vals))
        if vals else None
    )


def safe_median(values):
    vals = [
        float(v)
        for v in values
        if v is not None
        and not (
            isinstance(v, float)
            and math.isnan(v)
        )
    ]
    return (
        float(np.median(vals))
        if vals else None
    )


def safe_min(values):
    vals = [
        float(v)
        for v in values
        if v is not None
        and not (
            isinstance(v, float)
            and math.isnan(v)
        )
    ]
    return (
        float(np.min(vals))
        if vals else None
    )


def safe_max(values):
    vals = [
        float(v)
        for v in values
        if v is not None
        and not (
            isinstance(v, float)
            and math.isnan(v)
        )
    ]
    return (
        float(np.max(vals))
        if vals else None
    )


def safe_percentile(values, q):
    vals = [
        float(v)
        for v in values
        if v is not None
        and not (
            isinstance(v, float)
            and math.isnan(v)
        )
    ]
    return (
        float(np.percentile(vals, q))
        if vals else None
    )


def safe_iqr(values):
    vals = [
        float(v)
        for v in values
        if v is not None
        and not (
            isinstance(v, float)
            and math.isnan(v)
        )
    ]

    if not vals:
        return None

    return float(
        np.percentile(vals, 75)
        - np.percentile(vals, 25)
    )


def fmt(v, digits=6):
    if v is None:
        return ""
    try:
        if math.isnan(float(v)):
            return ""
    except Exception:
        pass
    return round(float(v), digits)


def join_list(values):
    return "|".join(
        str(v) for v in values
    )


# =====================================================================
# IMPORTAÇÃO DOS EXTRATORES VALIDADOS
# =====================================================================

def import_extractors():
    names = {
        "bars": "validate_color_bars_core_v5",
        "lines": "validate_color_lines_v4",
        "scatter": "validate_color_scatter_v7_markercore_final",
    }

    modules = {}

    for key, module_name in names.items():
        try:
            modules[key] = importlib.import_module(
                module_name
            )
        except Exception as exc:
            raise ImportError(
                f"Não foi possível importar '{module_name}.py'. "
                "Mantenha color_conformity.py e os três extratores "
                "na mesma pasta.\n"
                f"Erro original: {exc}"
            ) from exc

    return modules


# =====================================================================
# EXTRAÇÃO UNIFICADA
# =====================================================================

def extract_bar(path: Path, module):
    with Image.open(path) as img:
        (
            geometry,
            baseline,
            arr,
            n_initial
        ) = module.detect_bar_geometry(
            img.convert("RGB")
        )

    bars = module.measure_bars_v5(
        geometry,
        arr
    )

    colors = module.collapse_distinct_colors(
        bars
    )

    observed = []

    for idx, c in enumerate(colors):
        observed.append({
            "obs_index": idx,
            "observed_hex": c["observed_hex"],
            "representation": "BAR_CORE_DOMINANT",
            "family_type": "SOLID",
            "support_mean": c.get(
                "mean_support_exact"
            ),
            "support_min": c.get(
                "min_support_exact"
            ),
            "n_marks": c.get(
                "support_bars"
            ),
            "raw": c,
            "points": [],
        })

    support_mean = safe_mean([
        b.get("support_exact")
        for b in bars
    ])

    support_min = safe_min([
        b.get("support_exact")
        for b in bars
    ])

    return {
        "observed": observed,
        "extraction_status": (
            "OK" if observed
            else "NO_MARK"
        ),
        "n_marks": len(bars),
        "n_initial_candidates": n_initial,
        "extraction_support_mean": support_mean,
        "extraction_support_min": support_min,
        "n_solid_families": len(observed),
        "n_variable_families": 0,
        "n_insufficient_families": 0,
        "scatter_point_rows": [],
        "legend_suspected_count": 0,
        "n_overlap_derived_families": 0,
        "line_derived_hex": "",
    }


def extract_line(path: Path, module):
    with Image.open(path) as img:
        all_families = module.extract_line_color_families(
            img.convert("RGB"),
            include_derived=True
        )

    families = [
        f for f in all_families
        if f.get("family_role", "BASE") == "BASE"
    ]

    derived = [
        f for f in all_families
        if f.get("family_role") == "OVERLAP_DERIVED"
    ]

    observed = []

    for idx, fam in enumerate(families):
        observed.append({
            "obs_index": idx,
            "observed_hex": fam["observed_hex"],
            "representation": "LINE_COLOR_FAMILY_CORE",
            "family_type": "SOLID",
            "support_mean": fam.get(
                "core_support_exact"
            ),
            "support_min": fam.get(
                "core_support_exact"
            ),
            "n_marks": None,
            "raw": fam,
            "points": [],
        })

    supports = [
        f.get("core_support_exact")
        for f in families
    ]

    return {
        "observed": observed,
        "extraction_status": (
            "OK" if observed
            else "NO_MARK"
        ),
        "n_marks": None,
        "n_initial_candidates": None,
        "extraction_support_mean": (
            safe_mean(supports)
        ),
        "extraction_support_min": (
            safe_min(supports)
        ),
        "n_solid_families": len(observed),
        "n_variable_families": 0,
        "n_insufficient_families": 0,
        "scatter_point_rows": [],
        "legend_suspected_count": 0,
        "n_overlap_derived_families": len(derived),
        "line_derived_hex": join_list([
            f.get("observed_hex", "")
            for f in derived
        ]),
    }


def normalize_scatter_type(label):
    if label == "SOLID_CANDIDATE":
        return "SOLID"
    if label == "INSUFFICIENT_POINTS":
        return "INSUFFICIENT"
    if label in {
        "BRIGHTNESS_SCALE_CANDIDATE",
        "SATURATION_SCALE_CANDIDATE",
        "MIXED_SCALE_CANDIDATE",
    }:
        return "VARIATION"
    return label or "UNKNOWN"


def extract_scatter(path: Path, module):
    with Image.open(path) as opened:
        arr_u8 = np.asarray(
            opened.convert("RGB"),
            dtype=np.uint8
        )

    (
        plot_roi,
        roi_method,
        _n_hlines,
        _n_vlines
    ) = module.infer_plot_roi(
        arr_u8
    )

    (
        marker_candidates,
        hsv
    ) = module.detect_markers(
        arr_u8,
        plot_roi
    )

    # V7: nenhum candidato é removido.
    (
        kept_markers,
        suspected_legend
    ) = module.flag_probable_legend_markers(
        arr_u8,
        hsv,
        marker_candidates,
        plot_roi
    )

    families = module.finalize_families(
        module.group_markers_by_hue(
            kept_markers
        )
    )

    observed = []
    point_rows = []

    for idx, fam in enumerate(families):
        family_type = normalize_scatter_type(
            fam.get("encoding_candidate")
        )

        point_supports = [
            p.get("exact_color_support")
            for p in fam.get("points", [])
        ]

        observed.append({
            "obs_index": idx,
            "observed_hex": fam["medoid_hex"],
            "representation": "SCATTER_FAMILY_MEDOID",
            "family_type": family_type,
            "encoding_candidate": fam.get(
                "encoding_candidate"
            ),
            "solid_support_de4": fam.get(
                "solid_support_de4"
            ),
            "support_mean": safe_mean(
                point_supports
            ),
            "support_min": safe_min(
                point_supports
            ),
            "n_marks": fam.get(
                "n_points"
            ),
            "raw": fam,
            "points": fam.get(
                "points",
                []
            ),
        })

    n_solid = sum(
        o["family_type"] == "SOLID"
        for o in observed
    )
    n_variable = sum(
        o["family_type"] == "VARIATION"
        for o in observed
    )
    n_insufficient = sum(
        o["family_type"] == "INSUFFICIENT"
        for o in observed
    )

    all_marker_supports = [
        m.get("exact_color_support")
        for m in kept_markers
    ]

    return {
        "observed": observed,
        "extraction_status": (
            "OK" if observed
            else "NO_MARK"
        ),
        "n_marks": len(
            kept_markers
        ),
        "n_initial_candidates": len(
            marker_candidates
        ),
        "extraction_support_mean": (
            safe_mean(all_marker_supports)
        ),
        "extraction_support_min": (
            safe_min(all_marker_supports)
        ),
        "n_solid_families": n_solid,
        "n_variable_families": n_variable,
        "n_insufficient_families": (
            n_insufficient
        ),
        "scatter_point_rows": point_rows,
        "legend_suspected_count": len(
            suspected_legend
        ),
        "n_overlap_derived_families": 0,
        "line_derived_hex": "",
        "roi_method": roi_method,
        "roi": plot_roi,
    }


# =====================================================================
# PAREAMENTO OBSERVADO <-> ESPERADO
# =====================================================================

def best_one_to_one_matching(
    observed,
    expected_names,
    palette
):
    """
    Pareamento de custo mínimo.

    Como os perfis atuais têm no máximo duas cores esperadas, usa busca
    exata por combinações/permutação, evitando dependência de scipy.
    """
    n_obs = len(observed)
    n_exp = len(expected_names)

    if n_obs == 0 or n_exp == 0:
        return [], list(range(n_obs)), list(range(n_exp))

    expected_hex = [
        palette[name]
        for name in expected_names
    ]

    cost = np.zeros(
        (n_obs, n_exp),
        dtype=float
    )

    for i, obs in enumerate(observed):
        for j, hx in enumerate(expected_hex):
            cost[i, j] = de00_hex(
                obs["observed_hex"],
                hx
            )

    best_pairs = None
    best_cost = float("inf")

    if n_obs >= n_exp:
        # Escolhe quais observados serão pareados com todos os esperados.
        for obs_subset in itertools.combinations(
            range(n_obs),
            n_exp
        ):
            for obs_perm in itertools.permutations(
                obs_subset
            ):
                pairs = [
                    (obs_perm[j], j)
                    for j in range(n_exp)
                ]
                total = sum(
                    cost[i, j]
                    for i, j in pairs
                )
                if total < best_cost:
                    best_cost = total
                    best_pairs = pairs
    else:
        # Todos os observados serão pareados a um subconjunto dos esperados.
        for exp_subset in itertools.combinations(
            range(n_exp),
            n_obs
        ):
            for exp_perm in itertools.permutations(
                exp_subset
            ):
                pairs = [
                    (i, exp_perm[i])
                    for i in range(n_obs)
                ]
                total = sum(
                    cost[i, j]
                    for i, j in pairs
                )
                if total < best_cost:
                    best_cost = total
                    best_pairs = pairs

    best_pairs = best_pairs or []

    used_obs = {
        i for i, _ in best_pairs
    }
    used_exp = {
        j for _, j in best_pairs
    }

    unmatched_obs = [
        i for i in range(n_obs)
        if i not in used_obs
    ]

    unmatched_exp = [
        j for j in range(n_exp)
        if j not in used_exp
    ]

    pairs = []

    for i, j in best_pairs:
        pairs.append({
            "obs_index": i,
            "exp_index": j,
            "expected_name": (
                expected_names[j]
            ),
            "expected_hex": (
                expected_hex[j]
            ),
            "deltaE00": float(
                cost[i, j]
            ),
        })

    return (
        pairs,
        unmatched_obs,
        unmatched_exp
    )


# =====================================================================
# PROCESSAMENTO DE UMA IMAGEM
# =====================================================================

def process_image(
    path,
    profile,
    unit_number,
    repeat_number,
    cfg,
    modules
):
    technique_code = profile[0]
    task_code = profile[1]

    technique = TECHNIQUE_MAP[
        technique_code
    ]

    task = TASK_MAP[
        task_code
    ]

    with Image.open(path) as img:
        canvas_w, canvas_h = img.size

    if technique_code == "B":
        extraction = extract_bar(
            path,
            modules["bars"]
        )
    elif technique_code == "L":
        extraction = extract_line(
            path,
            modules["lines"]
        )
    elif technique_code == "S":
        extraction = extract_scatter(
            path,
            modules["scatter"]
        )
    else:
        raise ValueError(
            f"Técnica desconhecida: {profile}"
        )

    palette = cfg["palette_F"]
    expected_names = cfg[
        "expected_colors_by_profile"
    ][profile]

    observed = extraction[
        "observed"
    ]

    (
        matched,
        unmatched_obs,
        unmatched_exp
    ) = best_one_to_one_matching(
        observed,
        expected_names,
        palette
    )

    pair_rows = []
    point_rows = []
    dvals = []

    # ---------------------------------------------------------
    # Pares correspondidos
    # ---------------------------------------------------------
    for pair_rank, match in enumerate(
        matched,
        1
    ):
        obs = observed[
            match["obs_index"]
        ]

        de = match[
            "deltaE00"
        ]

        dvals.append(de)

        pair_rows.append({
            "pair_status": "MATCHED",
            "pair_rank": pair_rank,
            "observed_index": (
                match["obs_index"] + 1
            ),
            "expected_index": (
                match["exp_index"] + 1
            ),
            "observed_hex": obs[
                "observed_hex"
            ],
            "expected_name": match[
                "expected_name"
            ],
            "expected_hex": match[
                "expected_hex"
            ],
            "deltaE00": fmt(de),
            "representation": obs.get(
                "representation",
                ""
            ),
            "family_type": obs.get(
                "family_type",
                ""
            ),
            "encoding_candidate": obs.get(
                "encoding_candidate",
                ""
            ),
            "support_mean": fmt(
                obs.get("support_mean")
            ),
            "support_min": fmt(
                obs.get("support_min")
            ),
            "solid_support_de4": fmt(
                obs.get(
                    "solid_support_de4"
                )
            ),
            "n_marks_in_family": (
                obs.get("n_marks")
                if obs.get("n_marks")
                is not None else ""
            ),
        })

        # Para scatter: preserva distância de cada marcador/ponto
        # à cor esperada da família pareada.
        if technique_code == "S":
            for point_rank, p in enumerate(
                obs.get("points", []),
                1
            ):
                p_de = de00_hex(
                    p["observed_hex"],
                    match["expected_hex"]
                )

                point_rows.append({
                    "family_observed_index": (
                        match["obs_index"] + 1
                    ),
                    "point_rank": point_rank,
                    "family_type": obs.get(
                        "family_type",
                        ""
                    ),
                    "encoding_candidate": obs.get(
                        "encoding_candidate",
                        ""
                    ),
                    "observed_hex": p[
                        "observed_hex"
                    ],
                    "expected_name": match[
                        "expected_name"
                    ],
                    "expected_hex": match[
                        "expected_hex"
                    ],
                    "deltaE00_point_to_expected": (
                        fmt(p_de)
                    ),
                    "lab_L": fmt(
                        p.get("lab_L")
                    ),
                    "lab_chroma": fmt(
                        p.get("lab_chroma")
                    ),
                    "hsv_saturation": fmt(
                        p.get(
                            "hsv_saturation"
                        )
                    ),
                    "hsv_value": fmt(
                        p.get("hsv_value")
                    ),
                    "exact_color_support": fmt(
                        p.get(
                            "exact_color_support"
                        )
                    ),
                })

    # ---------------------------------------------------------
    # Cores extras
    # ---------------------------------------------------------
    for i in unmatched_obs:
        obs = observed[i]

        pair_rows.append({
            "pair_status": "EXTRA",
            "pair_rank": "",
            "observed_index": i + 1,
            "expected_index": "",
            "observed_hex": obs[
                "observed_hex"
            ],
            "expected_name": "",
            "expected_hex": "",
            "deltaE00": "",
            "representation": obs.get(
                "representation",
                ""
            ),
            "family_type": obs.get(
                "family_type",
                ""
            ),
            "encoding_candidate": obs.get(
                "encoding_candidate",
                ""
            ),
            "support_mean": fmt(
                obs.get("support_mean")
            ),
            "support_min": fmt(
                obs.get("support_min")
            ),
            "solid_support_de4": fmt(
                obs.get(
                    "solid_support_de4"
                )
            ),
            "n_marks_in_family": (
                obs.get("n_marks")
                if obs.get("n_marks")
                is not None else ""
            ),
        })

    # ---------------------------------------------------------
    # Cores esperadas ausentes
    # ---------------------------------------------------------
    for j in unmatched_exp:
        pair_rows.append({
            "pair_status": "MISSING",
            "pair_rank": "",
            "observed_index": "",
            "expected_index": j + 1,
            "observed_hex": "",
            "expected_name": (
                expected_names[j]
            ),
            "expected_hex": (
                palette[
                    expected_names[j]
                ]
            ),
            "deltaE00": "",
            "representation": "",
            "family_type": "",
            "encoding_candidate": "",
            "support_mean": "",
            "support_min": "",
            "solid_support_de4": "",
            "n_marks_in_family": "",
        })

    n_expected = len(
        expected_names
    )
    n_observed = len(
        observed
    )

    n_extra = len(
        unmatched_obs
    )
    n_missing = len(
        unmatched_exp
    )

    if n_observed == 0:
        count_status = "NO_MARK"
    elif n_extra > 0:
        count_status = "EXTRA"
    elif n_missing > 0:
        count_status = "MISSING"
    else:
        count_status = "EXACT"

    point_dvals = [
        float(r[
            "deltaE00_point_to_expected"
        ])
        for r in point_rows
        if r[
            "deltaE00_point_to_expected"
        ] != ""
    ]

    unit_id = (
        f"{profile}_{unit_number:03d}"
    )

    image_row = {
        "filename": path.name,
        "profile": profile,
        "technique": technique,
        "task": task,
        "unit_id": unit_id,
        "unit_number": unit_number,
        "repeat": repeat_number,

        "canvas_width": canvas_w,
        "canvas_height": canvas_h,

        "extraction_status": extraction[
            "extraction_status"
        ],

        "count_status": count_status,

        "n_expected_colors": n_expected,
        "n_observed_colors": n_observed,
        "n_matched_colors": len(matched),
        "n_extra_colors": n_extra,
        "n_missing_colors": n_missing,
        "count_error_abs": abs(
            n_observed - n_expected
        ),

        "expected_colors": join_list(
            expected_names
        ),
        "expected_hex": join_list([
            palette[n]
            for n in expected_names
        ]),
        "observed_hex": join_list([
            o["observed_hex"]
            for o in observed
        ]),

        "Dmean_color": fmt(
            safe_mean(dvals)
        ),
        "Dmedian_color": fmt(
            safe_median(dvals)
        ),
        "Dmax_color": fmt(
            safe_max(dvals)
        ),

        "extraction_support_mean": fmt(
            extraction.get(
                "extraction_support_mean"
            )
        ),
        "extraction_support_min": fmt(
            extraction.get(
                "extraction_support_min"
            )
        ),

        "n_marks_detected": (
            extraction.get("n_marks")
            if extraction.get("n_marks")
            is not None else ""
        ),
        "n_initial_candidates": (
            extraction.get(
                "n_initial_candidates"
            )
            if extraction.get(
                "n_initial_candidates"
            ) is not None
            else ""
        ),

        "n_solid_families": extraction.get(
            "n_solid_families",
            0
        ),
        "n_variable_families": extraction.get(
            "n_variable_families",
            0
        ),
        "n_insufficient_families": extraction.get(
            "n_insufficient_families",
            0
        ),
        "internal_variation": int(
            extraction.get(
                "n_variable_families",
                0
            ) > 0
        ),

        "scatter_point_Dmean": fmt(
            safe_mean(point_dvals)
        ),
        "scatter_point_Dmedian": fmt(
            safe_median(point_dvals)
        ),
        "scatter_point_Dp90": fmt(
            safe_percentile(
                point_dvals,
                90
            )
        ),
        "scatter_point_Dmax": fmt(
            safe_max(point_dvals)
        ),

        "legend_suspected_count": extraction.get(
            "legend_suspected_count",
            0
        ),

        "n_overlap_derived_families": extraction.get(
            "n_overlap_derived_families",
            0
        ),

        "line_derived_hex": extraction.get(
            "line_derived_hex",
            ""
        ),
    }

    # Identificadores comuns nos arquivos detalhados
    common = {
        "filename": path.name,
        "profile": profile,
        "technique": technique,
        "task": task,
        "unit_id": unit_id,
        "unit_number": unit_number,
        "repeat": repeat_number,
    }

    pair_rows = [
        {**common, **r}
        for r in pair_rows
    ]

    point_rows = [
        {**common, **r}
        for r in point_rows
    ]

    return (
        image_row,
        pair_rows,
        point_rows
    )


# =====================================================================
# AGREGAÇÃO DAS 10 REPETIÇÕES
# =====================================================================

def numeric_from_row(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def aggregate_units(
    image_rows,
    expected_repeats
):
    groups = defaultdict(list)

    for row in image_rows:
        groups[
            row["unit_id"]
        ].append(row)

    unit_rows = []

    for unit_id in sorted(groups):
        rows = groups[unit_id]
        first = rows[0]

        dmax = [
            numeric_from_row(
                r,
                "Dmax_color"
            )
            for r in rows
        ]

        dmean = [
            numeric_from_row(
                r,
                "Dmean_color"
            )
            for r in rows
        ]

        p_dmax = [
            numeric_from_row(
                r,
                "scatter_point_Dmax"
            )
            for r in rows
        ]

        p_dp90 = [
            numeric_from_row(
                r,
                "scatter_point_Dp90"
            )
            for r in rows
        ]

        n_repeats = len(rows)

        eval_color = sum(
            r["Dmax_color"] != ""
            for r in rows
        )

        unit_rows.append({
            "unit_id": unit_id,
            "profile": first["profile"],
            "technique": first["technique"],
            "task": first["task"],

            "n_repeats_found": n_repeats,
            "expected_repeats": (
                expected_repeats
            ),
            "repeats_complete": int(
                n_repeats
                == expected_repeats
            ),
            "n_color_evaluable": (
                eval_color
            ),

            "Dmax_mean": fmt(
                safe_mean(dmax)
            ),
            "Dmax_median": fmt(
                safe_median(dmax)
            ),
            "Dmax_IQR": fmt(
                safe_iqr(dmax)
            ),
            "Dmax_max": fmt(
                safe_max(dmax)
            ),

            "Dmean_mean": fmt(
                safe_mean(dmean)
            ),
            "Dmean_median": fmt(
                safe_median(dmean)
            ),
            "Dmean_IQR": fmt(
                safe_iqr(dmean)
            ),

            "p_extra_color": fmt(
                np.mean([
                    r["n_extra_colors"] > 0
                    for r in rows
                ])
            ),
            "p_missing_color": fmt(
                np.mean([
                    r["n_missing_colors"] > 0
                    for r in rows
                ])
            ),
            "p_exact_color_count": fmt(
                np.mean([
                    r["count_status"]
                    == "EXACT"
                    for r in rows
                ])
            ),
            "p_no_mark": fmt(
                np.mean([
                    r["extraction_status"]
                    == "NO_MARK"
                    for r in rows
                ])
            ),
            "p_internal_variation": fmt(
                np.mean([
                    r["internal_variation"]
                    == 1
                    for r in rows
                ])
            ),

            "p_line_overlap_derived": fmt(
                np.mean([
                    r.get(
                        "n_overlap_derived_families",
                        0
                    ) > 0
                    for r in rows
                ])
            ),

            "observed_colors_mean": fmt(
                np.mean([
                    r["n_observed_colors"]
                    for r in rows
                ])
            ),
            "extra_colors_mean": fmt(
                np.mean([
                    r["n_extra_colors"]
                    for r in rows
                ])
            ),
            "missing_colors_mean": fmt(
                np.mean([
                    r["n_missing_colors"]
                    for r in rows
                ])
            ),

            "extraction_support_mean": fmt(
                safe_mean([
                    numeric_from_row(
                        r,
                        "extraction_support_mean"
                    )
                    for r in rows
                ])
            ),
            "extraction_support_min": fmt(
                safe_min([
                    numeric_from_row(
                        r,
                        "extraction_support_min"
                    )
                    for r in rows
                ])
            ),

            "scatter_point_Dmax_mean": fmt(
                safe_mean(p_dmax)
            ),
            "scatter_point_Dp90_mean": fmt(
                safe_mean(p_dp90)
            ),
        })

    return unit_rows


# =====================================================================
# CSV
# =====================================================================

def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


# =====================================================================
# MAIN
# =====================================================================

def main():
    args = parse_args()

    root = args.root.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {root}"
        )

    cfg = read_config(
        args.config
    )

    modules = import_extractors()

    output = root / OUTPUT_DIRNAME
    output.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        args.config,
        output
        / "color_conformity_config_used.json"
    )

    allowed_profiles = None

    if args.profiles:
        allowed_profiles = {
            p.upper()
            for p in args.profiles
        }

    images = []

    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue

        m = IMAGE_RE.match(
            path.name
        )

        if not m:
            continue

        profile = m.group(1).upper()

        if (
            allowed_profiles is not None
            and profile not in allowed_profiles
        ):
            continue

        unit_number = int(
            m.group(2)
        )

        repeat_number = int(
            m.group(3)
        )

        images.append(
            (
                path,
                profile,
                unit_number,
                repeat_number
            )
        )

    # ---------------------------------------------------------
    # Amostragem por UNIDADES COMPLETAS
    # ---------------------------------------------------------
    if args.units_per_profile is not None:
        if args.units_per_profile <= 0:
            raise ValueError(
                "--units-per-profile deve ser maior que zero."
            )

        expected_repeats = int(
            cfg.get("expected_repeats_per_unit", 10)
        )

        # Organiza as imagens por perfil e unidade experimental.
        units_by_profile = defaultdict(
            lambda: defaultdict(list)
        )

        for item in images:
            path, profile, unit_number, repeat_number = item
            units_by_profile[profile][unit_number].append(item)

        requested_profiles = (
            sorted(allowed_profiles)
            if allowed_profiles is not None
            else ["BI", "BC", "LI", "LC", "SI", "SC"]
        )

        rng = random.Random(args.sample_seed)
        sampled_units = []

        for profile in requested_profiles:
            eligible = []

            for unit_number, items in units_by_profile.get(
                profile, {}
            ).items():
                repeats = sorted(
                    {it[3] for it in items}
                )

                expected_repeat_ids = list(
                    range(1, expected_repeats + 1)
                )

                if repeats == expected_repeat_ids:
                    # Ordena R01...R10
                    items_sorted = sorted(
                        items,
                        key=lambda x: x[3]
                    )
                    eligible.append(
                        (unit_number, items_sorted)
                    )

            if len(eligible) < args.units_per_profile:
                raise RuntimeError(
                    f"Perfil {profile}: há apenas {len(eligible)} "
                    f"unidades completas elegíveis, mas foram solicitadas "
                    f"{args.units_per_profile}."
                )

            chosen_units = rng.sample(
                eligible,
                args.units_per_profile
            )

            for unit_number, items_sorted in sorted(
                chosen_units,
                key=lambda x: x[0]
            ):
                sampled_units.extend(items_sorted)

        images = sampled_units

    # ---------------------------------------------------------
    # Amostragem estratificada opcional
    # ---------------------------------------------------------
    if args.sample_per_profile is not None:
        if args.units_per_profile is not None:
            raise ValueError(
                "Use apenas um dos modos de amostragem: "
                "--units-per-profile OU --sample-per-profile."
            )
        if args.sample_per_profile <= 0:
            raise ValueError(
                "--sample-per-profile deve ser maior que zero."
            )

        grouped = defaultdict(list)

        for item in images:
            grouped[item[1]].append(item)

        requested_profiles = (
            sorted(allowed_profiles)
            if allowed_profiles is not None
            else ["BI", "BC", "LI", "LC", "SI", "SC"]
        )

        rng = random.Random(args.sample_seed)
        sampled = []

        for profile in requested_profiles:
            pool = grouped.get(profile, [])

            if len(pool) < args.sample_per_profile:
                raise RuntimeError(
                    f"Perfil {profile}: disponíveis={len(pool)}, "
                    f"solicitadas={args.sample_per_profile}."
                )

            chosen = rng.sample(
                pool,
                args.sample_per_profile
            )

            sampled.extend(
                sorted(chosen, key=lambda x: x[0].name)
            )

        images = sampled

    # --limit, se usado junto, é aplicado depois da amostragem.
    if args.limit is not None:
        images = images[
            :max(0, args.limit)
        ]

    print("=" * 84)
    print(
        "AVALIAÇÃO UNIFICADA DE "
        "CONFORMIDADE DE COR"
    )
    print("=" * 84)
    print(f"Pasta: {root}")
    print(
        f"Imagens canônicas encontradas: "
        f"{len(images)}"
    )
    print(
        "Subpastas são ignoradas."
    )
    print(
        "Nenhum limiar final de ΔE00 "
        "será aplicado."
    )
    print()

    image_rows = []
    pair_rows = []
    point_rows = []
    error_rows = []

    started = time.time()

    for idx, (
        path,
        profile,
        unit_number,
        repeat_number
    ) in enumerate(images, 1):

        try:
            (
                image_row,
                pairs,
                points
            ) = process_image(
                path,
                profile,
                unit_number,
                repeat_number,
                cfg,
                modules
            )

            image_rows.append(
                image_row
            )

            pair_rows.extend(
                pairs
            )

            point_rows.extend(
                points
            )

            status = (
                f"{image_row['count_status']} "
                f"Dmax={image_row['Dmax_color'] or 'NA'}"
            )

        except Exception as exc:
            error_rows.append({
                "filename": path.name,
                "profile": profile,
                "unit_number": unit_number,
                "repeat": repeat_number,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            status = (
                f"ERRO: {type(exc).__name__}"
            )

        if (
            idx == 1
            or idx % 25 == 0
            or idx == len(images)
        ):
            print(
                f"{idx:04d}/{len(images):04d} | "
                f"{path.name} | {status}"
            )

    expected_repeats = int(
        cfg.get(
            "expected_repeats_per_unit",
            10
        )
    )

    unit_rows = aggregate_units(
        image_rows,
        expected_repeats
    )

    # ---------------------------------------------------------
    # Campos
    # ---------------------------------------------------------
    image_fields = [
        "filename",
        "profile",
        "technique",
        "task",
        "unit_id",
        "unit_number",
        "repeat",
        "canvas_width",
        "canvas_height",

        "extraction_status",
        "count_status",

        "n_expected_colors",
        "n_observed_colors",
        "n_matched_colors",
        "n_extra_colors",
        "n_missing_colors",
        "count_error_abs",

        "expected_colors",
        "expected_hex",
        "observed_hex",

        "Dmean_color",
        "Dmedian_color",
        "Dmax_color",

        "extraction_support_mean",
        "extraction_support_min",

        "n_marks_detected",
        "n_initial_candidates",

        "n_solid_families",
        "n_variable_families",
        "n_insufficient_families",
        "internal_variation",

        "scatter_point_Dmean",
        "scatter_point_Dmedian",
        "scatter_point_Dp90",
        "scatter_point_Dmax",

        "legend_suspected_count",
        "n_overlap_derived_families",
        "line_derived_hex",
    ]

    pair_fields = [
        "filename",
        "profile",
        "technique",
        "task",
        "unit_id",
        "unit_number",
        "repeat",

        "pair_status",
        "pair_rank",
        "observed_index",
        "expected_index",

        "observed_hex",
        "expected_name",
        "expected_hex",
        "deltaE00",

        "representation",
        "family_type",
        "encoding_candidate",

        "support_mean",
        "support_min",
        "solid_support_de4",
        "n_marks_in_family",
    ]

    point_fields = [
        "filename",
        "profile",
        "technique",
        "task",
        "unit_id",
        "unit_number",
        "repeat",

        "family_observed_index",
        "point_rank",
        "family_type",
        "encoding_candidate",

        "observed_hex",
        "expected_name",
        "expected_hex",
        "deltaE00_point_to_expected",

        "lab_L",
        "lab_chroma",
        "hsv_saturation",
        "hsv_value",
        "exact_color_support",
    ]

    unit_fields = [
        "unit_id",
        "profile",
        "technique",
        "task",

        "n_repeats_found",
        "expected_repeats",
        "repeats_complete",
        "n_color_evaluable",

        "Dmax_mean",
        "Dmax_median",
        "Dmax_IQR",
        "Dmax_max",

        "Dmean_mean",
        "Dmean_median",
        "Dmean_IQR",

        "p_extra_color",
        "p_missing_color",
        "p_exact_color_count",
        "p_no_mark",
        "p_internal_variation",
        "p_line_overlap_derived",

        "observed_colors_mean",
        "extra_colors_mean",
        "missing_colors_mean",

        "extraction_support_mean",
        "extraction_support_min",

        "scatter_point_Dmax_mean",
        "scatter_point_Dp90_mean",
    ]

    error_fields = [
        "filename",
        "profile",
        "unit_number",
        "repeat",
        "error_type",
        "error_message",
    ]

    # ---------------------------------------------------------
    # Escrita
    # ---------------------------------------------------------
    images_csv = (
        output
        / "color_conformity_images.csv"
    )

    pairs_csv = (
        output
        / "color_conformity_pairs.csv"
    )

    points_csv = (
        output
        / "color_conformity_scatter_points.csv"
    )

    units_csv = (
        output
        / "color_conformity_units.csv"
    )

    errors_csv = (
        output
        / "color_conformity_errors.csv"
    )

    write_csv(
        images_csv,
        image_rows,
        image_fields
    )

    write_csv(
        pairs_csv,
        pair_rows,
        pair_fields
    )

    write_csv(
        points_csv,
        point_rows,
        point_fields
    )

    write_csv(
        units_csv,
        unit_rows,
        unit_fields
    )

    write_csv(
        errors_csv,
        error_rows,
        error_fields
    )

    # ---------------------------------------------------------
    # Resumo
    # ---------------------------------------------------------
    elapsed = (
        time.time() - started
    )

    by_profile = defaultdict(int)
    for r in image_rows:
        by_profile[
            r["profile"]
        ] += 1

    n_exact = sum(
        r["count_status"] == "EXACT"
        for r in image_rows
    )

    n_extra = sum(
        r["n_extra_colors"] > 0
        for r in image_rows
    )

    n_missing = sum(
        r["n_missing_colors"] > 0
        for r in image_rows
    )

    n_no_mark = sum(
        r["extraction_status"] == "NO_MARK"
        for r in image_rows
    )

    n_variable = sum(
        r["internal_variation"] == 1
        for r in image_rows
    )

    n_line_overlap_derived = sum(
        r.get(
            "n_overlap_derived_families",
            0
        ) > 0
        for r in image_rows
    )

    summary = [
        "AVALIAÇÃO UNIFICADA DE CONFORMIDADE DE COR",
        "=" * 76,
        f"Pasta: {root}",
        f"Imagens selecionadas: {len(images)}",
        f"Imagens processadas: {len(image_rows)}",
        f"Erros de processamento: {len(error_rows)}",
        f"Unidades agregadas: {len(unit_rows)}",
        "",
        "POR PERFIL:",
    ]

    for profile in (
        "BI", "BC", "LI",
        "LC", "SI", "SC"
    ):
        summary.append(
            f"{profile}: "
            f"{by_profile.get(profile, 0)}"
        )

    summary.extend([
        "",
        "DIAGNÓSTICO DE CONTAGEM:",
        f"contagem exata: {n_exact}",
        f"com cor extra: {n_extra}",
        f"com cor esperada ausente: {n_missing}",
        f"sem marca cromática avaliável: {n_no_mark}",
        f"scatter com variação cromática interna: {n_variable}",
        f"linhas com família OVERLAP_DERIVED: {n_line_overlap_derived}",
        "",
        "MÉTRICAS PRODUZIDAS:",
        "- Dmean_color",
        "- Dmedian_color",
        "- Dmax_color",
        "- n_extra_colors",
        "- n_missing_colors",
        "- extraction_support_mean/min",
        "- scatter solid/variation/insufficient",
        "- scatter_point_Dmean/median/P90/max",
        "- n_overlap_derived_families / line_derived_hex",
        "",
        "AGREGAÇÃO POR UNIDADE:",
        "- média, mediana, IQR e máximo de Dmax",
        "- média, mediana e IQR de Dmean",
        "- proporção de repetições com cor extra/ausente",
        "- proporção de repetições com variação cromática",
        "- proporção de repetições com OVERLAP_DERIVED em linhas",
        "- métricas ponto a ponto para scatter",
        "",
        "IMPORTANTE:",
        "- nenhum limiar final de conformidade ΔE00 foi aplicado;",
        "- Dmax/Dmean usam somente pares cromáticos efetivamente correspondidos;",
        "- cores extras e ausentes permanecem variáveis separadas;",
        "- ausência de marca não recebe distância cromática artificial;",
        "",
        f"Tempo total: {elapsed/60.0:.2f} min",
        "",
        f"CSV imagens: {images_csv}",
        f"CSV pares: {pairs_csv}",
        f"CSV pontos scatter: {points_csv}",
        f"CSV unidades: {units_csv}",
        f"CSV erros: {errors_csv}",
    ])

    summary_path = (
        output
        / "color_conformity_summary.txt"
    )

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print()
    print("=" * 84)
    print("CONCLUÍDO")
    print("=" * 84)
    print(
        f"Imagens processadas: "
        f"{len(image_rows)}"
    )
    print(
        f"Unidades agregadas: "
        f"{len(unit_rows)}"
    )
    print(
        f"Erros: {len(error_rows)}"
    )
    print(
        f"Saída: {output}"
    )
    print()
    print(
        "Abra primeiro:"
    )
    print(
        summary_path
    )


if __name__ == "__main__":
    main()
