# -*- coding: utf-8 -*-
"""
visual_features_v4_plot_area_v6.py

Aplica o detector V6 da área de plotagem às 3.120 imagens experimentais.

DEFINIÇÃO METODOLÓGICA
----------------------
1. A área observada é detectada exclusivamente pela geometria da imagem.
2. A especificação F NÃO participa da detecção.
3. Somente depois da extração são calculadas as diferenças entre
   a área observada e a área esperada por F.
4. Nenhum limiar de conformidade é aplicado nesta etapa.

ENTRADA
-------
Somente arquivos .png diretamente na raiz:
C:/Users/Labvis/Downloads/imagens3120/imagens

As subpastas são ignoradas.

SAÍDAS
------
imagens/_visual_processing_v4/
    visual_features_v4_plot_area.csv
    plot_area_v4_summary.txt
    processing_errors.csv

    audit/
        audit_manifest.csv
        audit_contact_sheet.png
        overlays/*.png

AUDITORIA
---------
Por perfil BI, BC, LI, LC, SI e SC:
- 3 imagens com menor consistência interna;
- 3 imagens com maior dispersão dos grupos geométricos;
- 2 imagens aleatórias reprodutíveis.

As imagens de maior divergência de F NÃO são usadas para escolher a auditoria,
porque distância de F é uma variável experimental, não evidência de erro
do detector.

OVERLAY
-------
vermelho = área detectada
azul     = área esperada segundo F
verde    = grupos horizontais usados
laranja  = grupos verticais usados

DEPENDÊNCIAS
------------
numpy
pillow
opencv-python

Instalação:
    python -m pip install numpy pillow opencv-python
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

try:
    import numpy as np
except ImportError:
    raise SystemExit(
        "\nERRO: NumPy não instalado.\n"
        "Execute: python -m pip install numpy pillow opencv-python\n"
    )

try:
    import cv2
except ImportError:
    raise SystemExit(
        "\nERRO: OpenCV não instalado.\n"
        "Execute: python -m pip install opencv-python\n"
    )

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit(
        "\nERRO: Pillow não instalado.\n"
        "Execute: python -m pip install pillow\n"
    )


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path("C:/Users/Labvis/Downloads/imagens3120/imagens")
OUTPUT_DIR = ROOT_DIR / "_visual_processing_v4"
AUDIT_DIR = OUTPUT_DIR / "audit"
OVERLAY_DIR = AUDIT_DIR / "overlays"


# ============================================================
# IDENTIDADE
# ============================================================

EXPECTED_PROFILES = ("BI", "BC", "LI", "LC", "SI", "SC")

NAME_PATTERN = re.compile(
    r"^(?P<profile>BI|BC|LI|LC|SI|SC)_"
    r"(?P<unit>\d{3})_R(?P<replica>\d{2})\.png$",
    re.IGNORECASE,
)

TECHNIQUE_MAP = {
    "B": "barra",
    "L": "linha",
    "S": "scatterplot",
}

TASK_MAP = {
    "I": "identificacao",
    "C": "comparacao",
}


# ============================================================
# F - SOMENTE COMPARAÇÃO POSTERIOR
# ============================================================

F_LEFT = 0.08
F_RIGHT = 0.95
F_TOP_MPL = 0.88
F_BOTTOM_MPL = 0.15

# Sistema de coordenadas da imagem
F_X_LEFT = F_LEFT
F_X_RIGHT = F_RIGHT
F_Y_TOP_IMAGE = 1.0 - F_TOP_MPL
F_Y_BOTTOM_IMAGE = 1.0 - F_BOTTOM_MPL


# ============================================================
# DETECTOR V6
# ============================================================

MAX_CHROMA = 28
MIN_BG_DIFF = 1.5

H_KERNEL_FRAC = 0.055
V_KERNEL_FRAC = 0.055

MIN_H_SEGMENT_FRAC = 0.055
MIN_V_SEGMENT_FRAC = 0.055

MAX_LINE_THICKNESS_PX = 12

H_Y_TOL = 4
V_X_TOL = 4

MIN_H_GROUP_COVERAGE = 0.25
MIN_V_GROUP_COVERAGE = 0.25

X_PLAUSIBLE_RANGE = (0.015, 0.995)
Y_PLAUSIBLE_RANGE = (0.015, 0.985)


# ============================================================
# AUDITORIA
# ============================================================

LOW_CONF_PER_PROFILE = 3
HIGH_SPREAD_PER_PROFILE = 3
RANDOM_PER_PROFILE = 2
RANDOM_SEED = 20260827

PANEL_W = 490
PANEL_H = 370
THUMB_W = 455
THUMB_H = 290
COLS = 3


# ============================================================
# IDENTIDADE
# ============================================================

def parse_filename(filename: str) -> dict:
    m = NAME_PATTERN.match(filename)

    if not m:
        return {
            "name_valid": 0,
            "profile": "",
            "unit_number": "",
            "replica_number": "",
            "unit_id": "",
            "image_id": Path(filename).stem,
            "technique": "",
            "task": "",
        }

    profile = m.group("profile").upper()
    unit = int(m.group("unit"))
    replica = int(m.group("replica"))

    return {
        "name_valid": 1,
        "profile": profile,
        "unit_number": unit,
        "replica_number": replica,
        "unit_id": f"{profile}_{unit:03d}",
        "image_id": f"{profile}_{unit:03d}_R{replica:02d}",
        "technique": TECHNIQUE_MAP[profile[0]],
        "task": TASK_MAP[profile[1]],
    }


def safe(v, n=6):
    if v is None:
        return ""
    return round(float(v), n)


# ============================================================
# MÁSCARA
# ============================================================

def estimate_background_rgb(arr: np.ndarray) -> np.ndarray:
    h, w, _ = arr.shape
    bw = max(2, min(6, max(2, w // 100), max(2, h // 100)))

    parts = [
        arr[:bw, :, :].reshape(-1, 3),
        arr[h-bw:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3),
        arr[:, w-bw:, :].reshape(-1, 3),
    ]

    return np.median(np.concatenate(parts, axis=0), axis=0)


def make_acromatic_mask(arr: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    rgb = arr.astype(np.float32)

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    chroma = maxc - minc

    diff = np.sqrt(
        ((rgb - bg_rgb.reshape(1, 1, 3)) ** 2).sum(axis=2)
    )

    return (
        (chroma <= MAX_CHROMA)
        & (diff >= MIN_BG_DIFF)
    ).astype(np.uint8) * 255


# ============================================================
# SEGMENTOS MORFOLÓGICOS
# ============================================================

def extract_horizontal_segments(mask: np.ndarray):
    h, w = mask.shape

    klen = max(9, int(round(H_KERNEL_FRAC * w)))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (klen, 1)
    )

    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    horizontal = cv2.dilate(
        horizontal,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
        iterations=1
    )

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segments = []

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)

        if ww < MIN_H_SEGMENT_FRAC * w:
            continue

        if hh > MAX_LINE_THICKNESS_PX:
            continue

        segments.append({
            "x1": int(x),
            "x2": int(x + ww - 1),
            "y": int(round(y + (hh - 1) / 2)),
            "thickness": int(hh),
            "length": int(ww),
        })

    return segments


def extract_vertical_segments(mask: np.ndarray):
    h, w = mask.shape

    klen = max(9, int(round(V_KERNEL_FRAC * h)))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, klen)
    )

    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    vertical = cv2.dilate(
        vertical,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        iterations=1
    )

    contours, _ = cv2.findContours(
        vertical,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    segments = []

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)

        if hh < MIN_V_SEGMENT_FRAC * h:
            continue

        if ww > MAX_LINE_THICKNESS_PX:
            continue

        segments.append({
            "y1": int(y),
            "y2": int(y + hh - 1),
            "x": int(round(x + (ww - 1) / 2)),
            "thickness": int(ww),
            "length": int(hh),
        })

    return segments


# ============================================================
# AGRUPAMENTO
# ============================================================

def union_length(intervals):
    if not intervals:
        return 0

    intervals = sorted(intervals)
    total = 0
    s, e = intervals[0]

    for ns, ne in intervals[1:]:
        if ns <= e + 1:
            e = max(e, ne)
        else:
            total += e - s + 1
            s, e = ns, ne

    total += e - s + 1
    return total


def group_horizontal(segments, canvas_w):
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s["y"])
    groups = []

    for seg in segments:
        placed = False

        for group in groups:
            if abs(seg["y"] - group["y_mean"]) <= H_Y_TOL:
                group["segments"].append(seg)
                group["y_mean"] = mean(s["y"] for s in group["segments"])
                placed = True
                break

        if not placed:
            groups.append({
                "segments": [seg],
                "y_mean": float(seg["y"]),
            })

    result = []

    for group in groups:
        intervals = [
            (s["x1"], s["x2"])
            for s in group["segments"]
        ]

        coverage = union_length(intervals) / canvas_w
        x1 = min(s["x1"] for s in group["segments"])
        x2 = max(s["x2"] for s in group["segments"])

        if coverage < MIN_H_GROUP_COVERAGE:
            continue

        result.append({
            "y": int(round(group["y_mean"])),
            "x1": int(x1),
            "x2": int(x2),
            "coverage": float(coverage),
            "n_segments": len(group["segments"]),
        })

    return result


def group_vertical(segments, canvas_h):
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s["x"])
    groups = []

    for seg in segments:
        placed = False

        for group in groups:
            if abs(seg["x"] - group["x_mean"]) <= V_X_TOL:
                group["segments"].append(seg)
                group["x_mean"] = mean(s["x"] for s in group["segments"])
                placed = True
                break

        if not placed:
            groups.append({
                "segments": [seg],
                "x_mean": float(seg["x"]),
            })

    result = []

    for group in groups:
        intervals = [
            (s["y1"], s["y2"])
            for s in group["segments"]
        ]

        coverage = union_length(intervals) / canvas_h
        y1 = min(s["y1"] for s in group["segments"])
        y2 = max(s["y2"] for s in group["segments"])

        if coverage < MIN_V_GROUP_COVERAGE:
            continue

        result.append({
            "x": int(round(group["x_mean"])),
            "y1": int(y1),
            "y2": int(y2),
            "coverage": float(coverage),
            "n_segments": len(group["segments"]),
        })

    return result


def robust_median_int(values):
    if not values:
        return None
    return int(round(float(np.median(values))))


# ============================================================
# DETECTOR
# ============================================================

def detect_plot_bbox(img: Image.Image):
    arr = np.asarray(img.convert("RGB"))
    h, w, _ = arr.shape

    bg = estimate_background_rgb(arr)
    mask = make_acromatic_mask(arr, bg)

    h_segments = extract_horizontal_segments(mask)
    v_segments = extract_vertical_segments(mask)

    h_groups = group_horizontal(h_segments, w)
    v_groups = group_vertical(v_segments, h)

    h_groups = [
        g for g in h_groups
        if (
            X_PLAUSIBLE_RANGE[0] * w <= g["x1"] <= X_PLAUSIBLE_RANGE[1] * w
            and
            X_PLAUSIBLE_RANGE[0] * w <= g["x2"] <= X_PLAUSIBLE_RANGE[1] * w
            and
            Y_PLAUSIBLE_RANGE[0] * h <= g["y"] <= Y_PLAUSIBLE_RANGE[1] * h
        )
    ]

    v_groups = [
        g for g in v_groups
        if (
            X_PLAUSIBLE_RANGE[0] * w <= g["x"] <= X_PLAUSIBLE_RANGE[1] * w
            and
            Y_PLAUSIBLE_RANGE[0] * h <= g["y1"] <= Y_PLAUSIBLE_RANGE[1] * h
            and
            Y_PLAUSIBLE_RANGE[0] * h <= g["y2"] <= Y_PLAUSIBLE_RANGE[1] * h
        )
    ]

    methods = []

    x_left = None
    x_right = None
    y_top = None
    y_bottom = None

    # X pelo consenso horizontal
    if len(h_groups) >= 2:
        x_left = robust_median_int([g["x1"] for g in h_groups])
        x_right = robust_median_int([g["x2"] for g in h_groups])
        methods.append("x_from_horizontal_groups")

    # Fallback X pelas posições verticais
    if (x_left is None or x_right is None) and len(v_groups) >= 2:
        xs = sorted(g["x"] for g in v_groups)
        x_left = xs[0]
        x_right = xs[-1]
        methods.append("x_from_vertical_groups")

    # Y pelo consenso vertical
    if len(v_groups) >= 2:
        y_top = robust_median_int([g["y1"] for g in v_groups])
        y_bottom = robust_median_int([g["y2"] for g in v_groups])
        methods.append("y_from_vertical_groups")

    # Fallback Y pelas posições horizontais
    if (y_top is None or y_bottom is None) and len(h_groups) >= 2:
        ys = sorted(g["y"] for g in h_groups)
        y_top = ys[0]
        y_bottom = ys[-1]
        methods.append("y_from_horizontal_groups")

    reasons = []

    if x_left is None:
        reasons.append("left_not_found")
    if x_right is None:
        reasons.append("right_not_found")
    if y_top is None:
        reasons.append("top_not_found")
    if y_bottom is None:
        reasons.append("bottom_not_found")

    status = "OK" if not reasons else "INCOMPLETE"

    if status == "OK":
        if x_right <= x_left or y_bottom <= y_top:
            status = "INVALID_GEOMETRY"
            reasons.append("reversed_bounds")

        width_frac = (x_right - x_left) / w
        height_frac = (y_bottom - y_top) / h

        if width_frac < 0.35:
            reasons.append("plot_width_too_small")

        if height_frac < 0.35:
            reasons.append("plot_height_too_small")

        if reasons and status == "OK":
            status = "LOW_CONFIDENCE"
    else:
        width_frac = None
        height_frac = None

    # Dispersão interna dos grupos
    spreads = []

    h_left_sd = None
    h_right_sd = None
    v_top_sd = None
    v_bottom_sd = None

    if h_groups:
        h_left_sd = float(np.std([g["x1"] for g in h_groups]))
        h_right_sd = float(np.std([g["x2"] for g in h_groups]))

        spreads.extend([
            h_left_sd / max(1.0, w),
            h_right_sd / max(1.0, w),
        ])

    if v_groups:
        v_top_sd = float(np.std([g["y1"] for g in v_groups]))
        v_bottom_sd = float(np.std([g["y2"] for g in v_groups]))

        spreads.extend([
            v_top_sd / max(1.0, h),
            v_bottom_sd / max(1.0, h),
        ])

    spread_norm = mean(spreads) if spreads else None

    confidence = (
        max(0.0, 1.0 - 4.0 * spread_norm)
        if spread_norm is not None
        else 0.0
    )

    return {
        "status": status,
        "reasons": ";".join(reasons),
        "method": ";".join(methods),
        "confidence": confidence,

        "x_left": x_left,
        "x_right": x_right,
        "y_top": y_top,
        "y_bottom": y_bottom,

        "left_norm_image": x_left / w if x_left is not None else None,
        "right_norm_image": x_right / w if x_right is not None else None,
        "top_norm_image": y_top / h if y_top is not None else None,
        "bottom_norm_image": y_bottom / h if y_bottom is not None else None,

        # Convenção Matplotlib
        "left_norm_mpl": x_left / w if x_left is not None else None,
        "right_norm_mpl": x_right / w if x_right is not None else None,
        "top_norm_mpl": 1.0 - y_top / h if y_top is not None else None,
        "bottom_norm_mpl": 1.0 - y_bottom / h if y_bottom is not None else None,

        "plot_width_frac": width_frac,
        "plot_height_frac": height_frac,

        "n_h_segments": len(h_segments),
        "n_v_segments": len(v_segments),
        "n_h_groups": len(h_groups),
        "n_v_groups": len(v_groups),

        "h_left_sd_px": h_left_sd,
        "h_right_sd_px": h_right_sd,
        "v_top_sd_px": v_top_sd,
        "v_bottom_sd_px": v_bottom_sd,
        "spread_norm": spread_norm,

        "background_rgb": tuple(int(round(x)) for x in bg),

        # Guardados apenas em memória para overlays
        "h_groups": h_groups,
        "v_groups": v_groups,
    }


# ============================================================
# COMPARAÇÃO COM F
# ============================================================

def compare_to_F(result):
    keys = (
        "left_norm_mpl",
        "right_norm_mpl",
        "top_norm_mpl",
        "bottom_norm_mpl",
    )

    if any(result[k] is None for k in keys):
        return {
            "left_error_vs_F": None,
            "right_error_vs_F": None,
            "top_error_vs_F": None,
            "bottom_error_vs_F": None,
            "layout_error_sum_vs_F": None,
            "layout_error_mean_vs_F": None,
        }

    left = abs(result["left_norm_mpl"] - F_LEFT)
    right = abs(result["right_norm_mpl"] - F_RIGHT)
    top = abs(result["top_norm_mpl"] - F_TOP_MPL)
    bottom = abs(result["bottom_norm_mpl"] - F_BOTTOM_MPL)

    errors = [left, right, top, bottom]

    return {
        "left_error_vs_F": left,
        "right_error_vs_F": right,
        "top_error_vs_F": top,
        "bottom_error_vs_F": bottom,
        "layout_error_sum_vs_F": sum(errors),
        "layout_error_mean_vs_F": mean(errors),
    }


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(src, result, out):
    with Image.open(src) as opened:
        img = opened.convert("RGB")

    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Grupos horizontais
    for g in result["h_groups"]:
        draw.line(
            [(g["x1"], g["y"]), (g["x2"], g["y"])],
            fill=(30, 160, 80),
            width=max(1, w // 800),
        )

    # Grupos verticais
    for g in result["v_groups"]:
        draw.line(
            [(g["x"], g["y1"]), (g["x"], g["y2"])],
            fill=(240, 130, 20),
            width=max(1, w // 800),
        )

    # Área observada
    if all(
        result[k] is not None
        for k in ("x_left", "x_right", "y_top", "y_bottom")
    ):
        draw.rectangle(
            [
                result["x_left"],
                result["y_top"],
                result["x_right"],
                result["y_bottom"],
            ],
            outline=(220, 20, 60),
            width=max(2, w // 400),
        )

    # F
    draw.rectangle(
        [
            int(round(F_X_LEFT * w)),
            int(round(F_Y_TOP_IMAGE * h)),
            int(round(F_X_RIGHT * w)),
            int(round(F_Y_BOTTOM_IMAGE * h)),
        ],
        outline=(30, 110, 210),
        width=max(1, w // 600),
    )

    img.save(out)


def make_contact_sheet(items, out_path):
    if not items:
        return

    rows_n = math.ceil(len(items) / COLS)

    sheet = Image.new(
        "RGB",
        (COLS * PANEL_W, rows_n * PANEL_H),
        (255, 255, 255),
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, item in enumerate(items):
        col = idx % COLS
        row = idx // COLS

        x0 = col * PANEL_W
        y0 = row * PANEL_H

        draw.rectangle(
            [x0, y0, x0 + PANEL_W - 1, y0 + PANEL_H - 1],
            fill=(248, 248, 248),
            outline=(190, 190, 190),
        )

        with Image.open(item["overlay_path"]) as opened:
            img = opened.convert("RGB")
            img.thumbnail((THUMB_W, THUMB_H))

        px = x0 + (PANEL_W - img.width) // 2
        py = y0 + 63 + (THUMB_H - img.height) // 2
        sheet.paste(img, (px, py))

        label1 = (
            f"{item['profile']} | {item['audit_reason']} | "
            f"{item['filename']}"
        )

        label2 = (
            f"conf={item['confidence']:.3f} | "
            f"spread={item['spread_norm']:.4f} | "
            f"H={item['n_h_groups']} V={item['n_v_groups']}"
        )

        draw.text((x0 + 8, y0 + 8), label1, fill=(0, 0, 0), font=font)
        draw.text((x0 + 8, y0 + 28), label2, fill=(0, 0, 0), font=font)

    sheet.save(out_path)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("VISUAL FEATURES V4 - ÁREA DE PLOTAGEM / DETECTOR V6")
    print("=" * 78)
    print(f"Pasta:\n{ROOT_DIR}")
    print("\nSomente PNGs diretamente na raiz serão processados.\n")

    if not ROOT_DIR.exists():
        print("ERRO: pasta não encontrada.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in ROOT_DIR.glob("*.png")
        if p.is_file()
    )

    print(f"PNGs encontrados na raiz: {len(paths)}")

    if len(paths) != 3120:
        print(
            f"ATENÇÃO: eram esperadas 3120 imagens; "
            f"foram encontradas {len(paths)}."
        )

    rows = []
    errors = []
    internal_results = {}

    for idx, path in enumerate(paths, start=1):
        identity = parse_filename(path.name)

        row = {
            **identity,
            "filename": path.name,
            "file_size_bytes": path.stat().st_size,
            "processing_status": "OK",
            "processing_error": "",
        }

        try:
            with Image.open(path) as opened:
                img = opened.convert("RGB")
                w, h = img.size

            result = detect_plot_bbox(img)
            comp = compare_to_F(result)

            internal_results[path.name] = result

            row.update({
                "width": w,
                "height": h,
                "canvas_1200x800": int(w == 1200 and h == 800),

                "plot_detection_status": result["status"],
                "plot_detection_reasons": result["reasons"],
                "plot_detection_method": result["method"],
                "plot_detection_confidence": safe(result["confidence"]),

                "plot_left_px": result["x_left"] if result["x_left"] is not None else "",
                "plot_right_px": result["x_right"] if result["x_right"] is not None else "",
                "plot_top_px": result["y_top"] if result["y_top"] is not None else "",
                "plot_bottom_px": result["y_bottom"] if result["y_bottom"] is not None else "",

                "plot_left_norm_mpl": safe(result["left_norm_mpl"]),
                "plot_right_norm_mpl": safe(result["right_norm_mpl"]),
                "plot_top_norm_mpl": safe(result["top_norm_mpl"]),
                "plot_bottom_norm_mpl": safe(result["bottom_norm_mpl"]),

                "plot_width_frac": safe(result["plot_width_frac"]),
                "plot_height_frac": safe(result["plot_height_frac"]),

                "n_h_segments": result["n_h_segments"],
                "n_v_segments": result["n_v_segments"],
                "n_h_groups": result["n_h_groups"],
                "n_v_groups": result["n_v_groups"],

                "h_left_sd_px": safe(result["h_left_sd_px"]),
                "h_right_sd_px": safe(result["h_right_sd_px"]),
                "v_top_sd_px": safe(result["v_top_sd_px"]),
                "v_bottom_sd_px": safe(result["v_bottom_sd_px"]),
                "spread_norm": safe(result["spread_norm"]),

                "left_error_vs_F": safe(comp["left_error_vs_F"]),
                "right_error_vs_F": safe(comp["right_error_vs_F"]),
                "top_error_vs_F": safe(comp["top_error_vs_F"]),
                "bottom_error_vs_F": safe(comp["bottom_error_vs_F"]),
                "layout_error_sum_vs_F": safe(comp["layout_error_sum_vs_F"]),
                "layout_error_mean_vs_F": safe(comp["layout_error_mean_vs_F"]),

                "background_r": result["background_rgb"][0],
                "background_g": result["background_rgb"][1],
                "background_b": result["background_rgb"][2],
            })

        except Exception as exc:
            row["processing_status"] = "ERROR"
            row["processing_error"] = repr(exc)
            errors.append(row)

        rows.append(row)

        if idx == 1 or idx % 100 == 0 or idx == len(paths):
            print(f"Processadas: {idx}/{len(paths)}")

    # --------------------------------------------------------
    # CSV PRINCIPAL
    # --------------------------------------------------------
    fields = [
        "image_id",
        "unit_id",
        "profile",
        "technique",
        "task",
        "unit_number",
        "replica_number",
        "filename",
        "name_valid",
        "file_size_bytes",
        "width",
        "height",
        "canvas_1200x800",

        "plot_detection_status",
        "plot_detection_reasons",
        "plot_detection_method",
        "plot_detection_confidence",

        "plot_left_px",
        "plot_right_px",
        "plot_top_px",
        "plot_bottom_px",

        "plot_left_norm_mpl",
        "plot_right_norm_mpl",
        "plot_top_norm_mpl",
        "plot_bottom_norm_mpl",

        "plot_width_frac",
        "plot_height_frac",

        "n_h_segments",
        "n_v_segments",
        "n_h_groups",
        "n_v_groups",

        "h_left_sd_px",
        "h_right_sd_px",
        "v_top_sd_px",
        "v_bottom_sd_px",
        "spread_norm",

        "left_error_vs_F",
        "right_error_vs_F",
        "top_error_vs_F",
        "bottom_error_vs_F",
        "layout_error_sum_vs_F",
        "layout_error_mean_vs_F",

        "background_r",
        "background_g",
        "background_b",

        "processing_status",
        "processing_error",
    ]

    csv_path = OUTPUT_DIR / "visual_features_v4_plot_area.csv"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    error_path = OUTPUT_DIR / "processing_errors.csv"

    with error_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(errors)

    # --------------------------------------------------------
    # AUDITORIA
    # --------------------------------------------------------
    valid = [
        r for r in rows
        if r["processing_status"] == "OK"
        and r["filename"] in internal_results
        and r["plot_detection_status"] in {"OK", "LOW_CONFIDENCE"}
    ]

    rng = random.Random(RANDOM_SEED)
    selected = []
    seen = set()

    def add_audit(r, reason):
        if r["filename"] in seen:
            return

        seen.add(r["filename"])
        selected.append({
            **r,
            "audit_reason": reason,
        })

    for profile in EXPECTED_PROFILES:
        p_rows = [r for r in valid if r["profile"] == profile]

        # Menor confiança
        low_conf = sorted(
            p_rows,
            key=lambda r: float(r["plot_detection_confidence"])
        )[:LOW_CONF_PER_PROFILE]

        for r in low_conf:
            add_audit(r, "LOW_CONFIDENCE")

        # Maior dispersão
        spread_rows = [
            r for r in p_rows
            if str(r.get("spread_norm", "")).strip() != ""
        ]

        high_spread = sorted(
            spread_rows,
            key=lambda r: float(r["spread_norm"]),
            reverse=True
        )[:HIGH_SPREAD_PER_PROFILE]

        for r in high_spread:
            add_audit(r, "HIGH_SPREAD")

        # Aleatórias
        remaining = [
            r for r in p_rows
            if r["filename"] not in seen
        ]

        if remaining:
            k = min(RANDOM_PER_PROFILE, len(remaining))
            for r in rng.sample(remaining, k):
                add_audit(r, "RANDOM_STRATIFIED")

    audit_rows = []
    audit_items = []

    for r in selected:
        filename = r["filename"]
        result = internal_results[filename]
        src = ROOT_DIR / filename

        overlay_path = OVERLAY_DIR / f"{Path(filename).stem}__audit.png"
        make_overlay(src, result, overlay_path)

        audit_rows.append({
            "profile": r["profile"],
            "filename": filename,
            "audit_reason": r["audit_reason"],
            "plot_detection_status": r["plot_detection_status"],
            "plot_detection_confidence": r["plot_detection_confidence"],
            "spread_norm": r["spread_norm"],
            "layout_error_mean_vs_F": r["layout_error_mean_vs_F"],
            "n_h_groups": r["n_h_groups"],
            "n_v_groups": r["n_v_groups"],
            "overlay_file": str(overlay_path.relative_to(AUDIT_DIR)),
        })

        audit_items.append({
            "profile": r["profile"],
            "filename": filename,
            "audit_reason": r["audit_reason"],
            "confidence": float(r["plot_detection_confidence"]),
            "spread_norm": float(r["spread_norm"] or 0),
            "n_h_groups": int(r["n_h_groups"]),
            "n_v_groups": int(r["n_v_groups"]),
            "overlay_path": overlay_path,
        })

    audit_manifest_path = AUDIT_DIR / "audit_manifest.csv"

    audit_fields = [
        "profile",
        "filename",
        "audit_reason",
        "plot_detection_status",
        "plot_detection_confidence",
        "spread_norm",
        "layout_error_mean_vs_F",
        "n_h_groups",
        "n_v_groups",
        "overlay_file",
    ]

    with audit_manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    contact_path = AUDIT_DIR / "audit_contact_sheet.png"
    make_contact_sheet(audit_items, contact_path)

    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------
    profile_counts = Counter(r["profile"] for r in rows)

    status_counts = Counter(
        r.get("plot_detection_status", "")
        for r in rows
        if r["processing_status"] == "OK"
    )

    summary = [
        "VISUAL FEATURES V4 - ÁREA DE PLOTAGEM / DETECTOR V6",
        "=" * 72,
        f"Imagens processadas: {len(rows)}",
        f"Erros de processamento: {len(errors)}",
        "",
        "CONTAGEM POR PERFIL:",
    ]

    for p in EXPECTED_PROFILES:
        summary.append(f"  {p}: {profile_counts.get(p, 0)}")

    summary.extend([
        "",
        "STATUS DA DETECÇÃO:",
    ])

    for status, count in sorted(status_counts.items()):
        summary.append(f"  {status or '(vazio)'}: {count}")

    summary.extend([
        "",
        "ESTATÍSTICAS POR PERFIL:",
        "  confidence_mediana | spread_mediana | layout_error_mean_vs_F_mediana",
    ])

    for profile in EXPECTED_PROFILES:
        p_rows = [
            r for r in valid
            if r["profile"] == profile
        ]

        conf = [
            float(r["plot_detection_confidence"])
            for r in p_rows
            if str(r["plot_detection_confidence"]).strip() != ""
        ]

        spread = [
            float(r["spread_norm"])
            for r in p_rows
            if str(r["spread_norm"]).strip() != ""
        ]

        layout = [
            float(r["layout_error_mean_vs_F"])
            for r in p_rows
            if str(r["layout_error_mean_vs_F"]).strip() != ""
        ]

        summary.append(
            f"  {profile}: "
            f"{median(conf) if conf else float('nan'):.4f} | "
            f"{median(spread) if spread else float('nan'):.4f} | "
            f"{median(layout) if layout else float('nan'):.4f}"
        )

    summary.extend([
        "",
        "INTERPRETAÇÃO:",
        "- confidence = consistência interna do detector, não probabilidade.",
        "- spread = dispersão interna dos grupos geométricos.",
        "- layout_error_mean_vs_F = divergência observada em relação a F.",
        "- nenhum limiar de conformidade foi aplicado.",
        "- F NÃO participa da detecção.",
        "",
        f"CSV principal: {csv_path}",
        f"Auditoria visual: {contact_path}",
        f"Manifesto da auditoria: {audit_manifest_path}",
    ])

    summary_path = OUTPUT_DIR / "plot_area_v4_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n" + "=" * 78)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 78)
    print(f"CSV principal:\n{csv_path}")
    print(f"\nResumo:\n{summary_path}")
    print(f"\nAuditoria visual:\n{contact_path}")
    print("\nNenhuma imagem foi alterada.")


if __name__ == "__main__":
    main()
