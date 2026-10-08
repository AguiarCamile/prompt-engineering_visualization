# -*- coding: utf-8 -*-
"""
validate_color_bars_core_v5.py

TESTE V5 DO MÓDULO B-Cor
========================

Objetivo
--------
Testar, ANTES de incorporar à metodologia, a hipótese de que a melhor
estimativa da cor de uma barra é a cor dominante no núcleo visível da barra.

A V5 mantém a lógica geométrica da validação V4 e altera SOMENTE a etapa
de medição da cor, permitindo uma comparação mais limpa entre as abordagens.

Estratégia V5
-------------
Para cada barra detectada:

1. usa o bounding box já detectado;
2. recorta um núcleo central, removendo proporcionalmente as bordas;
3. considera apenas pixels cromáticos no núcleo;
4. conta as cores RGB exatas;
5. define como cor representativa a cor com MAIOR frequência;
6. calcula:
       support_exact = pixels da cor dominante / pixels cromáticos do núcleo
7. calcula também um suporte tolerante:
       support_near = pixels dentro de um pequeno raio RGB da cor dominante
                      / pixels cromáticos do núcleo
8. converte a cor dominante para CIELAB;
9. calcula CIEDE2000 para a cor normativa de F mais próxima.

A comparação principal será:
- V4: cor por moda do interior erodido
- V5: cor dominante no núcleo central + percentual de suporte

Importante
----------
- NÃO altera o detector geométrico de barras.
- NÃO usa layout.
- NÃO usa OCR.
- NÃO aplica limiar conforme/não conforme.
- Usa a MESMA seleção reprodutível de 40 imagens da validação V4:
    10 BI canvas OK
    10 BI canvas não OK
    10 BC canvas OK
    10 BC canvas não OK

Saídas
------
C:/Users/Labvis/Downloads/imagens3120/imagens/_color_bars_core_validation_v5/

    bar_core_validation_v5.csv
    bar_core_colors_v5.csv
    bar_core_contact_sheet_v5.png
    bar_core_summary_v5.txt
    overlays/*.png

Overlay
-------
magenta = barra detectada
ciano   = núcleo utilizado para medir a cor
verde   = linha de base

Dependências
------------
numpy
pillow
opencv-python
scikit-image

Instalação:
    python -m pip install numpy pillow opencv-python scikit-image
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab, deltaE_ciede2000


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path("C:/Users/Labvis/Downloads/imagens3120/imagens")
OUTPUT_DIR = ROOT_DIR / "_color_bars_core_validation_v5"
OVERLAY_DIR = OUTPUT_DIR / "overlays"


# ============================================================
# AMOSTRAGEM - IGUAL À V4
# ============================================================

RANDOM_SEED = 20260827
N_PER_PROFILE_CANVAS_CLASS = 10

NAME_RE = re.compile(
    r"^(BI|BC)_(\d{3})_R(\d{2})\.png$",
    re.I
)


# ============================================================
# PALETA F
# ============================================================

F_COLORS = {
    "azul": "#1F4FD8",
    "verde": "#2E7D32",
    "amarelo": "#FFFB0D",
    "laranja": "#EF6C00",
    "vermelho": "#C62828",
    "roxo": "#6A1B9A",
}


# ============================================================
# DETECTOR GEOMÉTRICO - MESMOS PARÂMETROS DA V4
# ============================================================

MIN_SATURATION = 0.24
MIN_MODE_PIXELS = 12
MAX_MODES = 18
RGB_RADIUS = 22.0

MIN_COMPONENT_AREA_FRAC = 0.000008
MIN_WIDTH_FRAC = 0.0025
MIN_HEIGHT_FRAC = 0.0015
MIN_FILL_RATIO = 0.58
MIN_LONG_DIMENSION_FRAC = 0.004

BASELINE_TOLERANCE_FRAC = 0.014
MIN_RELATIVE_AREA_TO_LARGEST = 0.002


# ============================================================
# NOVOS PARÂMETROS V5 - MEDIÇÃO DA COR NO NÚCLEO
# ============================================================

# Fração removida de cada lado do bounding box.
# 0.10 = remove 10% à esquerda, direita, topo e base.
CORE_MARGIN_X_FRAC = 0.10
CORE_MARGIN_Y_FRAC = 0.10

# Para barras muito pequenas, sempre tenta preservar ao menos esta
# dimensão interna.
MIN_CORE_WIDTH_PX = 2
MIN_CORE_HEIGHT_PX = 2

# Pixels pouco cromáticos são excluídos do cálculo da cor dominante.
CORE_MIN_SATURATION = 0.15

# Suporte tolerante ao redor da cor dominante.
# Representa pequenos desvios rasterizados muito próximos.
CORE_RGB_NEAR_RADIUS = 12.0


# ============================================================
# PRANCHA
# ============================================================

PANEL_W = 540
PANEL_H = 390
IMG_W = 500
IMG_H = 300
COLS = 4


# ============================================================
# COR
# ============================================================

def hex_to_rgb(v):
    v = v.lstrip("#")
    return tuple(int(v[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    vals = [
        max(0, min(255, int(round(float(v)))))
        for v in rgb
    ]
    return "#{:02X}{:02X}{:02X}".format(*vals)


def rgb_lab(rgb):
    arr = np.array(
        [[[rgb[0]/255, rgb[1]/255, rgb[2]/255]]],
        dtype=float
    )
    return rgb2lab(arr)[0, 0]


F_LAB = {
    name: rgb_lab(hex_to_rgb(h))
    for name, h in F_COLORS.items()
}


def nearest_f(rgb):
    lab = rgb_lab(rgb)

    distances = {
        name: float(
            deltaE_ciede2000(
                np.array([[lab]]),
                np.array([[target]])
            )[0, 0]
        )
        for name, target in F_LAB.items()
    }

    name = min(distances, key=distances.get)

    return (
        name,
        F_COLORS[name],
        distances[name]
    )


def saturation_map(arr):
    x = arr.astype(np.float32) / 255.0

    mx = x.max(axis=2)
    mn = x.min(axis=2)

    sat = np.zeros_like(mx)

    nz = mx > 0
    sat[nz] = (mx[nz] - mn[nz]) / mx[nz]

    return sat


# ============================================================
# DETECTOR DE BARRAS - V4
# ============================================================

def color_modes(arr):
    pix = arr[
        saturation_map(arr) >= MIN_SATURATION
    ]

    if len(pix) == 0:
        return []

    packed = (
        (pix[:, 0].astype(np.uint32) << 16)
        | (pix[:, 1].astype(np.uint32) << 8)
        | pix[:, 2].astype(np.uint32)
    )

    vals, counts = np.unique(
        packed,
        return_counts=True
    )

    modes = []

    for i in np.argsort(counts)[::-1]:
        if counts[i] < MIN_MODE_PIXELS:
            break

        v = int(vals[i])

        rgb = (
            (v >> 16) & 255,
            (v >> 8) & 255,
            v & 255
        )

        modes.append((
            rgb,
            int(counts[i])
        ))

        if len(modes) >= MAX_MODES:
            break

    return modes


def rgb_distance(arr, rgb):
    target = np.array(
        rgb,
        dtype=np.float32
    ).reshape(1, 1, 3)

    d = arr.astype(np.float32) - target

    return np.sqrt(
        (d * d).sum(axis=2)
    )


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)

    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    union = (
        aw * ah
        + bw * bh
        - inter
    )

    return inter / union if union else 0.0


def initial_components(img):
    arr = np.asarray(
        img.convert("RGB"),
        dtype=np.uint8
    )

    h, w, _ = arr.shape

    min_area = max(
        4,
        int(round(
            MIN_COMPONENT_AREA_FRAC * w * h
        ))
    )

    min_w = max(
        2,
        int(round(MIN_WIDTH_FRAC * w))
    )

    min_h = max(
        2,
        int(round(MIN_HEIGHT_FRAC * h))
    )

    min_long = max(
        3,
        int(round(
            MIN_LONG_DIMENSION_FRAC * h
        ))
    )

    out = []
    seen = []

    for seed_rgb, seed_count in color_modes(arr):

        mask = (
            rgb_distance(arr, seed_rgb)
            <= RGB_RADIUS
        ).astype(np.uint8) * 255

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3)
            )
        )

        n, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                8
            )
        )

        for lab in range(1, n):
            x, y, ww, hh, area = stats[lab]

            if (
                area < min_area
                or ww < min_w
                or hh < min_h
                or hh < min_long
            ):
                continue

            fill = (
                area / (ww * hh)
                if ww * hh else 0
            )

            if fill < MIN_FILL_RATIO:
                continue

            if hh < 0.25 * ww:
                continue

            box = (
                int(x),
                int(y),
                int(ww),
                int(hh)
            )

            if any(
                iou(box, old) >= 0.72
                for old in seen
            ):
                continue

            out.append({
                "x": int(x),
                "y": int(y),
                "w": int(ww),
                "h": int(hh),
                "bottom": int(y + hh),
                "center_x": float(
                    x + ww / 2
                ),
                "area": int(area),
                "fill_ratio": float(fill),
                "seed_rgb": seed_rgb,
                "seed_count": seed_count,
            })

            seen.append(box)

    return out, arr, (w, h)


def choose_baseline(cands, h):
    if not cands:
        return None

    tol = max(
        3,
        int(round(
            BASELINE_TOLERANCE_FRAC * h
        ))
    )

    clusters = []

    for c in sorted(
        cands,
        key=lambda z: z["bottom"]
    ):
        placed = False

        for cl in clusters:
            center = np.median(
                [x["bottom"] for x in cl]
            )

            if abs(
                c["bottom"] - center
            ) <= tol:
                cl.append(c)
                placed = True
                break

        if not placed:
            clusters.append([c])

    def score(cl):
        return (
            sum(c["area"] for c in cl)
            * (
                1
                + 0.10
                * max(0, len(cl) - 1)
            )
        )

    best = max(
        clusters,
        key=score
    )

    return int(round(
        float(np.median(
            [c["bottom"] for c in best]
        ))
    ))


def detect_bar_geometry(img):
    cands, arr, (_, h) = (
        initial_components(img)
    )

    if not cands:
        return [], None, arr, 0

    largest = max(
        c["area"]
        for c in cands
    )

    cands = [
        c for c in cands
        if c["area"]
        >= MIN_RELATIVE_AREA_TO_LARGEST
        * largest
    ]

    baseline = choose_baseline(
        cands,
        h
    )

    if baseline is None:
        return (
            [],
            None,
            arr,
            len(cands)
        )

    tol = max(
        3,
        int(round(
            BASELINE_TOLERANCE_FRAC * h
        ))
    )

    bars = [
        c for c in cands
        if abs(
            c["bottom"] - baseline
        ) <= tol
    ]

    bars.sort(
        key=lambda c: (
            c["center_x"],
            c["y"]
        )
    )

    return (
        bars,
        baseline,
        arr,
        len(cands)
    )


# ============================================================
# NOVA MEDIÇÃO V5
# ============================================================

def core_box(bar):
    """
    Calcula o retângulo central da barra.
    """
    x = bar["x"]
    y = bar["y"]
    w = bar["w"]
    h = bar["h"]

    mx = int(round(
        CORE_MARGIN_X_FRAC * w
    ))

    my = int(round(
        CORE_MARGIN_Y_FRAC * h
    ))

    # Em barras pequenas, remove no máximo o necessário
    # para preservar um núcleo mínimo.
    max_mx = max(
        0,
        (w - MIN_CORE_WIDTH_PX) // 2
    )

    max_my = max(
        0,
        (h - MIN_CORE_HEIGHT_PX) // 2
    )

    mx = min(mx, max_mx)
    my = min(my, max_my)

    cx1 = x + mx
    cy1 = y + my

    cx2 = x + w - mx
    cy2 = y + h - my

    if cx2 <= cx1:
        cx1 = x
        cx2 = x + w

    if cy2 <= cy1:
        cy1 = y
        cy2 = y + h

    return (
        int(cx1),
        int(cy1),
        int(cx2),
        int(cy2)
    )


def dominant_core_color(arr, bar):
    """
    Retorna:
    - cor RGB mais frequente no núcleo
    - support_exact
    - support_near
    - pixels cromáticos avaliados
    - bbox do núcleo
    """
    cx1, cy1, cx2, cy2 = (
        core_box(bar)
    )

    region = arr[
        cy1:cy2,
        cx1:cx2
    ]

    if region.size == 0:
        return None

    sat = saturation_map(region)

    chromatic = (
        sat >= CORE_MIN_SATURATION
    )

    pixels = region[chromatic]

    if len(pixels) == 0:
        return None

    packed = (
        (
            pixels[:, 0].astype(np.uint32)
            << 16
        )
        | (
            pixels[:, 1].astype(np.uint32)
            << 8
        )
        | pixels[:, 2].astype(np.uint32)
    )

    vals, counts = np.unique(
        packed,
        return_counts=True
    )

    best_idx = int(
        np.argmax(counts)
    )

    best = int(
        vals[best_idx]
    )

    dominant_rgb = (
        (best >> 16) & 255,
        (best >> 8) & 255,
        best & 255
    )

    exact_count = int(
        counts[best_idx]
    )

    total = int(
        len(pixels)
    )

    support_exact = (
        exact_count / total
        if total else 0.0
    )

    target = np.array(
        dominant_rgb,
        dtype=np.float32
    ).reshape(1, 3)

    distances = np.sqrt(
        (
            (
                pixels.astype(np.float32)
                - target
            ) ** 2
        ).sum(axis=1)
    )

    near_count = int(
        np.sum(
            distances
            <= CORE_RGB_NEAR_RADIUS
        )
    )

    support_near = (
        near_count / total
        if total else 0.0
    )

    nearest_name, nearest_hex, de = (
        nearest_f(dominant_rgb)
    )

    return {
        "core_x1": cx1,
        "core_y1": cy1,
        "core_x2": cx2,
        "core_y2": cy2,

        "dominant_rgb": dominant_rgb,
        "dominant_hex": rgb_to_hex(
            dominant_rgb
        ),

        "chromatic_core_pixels": total,
        "dominant_exact_pixels": exact_count,
        "support_exact": support_exact,

        "dominant_near_pixels": near_count,
        "support_near": support_near,

        "nearest_F_name": nearest_name,
        "nearest_F_hex": nearest_hex,
        "deltaE00": de,
    }


def measure_bars_v5(bars, arr):
    measured = []

    for bar in bars:
        color = dominant_core_color(
            arr,
            bar
        )

        if color is None:
            continue

        measured.append({
            **bar,
            **color,
        })

    return measured


# ============================================================
# CORES-BASE DISTINTAS
# ============================================================

def collapse_distinct_colors(bars):
    groups = defaultdict(
        lambda: {
            "support_bars": 0,
            "core_pixels": 0,
            "support_exact_values": [],
            "support_near_values": [],
            "nearest_F_name": "",
            "nearest_F_hex": "",
            "deltaE00": None,
        }
    )

    for b in bars:
        g = groups[
            b["dominant_hex"]
        ]

        g["support_bars"] += 1

        g["core_pixels"] += (
            b["chromatic_core_pixels"]
        )

        g["support_exact_values"].append(
            b["support_exact"]
        )

        g["support_near_values"].append(
            b["support_near"]
        )

        g["nearest_F_name"] = (
            b["nearest_F_name"]
        )

        g["nearest_F_hex"] = (
            b["nearest_F_hex"]
        )

        g["deltaE00"] = (
            b["deltaE00"]
        )

    result = []

    for hx, g in groups.items():
        result.append({
            "observed_hex": hx,
            "support_bars": (
                g["support_bars"]
            ),
            "support_core_pixels": (
                g["core_pixels"]
            ),
            "mean_support_exact": float(
                np.mean(
                    g[
                        "support_exact_values"
                    ]
                )
            ),
            "min_support_exact": float(
                np.min(
                    g[
                        "support_exact_values"
                    ]
                )
            ),
            "mean_support_near": float(
                np.mean(
                    g[
                        "support_near_values"
                    ]
                )
            ),
            "min_support_near": float(
                np.min(
                    g[
                        "support_near_values"
                    ]
                )
            ),
            "nearest_F_name": (
                g["nearest_F_name"]
            ),
            "nearest_F_hex": (
                g["nearest_F_hex"]
            ),
            "deltaE00": (
                g["deltaE00"]
            ),
        })

    result.sort(
        key=lambda x:
        x["support_core_pixels"],
        reverse=True
    )

    return result


# ============================================================
# SELEÇÃO DAS 40 IMAGENS
# ============================================================

def select_validation_images():
    rng = random.Random(
        RANDOM_SEED
    )

    groups = defaultdict(list)

    for p in sorted(
        ROOT_DIR.glob("*.png")
    ):
        m = NAME_RE.match(
            p.name
        )

        if not m:
            continue

        profile = (
            m.group(1).upper()
        )

        try:
            with Image.open(p) as img:
                canvas = (
                    "OK"
                    if img.size == (1200, 800)
                    else "NAO_OK"
                )
        except Exception:
            continue

        groups[
            (profile, canvas)
        ].append(p)

    selected = []

    for profile in ("BI", "BC"):
        for canvas in (
            "OK",
            "NAO_OK"
        ):
            pool = groups[
                (profile, canvas)
            ]

            k = min(
                N_PER_PROFILE_CANVAS_CLASS,
                len(pool)
            )

            for p in rng.sample(
                pool,
                k
            ):
                selected.append((
                    profile,
                    canvas,
                    p
                ))

    return selected


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(
    src,
    bars,
    baseline,
    out
):
    img = Image.open(
        src
    ).convert("RGB")

    draw = ImageDraw.Draw(
        img
    )

    font = ImageFont.load_default()

    if baseline is not None:
        draw.line(
            [
                (0, baseline),
                (
                    img.width - 1,
                    baseline
                )
            ],
            fill=(0, 150, 0),
            width=max(
                1,
                img.width // 700
            )
        )

    for rank, b in enumerate(
        bars,
        1
    ):
        x = b["x"]
        y = b["y"]
        w = b["w"]
        h = b["h"]

        # barra
        draw.rectangle(
            [
                x,
                y,
                x + w,
                y + h
            ],
            outline=(255, 0, 255),
            width=max(
                2,
                img.width // 500
            )
        )

        # núcleo
        draw.rectangle(
            [
                b["core_x1"],
                b["core_y1"],
                b["core_x2"],
                b["core_y2"]
            ],
            outline=(0, 200, 220),
            width=max(
                1,
                img.width // 700
            )
        )

        txt = (
            f"{rank} "
            f"{b['dominant_hex']} "
            f"sup={b['support_exact']:.2f} "
            f"near={b['support_near']:.2f} "
            f"dE={b['deltaE00']:.1f}"
        )

        draw.text(
            (
                x,
                max(2, y - 13)
            ),
            txt,
            fill=(0, 0, 0),
            font=font
        )

    img.save(out)


# ============================================================
# CONTACT SHEET
# ============================================================

def make_contact_sheet(
    items,
    out
):
    rows_n = math.ceil(
        len(items) / COLS
    )

    sheet = Image.new(
        "RGB",
        (
            COLS * PANEL_W,
            rows_n * PANEL_H
        ),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(
        sheet
    )

    font = ImageFont.load_default()

    for idx, item in enumerate(items):
        col = idx % COLS
        row = idx // COLS

        x0 = col * PANEL_W
        y0 = row * PANEL_H

        draw.rectangle(
            [
                x0,
                y0,
                x0 + PANEL_W - 1,
                y0 + PANEL_H - 1
            ],
            fill=(248, 248, 248),
            outline=(180, 180, 180)
        )

        img = Image.open(
            item["overlay"]
        ).convert("RGB")

        img.thumbnail(
            (IMG_W, IMG_H)
        )

        sheet.paste(
            img,
            (
                x0 + 15,
                y0 + 58
                + (
                    IMG_H
                    - img.height
                ) // 2
            )
        )

        title = (
            f"{item['profile']} | "
            f"CANVAS_{item['canvas']} | "
            f"{item['filename']} | "
            f"bars={item['n_bars']} | "
            f"colors={item['n_colors']}"
        )

        draw.text(
            (x0 + 8, y0 + 10),
            title,
            fill=(0, 0, 0),
            font=font
        )

        subtitle = (
            f"Dmax={item['Dmax']:.2f} | "
            f"support_min={item['support_min']:.2f}"
            if item["Dmax"] is not None
            and item["support_min"] is not None
            else
            "Dmax/support=NA"
        )

        draw.text(
            (x0 + 8, y0 + 30),
            subtitle,
            fill=(0, 0, 0),
            font=font
        )

    sheet.save(out)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print(
        "VALIDAÇÃO B-COR V5 "
        "- NÚCLEO + COR DOMINANTE"
    )
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    selected = (
        select_validation_images()
    )

    image_rows = []
    color_rows = []
    sheet_items = []

    for idx, (
        profile,
        canvas,
        path
    ) in enumerate(selected, 1):

        with Image.open(path) as img:
            (
                geometry,
                baseline,
                arr,
                n_initial
            ) = detect_bar_geometry(
                img.convert("RGB")
            )

        bars = measure_bars_v5(
            geometry,
            arr
        )

        colors = (
            collapse_distinct_colors(
                bars
            )
        )

        dvals = [
            c["deltaE00"]
            for c in colors
        ]

        Dmean = (
            float(np.mean(dvals))
            if dvals else None
        )

        Dmax = (
            float(np.max(dvals))
            if dvals else None
        )

        support_exact_all = [
            b["support_exact"]
            for b in bars
        ]

        support_near_all = [
            b["support_near"]
            for b in bars
        ]

        support_exact_mean = (
            float(
                np.mean(
                    support_exact_all
                )
            )
            if support_exact_all
            else None
        )

        support_exact_min = (
            float(
                np.min(
                    support_exact_all
                )
            )
            if support_exact_all
            else None
        )

        support_near_mean = (
            float(
                np.mean(
                    support_near_all
                )
            )
            if support_near_all
            else None
        )

        support_near_min = (
            float(
                np.min(
                    support_near_all
                )
            )
            if support_near_all
            else None
        )

        overlay = (
            OVERLAY_DIR
            / f"{path.stem}__v5.png"
        )

        make_overlay(
            path,
            bars,
            baseline,
            overlay
        )

        image_rows.append({
            "profile": profile,
            "canvas_class": canvas,
            "filename": path.name,

            "n_initial_candidates": (
                n_initial
            ),

            "n_bars_detected": (
                len(bars)
            ),

            "n_distinct_colors": (
                len(colors)
            ),

            "Dmean_distinct_colors": (
                round(Dmean, 6)
                if Dmean is not None
                else ""
            ),

            "Dmax_distinct_colors": (
                round(Dmax, 6)
                if Dmax is not None
                else ""
            ),

            "support_exact_mean_bars": (
                round(
                    support_exact_mean,
                    6
                )
                if support_exact_mean
                is not None
                else ""
            ),

            "support_exact_min_bars": (
                round(
                    support_exact_min,
                    6
                )
                if support_exact_min
                is not None
                else ""
            ),

            "support_near_mean_bars": (
                round(
                    support_near_mean,
                    6
                )
                if support_near_mean
                is not None
                else ""
            ),

            "support_near_min_bars": (
                round(
                    support_near_min,
                    6
                )
                if support_near_min
                is not None
                else ""
            ),

            "baseline_px": (
                baseline
                if baseline is not None
                else ""
            ),
        })

        for rank, c in enumerate(
            colors,
            1
        ):
            color_rows.append({
                "profile": profile,
                "canvas_class": canvas,
                "filename": path.name,
                "color_rank": rank,
                "observed_hex": (
                    c["observed_hex"]
                ),
                "support_bars": (
                    c["support_bars"]
                ),
                "support_core_pixels": (
                    c["support_core_pixels"]
                ),
                "mean_support_exact": (
                    round(
                        c[
                            "mean_support_exact"
                        ],
                        6
                    )
                ),
                "min_support_exact": (
                    round(
                        c[
                            "min_support_exact"
                        ],
                        6
                    )
                ),
                "mean_support_near": (
                    round(
                        c[
                            "mean_support_near"
                        ],
                        6
                    )
                ),
                "min_support_near": (
                    round(
                        c[
                            "min_support_near"
                        ],
                        6
                    )
                ),
                "nearest_F_name": (
                    c["nearest_F_name"]
                ),
                "nearest_F_hex": (
                    c["nearest_F_hex"]
                ),
                "deltaE00_to_nearest_F": (
                    round(
                        c["deltaE00"],
                        6
                    )
                ),
            })

        sheet_items.append({
            "profile": profile,
            "canvas": canvas,
            "filename": path.name,
            "overlay": overlay,
            "n_bars": len(bars),
            "n_colors": len(colors),
            "Dmax": Dmax,
            "support_min": (
                support_exact_min
            ),
        })

        print(
            f"{idx:02d}/{len(selected):02d} | "
            f"{path.name} | "
            f"bars={len(bars)} | "
            f"colors={len(colors)} | "
            f"Dmax="
            f"{Dmax if Dmax is not None else 'NA'} | "
            f"support_exact_mean="
            f"{support_exact_mean if support_exact_mean is not None else 'NA'}"
        )

    # --------------------------------------------------------
    # CSV POR IMAGEM
    # --------------------------------------------------------
    image_csv = (
        OUTPUT_DIR
        / "bar_core_validation_v5.csv"
    )

    image_fields = [
        "profile",
        "canvas_class",
        "filename",
        "n_initial_candidates",
        "n_bars_detected",
        "n_distinct_colors",
        "Dmean_distinct_colors",
        "Dmax_distinct_colors",
        "support_exact_mean_bars",
        "support_exact_min_bars",
        "support_near_mean_bars",
        "support_near_min_bars",
        "baseline_px",
    ]

    with image_csv.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=image_fields
        )
        writer.writeheader()
        writer.writerows(
            image_rows
        )

    # --------------------------------------------------------
    # CSV POR COR-BASE
    # --------------------------------------------------------
    color_csv = (
        OUTPUT_DIR
        / "bar_core_colors_v5.csv"
    )

    color_fields = [
        "profile",
        "canvas_class",
        "filename",
        "color_rank",
        "observed_hex",
        "support_bars",
        "support_core_pixels",
        "mean_support_exact",
        "min_support_exact",
        "mean_support_near",
        "min_support_near",
        "nearest_F_name",
        "nearest_F_hex",
        "deltaE00_to_nearest_F",
    ]

    with color_csv.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=color_fields
        )
        writer.writeheader()
        writer.writerows(
            color_rows
        )

    # --------------------------------------------------------
    # PRANCHA
    # --------------------------------------------------------
    contact = (
        OUTPUT_DIR
        / "bar_core_contact_sheet_v5.png"
    )

    make_contact_sheet(
        sheet_items,
        contact
    )

    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------
    no_color = sum(
        r["n_distinct_colors"] == 0
        for r in image_rows
    )

    supports = [
        float(
            r["support_exact_mean_bars"]
        )
        for r in image_rows
        if str(
            r["support_exact_mean_bars"]
        ).strip() != ""
    ]

    near_supports = [
        float(
            r["support_near_mean_bars"]
        )
        for r in image_rows
        if str(
            r["support_near_mean_bars"]
        ).strip() != ""
    ]

    summary = [
        "VALIDAÇÃO B-COR V5",
        "NÚCLEO + COR DOMINANTE",
        "=" * 72,

        f"Imagens analisadas: "
        f"{len(image_rows)}",

        f"Imagens sem cor-base detectada: "
        f"{no_color}",

        "",

        "NOVO TESTE:",
        "- núcleo central da barra;",
        "- exclusão das bordas;",
        "- cor RGB de maior ocorrência;",
        "- percentual exato de suporte;",
        "- percentual tolerante de suporte;",
        "- CIEDE2000 após extração.",

        "",

        "SUPORTE GLOBAL:",

        (
            f"support_exact_mean global: "
            f"{np.mean(supports):.4f}"
            if supports else
            "support_exact_mean global: NA"
        ),

        (
            f"support_exact_mediana global: "
            f"{np.median(supports):.4f}"
            if supports else
            "support_exact_mediana global: NA"
        ),

        (
            f"support_near_mean global: "
            f"{np.mean(near_supports):.4f}"
            if near_supports else
            "support_near_mean global: NA"
        ),

        (
            f"support_near_mediana global: "
            f"{np.median(near_supports):.4f}"
            if near_supports else
            "support_near_mediana global: NA"
        ),

        "",

        "Nenhum limiar de conformidade foi aplicado.",

        "",

        f"Prancha: {contact}",
        f"CSV imagem: {image_csv}",
        f"CSV cores: {color_csv}",
    ]

    summary_path = (
        OUTPUT_DIR
        / "bar_core_summary_v5.txt"
    )

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("CONCLUÍDO")
    print("=" * 80)

    print(
        f"Prancha:\n{contact}"
    )

    print(
        f"\nResumo:\n{summary_path}"
    )

    print(
        "\nNenhuma imagem original foi alterada."
    )


if __name__ == "__main__":
    main()
