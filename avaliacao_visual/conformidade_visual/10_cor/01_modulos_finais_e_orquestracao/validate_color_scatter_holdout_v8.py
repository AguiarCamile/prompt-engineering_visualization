# -*- coding: utf-8 -*-
"""
validate_color_scatter_v7_markercore.py

S-Cor HOLDOUT V8 — PARÂMETROS V7 CONGELADOS
================================

Objetivo
--------
Extrair a cor dos scatterplots usando NÚCLEOS ESPESSOS DAS MARCAS como fonte
primária da cor. Rótulos, texto da legenda, amostras de legenda, grade,
eixos e demais elementos não devem determinar as famílias cromáticas.

Fluxo
-----
1. estima a área de plotagem;
2. procura componentes cromáticos compactos dentro dessa área;
3. mantém somente componentes geometricamente compatíveis com marcadores;
4. sinaliza possíveis marcadores de legenda para auditoria, sem removê-los;
5. mede a cor de cada marcador usando apenas pixels cromáticos do próprio
   componente, reduzindo a influência de grade cinza;
6. agrupa os marcadores por família circular de matiz;
7. classifica cada família como:
       SOLID_CANDIDATE
       BRIGHTNESS_SCALE_CANDIDATE
       SATURATION_SCALE_CANDIDATE
       MIXED_SCALE_CANDIDATE
       INSUFFICIENT_POINTS
8. para famílias SOLID, registra a cor medóide e sua distância CIEDE2000
   para a cor de F mais próxima;
9. para famílias em escala, preserva os valores ponto a ponto. A métrica
   final de conformidade da escala NÃO é definida nesta versão.

Princípios
----------
- F não participa da descoberta das famílias.
- SI=1 / SC=2 é somente diagnóstico.
- Rótulos coloridos não são usados para descobrir a cor-base.
- A heurística de legenda é apenas diagnóstica e não exclui marcadores.
- Linhas de grade cinza são excluídas da medição do ponto por saturação.

Amostra
-------
40 imagens:
- 10 SI canvas 1200x800
- 10 SI canvas diferente
- 10 SC canvas 1200x800
- 10 SC canvas diferente

Saída
-----
C:/Users/Labvis/Downloads/imagens3120/imagens/_color_scatter_holdout_v8/

    scatter_holdout_image_v8.csv
    scatter_holdout_family_v8.csv
    scatter_holdout_points_v8.csv
    scatter_holdout_contact_sheet_v8.png
    scatter_holdout_summary_v8.txt
    overlays/*.png

Overlay
-------
verde  = área de plotagem estimada
ciano  = marcador de dado aceito
amarelo = candidato apenas sinalizado como possível legenda
caixa maior na cor da família = extensão dos marcadores daquela família

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
from skimage.color import rgb2hsv, rgb2lab, deltaE_ciede2000


# ============================================================
# CAMINHOS E AMOSTRAGEM
# ============================================================

ROOT_DIR = Path("C:/Users/Labvis/Downloads/imagens3120/imagens")
OUTPUT_DIR = ROOT_DIR / "_color_scatter_holdout_v8"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

# Conjunto de calibração V7: deve existir para garantir exclusão.
CALIBRATION_CSV = (
    ROOT_DIR
    / "_color_scatter_markercore_v7"
    / "scatter_markercore_image_v7.csv"
)

# Nova semente usada somente no holdout.
RANDOM_SEED = 20260828

# 20 por estrato:
# SI/SC x canvas OK/NAO_OK = 4 estratos -> 80 imagens.
N_PER_PROFILE_CANVAS_CLASS = 20

NAME_RE = re.compile(r"^(SI|SC)_(\d{3})_R(\d{2})\.png$", re.I)


# ============================================================
# PALETA F — APENAS COMPARAÇÃO POSTERIOR
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
# ÁREA DE PLOTAGEM
# ============================================================

# V6: fallback mais amplo. A V5 mostrou que marcadores legítimos nos
# extremos dos eixos podem ficar ligeiramente fora do fallback anterior.
FALLBACK_LEFT = 0.03
FALLBACK_RIGHT = 0.97
FALLBACK_TOP = 0.03
FALLBACK_BOTTOM = 0.97

# Marcadores centrados exatamente nos limites dos eixos podem ultrapassar
# visualmente o retângulo detectado. Por isso o ROI recebe pequena expansão.
ROI_PAD_X_FRAC = 0.012
ROI_PAD_Y_FRAC = 0.018

# Hough
HOUGH_THRESHOLD = 55
MIN_HORIZONTAL_LINE_FRAC = 0.28
MIN_VERTICAL_LINE_FRAC = 0.18


# ============================================================
# DETECÇÃO GEOMÉTRICA DOS MARCADORES
# ============================================================

MARKER_MIN_SATURATION = 0.10
MARKER_MAX_VALUE = 0.998

# Área proporcional ao canvas.
MIN_MARKER_AREA_FRAC = 0.000010
MIN_MARKER_AREA_ABS = 6
MAX_MARKER_AREA_FRAC = 0.0025

MIN_ASPECT_RATIO = 0.32
MAX_ASPECT_RATIO = 3.10
MIN_FILL_RATIO = 0.22
MIN_CIRCULARITY = 0.22

# Componentes maiores que isso, em relação ao menor lado da imagem,
# deixam de ser candidatos plausíveis a marcadores.
MAX_MARKER_DIM_FRAC = 0.055

# V5: em vez de usar o componente cromático inteiro, procura regiões
# espessas ("núcleos") no mapa de distância. Letras e rótulos tendem a ser
# traços finos; marcadores preenchidos possuem um núcleo espesso.
MIN_MARKER_CORE_RADIUS_PX = 2.40

# Raio usado para reconstruir a região do marcador ao redor do núcleo.
MARKER_EXTRACT_RADIUS_FACTOR = 1.35
MARKER_EXTRACT_RADIUS_PAD = 1.0

# Dois núcleos muito próximos podem ser metades do mesmo marcador quando
# uma linha de grade atravessa o ponto. A deduplicação usa distância entre
# centros e matiz representativo.
MARKER_SEED_MERGE_FACTOR = 0.72
MARKER_SEED_MERGE_HUE_DEG = 20.0

# Mantido apenas como referência histórica; a V5 não fecha o mapa cromático
# antes do distance transform para evitar unir marcador e rótulo.
MARKER_CLOSE_KERNEL = 1


# ============================================================
# MEDIÇÃO DA COR DO MARCADOR
# ============================================================

POINT_MIN_SATURATION = 0.055
POINT_MAX_VALUE = 0.999

# Usa os pixels mais saturados do marcador para encontrar o núcleo cromático.
POINT_CORE_SAT_QUANTILE = 0.55
MIN_POINT_COLOR_PIXELS = 3


# ============================================================
# HEURÍSTICA DE LEGENDA
# ============================================================

# Procura texto escuro imediatamente à direita do marcador.
LEGEND_DARK_MAX_VALUE = 0.50
LEGEND_DARK_MAX_SATURATION = 0.25

# Janela à direita, proporcional ao ROI.
LEGEND_TEXT_WINDOW_W_FRAC = 0.22
LEGEND_TEXT_WINDOW_H_FACTOR = 2.5
MIN_DARK_TEXT_PIXELS = 8

# Legendas costumam ficar em regiões periféricas.
LEGEND_TOP_FRAC = 0.38
LEGEND_SIDE_FRAC = 0.42

# Só remove se existir ao menos outro marcador cromaticamente semelhante
# fora da região suspeita.
LEGEND_HUE_MATCH_DEG = 14.0


# ============================================================
# AGRUPAMENTO EM FAMÍLIAS
# ============================================================

# Distância circular máxima entre marcadores de uma mesma família.
# Mantida abaixo da separação típica vermelho-laranja da paleta F.
FAMILY_HUE_LINK_DEG = 25.0

# Família precisa ter pelo menos um marcador.
MIN_MARKERS_PER_FAMILY = 1


# ============================================================
# CLASSIFICAÇÃO SÓLIDA / ESCALA
# ============================================================

SOLID_POINT_DE00_RADIUS = 4.0
MIN_SOLID_POINT_SUPPORT = 0.85

MIN_SCALE_RANGE = 7.0
DOMINANCE_RATIO = 1.50


# ============================================================
# PRANCHA
# ============================================================

PANEL_W = 590
PANEL_H = 435
IMG_W = 540
IMG_H = 325
COLS = 4


# ============================================================
# FUNÇÕES DE COR
# ============================================================

def hex_to_rgb(v):
    v = v.lstrip("#")
    return tuple(int(v[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    vals = [max(0, min(255, int(round(float(v))))) for v in rgb]
    return "#{:02X}{:02X}{:02X}".format(*vals)


def rgb_to_lab_one(rgb):
    arr = np.array(
        [[[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]]],
        dtype=float
    )
    return rgb2lab(arr)[0, 0]


def rgb_to_hsv_one(rgb):
    arr = np.array(
        [[[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]]],
        dtype=float
    )
    return rgb2hsv(arr)[0, 0]


F_LAB = {
    name: rgb_to_lab_one(hex_to_rgb(hx))
    for name, hx in F_COLORS.items()
}


def de00_lab(a, b):
    return float(
        deltaE_ciede2000(
            np.array([[a]], dtype=float),
            np.array([[b]], dtype=float)
        )[0, 0]
    )


def nearest_f(rgb):
    lab = rgb_to_lab_one(rgb)

    ds = {
        name: de00_lab(lab, target)
        for name, target in F_LAB.items()
    }

    nearest = min(ds, key=ds.get)

    return {
        "nearest_F_name": nearest,
        "nearest_F_hex": F_COLORS[nearest],
        "deltaE00_to_nearest_F": ds[nearest],
    }


def circular_deg_distance(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def circular_weighted_mean_deg(values, weights=None):
    vals = np.radians(np.asarray(values, dtype=float))

    if weights is None:
        weights = np.ones(len(vals), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    x = np.sum(np.cos(vals) * weights)
    y = np.sum(np.sin(vals) * weights)

    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float(np.degrees(vals[0]) % 360.0)

    return float(np.degrees(np.arctan2(y, x)) % 360.0)


# ============================================================
# ÁREA DE PLOTAGEM
# ============================================================

def pad_plot_roi(roi, w, h):
    """Expande o ROI para incluir marcadores centrados nos limites dos eixos."""
    x1, y1, x2, y2 = roi

    px = max(2, int(round(ROI_PAD_X_FRAC * w)))
    py = max(2, int(round(ROI_PAD_Y_FRAC * h)))

    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w - 1, x2 + px),
        min(h - 1, y2 + py),
    )


def infer_plot_roi(arr_u8):
    """
    Estima o retângulo da área de plotagem a partir de linhas longas
    horizontais/verticais. Se a evidência for insuficiente, usa fallback
    proporcional.

    Retorna:
        (x1, y1, x2, y2), metodo, n_horiz, n_vert
    """
    h, w, _ = arr_u8.shape

    gray = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 120)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=int(min(w, h) * 0.15),
        maxLineGap=8
    )

    horizontal_y = []
    vertical_x = []

    if lines is not None:
        # OpenCV pode retornar (N,1,4) ou (N,4), dependendo da versão.
        segments = np.asarray(lines).reshape(-1, 4)

        for line in segments:
            x1, y1, x2, y2 = map(int, line.tolist())

            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if (
                dy <= 3
                and dx >= MIN_HORIZONTAL_LINE_FRAC * w
            ):
                y = int(round((y1 + y2) / 2))
                if int(0.03*h) <= y <= int(0.97*h):
                    horizontal_y.append(y)

            if (
                dx <= 3
                and dy >= MIN_VERTICAL_LINE_FRAC * h
            ):
                x = int(round((x1 + x2) / 2))
                if int(0.03*w) <= x <= int(0.97*w):
                    vertical_x.append(x)

    # Agrupa posições muito próximas.
    def collapse(values, tol=5):
        if not values:
            return []

        vals = sorted(values)
        groups = [[vals[0]]]

        for v in vals[1:]:
            if abs(v - np.median(groups[-1])) <= tol:
                groups[-1].append(v)
            else:
                groups.append([v])

        return [
            int(round(float(np.median(g))))
            for g in groups
        ]

    ys = collapse(horizontal_y)
    xs = collapse(vertical_x)

    method = "FALLBACK"

    if len(xs) >= 2 and len(ys) >= 2:
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)

        # Aceita apenas retângulo plausível.
        if (
            (x2 - x1) >= 0.45*w
            and (y2 - y1) >= 0.35*h
        ):
            method = "HOUGH"
            roi = pad_plot_roi(
                (x1, y1, x2, y2),
                w,
                h
            )
            return (
                roi,
                method,
                len(ys),
                len(xs)
            )

    # Fallback
    x1 = int(round(FALLBACK_LEFT * w))
    x2 = int(round(FALLBACK_RIGHT * w))
    y1 = int(round(FALLBACK_TOP * h))
    y2 = int(round(FALLBACK_BOTTOM * h))

    roi = pad_plot_roi(
        (x1, y1, x2, y2),
        w,
        h
    )

    return (
        roi,
        method,
        len(ys),
        len(xs)
    )


# ============================================================
# COMPONENTES / MARCADORES
# ============================================================

def component_circularity(component_mask):
    contours, _ = cv2.findContours(
        component_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if perimeter <= 0:
        return 0.0

    return float(
        4.0 * math.pi * area
        / (perimeter * perimeter)
    )


def measure_marker_color(arr_u8, hsv, component_mask):
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    valid = (
        component_mask
        & (sat >= POINT_MIN_SATURATION)
        & (val <= POINT_MAX_VALUE)
    )

    pixels = arr_u8[valid]
    sats = sat[valid]

    if len(pixels) < MIN_POINT_COLOR_PIXELS:
        return None

    # Núcleo cromático: pixels relativamente mais saturados.
    threshold = float(
        np.quantile(
            sats,
            POINT_CORE_SAT_QUANTILE
        )
    )

    core = pixels[
        sats >= threshold
    ]

    if len(core) < MIN_POINT_COLOR_PIXELS:
        core = pixels

    # Moda RGB do núcleo.
    packed = (
        (core[:, 0].astype(np.uint32) << 16)
        | (core[:, 1].astype(np.uint32) << 8)
        | core[:, 2].astype(np.uint32)
    )

    vals, counts = np.unique(
        packed,
        return_counts=True
    )

    best_idx = int(np.argmax(counts))
    best = int(vals[best_idx])

    rgb = (
        (best >> 16) & 255,
        (best >> 8) & 255,
        best & 255
    )

    exact_support = (
        int(counts[best_idx]) / len(core)
        if len(core) else 0.0
    )

    # Hue robusto da marca completa, ponderado pela saturação.
    hsv_pixels = hsv[valid]

    hue_deg = circular_weighted_mean_deg(
        hsv_pixels[:, 0] * 360.0,
        weights=np.maximum(
            hsv_pixels[:, 1] ** 2,
            1e-6
        )
    )

    lab = rgb_to_lab_one(rgb)
    hsv_rep = rgb_to_hsv_one(rgb)

    L, a, b = map(float, lab)
    chroma = math.sqrt(a*a + b*b)

    nearest = nearest_f(rgb)

    return {
        "observed_rgb": rgb,
        "observed_hex": rgb_to_hex(rgb),

        "valid_color_pixels": int(len(pixels)),
        "core_color_pixels": int(len(core)),
        "exact_color_support": float(exact_support),

        "marker_hue_deg": float(hue_deg),

        "lab_L": L,
        "lab_a": a,
        "lab_b": b,
        "lab_chroma": float(chroma),
        "lab_hue_deg": float(
            math.degrees(math.atan2(b, a)) % 360.0
        ),

        "hsv_hue_deg": float(hsv_rep[0] * 360.0),
        "hsv_saturation": float(hsv_rep[1]),
        "hsv_value": float(hsv_rep[2]),

        **nearest,
    }


def detect_markers(arr_u8, plot_roi):
    """
    V5 MARKER-CORE.

    Em vez de assumir que cada componente cromático corresponde a um ponto,
    utiliza a espessura local do mapa cromático:

        máscara cromática
            -> distance transform
            -> núcleos espessos
            -> centro provável do marcador
            -> região circular local
            -> cor do marcador

    Isso reduz fortemente a inclusão de letras/rótulos coloridos, pois seus
    traços tendem a ser finos mesmo quando possuem a mesma cor dos pontos.

    Também ajuda quando texto colorido está conectado ao marcador: o máximo
    do distance transform tende a permanecer no interior do ponto.
    """
    h, w, _ = arr_u8.shape

    hsv = rgb2hsv(
        arr_u8.astype(float) / 255.0
    )

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    x1, y1, x2, y2 = plot_roi

    roi_mask = np.zeros(
        (h, w),
        dtype=bool
    )

    roi_mask[
        max(0, y1):min(h, y2+1),
        max(0, x1):min(w, x2+1)
    ] = True

    chromatic = (
        roi_mask
        & (sat >= MARKER_MIN_SATURATION)
        & (val <= MARKER_MAX_VALUE)
    )

    mask_u8 = (
        chromatic.astype(np.uint8)
        * 255
    )

    # Espessura local da região cromática.
    dist = cv2.distanceTransform(
        mask_u8,
        cv2.DIST_L2,
        5
    )

    core_seed_mask = (
        dist >= MIN_MARKER_CORE_RADIUS_PX
    ).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        core_seed_mask,
        connectivity=8
    )

    raw_candidates = []

    for label_id in range(1, n):
        seed = labels == label_id

        if not np.any(seed):
            continue

        seed_dist = np.where(
            seed,
            dist,
            -1.0
        )

        flat_idx = int(
            np.argmax(seed_dist)
        )

        cy, cx = np.unravel_index(
            flat_idx,
            seed_dist.shape
        )

        core_radius = float(
            dist[cy, cx]
        )

        if core_radius < MIN_MARKER_CORE_RADIUS_PX:
            continue

        extraction_radius = float(
            max(
                3.0,
                MARKER_EXTRACT_RADIUS_FACTOR
                * core_radius
                + MARKER_EXTRACT_RADIUS_PAD
            )
        )

        yy, xx = np.ogrid[
            :h,
            :w
        ]

        local_circle = (
            (xx - cx) ** 2
            + (yy - cy) ** 2
            <= extraction_radius ** 2
        )

        marker_region = (
            chromatic
            & local_circle
        )

        region_area = int(
            marker_region.sum()
        )

        if region_area < MIN_MARKER_AREA_ABS:
            continue

        ys, xs = np.where(
            marker_region
        )

        if len(xs) == 0:
            continue

        bx1 = int(xs.min())
        bx2 = int(xs.max())
        by1 = int(ys.min())
        by2 = int(ys.max())

        ww = bx2 - bx1 + 1
        hh = by2 - by1 + 1

        if ww <= 0 or hh <= 0:
            continue

        # A região reconstruída deve continuar compacta.
        aspect = ww / hh
        fill = (
            region_area / (ww * hh)
            if ww * hh else 0.0
        )

        if not (
            MIN_ASPECT_RATIO
            <= aspect
            <= MAX_ASPECT_RATIO
        ):
            continue

        if fill < MIN_FILL_RATIO:
            continue

        measured = measure_marker_color(
            arr_u8,
            hsv,
            marker_region
        )

        if measured is None:
            continue

        raw_candidates.append({
            "x": bx1,
            "y": by1,
            "w": int(ww),
            "h": int(hh),
            "area": region_area,

            "center_x": float(cx),
            "center_y": float(cy),

            "aspect_ratio": float(aspect),
            "fill_ratio": float(fill),

            # Para compatibilidade com as saídas anteriores.
            "circularity": float(
                min(
                    1.0,
                    region_area
                    / max(
                        1.0,
                        math.pi
                        * extraction_radius
                        * extraction_radius
                    )
                )
            ),

            "core_radius_px": core_radius,
            "extract_radius_px": extraction_radius,

            **measured,
        })

    # --------------------------------------------------------
    # Deduplica núcleos gerados pelo mesmo ponto.
    # --------------------------------------------------------
    raw_candidates.sort(
        key=lambda m: (
            m["core_radius_px"],
            m["valid_color_pixels"]
        ),
        reverse=True
    )

    kept = []

    for marker in raw_candidates:
        duplicate = False

        for old in kept:
            center_distance = math.hypot(
                marker["center_x"]
                - old["center_x"],
                marker["center_y"]
                - old["center_y"]
            )

            merge_distance = (
                MARKER_SEED_MERGE_FACTOR
                * (
                    marker["extract_radius_px"]
                    + old["extract_radius_px"]
                )
            )

            hue_distance = circular_deg_distance(
                marker["hsv_hue_deg"],
                old["hsv_hue_deg"]
            )

            if (
                center_distance
                <= merge_distance
                and hue_distance
                <= MARKER_SEED_MERGE_HUE_DEG
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(marker)

    kept.sort(
        key=lambda m: (
            m["center_x"],
            m["center_y"]
        )
    )

    return kept, hsv


# ============================================================
# LEGENDA
# ============================================================

def is_near_plot_periphery(marker, plot_roi):
    x1, y1, x2, y2 = plot_roi

    rw = max(1, x2 - x1)
    rh = max(1, y2 - y1)

    rx = (
        marker["center_x"] - x1
    ) / rw

    ry = (
        marker["center_y"] - y1
    ) / rh

    top = ry <= LEGEND_TOP_FRAC
    side = (
        rx <= LEGEND_SIDE_FRAC
        or rx >= (1.0 - LEGEND_SIDE_FRAC)
    )

    return top and side


def has_dark_text_to_right(arr_u8, hsv, marker, plot_roi):
    h, w, _ = arr_u8.shape

    x1, y1, x2, y2 = plot_roi
    roi_w = max(1, x2 - x1)

    mx2 = marker["x"] + marker["w"]

    tx1 = int(min(w-1, mx2 + 2))
    tx2 = int(min(
        w,
        tx1 + max(
            18,
            int(round(
                LEGEND_TEXT_WINDOW_W_FRAC
                * roi_w
            ))
        )
    ))

    half_h = int(round(
        LEGEND_TEXT_WINDOW_H_FACTOR
        * max(
            3,
            marker["h"]
        )
        / 2
    ))

    cy = int(round(
        marker["center_y"]
    ))

    ty1 = max(0, cy - half_h)
    ty2 = min(h, cy + half_h + 1)

    if tx2 <= tx1 or ty2 <= ty1:
        return False, 0

    sat = hsv[
        ty1:ty2,
        tx1:tx2,
        1
    ]

    val = hsv[
        ty1:ty2,
        tx1:tx2,
        2
    ]

    dark_text = (
        (val <= LEGEND_DARK_MAX_VALUE)
        & (sat <= LEGEND_DARK_MAX_SATURATION)
    )

    count = int(
        dark_text.sum()
    )

    return (
        count >= MIN_DARK_TEXT_PIXELS,
        count
    )


def flag_probable_legend_markers(
    arr_u8,
    hsv,
    markers,
    plot_roi
):
    """
    V7: a heurística de legenda NÃO remove marcadores da análise cromática.

    Ela apenas sinaliza candidatos para auditoria porque a V6 mostrou que
    rótulos junto a pontos podem satisfazer a mesma heurística usada para
    legenda, eliminando marcas de dados verdadeiras.

    Retorna:
        markers_all        -> todos os marcadores preservados
        suspected_legend   -> subconjunto apenas sinalizado
    """
    if not markers:
        return [], []

    suspicious = []

    for idx, marker in enumerate(markers):
        peripheral = is_near_plot_periphery(
            marker,
            plot_roi
        )

        text_ok, text_pixels = has_dark_text_to_right(
            arr_u8,
            hsv,
            marker,
            plot_roi
        )

        suspicious.append({
            "idx": idx,
            "peripheral": peripheral,
            "text_ok": text_ok,
            "text_pixels": text_pixels,
        })

    remove_ids = set()

    for info in suspicious:
        if not (
            info["peripheral"]
            and info["text_ok"]
        ):
            continue

        marker = markers[
            info["idx"]
        ]

        has_same_hue_elsewhere = any(
            j != info["idx"]
            and circular_deg_distance(
                marker["hsv_hue_deg"],
                other["hsv_hue_deg"]
            ) <= LEGEND_HUE_MATCH_DEG
            and not (
                suspicious[j]["peripheral"]
                and suspicious[j]["text_ok"]
            )
            for j, other in enumerate(markers)
        )

        if has_same_hue_elsewhere:
            remove_ids.add(
                info["idx"]
            )

    suspected = [
        marker
        for i, marker in enumerate(markers)
        if i in remove_ids
    ]

    # Nenhum marcador é excluído nesta etapa.
    return list(markers), suspected


# ============================================================
# AGRUPAMENTO MARKER-FIRST
# ============================================================

def group_markers_by_hue(markers):
    """
    União por conectividade circular de hue.
    """
    if not markers:
        return []

    n = len(markers)

    adjacency = [
        set()
        for _ in range(n)
    ]

    for i in range(n):
        for j in range(i+1, n):
            d = circular_deg_distance(
                markers[i]["hsv_hue_deg"],
                markers[j]["hsv_hue_deg"]
            )

            if d <= FAMILY_HUE_LINK_DEG:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    families = []

    for i in range(n):
        if i in visited:
            continue

        stack = [i]
        group = []

        while stack:
            k = stack.pop()

            if k in visited:
                continue

            visited.add(k)
            group.append(k)

            stack.extend(
                adjacency[k] - visited
            )

        pts = [
            markers[k]
            for k in group
        ]

        if len(pts) < MIN_MARKERS_PER_FAMILY:
            continue

        hue = circular_weighted_mean_deg(
            [
                p["hsv_hue_deg"]
                for p in pts
            ],
            weights=[
                max(
                    1,
                    p["valid_color_pixels"]
                )
                for p in pts
            ]
        )

        families.append({
            "family_hue_deg": hue,
            "points": sorted(
                pts,
                key=lambda p: (
                    p["center_x"],
                    p["center_y"]
                )
            ),
        })

    families.sort(
        key=lambda f:
        f["family_hue_deg"]
    )

    return families


# ============================================================
# CLASSIFICAÇÃO SÓLIDA / ESCALA
# ============================================================

def point_lab(p):
    return np.array(
        [
            p["lab_L"],
            p["lab_a"],
            p["lab_b"]
        ],
        dtype=float
    )


def robust_range(values):
    x = np.asarray(
        values,
        dtype=float
    )

    if len(x) == 0:
        return 0.0

    if len(x) <= 3:
        return float(
            x.max() - x.min()
        )

    return float(
        np.quantile(x, 0.90)
        - np.quantile(x, 0.10)
    )


def family_medoid(points):
    labs = [
        point_lab(p)
        for p in points
    ]

    n = len(labs)

    if n == 1:
        return 0, np.zeros(
            (1, 1),
            dtype=float
        )

    matrix = np.zeros(
        (n, n),
        dtype=float
    )

    for i in range(n):
        for j in range(i+1, n):
            de = de00_lab(
                labs[i],
                labs[j]
            )

            matrix[i, j] = de
            matrix[j, i] = de

    medoid_idx = int(
        np.argmin(
            matrix.mean(axis=1)
        )
    )

    return medoid_idx, matrix


def classify_family(points):
    n = len(points)

    medoid_idx, matrix = family_medoid(
        points
    )

    medoid = points[
        medoid_idx
    ]

    distances = (
        matrix[medoid_idx]
        if n > 1
        else np.array([0.0])
    )

    solid_support = float(
        np.mean(
            distances
            <= SOLID_POINT_DE00_RADIUS
        )
    )

    L_range = robust_range([
        p["lab_L"]
        for p in points
    ])

    C_range = robust_range([
        p["lab_chroma"]
        for p in points
    ])

    S_range = robust_range([
        p["hsv_saturation"]
        for p in points
    ])

    V_range = robust_range([
        p["hsv_value"]
        for p in points
    ])

    if n == 1:
        label = "INSUFFICIENT_POINTS"

    elif (
        solid_support
        >= MIN_SOLID_POINT_SUPPORT
    ):
        label = "SOLID_CANDIDATE"

    else:
        if (
            L_range >= MIN_SCALE_RANGE
            and
            L_range
            >= DOMINANCE_RATIO
            * max(C_range, 1e-6)
        ):
            label = (
                "BRIGHTNESS_SCALE_CANDIDATE"
            )

        elif (
            C_range >= MIN_SCALE_RANGE
            and
            C_range
            >= DOMINANCE_RATIO
            * max(L_range, 1e-6)
        ):
            label = (
                "SATURATION_SCALE_CANDIDATE"
            )

        else:
            label = (
                "MIXED_SCALE_CANDIDATE"
            )

    return {
        "encoding_candidate": label,

        "n_points": n,

        "solid_support_de4": solid_support,

        "medoid_hex": medoid[
            "observed_hex"
        ],

        "medoid_nearest_F_name": medoid[
            "nearest_F_name"
        ],

        "medoid_nearest_F_hex": medoid[
            "nearest_F_hex"
        ],

        "medoid_deltaE00_to_F": medoid[
            "deltaE00_to_nearest_F"
        ],

        "L_range": L_range,
        "C_range": C_range,
        "S_range": S_range,
        "V_range": V_range,

        "de_to_medoid_median": float(
            np.median(distances)
        ),

        "de_to_medoid_max": float(
            np.max(distances)
        ),
    }


def finalize_families(families):
    result = []

    for fam in families:
        classification = classify_family(
            fam["points"]
        )

        result.append({
            **fam,
            **classification,
        })

    return result


# ============================================================
# SELEÇÃO DAS IMAGENS
# ============================================================

def read_calibration_exclusions():
    """
    Lê as imagens utilizadas na calibração V7.

    O holdout é interrompido se esse arquivo não existir, pois sem a
    exclusão explícita das imagens de calibração não há validação
    independente.
    """
    if not CALIBRATION_CSV.exists():
        raise FileNotFoundError(
            "CSV da calibração V7 não encontrado:\n"
            f"{CALIBRATION_CSV}\n\n"
            "Mantenha a pasta _color_scatter_markercore_v7 no diretório "
            "de imagens antes de executar o holdout."
        )

    with CALIBRATION_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    excluded = {
        row["filename"]
        for row in rows
        if str(row.get("filename", "")).strip()
    }

    if len(excluded) == 0:
        raise RuntimeError(
            "O CSV da calibração V7 não contém nomes de imagens."
        )

    return excluded


def select_validation_images():
    """
    Seleciona 80 imagens independentes da calibração V7:
        20 SI canvas OK
        20 SI canvas NAO_OK
        20 SC canvas OK
        20 SC canvas NAO_OK

    Nenhum parâmetro do detector é alterado com base neste conjunto.
    """
    excluded = read_calibration_exclusions()

    rng = random.Random(
        RANDOM_SEED
    )

    groups = defaultdict(list)

    for path in sorted(
        ROOT_DIR.glob("*.png")
    ):
        m = NAME_RE.match(
            path.name
        )

        if not m:
            continue

        if path.name in excluded:
            continue

        profile = (
            m.group(1).upper()
        )

        try:
            with Image.open(path) as img:
                canvas = (
                    "OK"
                    if img.size == (1200, 800)
                    else "NAO_OK"
                )
        except Exception:
            continue

        groups[
            (profile, canvas)
        ].append(path)

    selected = []

    for profile in (
        "SI",
        "SC"
    ):
        for canvas in (
            "OK",
            "NAO_OK"
        ):
            pool = groups[
                (profile, canvas)
            ]

            if len(pool) < N_PER_PROFILE_CANVAS_CLASS:
                raise RuntimeError(
                    "Não há imagens suficientes no estrato "
                    f"{profile}/{canvas}. "
                    f"Disponíveis após excluir calibração: {len(pool)}; "
                    f"necessárias: {N_PER_PROFILE_CANVAS_CLASS}."
                )

            chosen = rng.sample(
                pool,
                N_PER_PROFILE_CANVAS_CLASS
            )

            for path in chosen:
                selected.append((
                    profile,
                    canvas,
                    path
                ))

    overlap = {
        path.name
        for _, _, path in selected
        if path.name in excluded
    }

    if overlap:
        raise RuntimeError(
            "ERRO: sobreposição calibração-holdout: "
            + ", ".join(sorted(overlap))
        )

    return selected, excluded


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(
    src,
    plot_roi,
    roi_method,
    kept_markers,
    suspected_legend,
    families,
    out
):
    img = Image.open(
        src
    ).convert("RGB")

    draw = ImageDraw.Draw(
        img
    )

    font = ImageFont.load_default()

    # Área de plotagem
    x1, y1, x2, y2 = plot_roi

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=(0, 160, 0),
        width=max(
            2,
            img.width // 600
        )
    )

    draw.text(
        (x1, max(2, y1-14)),
        f"plot_roi={roi_method}",
        fill=(0, 100, 0),
        font=font
    )

    # Marcadores aceitos
    for marker in kept_markers:
        x = marker["x"]
        y = marker["y"]
        w = marker["w"]
        h = marker["h"]

        draw.rectangle(
            [x, y, x+w, y+h],
            outline=(0, 190, 210),
            width=max(
                1,
                img.width // 900
            )
        )

    # Possível legenda (somente auditoria; NÃO removida)
    for marker in suspected_legend:
        x = marker["x"]
        y = marker["y"]
        w = marker["w"]
        h = marker["h"]

        draw.rectangle(
            [x, y, x+w, y+h],
            outline=(230, 190, 0),
            width=max(
                2,
                img.width // 700
            )
        )

    # Família
    for rank, fam in enumerate(
        families,
        1
    ):
        pts = fam["points"]

        medoid_rgb = hex_to_rgb(
            fam["medoid_hex"]
        )

        xs = [
            p["x"]
            for p in pts
        ]

        ys = [
            p["y"]
            for p in pts
        ]

        xe = [
            p["x"] + p["w"]
            for p in pts
        ]

        ye = [
            p["y"] + p["h"]
            for p in pts
        ]

        if not xs:
            continue

        fx1 = min(xs)
        fy1 = min(ys)
        fx2 = max(xe)
        fy2 = max(ye)

        draw.rectangle(
            [fx1, fy1, fx2, fy2],
            outline=medoid_rgb,
            width=max(
                2,
                img.width // 500
            )
        )

        txt = (
            f"{rank} "
            f"H={fam['family_hue_deg']:.0f} "
            f"pts={fam['n_points']} "
            f"{fam['encoding_candidate']} "
            f"solid={fam['solid_support_de4']:.2f}"
        )

        draw.text(
            (
                fx1,
                max(2, fy1-13)
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

    for idx, item in enumerate(
        items
    ):
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

        im = Image.open(
            item["overlay"]
        ).convert("RGB")

        im.thumbnail(
            (IMG_W, IMG_H)
        )

        sheet.paste(
            im,
            (
                x0 + 15,
                y0 + 70
                + (
                    IMG_H
                    - im.height
                ) // 2
            )
        )

        draw.text(
            (x0+8, y0+10),
            (
                f"{item['profile']} | "
                f"CANVAS_{item['canvas']} | "
                f"{item['filename']} | "
                f"families={item['n_families']} | "
                f"markers={item['n_markers']}"
            ),
            fill=(0,0,0),
            font=font
        )

        draw.text(
            (x0+8, y0+30),
            (
                f"ROI={item['roi_method']} | "
                f"legend_suspected="
                f"{item['n_legend_suspected']}"
            ),
            fill=(0,0,0),
            font=font
        )

        draw.text(
            (x0+8, y0+48),
            " | ".join(
                item["types"]
            )[:98],
            fill=(0,0,0),
            font=font
        )

    sheet.save(out)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 86)
    print(
        "S-COR V7 - "
        "MARKER-CORE SEM EXCLUSÃO PRECOCE DE LEGENDA"
    )
    print("=" * 86)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        selected,
        calibration_exclusions
    ) = select_validation_images()

    image_rows = []
    family_rows = []
    point_rows = []
    suspected_legend_rows = []
    sheet_items = []

    for idx, (
        profile,
        canvas,
        path
    ) in enumerate(
        selected,
        1
    ):
        with Image.open(path) as opened:
            arr_u8 = np.asarray(
                opened.convert("RGB"),
                dtype=np.uint8
            )

        (
            plot_roi,
            roi_method,
            n_hlines,
            n_vlines
        ) = infer_plot_roi(
            arr_u8
        )

        (
            marker_candidates,
            hsv
        ) = detect_markers(
            arr_u8,
            plot_roi
        )

        (
            kept_markers,
            suspected_legend
        ) = flag_probable_legend_markers(
            arr_u8,
            hsv,
            marker_candidates,
            plot_roi
        )

        for rm_rank, marker in enumerate(
            suspected_legend,
            1
        ):
            suspected_legend_rows.append({
                "dataset_role": "HOLDOUT",
                "profile": profile,
                "canvas_class": canvas,
                "filename": path.name,
                "suspected_rank": rm_rank,
                "x": marker["x"],
                "y": marker["y"],
                "w": marker["w"],
                "h": marker["h"],
                "area": marker["area"],
                "core_radius_px": round(marker["core_radius_px"], 6),
                "observed_hex": marker["observed_hex"],
                "hsv_hue_deg": round(marker["hsv_hue_deg"], 6),
                "nearest_F_name": marker["nearest_F_name"],
                "deltaE00_to_nearest_F": round(
                    marker["deltaE00_to_nearest_F"],
                    6
                ),
            })

        families = finalize_families(
            group_markers_by_hue(
                kept_markers
            )
        )

        overlay = (
            OVERLAY_DIR
            / f"{path.stem}__holdout_v8.png"
        )

        make_overlay(
            path,
            plot_roi,
            roi_method,
            kept_markers,
            suspected_legend,
            families,
            overlay
        )

        expected = (
            1 if profile == "SI"
            else 2
        )

        image_rows.append({
            "dataset_role": "HOLDOUT",
            "profile": profile,
            "canvas_class": canvas,
            "filename": path.name,

            "roi_method": roi_method,
            "roi_x1": plot_roi[0],
            "roi_y1": plot_roi[1],
            "roi_x2": plot_roi[2],
            "roi_y2": plot_roi[3],
            "n_hough_horizontal_lines": n_hlines,
            "n_hough_vertical_lines": n_vlines,

            "n_marker_candidates": len(
                marker_candidates
            ),

            "n_legend_markers_suspected": len(
                suspected_legend
            ),

            "n_data_markers_kept": len(
                kept_markers
            ),

            "expected_hue_family_count_diagnostic": (
                expected
            ),

            "n_hue_families_detected": len(
                families
            ),

            "count_matches_profile_diagnostic": int(
                len(families) == expected
            ),

            "n_solid_candidate": sum(
                f["encoding_candidate"]
                == "SOLID_CANDIDATE"
                for f in families
            ),

            "n_brightness_scale_candidate": sum(
                f["encoding_candidate"]
                == "BRIGHTNESS_SCALE_CANDIDATE"
                for f in families
            ),

            "n_saturation_scale_candidate": sum(
                f["encoding_candidate"]
                == "SATURATION_SCALE_CANDIDATE"
                for f in families
            ),

            "n_mixed_scale_candidate": sum(
                f["encoding_candidate"]
                == "MIXED_SCALE_CANDIDATE"
                for f in families
            ),

            "n_insufficient_points": sum(
                f["encoding_candidate"]
                == "INSUFFICIENT_POINTS"
                for f in families
            ),
        })

        for family_rank, fam in enumerate(
            families,
            1
        ):
            family_rows.append({
                "dataset_role": "HOLDOUT",
                "profile": profile,
                "canvas_class": canvas,
                "filename": path.name,
                "family_rank": family_rank,

                "family_hue_deg": round(
                    fam[
                        "family_hue_deg"
                    ],
                    6
                ),

                "n_points": fam[
                    "n_points"
                ],

                "encoding_candidate": fam[
                    "encoding_candidate"
                ],

                "solid_support_de4": round(
                    fam[
                        "solid_support_de4"
                    ],
                    6
                ),

                "medoid_hex": fam[
                    "medoid_hex"
                ],

                "medoid_nearest_F_name": fam[
                    "medoid_nearest_F_name"
                ],

                "medoid_nearest_F_hex": fam[
                    "medoid_nearest_F_hex"
                ],

                "medoid_deltaE00_to_F": round(
                    fam[
                        "medoid_deltaE00_to_F"
                    ],
                    6
                ),

                "L_range": round(
                    fam["L_range"],
                    6
                ),

                "C_range": round(
                    fam["C_range"],
                    6
                ),

                "S_range": round(
                    fam["S_range"],
                    6
                ),

                "V_range": round(
                    fam["V_range"],
                    6
                ),

                "de_to_medoid_median": round(
                    fam[
                        "de_to_medoid_median"
                    ],
                    6
                ),

                "de_to_medoid_max": round(
                    fam[
                        "de_to_medoid_max"
                    ],
                    6
                ),
            })

            for point_rank, p in enumerate(
                fam["points"],
                1
            ):
                point_rows.append({
                    "dataset_role": "HOLDOUT",
                    "profile": profile,
                    "canvas_class": canvas,
                    "filename": path.name,
                    "family_rank": family_rank,
                    "point_rank": point_rank,

                    "x": p["x"],
                    "y": p["y"],
                    "w": p["w"],
                    "h": p["h"],
                    "area": p["area"],

                    "aspect_ratio": round(
                        p[
                            "aspect_ratio"
                        ],
                        6
                    ),

                    "fill_ratio": round(
                        p[
                            "fill_ratio"
                        ],
                        6
                    ),

                    "circularity": round(
                        p[
                            "circularity"
                        ],
                        6
                    ),

                    "core_radius_px": round(
                        p["core_radius_px"],
                        6
                    ),

                    "extract_radius_px": round(
                        p["extract_radius_px"],
                        6
                    ),

                    "observed_hex": p[
                        "observed_hex"
                    ],

                    "marker_hue_deg": round(
                        p[
                            "marker_hue_deg"
                        ],
                        6
                    ),

                    "valid_color_pixels": p[
                        "valid_color_pixels"
                    ],

                    "core_color_pixels": p[
                        "core_color_pixels"
                    ],

                    "exact_color_support": round(
                        p[
                            "exact_color_support"
                        ],
                        6
                    ),

                    "lab_L": round(
                        p["lab_L"],
                        6
                    ),

                    "lab_a": round(
                        p["lab_a"],
                        6
                    ),

                    "lab_b": round(
                        p["lab_b"],
                        6
                    ),

                    "lab_chroma": round(
                        p[
                            "lab_chroma"
                        ],
                        6
                    ),

                    "lab_hue_deg": round(
                        p[
                            "lab_hue_deg"
                        ],
                        6
                    ),

                    "hsv_hue_deg": round(
                        p[
                            "hsv_hue_deg"
                        ],
                        6
                    ),

                    "hsv_saturation": round(
                        p[
                            "hsv_saturation"
                        ],
                        6
                    ),

                    "hsv_value": round(
                        p[
                            "hsv_value"
                        ],
                        6
                    ),

                    "nearest_F_name": p[
                        "nearest_F_name"
                    ],

                    "nearest_F_hex": p[
                        "nearest_F_hex"
                    ],

                    "deltaE00_to_nearest_F": round(
                        p[
                            "deltaE00_to_nearest_F"
                        ],
                        6
                    ),
                })

        sheet_items.append({
            "profile": profile,
            "canvas": canvas,
            "filename": path.name,
            "overlay": overlay,
            "n_families": len(
                families
            ),
            "n_markers": len(
                kept_markers
            ),
            "n_legend_suspected": len(
                suspected_legend
            ),
            "roi_method": roi_method,
            "types": [
                (
                    f"{f['encoding_candidate']}"
                    f"(pts={f['n_points']},"
                    f"solid={f['solid_support_de4']:.2f})"
                )
                for f in families
            ],
        })

        print(
            f"{idx:02d}/{len(selected):02d} | "
            f"{path.name} | "
            f"ROI={roi_method} | "
            f"markers={len(kept_markers)} | "
            f"legend_removed={len(suspected_legend)} | "
            f"families={len(families)} | "
            + ",".join(
                f["encoding_candidate"]
                for f in families
            )
        )

    # ========================================================
    # CSV IMAGEM
    # ========================================================

    image_csv = (
        OUTPUT_DIR
        / "scatter_holdout_image_v8.csv"
    )

    image_fields = [
        "dataset_role",
        "profile",
        "canvas_class",
        "filename",

        "roi_method",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "n_hough_horizontal_lines",
        "n_hough_vertical_lines",

        "n_marker_candidates",
        "n_legend_markers_suspected",
        "n_data_markers_kept",

        "expected_hue_family_count_diagnostic",
        "n_hue_families_detected",
        "count_matches_profile_diagnostic",

        "n_solid_candidate",
        "n_brightness_scale_candidate",
        "n_saturation_scale_candidate",
        "n_mixed_scale_candidate",
        "n_insufficient_points",
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

    # ========================================================
    # CSV FAMÍLIA
    # ========================================================

    family_csv = (
        OUTPUT_DIR
        / "scatter_holdout_family_v8.csv"
    )

    family_fields = [
        "dataset_role",
        "profile",
        "canvas_class",
        "filename",
        "family_rank",

        "family_hue_deg",
        "n_points",
        "encoding_candidate",
        "solid_support_de4",

        "medoid_hex",
        "medoid_nearest_F_name",
        "medoid_nearest_F_hex",
        "medoid_deltaE00_to_F",

        "L_range",
        "C_range",
        "S_range",
        "V_range",
        "de_to_medoid_median",
        "de_to_medoid_max",
    ]

    with family_csv.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=family_fields
        )

        writer.writeheader()
        writer.writerows(
            family_rows
        )

    # ========================================================
    # CSV PONTO
    # ========================================================

    point_csv = (
        OUTPUT_DIR
        / "scatter_holdout_points_v8.csv"
    )

    point_fields = [
        "dataset_role",
        "profile",
        "canvas_class",
        "filename",
        "family_rank",
        "point_rank",

        "x",
        "y",
        "w",
        "h",
        "area",
        "aspect_ratio",
        "fill_ratio",
        "circularity",
        "core_radius_px",
        "extract_radius_px",

        "observed_hex",
        "marker_hue_deg",
        "valid_color_pixels",
        "core_color_pixels",
        "exact_color_support",

        "lab_L",
        "lab_a",
        "lab_b",
        "lab_chroma",
        "lab_hue_deg",

        "hsv_hue_deg",
        "hsv_saturation",
        "hsv_value",

        "nearest_F_name",
        "nearest_F_hex",
        "deltaE00_to_nearest_F",
    ]

    with point_csv.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=point_fields
        )

        writer.writeheader()
        writer.writerows(
            point_rows
        )

    # ========================================================
    # CSV CANDIDATOS SINALIZADOS COMO POSSÍVEL LEGENDA
    # ========================================================

    removed_csv = (
        OUTPUT_DIR
        / "scatter_holdout_suspected_legend_v8.csv"
    )

    removed_fields = [
        "dataset_role",
        "profile",
        "canvas_class",
        "filename",
        "suspected_rank",
        "x",
        "y",
        "w",
        "h",
        "area",
        "core_radius_px",
        "observed_hex",
        "hsv_hue_deg",
        "nearest_F_name",
        "deltaE00_to_nearest_F",
    ]

    with removed_csv.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=removed_fields
        )
        writer.writeheader()
        writer.writerows(
            suspected_legend_rows
        )

    # ========================================================
    # PRANCHA
    # ========================================================

    contact = (
        OUTPUT_DIR
        / "scatter_holdout_contact_sheet_v8.png"
    )

    make_contact_sheet(
        sheet_items,
        contact
    )

    # ========================================================
    # RESUMO
    # ========================================================

    type_counts = defaultdict(int)

    for row in family_rows:
        type_counts[
            row[
                "encoding_candidate"
            ]
        ] += 1

    matches = sum(
        row[
            "count_matches_profile_diagnostic"
        ]
        for row in image_rows
    )

    roi_hough = sum(
        row["roi_method"] == "HOUGH"
        for row in image_rows
    )

    total_legend_suspected = sum(
        row[
            "n_legend_markers_suspected"
        ]
        for row in image_rows
    )

    si_rows = [
        row for row in image_rows
        if row["profile"] == "SI"
    ]

    sc_rows = [
        row for row in image_rows
        if row["profile"] == "SC"
    ]

    si_matches = sum(
        row["count_matches_profile_diagnostic"]
        for row in si_rows
    )

    sc_matches = sum(
        row["count_matches_profile_diagnostic"]
        for row in sc_rows
    )

    solid_supports = [
        float(row["solid_support_de4"])
        for row in family_rows
        if row["encoding_candidate"] == "SOLID_CANDIDATE"
    ]

    nonzero_medoid_de = sum(
        float(row["medoid_deltaE00_to_F"]) > 1e-9
        for row in family_rows
        if row["encoding_candidate"] == "SOLID_CANDIDATE"
    )

    summary = [
        "S-COR HOLDOUT V8 - PARÂMETROS V7 CONGELADOS",
        "=" * 78,

        f"Imagens analisadas: "
        f"{len(image_rows)}",

        f"Imagens excluídas da calibração V7: "
        f"{len(calibration_exclusions)}",

        "Sobreposição calibração-holdout: 0",

        f"SI/OK: {sum(1 for r in image_rows if r['profile']=='SI' and r['canvas_class']=='OK')}",
        f"SI/NAO_OK: {sum(1 for r in image_rows if r['profile']=='SI' and r['canvas_class']=='NAO_OK')}",
        f"SC/OK: {sum(1 for r in image_rows if r['profile']=='SC' and r['canvas_class']=='OK')}",
        f"SC/NAO_OK: {sum(1 for r in image_rows if r['profile']=='SC' and r['canvas_class']=='NAO_OK')}",

        f"SI com contagem diagnóstica coincidente: "
        f"{si_matches}/{len(si_rows)}",

        f"SC com contagem diagnóstica coincidente: "
        f"{sc_matches}/{len(sc_rows)}",

        f"ROI por Hough: "
        f"{roi_hough}/{len(image_rows)}",

        f"Candidatos sinalizados como possível legenda: "
        f"{total_legend_suspected}",

        f"Marcadores de dados mantidos: "
        f"{len(point_rows)}",

        f"Famílias finais: "
        f"{len(family_rows)}",

        f"Contagem coincidente com perfil "
        f"(diagnóstico): "
        f"{matches}/{len(image_rows)}",

        "",

        "CLASSIFICAÇÃO:",

        f"SOLID_CANDIDATE: "
        f"{type_counts['SOLID_CANDIDATE']}",

        f"BRIGHTNESS_SCALE_CANDIDATE: "
        f"{type_counts['BRIGHTNESS_SCALE_CANDIDATE']}",

        f"SATURATION_SCALE_CANDIDATE: "
        f"{type_counts['SATURATION_SCALE_CANDIDATE']}",

        f"MIXED_SCALE_CANDIDATE: "
        f"{type_counts['MIXED_SCALE_CANDIDATE']}",

        f"INSUFFICIENT_POINTS: "
        f"{type_counts['INSUFFICIENT_POINTS']}",

        "",

        (
            f"solid_support_de4 médio: {np.mean(solid_supports):.4f}"
            if solid_supports
            else "solid_support_de4 médio: NA"
        ),

        (
            f"solid_support_de4 mínimo: {np.min(solid_supports):.4f}"
            if solid_supports
            else "solid_support_de4 mínimo: NA"
        ),

        f"Famílias SOLID com medóide ΔE00 > 0: "
        f"{nonzero_medoid_de}",

        "",

        "DESENHO HOLDOUT:",
        "- todos os parâmetros do detector foram congelados na V7;",
        "- nenhuma regra é ajustada com base nos resultados deste conjunto;",
        "- 40 imagens da calibração V7 são excluídas antes da amostragem;",
        "- 20 novas imagens por estrato SI/SC x canvas OK/NAO_OK.",

        "",

        "PRINCÍPIO V7 CONGELADO:",
        "- marcas detectadas por núcleo espesso no distance transform;",
        "- ROI expandida para preservar marcadores nos limites dos eixos;",
        "- traços finos de rótulos tendem a não gerar marcador;",
        "- a cor é medida localmente ao redor do núcleo do ponto;",
        "- agrupamento de famílias usa o hue da cor-núcleo representativa;",
        "- possível legenda é apenas sinalizada; nenhum marcador é excluído por essa heurística;",
        "- grade cinza é excluída por baixa saturação;",
        "- famílias são classificadas como sólida ou escala.",

        "",

        "Nenhum limiar de conformidade final foi aplicado.",

        "",

        f"Prancha: {contact}",
        f"CSV imagem: {image_csv}",
        f"CSV família: {family_csv}",
        f"CSV pontos: {point_csv}",
        f"CSV candidatos sinalizados como possível legenda: {removed_csv}",
    ]

    summary_path = (
        OUTPUT_DIR
        / "scatter_holdout_summary_v8.txt"
    )

    summary_path.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print("\n" + "=" * 86)
    print("CONCLUÍDO")
    print("=" * 86)

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
