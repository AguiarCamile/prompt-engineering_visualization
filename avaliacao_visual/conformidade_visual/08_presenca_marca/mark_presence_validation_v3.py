# -*- coding: utf-8 -*-
r"""
mark_presence_validation_v3.py

VALIDAÇÃO EXPLORATÓRIA V3 — PRESENÇA DA ESTRUTURA GRÁFICA PRINCIPAL
===================================================================

Objetivo
--------
Detectar se a visualização contém a marca de dados principal correspondente
à técnica solicitada:

    B -> barras
    L -> linha(s)
    S -> pontos de dispersão

Esta V3 é uma etapa de CALIBRAÇÃO/INSPEÇÃO, não a versão final para as
3.120 imagens.

Princípio metodológico
----------------------
A detecção estrutural NÃO usa as cores esperadas da especificação F e NÃO
usa quantidade esperada de séries. O perfil do arquivo apenas seleciona o
detector apropriado à técnica (barra, linha ou dispersão).

A estratégia combina:
- contraste em relação ao fundo;
- geometria;
- alinhamento espacial;
- espessura/forma;
- extensão no eixo x.

Ela foi concebida para detectar marcas inclusive quando estiverem em preto,
cinza ou em cores diferentes das especificadas.

Amostra de calibração
---------------------
Por padrão:
- 10 imagens por perfil (BI, BC, LI, LC, SI, SC);
- tenta equilibrar CANVAS_OK / CANVAS_NAO_OK;
- acrescenta todos os casos NO_MARK encontrados na avaliação final de cor,
  porque são casos de estresse importantes para distinguir:
      "marca realmente ausente"
  de
      "marca presente, mas não cromática".

Pastas padrão
-------------
Imagens:
C:\Users\Labvis\Downloads\imagens3120\imagens

Canvas:
imagens\_canvas_conformity_v1\canvas_conformity_images.csv

Cor:
imagens\_color_conformity_final_v1\color_conformity_final_images.csv

Saída:
imagens\_mark_presence_validation_v3

Arquivos:
- mark_presence_validation_v3.csv
- mark_presence_validation_summary_v3.txt
- contact_<perfil>.png
- overlays\*.png

Interpretação
-------------
presence_status:
    PRESENT
    ABSENT_OR_UNCERTAIN

IMPORTANTE:
- ABSENT_OR_UNCERTAIN ainda NÃO deve ser tratado como não conformidade final.
- A calibração precisa ser conferida visualmente antes de congelar parâmetros.
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path(
    r"C:\Users\Labvis\Downloads\imagens3120\imagens"
)
DEFAULT_OUTPUT_DIRNAME = "_mark_presence_validation_v3"
DEFAULT_CANVAS_DIRNAME = "_canvas_conformity_v1"
DEFAULT_COLOR_DIRNAME = "_color_conformity_final_v1"

SEED = 20260828
N_RANDOM_PER_PROFILE = 10

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

NAME_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.I,
)

PROFILES = ["BI", "BC", "LI", "LC", "SI", "SC"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Validação exploratória da presença de marcas de dados."
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--sample-per-profile",
        type=int,
        default=N_RANDOM_PER_PROFILE,
    )
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def parse_name(path: Path):
    m = NAME_RE.match(path.name)
    if not m:
        return None

    profile = m.group(1).upper()
    unit_number = int(m.group(2))
    repeat = int(m.group(3))

    return {
        "profile": profile,
        "technique": profile[0],
        "task": profile[1],
        "unit_number": unit_number,
        "unit_id": f"{profile}_{unit_number:03d}",
        "repeat": repeat,
    }


def root_images(root: Path):
    rows = []
    for p in sorted(root.iterdir()):
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and NAME_RE.match(p.name)
        ):
            meta = parse_name(p)
            rows.append({
                "path": p,
                "filename": p.name,
                **meta,
            })
    return pd.DataFrame(rows)


def load_optional_labels(root: Path):
    canvas_path = (
        root
        / DEFAULT_CANVAS_DIRNAME
        / "canvas_conformity_images.csv"
    )
    color_path = (
        root
        / DEFAULT_COLOR_DIRNAME
        / "color_conformity_final_images.csv"
    )

    canvas = pd.DataFrame()
    color = pd.DataFrame()

    if canvas_path.exists():
        canvas = pd.read_csv(canvas_path)

    if color_path.exists():
        color = pd.read_csv(color_path)

    return canvas, color


def choose_sample(
    inventory: pd.DataFrame,
    canvas: pd.DataFrame,
    color: pd.DataFrame,
    n_per_profile: int,
    seed: int,
):
    df = inventory.copy()

    if not canvas.empty and "filename" in canvas.columns:
        keep = [
            c for c in [
                "filename",
                "canvas_status",
                "canvas_ok",
            ]
            if c in canvas.columns
        ]
        df = df.merge(
            canvas[keep],
            on="filename",
            how="left",
        )

    if not color.empty and "filename" in color.columns:
        keep = [
            c for c in [
                "filename",
                "extraction_status",
                "color_status_final",
            ]
            if c in color.columns
        ]
        df = df.merge(
            color[keep],
            on="filename",
            how="left",
        )

    rng = random.Random(seed)
    selected = set()

    for profile in PROFILES:
        g = df[df["profile"] == profile].copy()

        n = min(n_per_profile, len(g))

        # Se canvas disponível, tenta metade OK/metade não OK.
        if "canvas_ok" in g.columns and g["canvas_ok"].notna().any():
            n_ok = n // 2
            n_bad = n - n_ok

            ok_files = g[
                pd.to_numeric(
                    g["canvas_ok"],
                    errors="coerce"
                ) == 1
            ]["filename"].tolist()

            bad_files = g[
                pd.to_numeric(
                    g["canvas_ok"],
                    errors="coerce"
                ) == 0
            ]["filename"].tolist()

            rng.shuffle(ok_files)
            rng.shuffle(bad_files)

            chosen = ok_files[:n_ok] + bad_files[:n_bad]

            if len(chosen) < n:
                leftovers = [
                    x for x in g["filename"].tolist()
                    if x not in chosen
                ]
                rng.shuffle(leftovers)
                chosen += leftovers[:n - len(chosen)]

        else:
            files = g["filename"].tolist()
            rng.shuffle(files)
            chosen = files[:n]

        selected.update(chosen)

    # Inclui todos os NO_MARK da avaliação de cor como casos de estresse.
    if "extraction_status" in df.columns:
        stress = df[
            df["extraction_status"]
            .astype(str)
            .str.upper()
            .eq("NO_MARK")
        ]["filename"].tolist()

        selected.update(stress)

    out = df[df["filename"].isin(selected)].copy()

    out["is_color_no_mark_stress"] = 0

    if "extraction_status" in out.columns:
        out["is_color_no_mark_stress"] = (
            out["extraction_status"]
            .astype(str)
            .str.upper()
            .eq("NO_MARK")
        ).astype(int)

    return out.sort_values(
        ["profile", "unit_number", "repeat"]
    ).reset_index(drop=True)


def border_background_rgb(arr):
    h, w, _ = arr.shape
    b = max(2, int(round(min(h, w) * 0.02)))

    pixels = np.concatenate([
        arr[:b, :, :].reshape(-1, 3),
        arr[-b:, :, :].reshape(-1, 3),
        arr[:, :b, :].reshape(-1, 3),
        arr[:, -b:, :].reshape(-1, 3),
    ], axis=0)

    return np.median(
        pixels.astype(np.float32),
        axis=0
    )


def infer_plot_roi(arr):
    """
    Inferência da região interna do gráfico a partir das linhas dos eixos/spines.

    V2:
    - tenta detectar esquerda, direita, topo e base;
    - quando encontra o retângulo completo, exclui naturalmente legendas
      externas e títulos;
    - usa fallback conservador quando o retângulo não é recuperável.
    """
    h, w, _ = arr.shape
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    dark = (gray < 180).astype(np.uint8) * 255

    hk = max(20, int(round(0.14 * w)))
    vk = max(20, int(round(0.14 * h)))

    hlines = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (hk, 1)
        )
    )

    vlines = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, vk)
        )
    )

    # Contagens por coluna/linha.
    v_counts = (vlines > 0).sum(axis=0)
    h_counts = (hlines > 0).sum(axis=1)

    def choose_extreme(values, lo, hi, side):
        idx = np.arange(len(values))
        mask = (idx >= lo) & (idx <= hi)

        if not np.any(mask):
            return None

        local_idx = idx[mask]
        local_vals = values[mask]

        if local_vals.max() <= 0:
            return None

        mx = local_vals.max()
        strong = local_idx[
            local_vals >= 0.80 * mx
        ]

        if len(strong) == 0:
            return None

        if side in ("left", "top"):
            return int(strong.min())
        return int(strong.max())

    x_left = choose_extreme(
        v_counts,
        int(0.03 * w),
        int(0.45 * w),
        "left"
    )

    x_right = choose_extreme(
        v_counts,
        int(0.50 * w),
        int(0.985 * w),
        "right"
    )

    y_top = choose_extreme(
        h_counts,
        int(0.025 * h),
        int(0.45 * h),
        "top"
    )

    y_bottom = choose_extreme(
        h_counts,
        int(0.45 * h),
        int(0.965 * h),
        "bottom"
    )

    if (
        x_left is not None
        and x_right is not None
        and y_top is not None
        and y_bottom is not None
        and x_right - x_left >= 0.35 * w
        and y_bottom - y_top >= 0.30 * h
    ):
        pad = max(1, int(round(min(h, w) * 0.002)))

        return (
            min(w - 1, x_left + pad),
            min(h - 1, y_top + pad),
            max(0, x_right - pad),
            max(0, y_bottom - pad),
        ), "FULL_AXES_RECT"

    # Fallback baseado em esquerda/base.
    if x_left is not None and y_bottom is not None:
        x1 = max(0, x_left + 1)
        y2 = min(h - 1, y_bottom - 1)
        x2 = min(w - 1, int(round(0.94 * w)))
        y1 = max(0, int(round(0.07 * h)))

        if (
            x2 - x1 >= 0.40 * w
            and y2 - y1 >= 0.35 * h
        ):
            return (x1, y1, x2, y2), "AXIS_PARTIAL"

    return (
        int(round(0.08 * w)),
        int(round(0.08 * h)),
        int(round(0.94 * w)),
        int(round(0.90 * h)),
    ), "FALLBACK_CENTRAL"

def foreground_mask(arr, roi):
    h, w, _ = arr.shape
    bg = border_background_rgb(arr)

    diff = np.linalg.norm(
        arr.astype(np.float32) - bg[None, None, :],
        axis=2
    )

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    mask = (
        (diff >= 28.0)
        & (gray <= 246)
    )

    x1, y1, x2, y2 = roi

    roi_mask = np.zeros((h, w), dtype=bool)
    roi_mask[
        max(0, y1):min(h, y2 + 1),
        max(0, x1):min(w, x2 + 1)
    ] = True

    return mask & roi_mask, bg


def remove_long_axis_grid(mask, roi):
    """
    Remove traços longos estritamente horizontais/verticais.
    Mantém o restante para detecção estrutural.
    """
    x1, y1, x2, y2 = roi
    rh = max(1, y2 - y1 + 1)
    rw = max(1, x2 - x1 + 1)

    u8 = mask.astype(np.uint8) * 255

    hk = max(15, int(round(0.55 * rw)))
    vk = max(15, int(round(0.55 * rh)))

    horizontal = cv2.morphologyEx(
        u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (hk, 1)
        )
    )

    vertical = cv2.morphologyEx(
        u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, vk)
        )
    )

    long_lines = cv2.bitwise_or(horizontal, vertical)

    # dilata levemente antes de remover
    long_lines = cv2.dilate(
        long_lines,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    cleaned = u8.copy()
    cleaned[long_lines > 0] = 0

    return cleaned > 0


def detect_bars(arr, roi):
    """
    Detector estrutural de barras V3.

    Ideia central:
    1. obtém foreground sem usar cor normativa;
    2. aplica distance transform;
    3. mantém apenas o interior espesso dos objetos;
    4. procura corpos retangulares próximos à linha de base do plot.

    O núcleo espesso remove naturalmente grade, eixos, texto e contornos finos,
    evitando o problema da V2 em que gridlines conectavam o corpo das barras
    a outros elementos.
    """
    mask, _ = foreground_mask(arr, roi)

    x1, y1, x2, y2 = roi
    rw = x2 - x1 + 1
    rh = y2 - y1 + 1
    roi_area = rw * rh

    u8 = mask.astype(np.uint8)

    # Preenche microfalhas de rasterização sem unir regiões distantes.
    u8 = cv2.morphologyEx(
        u8,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    )

    dist = cv2.distanceTransform(
        u8,
        cv2.DIST_L2,
        5
    )

    core_radius = max(
        2.0,
        0.0035 * min(rw, rh)
    )

    core = (
        dist >= core_radius
    ).astype(np.uint8) * 255

    # Pequeno fechamento no núcleo recompõe barras com discretização irregular.
    core = cv2.morphologyEx(
        core,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        core,
        connectivity=8
    )

    baseline_tol = max(
        8,
        int(round(0.085 * rh))
    )

    candidates = []

    for lab in range(1, n):
        x, y, w, h, area = stats[lab]

        if area < max(20, int(0.00010 * roi_area)):
            continue

        if w < max(3, int(0.004 * rw)):
            continue

        if h < max(5, int(0.012 * rh)):
            continue

        if w > 0.42 * rw:
            continue

        fill = area / (w * h) if w * h else 0

        # Interior de uma barra deve ser compacto.
        if fill < 0.68:
            continue

        # Evita caixas muito horizontais.
        if h < 0.18 * w:
            continue

        bottom = y + h

        # Como usamos o núcleo, ele termina alguns pixels acima da base real.
        if abs(bottom - y2) > baseline_tol:
            continue

        candidates.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "fill": float(fill),
            "bottom": int(bottom),
            "baseline_gap": int(abs(bottom - y2)),
            "core_radius": float(core_radius),
        })

    if not candidates:
        return {
            "present": False,
            "confidence": 0.0,
            "candidates": [],
            "evidence": (
                f"NO_THICK_BASELINE_RECTANGLES;"
                f"CORE_R={core_radius:.2f}"
            ),
        }

    total_area = sum(c["area"] for c in candidates)
    max_height_frac = max(c["h"] / rh for c in candidates)

    strong_single = (
        len(candidates) == 1
        and total_area >= 0.0015 * roi_area
        and max_height_frac >= 0.055
    )

    present = (
        len(candidates) >= 2
        or strong_single
    )

    confidence = (
        min(
            1.0,
            0.35
            + 0.17 * min(len(candidates), 4)
            + 0.30 * min(
                1.0,
                total_area / max(1.0, 0.020 * roi_area)
            )
            + 0.18 * min(
                1.0,
                max_height_frac / 0.25
            )
        )
        if present
        else 0.20
    )

    return {
        "present": bool(present),
        "confidence": float(confidence),
        "candidates": candidates,
        "evidence": (
            f"THICK_BASELINE_RECTANGLES={len(candidates)};"
            f"MAX_H_FRAC={max_height_frac:.3f};"
            f"CORE_R={core_radius:.2f}"
        ),
    }

def detect_lines(arr, roi):
    """
    Detector estrutural de linhas V3.

    Correção principal da V2:
    - Hough é executado sobre a máscara estrutural LIMPA, não sobre a imagem
      original; assim gridlines/eixos removidos não reaparecem como evidência.
    - segmentos exatamente horizontais/verticais não contam, isoladamente,
      como evidência Hough, reduzindo falsos positivos de grade e barras.
    - usa cobertura espacial em X, em vez de soma bruta do comprimento dos
      segmentos.
    """
    mask, _ = foreground_mask(arr, roi)
    cleaned = remove_long_axis_grid(mask, roi)

    u8 = cleaned.astype(np.uint8) * 255

    u8 = cv2.morphologyEx(
        u8,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        u8,
        connectivity=8
    )

    x1, y1, x2, y2 = roi
    rw = x2 - x1 + 1
    rh = y2 - y1 + 1

    components = []

    for lab in range(1, n):
        x, y, w, h, area = stats[lab]

        x_span = w / rw
        y_span = h / rh

        if area < 18:
            continue

        if x_span < 0.16:
            continue

        fill = area / (w * h) if w * h else 0

        # Linhas/séries são esparsas dentro do bounding box.
        if fill > 0.50:
            continue

        if x_span > 0.95 and y_span > 0.85:
            continue

        components.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "x_span": float(x_span),
            "y_span": float(y_span),
            "fill": float(fill),
        })

    # Hough SOMENTE sobre foreground já limpo.
    crop_mask = u8[y1:y2+1, x1:x2+1]
    edges = cv2.Canny(crop_mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(15, int(0.020 * rw)),
        minLineLength=max(16, int(0.045 * rw)),
        maxLineGap=max(5, int(0.018 * rw)),
    )

    segments = []

    if lines is not None:
        for ln in np.asarray(lines).reshape(-1, 4):
            xa, ya, xb, yb = map(int, ln)

            dx = xb - xa
            dy = yb - ya
            length = math.hypot(dx, dy)

            if length <= 0:
                continue

            angle = abs(
                math.degrees(
                    math.atan2(dy, dx)
                )
            )
            angle = min(angle, 180 - angle)

            # Exclui bordas de barras, eixos e gridlines.
            # Um caminho de dados real costuma fornecer pelo menos alguns
            # segmentos oblíquos, mesmo quando contém trechos horizontais.
            if angle < 3.0 or angle > 82.0:
                continue

            ym = (ya + yb) / 2

            if ym < 0.03 * rh or ym > 0.97 * rh:
                continue

            segments.append({
                "x1": xa + x1,
                "y1": ya + y1,
                "x2": xb + x1,
                "y2": yb + y1,
                "length": float(length),
                "angle": float(angle),
            })

    best_x_span = max(
        [c["x_span"] for c in components],
        default=0.0
    )

    # Cobertura de X por segmentos oblíquos.
    n_bins = 40
    occupied = np.zeros(n_bins, dtype=bool)

    for s in segments:
        xa = min(s["x1"], s["x2"]) - x1
        xb = max(s["x1"], s["x2"]) - x1

        b1 = max(
            0,
            min(
                n_bins - 1,
                int(math.floor((xa / max(1, rw)) * n_bins))
            )
        )
        b2 = max(
            0,
            min(
                n_bins - 1,
                int(math.floor((xb / max(1, rw)) * n_bins))
            )
        )

        occupied[b1:b2+1] = True

    xbin_support = float(occupied.mean())

    structural_present = (
        best_x_span >= 0.24
        or xbin_support >= 0.20
    )

    confidence = (
        min(
            1.0,
            0.25
            + 0.80 * min(1.0, best_x_span)
            + 0.55 * min(1.0, xbin_support / 0.50)
        )
        if structural_present
        else 0.15
    )

    return {
        "present": bool(structural_present),
        "confidence": float(confidence),
        "components": components,
        "segments": segments,
        "xbin_support": float(xbin_support),
        "evidence": (
            f"BEST_X_SPAN={best_x_span:.3f};"
            f"XBIN_SUPPORT={xbin_support:.3f};"
            f"N_OBLIQUE_SEGMENTS={len(segments)}"
        ),
    }

def detect_scatter(arr, roi):
    mask, _ = foreground_mask(arr, roi)
    cleaned = remove_long_axis_grid(mask, roi)

    u8 = cleaned.astype(np.uint8) * 255

    # Pequeno fechamento preserva pontos, conecta antialiasing.
    u8 = cv2.morphologyEx(
        u8,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8)
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        u8,
        connectivity=8
    )

    x1, y1, x2, y2 = roi
    rw = x2 - x1 + 1
    rh = y2 - y1 + 1
    roi_area = rw * rh

    markers = []

    for lab in range(1, n):
        x, y, w, h, area = stats[lab]

        if area < max(10, int(0.000012 * roi_area)):
            continue

        if w < 3 or h < 3:
            continue

        if w > 0.08 * rw or h > 0.10 * rh:
            continue

        aspect = w / h if h else 0

        if not (0.35 <= aspect <= 2.8):
            continue

        component = (labels == lab).astype(np.uint8)

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        cnt_area = cv2.contourArea(cnt)

        circularity = (
            4 * math.pi * cnt_area / (perimeter * perimeter)
            if perimeter > 0
            else 0
        )

        fill = area / (w * h) if w * h else 0

        # Texto tende a ter fill menor e circularidade baixa,
        # mas não fazemos exigência excessiva para aceitar marcadores quadrados.
        if fill < 0.24:
            continue

        if circularity < 0.10 and fill < 0.45:
            continue

        markers.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
            "aspect": float(aspect),
            "fill": float(fill),
            "circularity": float(circularity),
            "cx": float(x + w / 2),
            "cy": float(y + h / 2),
        })

    # Evita considerar um único glifo/texto isolado como scatter.
    if len(markers) >= 2:
        xs = [m["cx"] for m in markers]
        ys = [m["cy"] for m in markers]

        x_disp = (
            (max(xs) - min(xs)) / rw
            if len(xs) > 1
            else 0
        )

        y_disp = (
            (max(ys) - min(ys)) / rh
            if len(ys) > 1
            else 0
        )
    else:
        x_disp = 0.0
        y_disp = 0.0

    # >=2 candidatos com alguma dispersão espacial.
    present = (
        len(markers) >= 2
        and (
            x_disp >= 0.03
            or y_disp >= 0.03
        )
    )

    # Um único marcador muito circular e grande pode ser evidência fraca,
    # mas nesta V2 não é suficiente para PRESENT.
    confidence = min(
        1.0,
        0.25
        + 0.12 * min(len(markers), 5)
        + 0.40 * min(1.0, x_disp + y_disp)
    ) if present else (
        0.30
        if len(markers) == 1
        else 0.10
    )

    return {
        "present": bool(present),
        "confidence": float(confidence),
        "markers": markers,
        "x_dispersion": float(x_disp),
        "y_dispersion": float(y_disp),
        "evidence": (
            f"N_MARKERS={len(markers)};"
            f"XDISP={x_disp:.3f};YDISP={y_disp:.3f}"
        ),
    }


def analyze_one(path: Path, technique: str):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    roi, roi_method = infer_plot_roi(arr)

    if technique == "B":
        det = detect_bars(arr, roi)
        n_candidates = len(det["candidates"])
    elif technique == "L":
        det = detect_lines(arr, roi)
        n_candidates = len(det["components"])
    elif technique == "S":
        det = detect_scatter(arr, roi)
        n_candidates = len(det["markers"])
    else:
        raise ValueError(f"Técnica desconhecida: {technique}")

    return {
        "image": img,
        "roi": roi,
        "roi_method": roi_method,
        "det": det,
        "n_candidates": n_candidates,
    }


def make_overlay(
    img: Image.Image,
    technique: str,
    roi,
    det,
):
    out = img.copy()
    draw = ImageDraw.Draw(out)

    x1, y1, x2, y2 = roi
    draw.rectangle(
        [x1, y1, x2, y2],
        outline="cyan",
        width=max(2, int(round(out.width / 500)))
    )

    if technique == "B":
        for c in det["candidates"]:
            draw.rectangle(
                [
                    c["x"],
                    c["y"],
                    c["x"] + c["w"],
                    c["y"] + c["h"],
                ],
                outline="magenta",
                width=max(2, int(round(out.width / 500)))
            )

    elif technique == "L":
        for c in det["components"]:
            draw.rectangle(
                [
                    c["x"],
                    c["y"],
                    c["x"] + c["w"],
                    c["y"] + c["h"],
                ],
                outline="magenta",
                width=max(2, int(round(out.width / 500)))
            )

        for s in det["segments"]:
            draw.line(
                [
                    s["x1"], s["y1"],
                    s["x2"], s["y2"],
                ],
                fill="yellow",
                width=max(2, int(round(out.width / 600)))
            )

    elif technique == "S":
        for m in det["markers"]:
            draw.ellipse(
                [
                    m["x"],
                    m["y"],
                    m["x"] + m["w"],
                    m["y"] + m["h"],
                ],
                outline="magenta",
                width=max(2, int(round(out.width / 500)))
            )

    return out


def font_default():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        return ImageFont.load_default()


def contact_sheet(profile_df, overlays_dir, output_path):
    if profile_df.empty:
        return

    cols = 3
    panel_w = 500
    panel_h = 390
    img_h = 315
    rows = math.ceil(len(profile_df) / cols)

    sheet = Image.new(
        "RGB",
        (cols * panel_w, rows * panel_h),
        "white"
    )

    draw = ImageDraw.Draw(sheet)
    font = font_default()

    for idx, row in profile_df.reset_index(drop=True).iterrows():
        overlay_path = overlays_dir / row["overlay_filename"]

        if not overlay_path.exists():
            continue

        im = Image.open(overlay_path).convert("RGB")
        im.thumbnail((panel_w - 20, img_h - 10))

        col = idx % cols
        r = idx // cols

        px = col * panel_w + (panel_w - im.width) // 2
        py = r * panel_h + 5

        sheet.paste(im, (px, py))

        status = row["presence_status"]
        stress = (
            " | COLOR_NO_MARK"
            if int(row.get("is_color_no_mark_stress", 0)) == 1
            else ""
        )

        text = (
            f"{row['filename']} | {status} | "
            f"n={row['n_candidates']} | "
            f"conf={row['presence_confidence']:.2f}"
            f"{stress}"
        )

        draw.text(
            (col * panel_w + 10, r * panel_h + img_h + 8),
            text,
            fill="black",
            font=font
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

    overlays_dir = output_dir / "overlays"

    if not root.is_dir():
        raise FileNotFoundError(
            f"Pasta não encontrada: {root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("VALIDAÇÃO V3 — PRESENÇA DA ESTRUTURA GRÁFICA PRINCIPAL")
    print("=" * 84)
    print(f"Raiz: {root}")
    print(f"Saída: {output_dir}")
    print()

    inventory = root_images(root)
    canvas, color = load_optional_labels(root)

    sample = choose_sample(
        inventory,
        canvas,
        color,
        n_per_profile=int(args.sample_per_profile),
        seed=int(args.seed),
    )

    print(f"Inventário canônico: {len(inventory)}")
    print(f"Amostra de calibração: {len(sample)}")

    rows = []
    errors = []

    for idx, row in sample.iterrows():
        path = Path(row["path"])

        try:
            result = analyze_one(
                path,
                row["technique"]
            )

            present = bool(
                result["det"]["present"]
            )

            overlay = make_overlay(
                result["image"],
                row["technique"],
                result["roi"],
                result["det"],
            )

            overlay_name = (
                path.stem
                + "_presence_overlay.png"
            )

            overlay.save(
                overlays_dir / overlay_name
            )

            out = {
                "filename": row["filename"],
                "profile": row["profile"],
                "technique": row["technique"],
                "task": row["task"],
                "unit_id": row["unit_id"],
                "unit_number": row["unit_number"],
                "repeat": row["repeat"],

                "presence_status": (
                    "PRESENT"
                    if present
                    else "ABSENT_OR_UNCERTAIN"
                ),
                "presence_detected": int(present),
                "presence_confidence": float(
                    result["det"]["confidence"]
                ),
                "n_candidates": int(
                    result["n_candidates"]
                ),
                "evidence": result["det"]["evidence"],

                "roi_method": result["roi_method"],
                "roi_x1": result["roi"][0],
                "roi_y1": result["roi"][1],
                "roi_x2": result["roi"][2],
                "roi_y2": result["roi"][3],

                "is_color_no_mark_stress": int(
                    row.get(
                        "is_color_no_mark_stress",
                        0
                    )
                ),

                "canvas_status": (
                    row.get("canvas_status", "")
                    if pd.notna(
                        row.get("canvas_status", "")
                    )
                    else ""
                ),

                "color_extraction_status": (
                    row.get("extraction_status", "")
                    if pd.notna(
                        row.get("extraction_status", "")
                    )
                    else ""
                ),

                "color_status_final": (
                    row.get("color_status_final", "")
                    if pd.notna(
                        row.get("color_status_final", "")
                    )
                    else ""
                ),

                "overlay_filename": overlay_name,
            }

            if row["technique"] == "S":
                out["scatter_x_dispersion"] = (
                    result["det"]["x_dispersion"]
                )
                out["scatter_y_dispersion"] = (
                    result["det"]["y_dispersion"]
                )
            else:
                out["scatter_x_dispersion"] = np.nan
                out["scatter_y_dispersion"] = np.nan

            rows.append(out)

        except Exception as exc:
            errors.append({
                "filename": row["filename"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

        done = len(rows) + len(errors)

        if done % 20 == 0 or done == len(sample):
            print(
                f"Processadas: "
                f"{done}/{len(sample)}"
            )

    result_df = pd.DataFrame(rows)
    error_df = pd.DataFrame(errors)

    csv_path = (
        output_dir
        / "mark_presence_validation_v3.csv"
    )

    result_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    error_df.to_csv(
        output_dir
        / "mark_presence_validation_errors_v3.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for profile in PROFILES:
        g = result_df[
            result_df["profile"] == profile
        ]

        contact_sheet(
            g,
            overlays_dir,
            output_dir
            / f"contact_{profile}.png"
        )

    lines = [
        "VALIDAÇÃO V3 — PRESENÇA DA ESTRUTURA GRÁFICA PRINCIPAL",
        "=" * 78,
        f"Inventário disponível: {len(inventory)}",
        f"Amostra processada: {len(result_df)}",
        f"Erros: {len(error_df)}",
        "",
        "RESULTADOS DIAGNÓSTICOS:",
    ]

    for profile in PROFILES:
        g = result_df[
            result_df["profile"] == profile
        ]

        if len(g):
            lines.append(
                f"{profile}: "
                f"PRESENT={int(g['presence_detected'].sum())}/{len(g)}; "
                f"ABSENT_OR_UNCERTAIN="
                f"{int((g['presence_detected'] == 0).sum())}"
            )

    stress = result_df[
        result_df[
            "is_color_no_mark_stress"
        ] == 1
    ]

    lines.extend([
        "",
        "CASOS DE ESTRESSE COLOR_NO_MARK:",
        f"N: {len(stress)}",
    ])

    if len(stress):
        lines.append(
            "Recuperados como estrutura PRESENT: "
            f"{int(stress['presence_detected'].sum())}/{len(stress)}"
        )

        for _, r in stress.iterrows():
            lines.append(
                f"- {r['filename']}: "
                f"{r['presence_status']}; "
                f"n={int(r['n_candidates'])}; "
                f"conf={float(r['presence_confidence']):.3f}"
            )

    lines.extend([
        "",
        "IMPORTANTE:",
        "- estes resultados são diagnósticos de calibração V3;",
        "- PRESENT/ABSENT_OR_UNCERTAIN ainda não são resultados finais;",
        "- revisar visualmente os contact sheets e overlays;",
        "- o detector não usa as cores normativas de F;",
        "- a quantidade esperada de séries/marcas não é usada para forçar a decisão.",
    ])

    summary_path = (
        output_dir
        / "mark_presence_validation_summary_v3.txt"
    )

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print()
    print("Abra primeiro:")
    print(summary_path)
    print()
    print("Depois revise:")
    for profile in PROFILES:
        print(output_dir / f"contact_{profile}.png")


if __name__ == "__main__":
    main()
