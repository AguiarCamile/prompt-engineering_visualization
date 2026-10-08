# -*- coding: utf-8 -*-
r"""
axis_conformity_calibration_v3.py

CALIBRAÇÃO V3 — EIXOS X/Y: NÚCLEO GEOMÉTRICO E COR OBSERVADA
=============================================================

Objetivo
--------
Corrigir os dois problemas remanescentes da V2:

1. BI: o eixo Y ainda podia ser confundido com bordas de barras ou com uma
   linha vertical muito à esquerda.
2. A cor observada era dominada por grade/fundo (#F1F1F1), porque a V2
   selecionava uma família cromática presente em uma faixa ao redor do eixo.

A V3 procura primeiro o NÚCLEO GEOMÉTRICO da linha do eixo, usando contraste
perpendicular local. A cor é extraída SOMENTE dos pixels que constituem esse
núcleo.

Princípio metodológico
----------------------
A especificação formal F não participa da detecção:

imagem
→ região geométrica aproximada
→ núcleo da linha X/Y
→ cor observada do núcleo
→ comparação posterior com F (#4A4A4A).

O valor #4A4A4A é usado apenas no cálculo final de ΔE00.

Estratégia V3
-------------
Eixo X:
- busca linhas próximas ao limite inferior estimado;
- amplia a busca horizontal para a ESQUERDA da ROI V6;
- identifica pixels que formam uma crista/ridge horizontal:
  o pixel da linha deve diferir dos pixels imediatamente acima/abaixo;
- escolhe o candidato com maior continuidade;
- identifica o maior segmento contínuo da linha;
- usa o início desse segmento como âncora geométrica do eixo Y.

Eixo Y:
- busca apenas numa vizinhança da âncora obtida no eixo X;
- usa o mesmo princípio de contraste perpendicular, agora esquerda/direita;
- isso reduz confusão com a borda da primeira barra.

Cor:
- para cada posição longitudinal do eixo, seleciona o pixel de maior
  contraste perpendicular no núcleo;
- descarta posições sem evidência de crista;
- agrupa cores observadas por quantização RGB, sem usar F;
- a família cromática com maior suporte longitudinal é a cor observada;
- barras/linhas que cruzam o eixo ocupam apenas parte do comprimento e,
  portanto, tendem a não dominar.

Esta continua sendo uma etapa de calibração. Nenhum limiar final de
conformidade cromática é aplicado aqui.

Entrada padrão
--------------
C:\Users\Labvis\Downloads\imagens3120\imagens

Reutiliza a MESMA amostra de 60 imagens da V1:
_axis_conformity_calibration_v1\axis_calibration_manifest_v1.csv

Dependência:
visual_features_v4_plot_area_v6.py
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.color import rgb2lab, deltaE_ciede2000
except ImportError:
    raise SystemExit(
        "Instale as dependências:\n"
        "python -m pip install numpy pandas pillow opencv-python scikit-image"
    )

try:
    import visual_features_v4_plot_area_v6 as plot_v6
except Exception as exc:
    raise SystemExit(
        "Coloque visual_features_v4_plot_area_v6.py na mesma pasta.\n"
        f"{type(exc).__name__}: {exc}"
    )


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
V1_DIRNAME = "_axis_conformity_calibration_v1"
OUT_DIRNAME = "_axis_conformity_calibration_v3"
PROFILES = ("BI", "BC", "LI", "LC", "SI", "SC")

# F — SOMENTE comparação posterior
F_AXIS_HEX = "#4A4A4A"

# ------------------------------------------------------------
# Detector de crista/linha — independentes de F
# ------------------------------------------------------------

# Diferença local mínima entre o pixel candidato e sua vizinhança
# perpendicular.
RIDGE_CONTRAST_MIN = 12.0

# Pixel candidato também deve diferir minimamente do fundo global.
MIN_BG_DIFF = 8.0

# Distância, em pixels, usada para estimar a vizinhança perpendicular.
PERP_OFFSET_MIN = 3
PERP_OFFSET_MAX = 5

# Permite eixo antialiasado com 1–2 pixels.
CENTER_RADIUS = 1

# Fecha interrupções pequenas na linha.
MAX_GAP = 3

# Busca vertical do eixo X ao redor do bottom fornecido pelo V6.
X_Y_SEARCH_PX = 18

# A busca horizontal do X é expandida para a esquerda, porque o x_left do
# V6 pode cair na primeira barra em BI.
X_LEFT_EXPAND_FRAC = 0.30
X_LEFT_EXPAND_MIN_PX = 120
X_RIGHT_EXPAND_PX = 12

# O Y será procurado ao redor do início real do segmento do eixo X.
Y_ANCHOR_SEARCH_PX = 28

# Caso a âncora do X não seja suficientemente estável, há fallback amplo.
Y_FALLBACK_EXPAND_FRAC = 0.22
Y_FALLBACK_MIN_PX = 100

# Presença preliminar V3.
RIDGE_SUPPORT_MIN = 0.32
RIDGE_RUN_MIN = 0.28

# Segmento horizontal usado para ancorar Y precisa ser minimamente longo.
X_SEGMENT_MIN_FRAC = 0.25

# Cor observada
COLOR_QUANT_STEP = 8
CORE_COLOR_MIN_SUPPORT = 0.20

# Remover pontas do segmento para reduzir interseções com os spines.
END_TRIM_FRAC = 0.015

# Contact sheets
COLS = 3
PANEL_W = 530
PANEL_H = 425
IMG_H = 305


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


# ============================================================
# Utilidades básicas
# ============================================================

def estimate_background(arr):
    h, w, _ = arr.shape
    bw = max(2, min(6, max(2, w // 100), max(2, h // 100)))
    parts = [
        arr[:bw].reshape(-1, 3),
        arr[-bw:].reshape(-1, 3),
        arr[:, :bw].reshape(-1, 3),
        arr[:, -bw:].reshape(-1, 3),
    ]
    return np.median(np.concatenate(parts, axis=0), axis=0)


def rgb_dist(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sqrt(((a - b) ** 2).sum()))


def bridge(v, max_gap=MAX_GAP):
    v = np.asarray(v, dtype=bool).copy()
    n = len(v)
    i = 0
    while i < n:
        if v[i]:
            i += 1
            continue

        s = i
        while i < n and not v[i]:
            i += 1

        if s > 0 and i < n and (i - s) <= max_gap:
            v[s:i] = True

    return v


def runs(v):
    v = np.asarray(v, dtype=bool)
    out = []
    i = 0
    while i < len(v):
        if not v[i]:
            i += 1
            continue

        s = i
        while i < len(v) and v[i]:
            i += 1
        e = i - 1
        out.append((s, e, e - s + 1))

    return out


def longest_run(v):
    rr = runs(v)
    if not rr:
        return None
    return max(rr, key=lambda x: x[2])


def quantize_color(rgb):
    a = np.asarray(rgb, dtype=np.uint8)
    q = (
        (a.astype(np.int16) // COLOR_QUANT_STEP)
        * COLOR_QUANT_STEP
    ).clip(0, 255).astype(np.uint8)
    return tuple(q.tolist())


def hex_to_rgb(h):
    h = str(h).strip().lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    if rgb is None:
        return ""
    return "#{:02X}{:02X}{:02X}".format(*map(int, rgb))


def lab_one(rgb):
    a = np.array(
        [[[rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]]],
        dtype=float,
    )
    return rgb2lab(a)[0, 0]


def de00(rgb1, rgb2):
    if rgb1 is None or rgb2 is None:
        return None
    a, b = lab_one(rgb1), lab_one(rgb2)
    return float(
        deltaE_ciede2000(
            np.array([[a]], dtype=float),
            np.array([[b]], dtype=float),
        )[0, 0]
    )


# ============================================================
# Núcleo geométrico: contraste perpendicular
# ============================================================

def horizontal_core_sample(arr, bg, x, y):
    """
    Para a posição longitudinal x de uma linha horizontal em torno de y,
    escolhe o pixel central (y-1..y+1) que mais difere da vizinhança
    perpendicular acima/abaixo.

    Retorna:
      (is_core, best_rgb, contrast)
    """
    h, w, _ = arr.shape

    if x < 0 or x >= w:
        return False, None, 0.0

    candidates = []

    for yy in range(max(0, y - CENTER_RADIUS), min(h, y + CENTER_RADIUS + 1)):
        neigh = []

        for off in range(PERP_OFFSET_MIN, PERP_OFFSET_MAX + 1):
            if yy - off >= 0:
                neigh.append(arr[yy - off, x])
            if yy + off < h:
                neigh.append(arr[yy + off, x])

        if not neigh:
            continue

        ref = np.median(np.asarray(neigh, dtype=np.float32), axis=0)
        px = arr[yy, x].astype(np.float32)

        local_contrast = rgb_dist(px, ref)
        bg_diff = rgb_dist(px, bg)

        candidates.append((local_contrast, bg_diff, tuple(arr[yy, x].tolist())))

    if not candidates:
        return False, None, 0.0

    best = max(candidates, key=lambda z: z[0])

    is_core = (
        best[0] >= RIDGE_CONTRAST_MIN
        and best[1] >= MIN_BG_DIFF
    )

    return is_core, best[2], float(best[0])


def vertical_core_sample(arr, bg, x, y):
    """
    Análogo ao horizontal_core_sample, mas para uma linha vertical:
    compara o pixel central com vizinhos à esquerda/direita.
    """
    h, w, _ = arr.shape

    if y < 0 or y >= h:
        return False, None, 0.0

    candidates = []

    for xx in range(max(0, x - CENTER_RADIUS), min(w, x + CENTER_RADIUS + 1)):
        neigh = []

        for off in range(PERP_OFFSET_MIN, PERP_OFFSET_MAX + 1):
            if xx - off >= 0:
                neigh.append(arr[y, xx - off])
            if xx + off < w:
                neigh.append(arr[y, xx + off])

        if not neigh:
            continue

        ref = np.median(np.asarray(neigh, dtype=np.float32), axis=0)
        px = arr[y, xx].astype(np.float32)

        local_contrast = rgb_dist(px, ref)
        bg_diff = rgb_dist(px, bg)

        candidates.append((local_contrast, bg_diff, tuple(arr[y, xx].tolist())))

    if not candidates:
        return False, None, 0.0

    best = max(candidates, key=lambda z: z[0])

    is_core = (
        best[0] >= RIDGE_CONTRAST_MIN
        and best[1] >= MIN_BG_DIFF
    )

    return is_core, best[2], float(best[0])


def horizontal_ridge(arr, bg, y, x1, x2):
    h, w, _ = arr.shape
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))

    if x2 <= x1:
        return None

    raw = []
    rgbs = []
    contrasts = []

    for x in range(x1, x2 + 1):
        ok, rgb, c = horizontal_core_sample(arr, bg, x, int(y))
        raw.append(ok)
        rgbs.append(rgb)
        contrasts.append(c)

    raw = np.asarray(raw, dtype=bool)
    bridged = bridge(raw)

    rr = longest_run(bridged)
    span = len(raw)

    support = float(raw.mean())
    run_frac = float(rr[2] / span) if rr else 0.0

    return {
        "coord": int(y),
        "x1": x1,
        "x2": x2,
        "raw_mask": raw,
        "mask": bridged,
        "rgbs": rgbs,
        "contrasts": contrasts,
        "support": support,
        "run_frac": run_frac,
        "score": max(support, run_frac),
        "longest_run": rr,
    }


def vertical_ridge(arr, bg, x, y1, y2):
    h, w, _ = arr.shape
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))

    if y2 <= y1:
        return None

    raw = []
    rgbs = []
    contrasts = []

    for y in range(y1, y2 + 1):
        ok, rgb, c = vertical_core_sample(arr, bg, int(x), y)
        raw.append(ok)
        rgbs.append(rgb)
        contrasts.append(c)

    raw = np.asarray(raw, dtype=bool)
    bridged = bridge(raw)

    rr = longest_run(bridged)
    span = len(raw)

    support = float(raw.mean())
    run_frac = float(rr[2] / span) if rr else 0.0

    return {
        "coord": int(x),
        "y1": y1,
        "y2": y2,
        "raw_mask": raw,
        "mask": bridged,
        "rgbs": rgbs,
        "contrasts": contrasts,
        "support": support,
        "run_frac": run_frac,
        "score": max(support, run_frac),
        "longest_run": rr,
    }


def ridge_present(r):
    if r is None:
        return False
    return (
        r["support"] >= RIDGE_SUPPORT_MIN
        or r["run_frac"] >= RIDGE_RUN_MIN
    )


# ============================================================
# Escolha geométrica dos eixos
# ============================================================

def choose_x_axis(arr, bg, plot):
    h, w, _ = arr.shape

    xl = int(plot["x_left"])
    xr = int(plot["x_right"])
    yb = int(plot["y_bottom"])

    plot_w = max(1, xr - xl)

    expand_left = max(
        X_LEFT_EXPAND_MIN_PX,
        int(round(X_LEFT_EXPAND_FRAC * plot_w)),
    )

    scan_left = max(1, xl - expand_left)
    scan_right = min(w - 2, xr + X_RIGHT_EXPAND_PX)

    candidates = []

    for y in range(
        max(1, yb - X_Y_SEARCH_PX),
        min(h - 1, yb + X_Y_SEARCH_PX + 1),
    ):
        r = horizontal_ridge(arr, bg, y, scan_left, scan_right)
        if r is not None:
            candidates.append(r)

    if not candidates:
        return None

    best_score = max(r["score"] for r in candidates)

    # Candidatos próximos do melhor; em empate, o eixo X tende a ser o mais baixo.
    near = [
        r for r in candidates
        if r["score"] >= 0.85 * best_score
    ]

    chosen = max(near, key=lambda r: (r["score"], r["coord"]))

    # Dentro da linha escolhida, identificar o maior segmento contínuo.
    rr = chosen["longest_run"]

    if rr:
        s, e, length = rr
        chosen["segment_start_x"] = chosen["x1"] + s
        chosen["segment_end_x"] = chosen["x1"] + e
        chosen["segment_frac"] = length / max(1, chosen["x2"] - chosen["x1"] + 1)
    else:
        chosen["segment_start_x"] = None
        chosen["segment_end_x"] = None
        chosen["segment_frac"] = 0.0

    return chosen


def choose_y_axis(arr, bg, plot, x_axis):
    h, w, _ = arr.shape

    yt = int(plot["y_top"])
    yb = int(plot["y_bottom"])
    xl = int(plot["x_left"])
    xr = int(plot["x_right"])

    anchor = None
    anchor_source = "fallback_plot_left"

    if (
        x_axis is not None
        and x_axis.get("segment_start_x") is not None
        and x_axis.get("segment_frac", 0.0) >= X_SEGMENT_MIN_FRAC
    ):
        anchor = int(x_axis["segment_start_x"])
        anchor_source = "x_axis_segment_start"

    candidates = []

    if anchor is not None:
        xa = max(1, anchor - Y_ANCHOR_SEARCH_PX)
        xb = min(w - 2, anchor + Y_ANCHOR_SEARCH_PX)

    else:
        plot_w = max(1, xr - xl)
        expand = max(
            Y_FALLBACK_MIN_PX,
            int(round(Y_FALLBACK_EXPAND_FRAC * plot_w)),
        )
        xa = max(1, xl - expand)
        xb = min(w - 2, xl + Y_ANCHOR_SEARCH_PX)

    for x in range(xa, xb + 1):
        r = vertical_ridge(arr, bg, x, yt, yb)
        if r is not None:
            candidates.append(r)

    if not candidates:
        return None, anchor_source

    best_score = max(r["score"] for r in candidates)

    # Mantém candidatos quase tão bons quanto o melhor.
    near = [
        r for r in candidates
        if r["score"] >= 0.88 * best_score
    ]

    if anchor is not None:
        # Entre candidatos fortes, preferir proximidade à origem do eixo X.
        chosen = min(
            near,
            key=lambda r: (
                abs(r["coord"] - anchor),
                -r["score"],
            )
        )
    else:
        # Fallback: melhor continuidade; não escolher automaticamente o mais à esquerda.
        chosen = max(
            near,
            key=lambda r: (
                r["score"],
                -abs(r["coord"] - xl),
            )
        )

    return chosen, anchor_source


# ============================================================
# Cor apenas a partir do núcleo
# ============================================================

def core_color_from_ridge(ridge, trim_frac=END_TRIM_FRAC):
    if ridge is None:
        return None, 0, 0.0, 0

    raw = ridge["raw_mask"]
    rgbs = ridge["rgbs"]

    n = len(raw)
    trim = max(1, int(round(trim_frac * n)))

    lo = trim
    hi = n - trim

    if hi <= lo:
        lo, hi = 0, n

    samples = []

    for i in range(lo, hi):
        if not raw[i]:
            continue
        rgb = rgbs[i]
        if rgb is None:
            continue
        samples.append((i, tuple(map(int, rgb))))

    if not samples:
        return None, 0, 0.0, 0

    # Suporte longitudinal por família cromática.
    by_color = {}
    for pos, rgb in samples:
        q = quantize_color(rgb)
        by_color.setdefault(q, []).append((pos, rgb))

    best_q, members = max(
        by_color.items(),
        key=lambda kv: len({p for p, _ in kv[1]})
    )

    positions = {p for p, _ in members}
    support = len(positions) / max(1, hi - lo)

    cluster_rgb = np.asarray(
        [rgb for _, rgb in members],
        dtype=np.float32,
    )

    observed = tuple(
        int(round(v))
        for v in np.median(cluster_rgb, axis=0)
    )

    return observed, len(members), float(support), len(samples)


# ============================================================
# Processamento
# ============================================================

def analyze(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    bg = estimate_background(arr)

    plot = plot_v6.detect_plot_bbox(img)

    base = {
        "width": img.width,
        "height": img.height,
        "plot_status": plot["status"],
        "plot_confidence": plot["confidence"],
        "plot_x_left": plot["x_left"],
        "plot_x_right": plot["x_right"],
        "plot_y_top": plot["y_top"],
        "plot_y_bottom": plot["y_bottom"],
    }

    if any(
        plot[k] is None
        for k in ("x_left", "x_right", "y_top", "y_bottom")
    ):
        return {
            **base,
            "axis_x_evaluable": 0,
            "axis_y_evaluable": 0,
        }

    xd = choose_x_axis(arr, bg, plot)
    yd, anchor_source = choose_y_axis(arr, bg, plot, xd)

    xp = ridge_present(xd)
    yp = ridge_present(yd)

    xrgb = yrgb = None
    x_color_n = y_color_n = 0
    x_color_support = y_color_support = 0.0
    x_core_samples = y_core_samples = 0

    if xp:
        xrgb, x_color_n, x_color_support, x_core_samples = core_color_from_ridge(xd)

    if yp:
        yrgb, y_color_n, y_color_support, y_core_samples = core_color_from_ridge(yd)

    f_rgb = hex_to_rgb(F_AXIS_HEX)

    return {
        **base,

        "axis_x_evaluable": 1,
        "axis_y_evaluable": 1,

        "axis_x_present_candidate": int(xp),
        "axis_y_present_candidate": int(yp),

        "axis_x_coord": xd["coord"] if xd else "",
        "axis_y_coord": yd["coord"] if yd else "",

        "axis_x_ridge_support": xd["support"] if xd else "",
        "axis_y_ridge_support": yd["support"] if yd else "",

        "axis_x_ridge_run_frac": xd["run_frac"] if xd else "",
        "axis_y_ridge_run_frac": yd["run_frac"] if yd else "",

        "axis_x_segment_start": (
            xd.get("segment_start_x", "") if xd else ""
        ),
        "axis_x_segment_end": (
            xd.get("segment_end_x", "") if xd else ""
        ),
        "axis_x_segment_frac": (
            xd.get("segment_frac", "") if xd else ""
        ),

        "axis_y_anchor_source": anchor_source,

        "axis_x_observed_hex": rgb_to_hex(xrgb),
        "axis_y_observed_hex": rgb_to_hex(yrgb),

        "axis_x_core_color_pixels": x_color_n,
        "axis_y_core_color_pixels": y_color_n,

        "axis_x_core_color_support": x_color_support,
        "axis_y_core_color_support": y_color_support,

        "axis_x_total_core_samples": x_core_samples,
        "axis_y_total_core_samples": y_core_samples,

        "axis_x_deltaE00_vs_F": (
            de00(xrgb, f_rgb) if xrgb is not None else ""
        ),
        "axis_y_deltaE00_vs_F": (
            de00(yrgb, f_rgb) if yrgb is not None else ""
        ),
    }


# ============================================================
# Auditoria visual
# ============================================================

def font_default(size=13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_overlay(path, row, out_path):
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)

    xl = row.get("plot_x_left")
    xr = row.get("plot_x_right")
    yt = row.get("plot_y_top")
    yb = row.get("plot_y_bottom")

    valid_box = all(
        v not in ("", None) and not pd.isna(v)
        for v in (xl, xr, yt, yb)
    )

    if valid_box:
        d.rectangle(
            [int(xl), int(yt), int(xr), int(yb)],
            outline=(120, 120, 120),
            width=2,
        )

    # X realçado em vermelho.
    if row.get("axis_x_coord") not in ("", None) and not pd.isna(row.get("axis_x_coord")):
        y = int(row["axis_x_coord"])
        xs = row.get("axis_x_segment_start")
        xe = row.get("axis_x_segment_end")

        if xs not in ("", None) and xe not in ("", None) and not pd.isna(xs) and not pd.isna(xe):
            d.line(
                [(int(xs), y), (int(xe), y)],
                fill=(220, 0, 0),
                width=3,
            )

            # Âncora do Y.
            d.ellipse(
                [int(xs)-4, y-4, int(xs)+4, y+4],
                outline=(0, 150, 0),
                width=2,
            )
        elif valid_box:
            d.line(
                [(int(xl), y), (int(xr), y)],
                fill=(220, 0, 0),
                width=3,
            )

    # Y realçado em azul.
    if row.get("axis_y_coord") not in ("", None) and not pd.isna(row.get("axis_y_coord")):
        x = int(row["axis_y_coord"])
        if valid_box:
            d.line(
                [(x, int(yt)), (x, int(yb))],
                fill=(0, 80, 220),
                width=3,
            )

    # Swatches: X e Y observados.
    for idx, key in enumerate(("axis_x_observed_hex", "axis_y_observed_hex")):
        hx = str(row.get(key, ""))
        x0 = 8 + idx * 38
        if hx.startswith("#") and len(hx) == 7:
            try:
                c = hex_to_rgb(hx)
                d.rectangle(
                    [x0, 8, x0 + 30, 28],
                    fill=c,
                    outline=(0, 0, 0),
                )
            except Exception:
                pass

    im.save(out_path)


def fmt(v, nd=2):
    try:
        if pd.isna(v):
            return "-"
        return f"{float(v):.{nd}f}"
    except Exception:
        return "-"


def contact_sheet(g, root, outdir, out_path):
    if g.empty:
        return

    rows = math.ceil(len(g) / COLS)
    sheet = Image.new(
        "RGB",
        (COLS * PANEL_W, rows * PANEL_H),
        "white",
    )
    d = ImageDraw.Draw(sheet)
    f = font_default(12)

    for idx, row in g.reset_index(drop=True).iterrows():
        col = idx % COLS
        rr = idx // COLS

        x0 = col * PANEL_W
        y0 = rr * PANEL_H

        ov = outdir / "overlays" / row["filename"]
        src = ov if ov.exists() else root / row["filename"]

        im = Image.open(src).convert("RGB")
        im.thumbnail((PANEL_W - 20, IMG_H - 10))

        sheet.paste(
            im,
            (
                x0 + (PANEL_W - im.width) // 2,
                y0 + 5,
            ),
        )

        txt = (
            f"{row['filename']} | plot={row['plot_status']}\n"
            f"X={row.get('axis_x_present_candidate','-')} "
            f"ridge={fmt(row.get('axis_x_ridge_support'))}/"
            f"{fmt(row.get('axis_x_ridge_run_frac'))} "
            f"seg={fmt(row.get('axis_x_segment_frac'))} "
            f"cor={row.get('axis_x_observed_hex','')} "
            f"cs={fmt(row.get('axis_x_core_color_support'))} "
            f"ΔE={fmt(row.get('axis_x_deltaE00_vs_F'))}\n"
            f"Y={row.get('axis_y_present_candidate','-')} "
            f"ridge={fmt(row.get('axis_y_ridge_support'))}/"
            f"{fmt(row.get('axis_y_ridge_run_frac'))} "
            f"cor={row.get('axis_y_observed_hex','')} "
            f"cs={fmt(row.get('axis_y_core_color_support'))} "
            f"ΔE={fmt(row.get('axis_y_deltaE00_vs_F'))}"
        )

        d.multiline_text(
            (x0 + 10, y0 + IMG_H + 3),
            txt,
            fill="black",
            font=f,
            spacing=2,
        )

    sheet.save(out_path)


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()

    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else root / V1_DIRNAME / "axis_calibration_manifest_v1.csv"
    )

    outdir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / OUT_DIRNAME
    )

    (outdir / "overlays").mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifesto da calibração V1 não encontrado: {manifest}"
        )

    sample = pd.read_csv(manifest)

    rows = []
    errors = []

    print("=" * 80)
    print("CALIBRAÇÃO V3 — EIXOS X/Y")
    print("=" * 80)
    print(f"Mesmas imagens da V1/V2: {len(sample)}")
    print(f"F={F_AXIS_HEX} usado somente na comparação posterior")
    print()

    for _, r in sample.iterrows():
        path = root / r["filename"]

        try:
            res = analyze(path)

            row = {
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "repeat": r["repeat"],
                "F_axis_hex": F_AXIS_HEX,
                **res,
            }

            rows.append(row)

            make_overlay(
                path,
                row,
                outdir / "overlays" / r["filename"],
            )

        except Exception as exc:
            errors.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

        done = len(rows) + len(errors)
        if done % 10 == 0 or done == len(sample):
            print(f"Processadas: {done}/{len(sample)}")

    df = pd.DataFrame(rows)
    err = pd.DataFrame(errors)

    df.to_csv(
        outdir / "axis_calibration_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    err.to_csv(
        outdir / "axis_calibration_errors_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for p in PROFILES:
        contact_sheet(
            df[df["profile"] == p],
            root,
            outdir,
            outdir / f"contact_{p}.png",
        )

    lines = [
        "CALIBRAÇÃO V3 — EIXOS X/Y",
        "=" * 78,
        f"Processadas: {len(df)}",
        f"Erros: {len(err)}",
        f"F_axis_hex (comparação posterior): {F_AXIS_HEX}",
        "",
        "POR PERFIL:",
    ]

    for p in PROFILES:
        g = df[df["profile"] == p]
        if g.empty:
            continue

        xp = pd.to_numeric(
            g["axis_x_present_candidate"],
            errors="coerce",
        ).fillna(0)

        yp = pd.to_numeric(
            g["axis_y_present_candidate"],
            errors="coerce",
        ).fillna(0)

        xcs = pd.to_numeric(
            g["axis_x_core_color_support"],
            errors="coerce",
        ).dropna()

        ycs = pd.to_numeric(
            g["axis_y_core_color_support"],
            errors="coerce",
        ).dropna()

        xde = pd.to_numeric(
            g["axis_x_deltaE00_vs_F"],
            errors="coerce",
        ).dropna()

        yde = pd.to_numeric(
            g["axis_y_deltaE00_vs_F"],
            errors="coerce",
        ).dropna()

        lines.append(
            f"{p}: "
            f"X={int(xp.sum())}/{len(g)}; "
            f"Y={int(yp.sum())}/{len(g)}; "
            f"core-color-support med X="
            f"{(xcs.median() if len(xcs) else float('nan')):.3f}; "
            f"Y={(ycs.median() if len(ycs) else float('nan')):.3f}; "
            f"ΔE med X={(xde.median() if len(xde) else float('nan')):.3f}; "
            f"Y={(yde.median() if len(yde) else float('nan')):.3f}"
        )

    lines.extend([
        "",
        "CORES OBSERVADAS MAIS FREQUENTES:",
    ])

    for axis in ("x", "y"):
        col = f"axis_{axis}_observed_hex"
        vc = (
            df[col]
            .fillna("")
            .astype(str)
            .loc[lambda s: s.str.startswith("#")]
            .value_counts()
            .head(10)
        )

        lines.append(f"Eixo {axis.upper()}:")
        for hx, n in vc.items():
            lines.append(f"  {hx}: {n}")

    lines.extend([
        "",
        "AUDITORIA VISUAL:",
        "- vermelho = segmento do núcleo do eixo X;",
        "- azul = candidato ao eixo Y;",
        "- círculo verde = início do segmento X usado como âncora para Y;",
        "- quadrados no canto superior esquerdo = cores observadas X e Y;",
        "- verificar BI com atenção especial;",
        "- verificar se as cores deixam de acompanhar grade/barras/marcas;",
        "- NÃO aplicar ainda limiar final de ΔE.",
    ])

    summary = outdir / "axis_calibration_summary_v3.txt"
    summary.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("Concluído.")
    print("Resumo:", summary)
    print("CSV:", outdir / "axis_calibration_v3.csv")
    print("Envie o resumo e os seis contact sheets.")


if __name__ == "__main__":
    main()
