# -*- coding: utf-8 -*-
r"""
B-A-CAT V1 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
================================================

Objetivo
--------
Desenvolver um módulo específico para extração de tick labels categóricos
em gráficos de barras BI/BC, preservando integralmente a B-A V7 congelada
para os componentes já validados.

Princípios metodológicos
------------------------
1. NÃO usa a especificação F.
2. NÃO recebe categorias esperadas, número esperado de categorias ou ordem esperada.
3. NÃO usa imagens do holdout independente B-A V7.
4. Usa SOMENTE as imagens BI/BC já pertencentes à calibração B-A V7.
5. A V7 congelada é importada apenas como infraestrutura auxiliar e para
   recuperar a segmentação/caixas já produzidas; seus arquivos não são alterados.
6. A orientação observada é inferida a partir da própria imagem:
      VERTICAL   -> X categórico, Y numérico
      HORIZONTAL -> X numérico, Y categórico
      AMBIGUOUS  -> não força a orientação.
7. O categórico é reconstruído por slots espaciais e múltiplas evidências OCR.
8. Falta de evidência suficiente produz AMBIGUOUS/UNREADABLE, nunca um falso
   rótulo "corrigido" por conhecimento de F.

Saídas principais
-----------------
ba_cat_v1_manifest.csv
ba_cat_v1_images.csv
ba_cat_v1_slots.csv
ba_cat_v1_candidates.csv
ba_cat_v1_audit_all.csv
ba_cat_v1_priority_review.csv
ba_cat_v1_errors.csv
ba_cat_v1_summary.txt
contact_BA_CAT_V1_BI.png
contact_BA_CAT_V1_BC.png
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageDraw, ImageFont

import ticklabel_content_extraction_calibration_ba_v7 as ba_v7


# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------

DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
OUT_DIRNAME = "_ticklabel_categorical_extraction_ba_cat_v1"

EXPECTED_V7_SHA256 = "21423b59ce40eb351693f00b6a197936d936fe53131467f47c1031da864fd688"
EXPECTED_PRESENCE_V4_SHA256 = "d15c325290c7596632f9f4c62d907e7a78d0a794862b690c4c4691ed87ea86b2"

PROFILES = ("BI", "BC")

# Parâmetros de desenvolvimento. A versão ainda NÃO está congelada.
OCR_SCALE = 3.0
OCR_ROTATIONS = (0, -45, 45, -90, 90)
OCR_PSMS = (6, 7, 11, 13)
OCR_PREPROCESS = ("gray", "otsu", "adaptive")

# Consenso entre leituras independentes da MESMA imagem.
SIMILARITY_THRESHOLD = 0.80
CONSENSUS_MIN_SUPPORT = 2
SINGLE_HIGH_CONF = 88.0
AMBIGUITY_SCORE_MARGIN = 5.0

# Lane categórica: região imediatamente adjacente ao plot.
X_LANE_FRAC = 0.22
Y_LANE_FRAC = 0.26

# Heurística de orientação.
KIND_SCORE_MIN = 0.58
ORIENTATION_MARGIN = 0.12

# Geometria de barras.
BAR_MIN_AREA_FRAC = 0.002
BAR_MAX_AREA_FRAC = 0.40
BAR_ASPECT_MIN = 1.25


# ---------------------------------------------------------------------------
# UTILITÁRIOS
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--calibration-manifest", type=Path, default=None)
    p.add_argument("--tesseract", type=Path, default=None)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_dependencies():
    v7_path = Path(ba_v7.__file__).resolve()
    v7_hash = sha256(v7_path)
    if v7_hash != EXPECTED_V7_SHA256:
        raise RuntimeError(
            "A B-A V7 não corresponde à versão congelada.\n"
            f"Arquivo: {v7_path}\n"
            f"SHA esperado: {EXPECTED_V7_SHA256}\n"
            f"SHA encontrado: {v7_hash}\n"
            "Não execute a B-A-CAT V1 sobre uma V7 modificada."
        )

    presence_path = Path(ba_v7.v4.__file__).resolve()
    presence_hash = sha256(presence_path)
    if presence_hash != EXPECTED_PRESENCE_V4_SHA256:
        raise RuntimeError(
            "A V4 de presença não corresponde à versão congelada.\n"
            f"Arquivo: {presence_path}\n"
            f"SHA esperado: {EXPECTED_PRESENCE_V4_SHA256}\n"
            f"SHA encontrado: {presence_hash}"
        )

    ba_v7.check_v4_hash()
    return v7_path, v7_hash, presence_path, presence_hash


def find_calibration_manifest(root: Path, explicit=None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manifesto de calibração não encontrado: {p}")
        return p

    p = root / "_ticklabel_content_extraction_ba_v7" / "ticklabel_ba_v7_manifest.csv"
    if not p.exists():
        raise FileNotFoundError(
            "Manifesto da calibração B-A V7 não encontrado.\n"
            f"Esperado: {p}\n"
            "Use --calibration-manifest se estiver em outro local."
        )
    return p.resolve()


def load_calibration_bars(cal_path: Path, inv: pd.DataFrame) -> pd.DataFrame:
    cal = pd.read_csv(cal_path, dtype=str)
    if "filename" not in cal.columns:
        raise RuntimeError("Manifesto da B-A V7 não contém a coluna filename.")

    names = cal[["filename"]].drop_duplicates()
    inv2 = inv.copy()
    inv2["filename"] = inv2["filename"].astype(str)

    cols = [c for c in
            ["filename", "profile", "unit_id", "unit_number", "repeat"]
            if c in inv2.columns]

    m = names.merge(inv2[cols], on="filename", how="left", validate="one_to_one")
    if m["profile"].isna().any():
        miss = m.loc[m["profile"].isna(), "filename"].tolist()
        raise RuntimeError("Imagens da calibração ausentes no inventário: " + ", ".join(miss[:20]))

    bars = m[m["profile"].isin(PROFILES)].copy()
    if len(bars) != 24:
        raise RuntimeError(
            f"Calibração BI/BC contém {len(bars)} imagens; esperado=24 "
            "(12 BI + 12 BC)."
        )
    counts = bars["profile"].value_counts().to_dict()
    if counts.get("BI", 0) != 12 or counts.get("BC", 0) != 12:
        raise RuntimeError(f"Balanceamento BI/BC inesperado: {counts}")

    return bars.sort_values(["profile", "unit_id", "repeat"]).reset_index(drop=True)


def normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^A-Z0-9/%+\-.,() ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def alpha_ratio(s: str) -> float:
    s = normalize_text(s)
    chars = [c for c in s if c.isalnum()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


def numeric_ratio(s: str) -> float:
    s = normalize_text(s)
    chars = [c for c in s if c.isalnum()]
    if not chars:
        return 0.0
    return sum(c.isdigit() for c in chars) / len(chars)


def bbox_from_row(row):
    try:
        return (
            int(float(row["bbox_x1"])), int(float(row["bbox_y1"])),
            int(float(row["bbox_x2"])), int(float(row["bbox_y2"]))
        )
    except Exception:
        return None


def safe_get(obj, *names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _bbox_candidate(v):
    if v is None:
        return None
    if isinstance(v, str):
        parts = re.split(r"[|,; ]+", v.strip())
        if len(parts) >= 4:
            try:
                return tuple(int(float(x)) for x in parts[:4])
            except Exception:
                return None
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 4:
        try:
            return tuple(int(float(x)) for x in v[:4])
        except Exception:
            return None
    if isinstance(v, dict):
        keysets = [
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
            ("plot_x1", "plot_y1", "plot_x2", "plot_y2"),
        ]
        for ks in keysets:
            if all(k in v for k in ks):
                try:
                    return tuple(int(float(v[k])) for k in ks)
                except Exception:
                    pass
    return None


def extract_plot_bbox(pres, axes, w: int, h: int):
    """
    Tenta usar a área de plotagem já detectada pelos módulos congelados.
    Se a estrutura concreta do objeto mudar, usa fallback geométrico
    conservador e registra isso na saída.
    """
    candidates = []
    for obj in (pres, axes):
        for name in (
            "plot_bbox", "plot_area_bbox", "plot_box", "bbox_plot",
            "plot_rect", "plot_area", "plot"
        ):
            v = safe_get(obj, name)
            b = _bbox_candidate(v)
            if b:
                candidates.append(b)

        # Alguns módulos podem devolver chaves x1/y1/x2/y2 diretamente.
        if isinstance(obj, dict):
            for ks in [
                ("plot_x1", "plot_y1", "plot_x2", "plot_y2"),
                ("x1", "y1", "x2", "y2"),
            ]:
                if all(k in obj for k in ks):
                    try:
                        candidates.append(tuple(int(float(obj[k])) for k in ks))
                    except Exception:
                        pass

    for x1, y1, x2, y2 in candidates:
        x1 = max(0, min(w - 2, x1))
        y1 = max(0, min(h - 2, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        if (x2 - x1) > 0.35 * w and (y2 - y1) > 0.30 * h:
            return (x1, y1, x2, y2), "FROZEN_DETECTOR"

    # Fallback apenas operacional; deve aparecer no audit.
    return (
        int(round(0.10 * w)),
        int(round(0.08 * h)),
        int(round(0.96 * w)),
        int(round(0.82 * h)),
    ), "FALLBACK_GEOMETRIC"


def categorical_lane_bbox(plot_bbox, axis, w, h):
    x1, y1, x2, y2 = plot_bbox
    pw, ph = x2 - x1, y2 - y1

    if axis == "X":
        # Faixa logo abaixo do plot, evitando o rodapé mais distante.
        ly1 = max(0, y2 - int(0.015 * ph))
        ly2 = min(h, y2 + int(X_LANE_FRAC * h))
        return (max(0, x1 - int(0.02 * pw)), ly1,
                min(w, x2 + int(0.02 * pw)), ly2)

    lx1 = max(0, x1 - int(Y_LANE_FRAC * w))
    lx2 = min(w, x1 + int(0.015 * pw))
    return (lx1, max(0, y1 - int(0.02 * ph)),
            lx2, min(h, y2 + int(0.02 * ph)))


def crop_pil(img: Image.Image, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return img.crop((x1, y1, x2, y2))


def preprocess_variant(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "gray":
        return gray
    if mode == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 15
        )
    raise ValueError(mode)


def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return img
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))
    M[0, 2] += (nw / 2) - center[0]
    M[1, 2] += (nh / 2) - center[1]
    return cv2.warpAffine(
        img, M, (nw, nh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )


def ocr_once(img_arr: np.ndarray, lang: str, psm: int):
    cfg = f"--psm {psm}"
    d = pytesseract.image_to_data(
        img_arr, lang=lang, config=cfg,
        output_type=pytesseract.Output.DATAFRAME
    )
    if d is None or len(d) == 0:
        return "", -1.0, []

    d = d.copy()
    d["text"] = d["text"].fillna("").astype(str)
    d["conf"] = pd.to_numeric(d["conf"], errors="coerce").fillna(-1)

    good = d[(d["text"].str.strip() != "") & (d["conf"] >= 0)].copy()
    if good.empty:
        return "", -1.0, []

    tokens = good["text"].str.strip().tolist()
    confs = good["conf"].astype(float).tolist()
    text = " ".join(tokens).strip()
    conf = float(np.average(confs)) if confs else -1.0
    return text, conf, tokens


def run_ocr_variants(crop: Image.Image, lang: str, source: str):
    rgb = np.asarray(crop.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Upscale antes dos preprocessamentos.
    gray = cv2.resize(
        gray, None, fx=OCR_SCALE, fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC
    )

    rows = []
    for prep in OCR_PREPROCESS:
        base = preprocess_variant(gray, prep)
        for angle in OCR_ROTATIONS:
            rotated = rotate_bound(base, angle)
            for psm in OCR_PSMS:
                raw, conf, tokens = ocr_once(rotated, lang, psm)
                norm = normalize_text(raw)
                if not norm:
                    continue

                ar = alpha_ratio(norm)
                nr = numeric_ratio(norm)
                # O score NÃO conhece F; usa apenas evidência OCR/lexical.
                lexical = min(1.0, len([c for c in norm if c.isalpha()]) / 12.0)
                score = (
                    max(0.0, conf) * 0.62
                    + 18.0 * ar
                    + 8.0 * lexical
                    - 10.0 * nr
                )
                rows.append({
                    "source": source,
                    "preprocess": prep,
                    "rotation": angle,
                    "psm": psm,
                    "ocr_raw": raw,
                    "ocr_normalized": norm,
                    "ocr_confidence": conf,
                    "alpha_ratio": ar,
                    "numeric_ratio": nr,
                    "base_score": score,
                    "tokens": " | ".join(tokens),
                })
    return rows


def lane_kind_evidence(img: Image.Image, lane_bbox, lang: str):
    crop = crop_pil(img, lane_bbox)
    # Para natureza do eixo, usamos leitura ampla e barata.
    candidates = run_ocr_variants(crop, lang, "LANE_KIND")
    if not candidates:
        return {
            "alpha_score": 0.0, "numeric_score": 0.0,
            "best_text": "", "best_confidence": -1.0
        }

    # Melhor evidência alfabética e numérica separadamente.
    best_alpha = max(
        candidates,
        key=lambda r: (r["alpha_ratio"] * max(0.0, r["ocr_confidence"]), r["base_score"])
    )
    best_num = max(
        candidates,
        key=lambda r: (r["numeric_ratio"] * max(0.0, r["ocr_confidence"]), r["base_score"])
    )

    a = min(1.0, 0.65 * best_alpha["alpha_ratio"] + 0.35 * max(0.0, best_alpha["ocr_confidence"]) / 100.0)
    n = min(1.0, 0.65 * best_num["numeric_ratio"] + 0.35 * max(0.0, best_num["ocr_confidence"]) / 100.0)

    return {
        "alpha_score": float(a),
        "numeric_score": float(n),
        "best_text": best_alpha["ocr_normalized"],
        "best_confidence": float(best_alpha["ocr_confidence"]),
    }


def bar_geometry_orientation(img: Image.Image, plot_bbox):
    """
    Evidência auxiliar: procura componentes cromaticamente diferentes do fundo.
    Não decide sozinho a orientação.
    """
    x1, y1, x2, y2 = plot_bbox
    arr = np.asarray(img.convert("RGB"))
    roi = arr[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return {"vertical_score": 0.0, "horizontal_score": 0.0,
                "vertical_centers": [], "horizontal_centers": []}

    h, w = roi.shape[:2]
    border = np.concatenate([
        roi[:max(1, h//30), :, :].reshape(-1, 3),
        roi[-max(1, h//30):, :, :].reshape(-1, 3),
        roi[:, :max(1, w//30), :].reshape(-1, 3),
        roi[:, -max(1, w//30):, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(roi.astype(np.float32) - bg.astype(np.float32), axis=2)
    mask = (dist > 28).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    plot_area = float(max(1, w * h))
    vertical, horizontal = [], []

    for i in range(1, n):
        xx, yy, ww, hh, area = stats[i]
        frac = area / plot_area
        if frac < BAR_MIN_AREA_FRAC or frac > BAR_MAX_AREA_FRAC:
            continue
        if ww <= 2 or hh <= 2:
            continue

        aspect_v = hh / max(1.0, ww)
        aspect_h = ww / max(1.0, hh)

        cx = x1 + xx + ww / 2.0
        cy = y1 + yy + hh / 2.0

        if aspect_v >= BAR_ASPECT_MIN and hh >= 0.08 * h:
            vertical.append(cx)
        if aspect_h >= BAR_ASPECT_MIN and ww >= 0.08 * w:
            horizontal.append(cy)

    def merge_centers(vals, tol):
        vals = sorted(float(v) for v in vals)
        if not vals:
            return []
        groups = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - np.mean(groups[-1])) <= tol:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [float(np.mean(g)) for g in groups]

    vertical = merge_centers(vertical, max(8.0, 0.035 * w))
    horizontal = merge_centers(horizontal, max(8.0, 0.035 * h))

    # 3–8 barras é faixa plausível sem conhecer o número esperado.
    def plausibility(k):
        if 2 <= k <= 8:
            return min(1.0, 0.45 + 0.10 * k)
        if k == 1:
            return 0.25
        return 0.0

    return {
        "vertical_score": plausibility(len(vertical)),
        "horizontal_score": plausibility(len(horizontal)),
        "vertical_centers": vertical,
        "horizontal_centers": horizontal,
    }


def infer_orientation(img, plot_bbox, lang):
    x_lane = categorical_lane_bbox(plot_bbox, "X", img.width, img.height)
    y_lane = categorical_lane_bbox(plot_bbox, "Y", img.width, img.height)

    xe = lane_kind_evidence(img, x_lane, lang)
    ye = lane_kind_evidence(img, y_lane, lang)
    geom = bar_geometry_orientation(img, plot_bbox)

    # H1: barras verticais -> X categórico, Y numérico.
    vertical = (
        0.42 * xe["alpha_score"]
        + 0.28 * ye["numeric_score"]
        + 0.20 * geom["vertical_score"]
        + 0.10 * (1.0 - min(1.0, xe["numeric_score"]))
    )

    # H2: barras horizontais -> X numérico, Y categórico.
    horizontal = (
        0.42 * ye["alpha_score"]
        + 0.28 * xe["numeric_score"]
        + 0.20 * geom["horizontal_score"]
        + 0.10 * (1.0 - min(1.0, ye["numeric_score"]))
    )

    diff = vertical - horizontal
    if max(vertical, horizontal) < KIND_SCORE_MIN or abs(diff) < ORIENTATION_MARGIN:
        orientation = "AMBIGUOUS"
        categorical_axis = ""
    elif diff > 0:
        orientation = "VERTICAL"
        categorical_axis = "X"
    else:
        orientation = "HORIZONTAL"
        categorical_axis = "Y"

    return {
        "orientation": orientation,
        "categorical_axis": categorical_axis,
        "vertical_score": float(vertical),
        "horizontal_score": float(horizontal),
        "x_alpha_score": xe["alpha_score"],
        "x_numeric_score": xe["numeric_score"],
        "y_alpha_score": ye["alpha_score"],
        "y_numeric_score": ye["numeric_score"],
        "x_lane_best_text": xe["best_text"],
        "y_lane_best_text": ye["best_text"],
        "vertical_bar_centers": geom["vertical_centers"],
        "horizontal_bar_centers": geom["horizontal_centers"],
        "x_lane_bbox": x_lane,
        "y_lane_bbox": y_lane,
    }


def run_v7_axis_boxes(full_img, pres, axes, profile, axis, lang, crop_dir, filename):
    """
    Reutiliza a segmentação congelada da V7 somente para obter caixas espaciais.
    O conteúdo categórico final NÃO é o candidate_text da V7.
    """
    ar, lrs, svrs, ncrs = ba_v7.axis_extraction(
        full_img, pres, axes, profile, axis, lang,
        crop_dir, filename
    )
    return ar, lrs


def centers_from_v7_labels(label_rows, axis):
    vals = []
    boxes = []
    for r in label_rows:
        b = bbox_from_row(r)
        if not b:
            continue
        x1, y1, x2, y2 = b
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(b)
        vals.append((x1 + x2) / 2.0 if axis == "X" else (y1 + y2) / 2.0)
    return vals, boxes


def merge_slot_centers(primary, secondary, span, tol_frac=0.045):
    vals = sorted([float(x) for x in primary] + [float(x) for x in secondary])
    if not vals:
        return []
    tol = max(8.0, tol_frac * span)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - np.mean(groups[-1])) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def slot_bboxes_from_centers(centers, lane_bbox, axis):
    if not centers:
        return []
    centers = sorted(centers)
    x1, y1, x2, y2 = lane_bbox

    bounds = []
    for i, c in enumerate(centers):
        prev_c = centers[i - 1] if i > 0 else None
        next_c = centers[i + 1] if i + 1 < len(centers) else None

        if axis == "X":
            left = x1 if prev_c is None else int(round((prev_c + c) / 2))
            right = x2 if next_c is None else int(round((c + next_c) / 2))
            # Limita slot excessivamente largo.
            half = max(28, int(0.58 * (right - left)))
            left = max(x1, int(c - half))
            right = min(x2, int(c + half))
            bounds.append((left, y1, right, y2))
        else:
            top = y1 if prev_c is None else int(round((prev_c + c) / 2))
            bottom = y2 if next_c is None else int(round((c + next_c) / 2))
            half = max(22, int(0.58 * (bottom - top)))
            top = max(y1, int(c - half))
            bottom = min(y2, int(c + half))
            bounds.append((x1, top, x2, bottom))
    return bounds


def overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area


def expand_bbox(b, w, h, frac=0.18):
    x1, y1, x2, y2 = b
    dx = int((x2 - x1) * frac)
    dy = int((y2 - y1) * frac)
    return (
        max(0, x1 - dx), max(0, y1 - dy),
        min(w, x2 + dx), min(h, y2 + dy)
    )


def cluster_candidates(rows):
    """
    Agrupa somente leituras produzidas da MESMA imagem/slot.
    Não há vocabulário externo nem comparação com F.
    """
    if not rows:
        return [], None

    # Remove duplicatas exatas da mesma configuração.
    clean = []
    seen = set()
    for r in rows:
        text = normalize_text(r.get("ocr_normalized", ""))
        if len(re.sub(r"[^A-Z]", "", text)) < 2:
            continue
        key = (
            r.get("source"), r.get("preprocess"),
            r.get("rotation"), r.get("psm"), text
        )
        if key in seen:
            continue
        seen.add(key)
        rr = dict(r)
        rr["ocr_normalized"] = text
        clean.append(rr)

    clusters = []
    for r in sorted(clean, key=lambda x: x["base_score"], reverse=True):
        t = r["ocr_normalized"]
        placed = False
        for c in clusters:
            sim = SequenceMatcher(None, t, c["representative"]).ratio()
            if sim >= SIMILARITY_THRESHOLD:
                c["members"].append(r)
                # Mantém como representante o membro de maior base_score.
                best = max(c["members"], key=lambda x: x["base_score"])
                c["representative"] = best["ocr_normalized"]
                placed = True
                break
        if not placed:
            clusters.append({"representative": t, "members": [r]})

    for c in clusters:
        members = c["members"]
        sources = set(m["source"] for m in members)
        variants = set(
            (m["preprocess"], m["rotation"], m["psm"])
            for m in members
        )
        confs = [max(0.0, float(m["ocr_confidence"])) for m in members]
        base = [float(m["base_score"]) for m in members]
        support = len(variants)
        source_support = len(sources)
        rep = c["representative"]

        # Consenso independente: recorrência + confiança + qualidade lexical.
        consensus_bonus = min(22.0, 4.0 * support + 3.0 * source_support)
        c["support"] = support
        c["source_support"] = source_support
        c["mean_confidence"] = float(np.mean(confs)) if confs else 0.0
        c["max_confidence"] = float(np.max(confs)) if confs else 0.0
        c["score"] = float(np.max(base) + consensus_bonus)
        c["alpha_ratio"] = alpha_ratio(rep)

    clusters.sort(key=lambda c: c["score"], reverse=True)
    best = clusters[0] if clusters else None
    return clusters, best


def select_slot(rows):
    clusters, best = cluster_candidates(rows)
    if not best:
        return {
            "status": "UNREADABLE", "selected_text": "",
            "score": 0.0, "support": 0, "confidence": -1.0,
            "clusters": clusters,
        }

    second = clusters[1] if len(clusters) > 1 else None
    margin = best["score"] - second["score"] if second else math.inf

    if best["support"] >= CONSENSUS_MIN_SUPPORT and best["alpha_ratio"] >= 0.55:
        if second is not None and margin < AMBIGUITY_SCORE_MARGIN:
            status = "AMBIGUOUS"
        else:
            status = "CONSENSUS"
    elif best["max_confidence"] >= SINGLE_HIGH_CONF and best["alpha_ratio"] >= 0.65:
        if second is not None and margin < AMBIGUITY_SCORE_MARGIN:
            status = "AMBIGUOUS"
        else:
            status = "SINGLE_HIGH_CONF"
    else:
        status = "UNREADABLE"

    selected = best["representative"] if status in ("CONSENSUS", "SINGLE_HIGH_CONF") else ""

    return {
        "status": status,
        "selected_text": selected,
        "score": best["score"],
        "support": best["support"],
        "confidence": best["max_confidence"],
        "clusters": clusters,
    }


def slot_candidates(full_img, slot_bbox, v7_boxes, lang):
    rows = []

    # Evidência 1: slot geométrico.
    slot_crop = crop_pil(full_img, slot_bbox)
    rows.extend(run_ocr_variants(slot_crop, lang, "SLOT"))

    # Evidência 2: caixas V7 sobrepostas ao slot.
    for j, b in enumerate(v7_boxes, start=1):
        if overlap_ratio(b, slot_bbox) < 0.18 and overlap_ratio(slot_bbox, b) < 0.18:
            continue

        c = crop_pil(full_img, b)
        rows.extend(run_ocr_variants(c, lang, f"V7_BOX_{j}"))

        eb = expand_bbox(b, full_img.width, full_img.height, frac=0.22)
        ec = crop_pil(full_img, eb)
        rows.extend(run_ocr_variants(ec, lang, f"V7_BOX_EXPANDED_{j}"))

    return rows


def axis_status(slot_results):
    if not slot_results:
        return "CATEGORICAL_FAILED"

    st = [r["selection_status"] for r in slot_results]
    if all(x in ("CONSENSUS", "SINGLE_HIGH_CONF") for x in st):
        return "CATEGORICAL_COMPLETE"
    if any(x == "AMBIGUOUS" for x in st):
        return "CATEGORICAL_AMBIGUOUS"
    if any(x == "UNREADABLE" for x in st):
        if any(x in ("CONSENSUS", "SINGLE_HIGH_CONF") for x in st):
            return "CATEGORICAL_PARTIAL"
        return "CATEGORICAL_FAILED"
    return "CATEGORICAL_PARTIAL"


def bbox_to_text(b):
    return "|".join(str(int(v)) for v in b)


def draw_overlay(img, plot_bbox, lane_bbox, slot_results, out_path, title):
    canvas = img.convert("RGB").copy()
    d = ImageDraw.Draw(canvas)

    # Sem cores especificadas manualmente: usa contornos padrão simples em escala de cinza.
    d.rectangle(plot_bbox, outline=(40, 40, 40), width=2)
    d.rectangle(lane_bbox, outline=(110, 110, 110), width=2)

    for r in slot_results:
        b = r["slot_bbox"]
        d.rectangle(b, outline=(0, 0, 0), width=2)
        label = f"{r['slot_index']}:{r['selection_status']}"
        d.text((b[0] + 2, b[1] + 2), label, fill=(0, 0, 0))

    d.text((8, 8), title, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def process_image(path, profile, lang, outdir):
    img = Image.open(path).convert("RGB")
    pres = ba_v7.v4.analyze(path)
    axes = ba_v7.v4.axis_v3.analyze(path)

    plot_bbox, plot_source = extract_plot_bbox(
        pres, axes, img.width, img.height
    )

    orientation = infer_orientation(img, plot_bbox, lang)
    cat_axis = orientation["categorical_axis"]

    # A V7 é executada para os dois eixos apenas para aproveitar as caixas
    # espaciais congeladas. Seu candidate_text não é usado como decisão final.
    v7_crop_dir = outdir / "v7_aux_crops"
    ar_x, lrs_x = run_v7_axis_boxes(
        img, pres, axes, profile, "X", lang, v7_crop_dir, path.name
    )
    ar_y, lrs_y = run_v7_axis_boxes(
        img, pres, axes, profile, "Y", lang, v7_crop_dir, path.name
    )

    v7_by_axis = {"X": lrs_x, "Y": lrs_y}

    image_row = {
        "filename": path.name,
        "profile": profile,
        "plot_bbox": bbox_to_text(plot_bbox),
        "plot_bbox_source": plot_source,
        "bar_orientation_observed": orientation["orientation"],
        "categorical_axis_observed": cat_axis,
        "orientation_vertical_score": orientation["vertical_score"],
        "orientation_horizontal_score": orientation["horizontal_score"],
        "observed_x_alpha_score": orientation["x_alpha_score"],
        "observed_x_numeric_score": orientation["x_numeric_score"],
        "observed_y_alpha_score": orientation["y_alpha_score"],
        "observed_y_numeric_score": orientation["y_numeric_score"],
        "x_lane_best_text": orientation["x_lane_best_text"],
        "y_lane_best_text": orientation["y_lane_best_text"],
    }

    if not cat_axis:
        image_row.update({
            "categorical_axis_status": "ORIENTATION_AMBIGUOUS",
            "categorical_n_slots": 0,
            "categorical_selected_labels_spatial": "",
        })
        return image_row, [], []

    lane_bbox = orientation["x_lane_bbox"] if cat_axis == "X" else orientation["y_lane_bbox"]
    v7_centers, v7_boxes = centers_from_v7_labels(v7_by_axis[cat_axis], cat_axis)

    if cat_axis == "X":
        bar_centers = orientation["vertical_bar_centers"]
        span = plot_bbox[2] - plot_bbox[0]
    else:
        bar_centers = orientation["horizontal_bar_centers"]
        span = plot_bbox[3] - plot_bbox[1]

    centers = merge_slot_centers(v7_centers, bar_centers, span)

    # Se a geometria não der slots, ainda tentamos usar as caixas V7 existentes.
    if not centers:
        centers = sorted(v7_centers)

    slots = slot_bboxes_from_centers(centers, lane_bbox, cat_axis)

    slot_rows = []
    cand_rows = []

    for idx, b in enumerate(slots, start=1):
        rows = slot_candidates(img, b, v7_boxes, lang)
        sel = select_slot(rows)

        for c in rows:
            cand_rows.append({
                "filename": path.name,
                "profile": profile,
                "categorical_axis_observed": cat_axis,
                "slot_index": idx,
                "slot_bbox": bbox_to_text(b),
                **{k: v for k, v in c.items() if k != "tokens"},
                "tokens": c.get("tokens", ""),
            })

        slot_rows.append({
            "filename": path.name,
            "profile": profile,
            "categorical_axis_observed": cat_axis,
            "slot_index": idx,
            "slot_center": centers[idx - 1],
            "slot_bbox": bbox_to_text(b),
            "selection_status": sel["status"],
            "categorical_selected_text": sel["selected_text"],
            "categorical_selection_score": sel["score"],
            "categorical_selection_support": sel["support"],
            "categorical_selection_confidence": sel["confidence"],
            "n_raw_candidates": len(rows),
        })

    status = axis_status(slot_rows)
    selected_spatial = " | ".join(
        (r["categorical_selected_text"]
         if r["categorical_selected_text"]
         else f"<{r['selection_status']}>")
        for r in slot_rows
    )

    image_row.update({
        "categorical_lane_bbox": bbox_to_text(lane_bbox),
        "categorical_axis_status": status,
        "categorical_n_slots": len(slot_rows),
        "categorical_selected_labels_spatial": selected_spatial,
        "categorical_n_selected": sum(
            r["selection_status"] in ("CONSENSUS", "SINGLE_HIGH_CONF")
            for r in slot_rows
        ),
        "categorical_n_ambiguous": sum(
            r["selection_status"] == "AMBIGUOUS" for r in slot_rows
        ),
        "categorical_n_unreadable": sum(
            r["selection_status"] == "UNREADABLE" for r in slot_rows
        ),
    })

    overlay_path = (
        outdir / "overlays" / profile /
        f"{path.stem}_BA_CAT_V1.png"
    )
    draw_overlay(
        img, plot_bbox, lane_bbox, slot_rows, overlay_path,
        f"{path.name} | {orientation['orientation']} | {status}"
    )
    image_row["overlay_path"] = str(overlay_path)

    return image_row, slot_rows, cand_rows


def make_audit_all(images_df):
    cols = [
        "filename", "profile",
        "plot_bbox_source",
        "bar_orientation_observed",
        "categorical_axis_observed",
        "orientation_vertical_score",
        "orientation_horizontal_score",
        "observed_x_alpha_score",
        "observed_x_numeric_score",
        "observed_y_alpha_score",
        "observed_y_numeric_score",
        "categorical_axis_status",
        "categorical_n_slots",
        "categorical_n_selected",
        "categorical_n_ambiguous",
        "categorical_n_unreadable",
        "categorical_selected_labels_spatial",
        "overlay_path",
    ]
    cols = [c for c in cols if c in images_df.columns]
    audit = images_df[cols].copy()
    audit["manual_visible_categories"] = ""
    audit["manual_orientation_result"] = ""
    audit["manual_extraction_result"] = ""
    audit["manual_missing_labels"] = ""
    audit["manual_false_labels"] = ""
    audit["manual_notes"] = ""
    return audit


def make_priority_review(audit):
    if audit.empty:
        return audit.copy()
    mask = (
        audit["bar_orientation_observed"].eq("AMBIGUOUS")
        | ~audit["categorical_axis_status"].eq("CATEGORICAL_COMPLETE")
        | audit["plot_bbox_source"].eq("FALLBACK_GEOMETRIC")
    )
    return audit[mask].copy()


def write_empty_csv(path, columns):
    pd.DataFrame(columns=list(columns)).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    outdir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / OUT_DIRNAME
    )
    outdir.mkdir(parents=True, exist_ok=True)

    v7_path, v7_hash, presence_path, presence_hash = verify_frozen_dependencies()
    t_version, lang, langs = ba_v7.configure_tesseract(args.tesseract)

    inv = ba_v7.inventory(root)
    if len(inv) != 3120:
        raise RuntimeError(f"Inventário canônico na RAIZ={len(inv)}; esperado=3120.")
    if inv["unit_id"].nunique() != 312:
        raise RuntimeError(
            f"Unidades canônicas={inv['unit_id'].nunique()}; esperado=312."
        )

    cal_path = find_calibration_manifest(root, args.calibration_manifest)
    manifest = load_calibration_bars(cal_path, inv)
    manifest.to_csv(
        outdir / "ba_cat_v1_manifest.csv",
        index=False, encoding="utf-8-sig"
    )

    print("=" * 80)
    print("B-A-CAT V1 — CALIBRAÇÃO")
    print("=" * 80)
    print(f"B-A V7 congelada: {v7_hash}")
    print(f"V4 presença congelada: {presence_hash}")
    print(f"Tesseract: {t_version}")
    print(f"Idioma OCR: {lang}")
    print(f"Imagens de desenvolvimento: {len(manifest)} (12 BI + 12 BC)")
    print("Nenhuma imagem nova de holdout é selecionada por este programa.")
    print("A especificação F não é carregada.")
    print()

    image_rows = []
    slot_rows = []
    candidate_rows = []
    errors = []

    for i, r in manifest.iterrows():
        path = root / r["filename"]
        print(f"[{i+1:02d}/{len(manifest)}] {r['filename']} ...", end=" ", flush=True)
        try:
            imr, sr, cr = process_image(
                path, r["profile"], lang, outdir
            )
            for c in ("unit_id", "unit_number", "repeat"):
                if c in r.index:
                    imr[c] = r[c]
                    for rr in sr:
                        rr[c] = r[c]
                    for rr in cr:
                        rr[c] = r[c]

            image_rows.append(imr)
            slot_rows.extend(sr)
            candidate_rows.extend(cr)
            print("OK")
        except Exception as exc:
            errors.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            print(f"ERRO: {type(exc).__name__}: {exc}")

    images_df = pd.DataFrame(image_rows)
    slots_df = pd.DataFrame(slot_rows)
    candidates_df = pd.DataFrame(candidate_rows)
    errors_df = pd.DataFrame(errors)

    images_df.to_csv(
        outdir / "ba_cat_v1_images.csv",
        index=False, encoding="utf-8-sig"
    )
    slots_df.to_csv(
        outdir / "ba_cat_v1_slots.csv",
        index=False, encoding="utf-8-sig"
    )
    candidates_df.to_csv(
        outdir / "ba_cat_v1_candidates.csv",
        index=False, encoding="utf-8-sig"
    )

    if errors_df.empty:
        write_empty_csv(
            outdir / "ba_cat_v1_errors.csv",
            ["filename", "profile", "error_type", "error_message"]
        )
    else:
        errors_df.to_csv(
            outdir / "ba_cat_v1_errors.csv",
            index=False, encoding="utf-8-sig"
        )

    audit = make_audit_all(images_df)
    audit.to_csv(
        outdir / "ba_cat_v1_audit_all.csv",
        index=False, encoding="utf-8-sig"
    )

    priority = make_priority_review(audit)
    priority.to_csv(
        outdir / "ba_cat_v1_priority_review.csv",
        index=False, encoding="utf-8-sig"
    )

    # Contatos: reutiliza o formato já empregado na B-A V7.
    if not images_df.empty:
        for profile in PROFILES:
            g = images_df[images_df["profile"] == profile]
            if len(g):
                ba_v7.make_contact_sheet(
                    g, root,
                    outdir / f"contact_BA_CAT_V1_{profile}.png"
                )

    # Resumo automático — diagnóstico, não acurácia.
    lines = [
        "B-A-CAT V1 — CALIBRAÇÃO",
        "=" * 78,
        f"Arquivo B-A V7: {v7_path}",
        f"B-A V7 SHA-256: {v7_hash}",
        f"V4 presença SHA-256: {presence_hash}",
        f"Manifesto de calibração: {cal_path}",
        f"Tesseract: {t_version}",
        f"Idioma OCR: {lang}",
        "",
        "DESENHO:",
        "- somente BI/BC pertencentes à calibração B-A V7;",
        f"- imagens processadas: {len(images_df)}/{len(manifest)};",
        f"- erros de execução: {len(errors_df)};",
        "- nenhuma categoria esperada é fornecida ao programa;",
        "- nenhuma imagem do holdout independente B-A V7 é selecionada;",
        "",
        "ORIENTAÇÃO OBSERVADA:",
    ]

    if not images_df.empty:
        for k, n in images_df["bar_orientation_observed"].value_counts().items():
            lines.append(f"- {k}: {int(n)}")

        lines.extend(["", "STATUS CATEGÓRICO:"])
        for k, n in images_df["categorical_axis_status"].value_counts().items():
            lines.append(f"- {k}: {int(n)}")

        lines.extend([
            "",
            f"Casos em revisão prioritária: {len(priority)}",
            "",
            "IMPORTANTE:",
            "- os estados automáticos não equivalem a acurácia;",
            "- auditar visualmente todas as 24 imagens antes de congelar a CAT V1;",
            "- parâmetros podem ser ajustados somente nesta amostra de desenvolvimento;",
            "- depois do congelamento, usar novo holdout BI/BC independente.",
        ])

    summary_path = outdir / "ba_cat_v1_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Concluído.")
    print("Resumo:", summary_path)
    print("Auditoria:", outdir / "ba_cat_v1_audit_all.csv")
    print("Slots:", outdir / "ba_cat_v1_slots.csv")
    print("Candidatos:", outdir / "ba_cat_v1_candidates.csv")
    print("Prioridade:", outdir / "ba_cat_v1_priority_review.csv")
    print("Envie também contact_BA_CAT_V1_BI.png e contact_BA_CAT_V1_BC.png.")


if __name__ == "__main__":
    main()
