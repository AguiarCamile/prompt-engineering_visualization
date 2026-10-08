# -*- coding: utf-8 -*-
r"""
B-A-CAT V9 — CALIBRAÇÃO DO EXTRATOR CATEGÓRICO
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
ba_cat_v9_manifest.csv
ba_cat_v9_images.csv
ba_cat_v9_slots.csv
ba_cat_v9_candidates.csv
ba_cat_v9_audit_all.csv
ba_cat_v9_priority_review.csv
ba_cat_v9_errors.csv
ba_cat_v9_summary.txt
contact_BA_CAT_V9_BI.png
contact_BA_CAT_V9_BC.png
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
OUT_DIRNAME = "_ticklabel_categorical_extraction_ba_cat_v9"

EXPECTED_V7_SHA256 = "21423b59ce40eb351693f00b6a197936d936fe53131467f47c1031da864fd688"
EXPECTED_PRESENCE_V4_SHA256 = "d15c325290c7596632f9f4c62d907e7a78d0a794862b690c4c4691ed87ea86b2"

PROFILES = ("BI", "BC")

# Parâmetros de desenvolvimento. A versão ainda NÃO está congelada.
OCR_SCALE = 3.0
OCR_ROTATIONS = (0, -15, 15, -45, 45, -90, 90)
OCR_PSMS = (6, 7, 11, 13)
OCR_PREPROCESS = ("gray", "otsu", "adaptive")

# Consenso entre leituras independentes da MESMA imagem.
SIMILARITY_THRESHOLD = 0.72
CONSENSUS_MIN_SUPPORT = 2
SINGLE_HIGH_CONF = 88.0
AMBIGUITY_SCORE_MARGIN = 5.0

# Lane categórica: região imediatamente adjacente ao plot.
X_LANE_FRAC = 0.22
Y_LANE_FRAC = 0.26

# Heurística de orientação.
KIND_SCORE_MIN = 0.58
ORIENTATION_MARGIN = 0.08

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
            "Não execute a B-A-CAT V9 sobre uma V7 modificada."
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
    Recupera primeiro a área de plotagem já produzida pelo detector de eixos
    congelado. A interface real usa:
        plot_x_left, plot_x_right, plot_y_top, plot_y_bottom

    Somente se essas coordenadas não existirem é tentado o fallback.
    """
    # Interface efetiva do axis_v3 / V4.
    for obj, source in ((axes, "AXIS_V3"), (pres, "PRESENCE_V4")):
        if isinstance(obj, dict):
            keys = ("plot_x_left", "plot_y_top", "plot_x_right", "plot_y_bottom")
            vals = [obj.get(k) for k in keys]
            if all(v not in ("", None) and not pd.isna(v) for v in vals):
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in vals]
                    b = sanitize_bbox((x1, y1, x2, y2), w, h)
                    if (b[2]-b[0]) > 0.25*w and (b[3]-b[1]) > 0.20*h:
                        return b, source
                except Exception:
                    pass

    # Compatibilidade com possíveis interfaces alternativas.
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

    for b0 in candidates:
        try:
            b = sanitize_bbox(b0, w, h)
        except Exception:
            continue
        if (b[2]-b[0]) > 0.25*w and (b[3]-b[1]) > 0.20*h:
            return b, "ALTERNATIVE_DETECTOR"

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


def sanitize_bbox(bbox, w=None, h=None, min_size=2):
    """
    Converte coordenadas para int Python, corrige ordem invertida,
    remove NaN/inf e opcionalmente limita aos limites da imagem.
    """
    if bbox is None or len(bbox) < 4:
        raise ValueError(f"BBox inválido: {bbox}")

    vals = []
    for v in bbox[:4]:
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError(f"Coordenada não finita em bbox: {bbox}")
        vals.append(int(round(fv)))

    x1, y1, x2, y2 = vals

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if w is not None:
        x1 = max(0, min(int(w) - 1, x1))
        x2 = max(0, min(int(w), x2))
    if h is not None:
        y1 = max(0, min(int(h) - 1, y1))
        y2 = max(0, min(int(h), y2))

    if x2 - x1 < min_size:
        if w is None:
            x2 = x1 + min_size
        else:
            x2 = min(int(w), x1 + min_size)
            x1 = max(0, x2 - min_size)

    if y2 - y1 < min_size:
        if h is None:
            y2 = y1 + min_size
        else:
            y2 = min(int(h), y1 + min_size)
            y1 = max(0, y2 - min_size)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"BBox sem área após saneamento: {(x1, y1, x2, y2)}")

    return (int(x1), int(y1), int(x2), int(y2))


def crop_pil(img: Image.Image, bbox):
    b = sanitize_bbox(bbox, img.width, img.height)
    return img.crop(b)


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



def _clean_numeric_like_text(text: str):
    """
    Evidência de natureza numérica baseada apenas no texto observado.
    Aceita símbolos/unidades comuns de escala sem conhecer valores esperados.
    """
    t = normalize_text(text)
    if not t:
        return 0.0

    digits = sum(ch.isdigit() for ch in t)
    letters = [ch for ch in t if ch.isalpha()]
    n_letters = len(letters)

    if digits == 0:
        return 0.0

    # Letras muito curtas são compatíveis com moeda/unidade: R, K, M, B.
    allowed = set("RKMB")
    non_unit_letters = sum(ch not in allowed for ch in letters)

    core = digits / max(1.0, digits + n_letters)
    score = 0.55 + 0.45 * core
    if non_unit_letters >= 3:
        score *= 0.35
    return float(max(0.0, min(1.0, score)))


def light_box_ocr(img: Image.Image, bbox, lang: str):
    """
    OCR leve para decidir a natureza observada de um eixo.
    Não usa F e não tenta selecionar o rótulo final.
    """
    b = expand_bbox(bbox, img.width, img.height, frac=0.12)
    crop = crop_pil(img, b)
    rgb = np.asarray(crop.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)

    rows = []
    for prep in ("gray", "otsu"):
        base = preprocess_variant(gray, prep)
        for angle in (0, -90, 90):
            rr = rotate_bound(base, angle)
            raw, conf, tokens = ocr_once(rr, lang, 7)
            norm = normalize_text(raw)
            if not norm:
                continue

            letters = sum(ch.isalpha() for ch in norm)
            digits = sum(ch.isdigit() for ch in norm)
            ar = alpha_ratio(norm)
            nr = _clean_numeric_like_text(norm)

            cat = 0.0
            if letters >= 2:
                cat = (
                    max(0.0, conf) / 100.0
                    * ar
                    * min(1.0, letters / 7.0)
                )
                if digits > letters:
                    cat *= 0.45

            num = max(0.0, conf) / 100.0 * nr

            rows.append({
                "text": norm,
                "confidence": float(conf),
                "categorical_score": float(cat),
                "numeric_score": float(num),
            })

    if not rows:
        return {
            "text": "",
            "confidence": -1.0,
            "categorical_score": 0.0,
            "numeric_score": 0.0,
        }

    best_cat = max(rows, key=lambda r: r["categorical_score"])
    best_num = max(rows, key=lambda r: r["numeric_score"])

    return {
        "text": best_cat["text"] if best_cat["categorical_score"] >= best_num["numeric_score"] else best_num["text"],
        "confidence": max(best_cat["confidence"], best_num["confidence"]),
        "categorical_score": best_cat["categorical_score"],
        "numeric_score": best_num["numeric_score"],
    }


def axis_box_kind_evidence(img: Image.Image, label_rows, lang: str):
    """
    Caracteriza um eixo a partir das próprias caixas de tick labels detectadas
    pela segmentação congelada, em vez de uma faixa ampla da figura.
    """
    evid = []
    for r in label_rows:
        b = bbox_from_row(r)
        if not b:
            continue
        try:
            b = sanitize_bbox(b, img.width, img.height)
        except Exception:
            continue
        e = light_box_ocr(img, b, lang)
        if e["text"]:
            evid.append(e)

    if not evid:
        return {
            "categorical_score": 0.0,
            "numeric_score": 0.0,
            "n_boxes": 0,
            "best_texts": "",
        }

    cat_scores = sorted((e["categorical_score"] for e in evid), reverse=True)
    num_scores = sorted((e["numeric_score"] for e in evid), reverse=True)

    # Média robusta das melhores evidências. Uma caixa ruim não deve dominar.
    k = min(max(2, len(evid) // 2 + 1), len(evid))
    cat = float(np.mean(cat_scores[:k]))
    num = float(np.mean(num_scores[:k]))

    # Cobertura: eixos com várias caixas coerentes recebem pequeno reforço.
    cat_cov = sum(e["categorical_score"] >= 0.45 for e in evid) / len(evid)
    num_cov = sum(e["numeric_score"] >= 0.45 for e in evid) / len(evid)

    cat = min(1.0, 0.82 * cat + 0.18 * cat_cov)
    num = min(1.0, 0.82 * num + 0.18 * num_cov)

    texts = " || ".join(e["text"] for e in evid[:8])

    return {
        "categorical_score": float(cat),
        "numeric_score": float(num),
        "n_boxes": len(evid),
        "best_texts": texts,
    }


def v7_numeric_coherence(ar):
    status = str(safe_get(ar, "numeric_selection_status") or "").upper()
    nsel = safe_get(ar, "numeric_selection_n_selected")
    try:
        nsel = int(float(nsel))
    except Exception:
        nsel = 0

    if status.startswith("COHERENT") and nsel >= 2:
        return 1.0
    if "PARTIAL" in status and nsel >= 2:
        return 0.65
    if nsel >= 3:
        return 0.50
    return 0.0


def v7_category_band_strength(ar):
    status = str(safe_get(ar, "category_band_status") or "").upper()
    usable = safe_get(ar, "band_usable", "category_band_usable")
    ngrp = safe_get(ar, "category_band_n_groups")
    try:
        ngrp = int(float(ngrp))
    except Exception:
        ngrp = 0

    if status == "FOUND" and ngrp >= 2:
        return 1.0
    if usable in (1, True, "1", "True", "TRUE") and ngrp >= 1:
        return 0.75
    if ngrp >= 2:
        return 0.60
    return 0.0


def infer_orientation_from_tick_boxes(img, ar_x, lrs_x, ar_y, lrs_y, plot_bbox, lang):
    """
    Hipóteses concorrentes:
      VERTICAL   = X categórico + Y numérico
      HORIZONTAL = X numérico   + Y categórico

    A decisão usa somente evidência observada na própria imagem:
    caixas de tick labels, coerência numérica e geometria de barras.
    """
    xe = axis_box_kind_evidence(img, lrs_x, lang)
    ye = axis_box_kind_evidence(img, lrs_y, lang)
    geom = bar_geometry_orientation(img, plot_bbox)

    x_band = v7_category_band_strength(ar_x)
    y_num_coh = v7_numeric_coherence(ar_y)

    vertical = (
        0.38 * xe["categorical_score"]
        + 0.32 * max(ye["numeric_score"], y_num_coh)
        + 0.15 * x_band
        + 0.15 * geom["vertical_score"]
    )

    horizontal = (
        0.43 * ye["categorical_score"]
        + 0.37 * xe["numeric_score"]
        + 0.20 * geom["horizontal_score"]
    )

    diff = vertical - horizontal

    if max(vertical, horizontal) < 0.48 or abs(diff) < ORIENTATION_MARGIN:
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
        "x_alpha_score": xe["categorical_score"],
        "x_numeric_score": xe["numeric_score"],
        "y_alpha_score": ye["categorical_score"],
        "y_numeric_score": ye["numeric_score"],
        "x_lane_best_text": xe["best_texts"],
        "y_lane_best_text": ye["best_texts"],
        "vertical_bar_centers": geom["vertical_centers"],
        "horizontal_bar_centers": geom["horizontal_centers"],
        "orientation_evidence_source": "TICK_BOXES+V7_COHERENCE+BAR_GEOMETRY",
    }


def v7_category_band_bbox(ar, img: Image.Image):
    for name in ("category_band_bbox", "band_bbox"):
        b = _bbox_candidate(safe_get(ar, name))
        if b:
            try:
                return sanitize_bbox(b, img.width, img.height)
            except Exception:
                pass
    return None


def tight_lane_from_boxes(boxes, img: Image.Image, axis: str):
    good = []
    for b in boxes:
        try:
            good.append(sanitize_bbox(b, img.width, img.height))
        except Exception:
            pass
    if not good:
        return None

    x1 = min(b[0] for b in good)
    y1 = min(b[1] for b in good)
    x2 = max(b[2] for b in good)
    y2 = max(b[3] for b in good)

    if axis == "X":
        # Mantém a faixa vertical estreita para excluir valores sobre barras
        # e o título do eixo abaixo dos ticks.
        pad_x = max(8, int(0.025 * img.width))
        pad_y = max(3, int(0.008 * img.height))
    else:
        pad_x = max(4, int(0.010 * img.width))
        pad_y = max(6, int(0.018 * img.height))

    return sanitize_bbox(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        img.width, img.height
    )


def slot_specs_from_v7_boxes(label_rows, lane_bbox, axis, img: Image.Image):
    """
    Um slot nasce primeiro de uma caixa de tick label real.
    Isso evita os slots excessivamente largos da V1.
    """
    specs = []
    for r in label_rows:
        b = bbox_from_row(r)
        if not b:
            continue
        try:
            b = sanitize_bbox(b, img.width, img.height)
        except Exception:
            continue

        # Crop final ligeiramente expandido, mas limitado à categorical lane.
        eb = expand_bbox(b, img.width, img.height, frac=0.20)
        lx1, ly1, lx2, ly2 = sanitize_bbox(lane_bbox, img.width, img.height)

        sx1 = max(eb[0], lx1)
        sy1 = max(eb[1], ly1)
        sx2 = min(eb[2], lx2)
        sy2 = min(eb[3], ly2)

        try:
            sb = sanitize_bbox((sx1, sy1, sx2, sy2), img.width, img.height)
        except Exception:
            sb = b

        center = ((b[0] + b[2]) / 2.0) if axis == "X" else ((b[1] + b[3]) / 2.0)
        specs.append((float(center), sb))

    specs.sort(key=lambda x: x[0])
    return specs


def internal_box_to_bbox(b, img: Image.Image):
    if b is None:
        return None
    if isinstance(b, dict):
        try:
            return sanitize_bbox(
                (b["x1"], b["y1"], b["x2"], b["y2"]),
                img.width, img.height
            )
        except Exception:
            return None
    return _bbox_candidate(b)


def v4_layer_bbox(pres, axis: str, img: Image.Image):
    key = "_x_layer_rect" if axis == "X" else "_y_layer_rect"
    b = internal_box_to_bbox(pres.get(key) if isinstance(pres, dict) else None, img)
    if b:
        return b
    return None


def v4_axislabel_bboxes(pres, axis: str, img: Image.Image):
    key = "_x_axislabel" if axis == "X" else "_y_axislabel"
    rows = pres.get(key, []) if isinstance(pres, dict) else []
    out = []
    for r in rows:
        b = internal_box_to_bbox(r, img)
        if b:
            out.append(b)
    return out


def v4_layer_group_bboxes(pres, axis: str, img: Image.Image):
    key = "_x_layer_groups" if axis == "X" else "_y_layer_groups"
    rows = pres.get(key, []) if isinstance(pres, dict) else []
    out = []
    for r in rows:
        b = internal_box_to_bbox(r, img)
        if b:
            out.append(b)
    return out


def mask_rectangles(img: Image.Image, boxes):
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for b in boxes:
        try:
            bb = sanitize_bbox(b, out.width, out.height)
            d.rectangle(bb, fill="white")
        except Exception:
            continue
    return out


def bbox_center(b, axis="X"):
    x1, y1, x2, y2 = b
    return (x1+x2)/2.0 if axis == "X" else (y1+y2)/2.0


def bbox_axis_size(b, axis="X"):
    x1, y1, x2, y2 = b
    return (x2-x1) if axis == "X" else (y2-y1)


def robust_filter_text_boxes(boxes, axis="X"):
    """
    Remove caixas minúsculas isoladas como IM/VV/AN/CC/VL observadas na V2.
    A regra é puramente geométrica e relativa às demais caixas da mesma imagem.
    """
    if not boxes:
        return []

    sizes = [bbox_axis_size(b, axis) for b in boxes if bbox_axis_size(b, axis) > 0]
    if not sizes:
        return []

    med = float(np.median(sizes))
    # Só elimina caixas claramente minúsculas em relação ao conjunto.
    cutoff = max(8.0, 0.22 * med)

    out = []
    for b in boxes:
        s = bbox_axis_size(b, axis)
        cross = (b[3]-b[1]) if axis == "X" else (b[2]-b[0])
        if s < cutoff and cross < 0.70 * max(8.0, np.median([
            (z[3]-z[1]) if axis == "X" else (z[2]-z[0]) for z in boxes
        ])):
            continue
        out.append(b)
    return out


def group_bar_centers(centers, span):
    """
    Converte centros de barras individuais em centros de grupos categóricos.
    Se os gaps têm duas escalas claras, gaps pequenos são tratados como barras
    do mesmo grupo; se os gaps são aproximadamente homogêneos, cada barra é
    considerada um grupo separado.

    Não usa número esperado de categorias.
    """
    vals = sorted(float(v) for v in centers if math.isfinite(float(v)))
    if not vals:
        return []
    if len(vals) <= 2:
        return vals

    gaps = np.diff(vals)
    pos = [float(g) for g in gaps if g > 1]
    if not pos:
        return [float(np.mean(vals))]

    gmin = min(pos)
    gmax = max(pos)

    if gmax / max(1.0, gmin) >= 2.2:
        thr = math.sqrt(gmin * gmax)
    else:
        # Gaps homogêneos: barras provavelmente representam categorias distintas.
        thr = -1.0

    groups = [[vals[0]]]
    for i, v in enumerate(vals[1:], start=1):
        gap = vals[i] - vals[i-1]
        if thr > 0 and gap <= thr:
            groups[-1].append(v)
        else:
            groups.append([v])

    return [float(np.mean(g)) for g in groups]


def merge_anchor_candidates(anchor_sets, span):
    vals = []
    for name, seq, weight in anchor_sets:
        for v in seq:
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isfinite(fv):
                vals.append((fv, name, float(weight)))

    if not vals:
        return []

    vals.sort(key=lambda x: x[0])
    tol = max(10.0, 0.035 * span)

    groups = [[vals[0]]]
    for item in vals[1:]:
        center = np.average(
            [x[0] for x in groups[-1]],
            weights=[x[2] for x in groups[-1]]
        )
        if abs(item[0] - center) <= tol:
            groups[-1].append(item)
        else:
            groups.append([item])

    anchors = []
    for g in groups:
        c = float(np.average([x[0] for x in g], weights=[x[2] for x in g]))
        sources = sorted(set(x[1] for x in g))
        strength = float(sum(x[2] for x in g))
        anchors.append({
            "center": c,
            "sources": "+".join(sources),
            "strength": strength,
        })
    return anchors


def category_anchor_specs(
    pres, v7_boxes, orientation, lane_bbox, axis, img: Image.Image, plot_bbox
):
    """
    Combina três fontes independentes da própria imagem:
    - geometria das barras;
    - grupos coarse da camada textual V4;
    - caixas segmentadas V7/V4.

    O número de categorias não é conhecido a priori.
    """
    span = (plot_bbox[2]-plot_bbox[0]) if axis == "X" else (plot_bbox[3]-plot_bbox[1])

    # 1) Barras -> grupos categóricos.
    raw_bars = (
        orientation.get("vertical_bar_centers", [])
        if axis == "X"
        else orientation.get("horizontal_bar_centers", [])
    )
    bar_groups = group_bar_centers(raw_bars, span)

    # 2) Grupos da camada textual V4.
    layer_groups = robust_filter_text_boxes(
        v4_layer_group_bboxes(pres, axis, img), axis
    )
    layer_centers = [bbox_center(b, axis) for b in layer_groups]

    # 3) Caixas individuais, eliminando microcaixas.
    boxes = robust_filter_text_boxes(v7_boxes, axis)
    box_centers = [bbox_center(b, axis) for b in boxes]

    anchors = merge_anchor_candidates(
        [
            ("BAR", bar_groups, 2.0),
            ("LAYER_GROUP", layer_centers, 1.5),
            ("TEXT_BOX", box_centers, 1.0),
        ],
        span
    )

    if not anchors:
        return []

    # Mantém somente anchors dentro da extensão principal do plot/lane.
    lx1, ly1, lx2, ly2 = lane_bbox
    low, high = (plot_bbox[0], plot_bbox[2]) if axis == "X" else (plot_bbox[1], plot_bbox[3])
    anchors = [a for a in anchors if low - 0.03*span <= a["center"] <= high + 0.03*span]
    anchors.sort(key=lambda a: a["center"])

    if not anchors:
        return []

    # Voronoi espacial entre anchors, mas verticalmente limitado à camada V4.
    specs = []
    centers = [a["center"] for a in anchors]

    for i, a in enumerate(anchors):
        c = a["center"]
        prev_c = centers[i-1] if i > 0 else None
        next_c = centers[i+1] if i+1 < len(centers) else None

        if axis == "X":
            left = lx1 if prev_c is None else int(round((prev_c+c)/2))
            right = lx2 if next_c is None else int(round((c+next_c)/2))
            # não deixa o slot invadir exageradamente vizinhos
            half = max(28, int(0.48 * max(2, right-left)))
            b = (
                max(lx1, int(round(c-half))),
                ly1,
                min(lx2, int(round(c+half))),
                ly2,
            )
        else:
            top = ly1 if prev_c is None else int(round((prev_c+c)/2))
            bottom = ly2 if next_c is None else int(round((c+next_c)/2))
            half = max(22, int(0.48 * max(2, bottom-top)))
            b = (
                lx1,
                max(ly1, int(round(c-half))),
                lx2,
                min(ly2, int(round(c+half))),
            )

        try:
            b = sanitize_bbox(b, img.width, img.height)
        except Exception:
            continue

        specs.append({
            "center": float(c),
            "bbox": b,
            "anchor_sources": a["sources"],
            "anchor_strength": a["strength"],
        })

    return specs


def slot_candidates_v3(full_img, slot_bbox, v7_boxes, lang, axislabel_boxes):
    """
    Igual à evidência multi-OCR anterior, mas com prováveis axis labels
    mascarados antes de qualquer crop.
    """
    masked = mask_rectangles(full_img, axislabel_boxes)
    rows = []

    slot_crop = crop_pil(masked, slot_bbox)
    rows.extend(run_ocr_variants(slot_crop, lang, "SLOT"))

    for j, b in enumerate(v7_boxes, start=1):
        if overlap_ratio(b, slot_bbox) < 0.18 and overlap_ratio(slot_bbox, b) < 0.18:
            continue

        # Intersecta a caixa ao slot para evitar concatenar a categoria vizinha.
        ix1 = max(b[0], slot_bbox[0])
        iy1 = max(b[1], slot_bbox[1])
        ix2 = min(b[2], slot_bbox[2])
        iy2 = min(b[3], slot_bbox[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        ib = sanitize_bbox((ix1, iy1, ix2, iy2), masked.width, masked.height)
        rows.extend(run_ocr_variants(crop_pil(masked, ib), lang, f"V7_INTERSECT_{j}"))

    return rows


def detect_bar_rectangles_hsv(img: Image.Image, plot_bbox, orientation="VERTICAL"):
    """
    Detecta retângulos preenchidos das barras usando saturação/cor e
    alinhamento com uma linha de base comum.

    Não usa número esperado de categorias nem conteúdo de F.
    """
    x1, y1, x2, y2 = sanitize_bbox(plot_bbox, img.width, img.height)
    arr = np.asarray(img.convert("RGB"))
    roi = arr[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Barras coloridas são significativamente mais saturadas que fundo,
    # grades e texto preto/cinza.
    mask = ((sat >= 48) & (val >= 45)).astype(np.uint8) * 255

    # Remove ruído pequeno sem apagar barras finas.
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    ph, pw = roi.shape[:2]
    parea = max(1.0, float(ph * pw))

    comps = []
    for i in range(1, n):
        xx, yy, ww, hh, area = [int(v) for v in stats[i]]
        if ww < 3 or hh < 3:
            continue

        frac = area / parea
        if frac < 0.00012 or frac > 0.35:
            continue

        fill = area / max(1.0, ww * hh)
        if fill < 0.42:
            continue

        # Evita componentes extremamente finos típicos de linhas/legenda.
        if orientation == "VERTICAL":
            if ww > 0.30 * pw:
                continue
        else:
            if hh > 0.30 * ph:
                continue

        comps.append({
            "x": xx, "y": yy, "w": ww, "h": hh, "area": area,
            "fill": float(fill),
            "cx": x1 + xx + ww / 2.0,
            "cy": y1 + yy + hh / 2.0,
            "right": x1 + xx + ww,
            "bottom": y1 + yy + hh,
        })

    if not comps:
        return []

    # Seleciona o cluster que compartilha a linha de base das barras.
    if orientation == "VERTICAL":
        coord = "bottom"
        tol = max(6.0, 0.045 * ph)
        # Baseline tende a ser o cluster inferior com maior área agregada.
        values = sorted(set(round(c[coord], 1) for c in comps))
    else:
        coord = "right"
        tol = max(6.0, 0.045 * pw)
        values = sorted(set(round(c[coord], 1) for c in comps))

    clusters = []
    for c in sorted(comps, key=lambda z: z[coord]):
        if not clusters:
            clusters.append([c])
            continue
        center = np.average(
            [z[coord] for z in clusters[-1]],
            weights=[max(1, z["area"]) for z in clusters[-1]]
        )
        if abs(c[coord] - center) <= tol:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    # Pontuação: área agregada + preferência por baseline mais externa.
    scored = []
    for g in clusters:
        total_area = sum(z["area"] for z in g)
        base = np.average(
            [z[coord] for z in g],
            weights=[max(1, z["area"]) for z in g]
        )
        norm_base = (
            (base - y1) / max(1.0, y2 - y1)
            if orientation == "VERTICAL"
            else (base - x1) / max(1.0, x2 - x1)
        )
        score = total_area * (0.75 + 0.25 * norm_base)
        scored.append((score, g))

    best = max(scored, key=lambda t: t[0])[1]

    # Remove pequenos patches de legenda se ainda sobraram:
    # barra deve ter dimensão útil na direção do valor.
    dims = [z["h"] if orientation == "VERTICAL" else z["w"] for z in best]
    med_dim = float(np.median(dims)) if dims else 0.0

    kept = []
    for z in best:
        d = z["h"] if orientation == "VERTICAL" else z["w"]
        # Mantém também barras pequenas, mas exclui patches muito menores
        # que a distribuição dominante.
        if med_dim > 0 and d < max(3.0, 0.08 * med_dim):
            continue
        kept.append(z)

    return kept


def bar_category_centers_v4(img: Image.Image, plot_bbox, orientation):
    rects = detect_bar_rectangles_hsv(img, plot_bbox, orientation)
    if not rects:
        return {
            "raw_centers": [],
            "group_centers": [],
            "n_rects": 0,
            "confidence": 0.0,
        }

    if orientation == "VERTICAL":
        raw = sorted(z["cx"] for z in rects)
        span = plot_bbox[2] - plot_bbox[0]
    else:
        raw = sorted(z["cy"] for z in rects)
        span = plot_bbox[3] - plot_bbox[1]

    groups = group_bar_centers(raw, span)

    # Confiança apenas estrutural: quantidade plausível de barras e grupos.
    conf = 0.0
    if 2 <= len(raw) <= 16 and 2 <= len(groups) <= 8:
        conf = 0.85
        if len(groups) <= len(raw):
            conf += 0.10
        conf = min(1.0, conf)

    return {
        "raw_centers": [float(v) for v in raw],
        "group_centers": [float(v) for v in groups],
        "n_rects": len(raw),
        "confidence": float(conf),
    }


def ocr_word_boxes_lane(img: Image.Image, lane_bbox, lang: str):
    """
    OCR de palavras da lane para separar a linha de tick labels do título do eixo.
    """
    b = sanitize_bbox(lane_bbox, img.width, img.height)
    crop = crop_pil(img, b)
    arr = np.asarray(crop.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    scale = 2.2
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    d = pytesseract.image_to_data(
        up, lang=lang, config="--psm 11",
        output_type=pytesseract.Output.DATAFRAME
    )
    if d is None or len(d) == 0:
        return []

    d = d.copy()
    d["text"] = d["text"].fillna("").astype(str)
    d["conf"] = pd.to_numeric(d["conf"], errors="coerce").fillna(-1)

    out = []
    for _, r in d.iterrows():
        txt = normalize_text(r["text"])
        if not txt or float(r["conf"]) < 20:
            continue
        try:
            left = float(r["left"]) / scale + b[0]
            top = float(r["top"]) / scale + b[1]
            width = float(r["width"]) / scale
            height = float(r["height"]) / scale
        except Exception:
            continue
        if width <= 1 or height <= 1:
            continue
        out.append({
            "text": txt,
            "conf": float(r["conf"]),
            "bbox": (
                int(round(left)),
                int(round(top)),
                int(round(left + width)),
                int(round(top + height)),
            ),
            "cx": left + width / 2.0,
            "cy": top + height / 2.0,
            "h": height,
            "w": width,
        })
    return out


def refine_categorical_lane_row(img: Image.Image, lane_bbox, plot_bbox, axis, lang):
    """
    Para X categórico, identifica a faixa de texto mais próxima do eixo X e
    com distribuição horizontal ampla. Isso tende a preservar tick labels e
    excluir o título do eixo, geralmente em uma linha inferior centralizada.

    Se não houver evidência suficiente, mantém a lane original.
    """
    b = sanitize_bbox(lane_bbox, img.width, img.height)
    words = ocr_word_boxes_lane(img, b, lang)
    if len(words) < 2:
        return b, "ORIGINAL_LANE"

    if axis != "X":
        return b, "ORIGINAL_LANE"

    hs = [w["h"] for w in words]
    tol = max(5.0, 0.90 * float(np.median(hs)))

    ordered = sorted(words, key=lambda z: z["cy"])
    clusters = [[ordered[0]]]
    for w in ordered[1:]:
        center = np.mean([z["cy"] for z in clusters[-1]])
        if abs(w["cy"] - center) <= tol:
            clusters[-1].append(w)
        else:
            clusters.append([w])

    px1, py1, px2, py2 = plot_bbox
    lane_w = max(1.0, b[2] - b[0])

    scored = []
    for g in clusters:
        gy = float(np.mean([z["cy"] for z in g]))
        gx1 = min(z["bbox"][0] for z in g)
        gx2 = max(z["bbox"][2] for z in g)
        coverage = (gx2 - gx1) / lane_w
        n = len(g)
        alpha = sum(sum(c.isalpha() for c in z["text"]) for z in g)

        # Tick labels tendem a ser a primeira linha textual logo abaixo do plot.
        proximity = 1.0 / (1.0 + max(0.0, gy - py2) / 20.0)
        multi = min(1.0, n / 3.0)
        spread = min(1.0, coverage / 0.35)
        lexical = min(1.0, alpha / 12.0)
        score = 0.40 * proximity + 0.25 * multi + 0.25 * spread + 0.10 * lexical
        scored.append((score, g))

    if not scored:
        return b, "ORIGINAL_LANE"

    best_score, g = max(scored, key=lambda t: t[0])
    if best_score < 0.48:
        return b, "ORIGINAL_LANE"

    y1 = min(z["bbox"][1] for z in g)
    y2 = max(z["bbox"][3] for z in g)
    pad = max(3, int(round(0.35 * np.median([z["h"] for z in g]))))

    rb = sanitize_bbox(
        (b[0], y1 - pad, b[2], y2 + pad),
        img.width, img.height
    )
    return rb, "OCR_ROW_REFINED"


def bar_primary_slot_specs(bar_info, lane_bbox, plot_bbox, axis, img: Image.Image):
    centers = sorted(bar_info.get("group_centers", []))
    if bar_info.get("confidence", 0.0) < 0.80 or len(centers) < 2:
        return []

    lx1, ly1, lx2, ly2 = sanitize_bbox(lane_bbox, img.width, img.height)

    specs = []
    for i, c in enumerate(centers):
        prev_c = centers[i-1] if i > 0 else None
        next_c = centers[i+1] if i+1 < len(centers) else None

        if axis == "X":
            left = lx1 if prev_c is None else int(round((prev_c + c) / 2.0))
            right = lx2 if next_c is None else int(round((c + next_c) / 2.0))
            # pequena retração para evitar texto da categoria vizinha
            width = max(2, right-left)
            inset = int(round(0.04 * width))
            b = (left + inset, ly1, right - inset, ly2)
        else:
            top = ly1 if prev_c is None else int(round((prev_c + c) / 2.0))
            bottom = ly2 if next_c is None else int(round((c + next_c) / 2.0))
            height = max(2, bottom-top)
            inset = int(round(0.04 * height))
            b = (lx1, top + inset, lx2, bottom - inset)

        try:
            bb = sanitize_bbox(b, img.width, img.height)
        except Exception:
            continue

        specs.append({
            "center": float(c),
            "bbox": bb,
            "anchor_sources": "BAR_PRIMARY",
            "anchor_strength": 5.0,
        })

    return specs


def slot_candidates_v4(full_img, slot_bbox, v7_boxes, lang, axislabel_boxes):
    """
    Prioriza o slot completo. Interseções V7 entram apenas como evidência
    adicional, sem criar novos slots.
    """
    masked = mask_rectangles(full_img, axislabel_boxes)
    rows = []

    # Evidência principal.
    rows.extend(
        run_ocr_variants(crop_pil(masked, slot_bbox), lang, "BAR_SLOT")
    )

    # Evidência auxiliar de caixas já detectadas que realmente intersectam o slot.
    for j, b in enumerate(v7_boxes, start=1):
        if overlap_ratio(b, slot_bbox) < 0.20 and overlap_ratio(slot_bbox, b) < 0.20:
            continue
        ix1 = max(b[0], slot_bbox[0])
        iy1 = max(b[1], slot_bbox[1])
        ix2 = min(b[2], slot_bbox[2])
        iy2 = min(b[3], slot_bbox[3])
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        ib = sanitize_bbox((ix1, iy1, ix2, iy2), masked.width, masked.height)
        rows.extend(
            run_ocr_variants(crop_pil(masked, ib), lang, f"TEXT_SUPPORT_{j}")
        )
    return rows


def widen_x_tick_lane(lane_bbox, img: Image.Image, axis: str):
    """
    A V4 mostrou que plot/layer X pode começar depois da primeira categoria.
    Para X categórico, preserva a faixa vertical já estimada, mas amplia
    horizontalmente a busca quase até as bordas da imagem.

    Isso permite recuperar categorias com barra zero/muito pequena e labels
    anteriores ao plot_bbox detectado.
    """
    b = sanitize_bbox(lane_bbox, img.width, img.height)
    if axis != "X":
        return b

    margin = max(8, int(round(0.025 * img.width)))
    return sanitize_bbox(
        (margin, b[1], img.width - margin, b[3]),
        img.width, img.height
    )


def row_word_groups(img: Image.Image, lane_bbox, lang: str):
    """
    Extrai palavras da linha de tick labels e agrupa palavras adjacentes em
    frases categóricas, sem qualquer vocabulário de F.

    Retorna grupos com texto, bbox, centro e confiança.
    """
    b = sanitize_bbox(lane_bbox, img.width, img.height)
    crop = crop_pil(img, b)
    arr = np.asarray(crop.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    scale = 2.8
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # psm 6 tende a preservar uma linha de labels; psm 11 ajuda quando há
    # espaçamento irregular. Mantemos o melhor conjunto por qualidade.
    all_sets = []

    for psm in (6, 11):
        d = pytesseract.image_to_data(
            up, lang=lang, config=f"--psm {psm}",
            output_type=pytesseract.Output.DATAFRAME
        )
        if d is None or len(d) == 0:
            continue

        d = d.copy()
        d["text"] = d["text"].fillna("").astype(str)
        d["conf"] = pd.to_numeric(d["conf"], errors="coerce").fillna(-1)

        words = []
        for _, r in d.iterrows():
            txt = normalize_text(r["text"])
            conf = float(r["conf"])
            if not txt or conf < 28:
                continue

            # Exige algum conteúdo lexical; números sobre barras não entram
            # como âncoras categóricas primárias.
            letters = sum(ch.isalpha() for ch in txt)
            if letters < 1:
                continue

            try:
                left = float(r["left"]) / scale + b[0]
                top = float(r["top"]) / scale + b[1]
                width = float(r["width"]) / scale
                height = float(r["height"]) / scale
            except Exception:
                continue

            if width <= 1 or height <= 1:
                continue

            words.append({
                "text": txt,
                "conf": conf,
                "bbox": (
                    int(round(left)),
                    int(round(top)),
                    int(round(left + width)),
                    int(round(top + height)),
                ),
                "cx": left + width / 2.0,
                "cy": top + height / 2.0,
                "w": width,
                "h": height,
            })

        if len(words) < 2:
            continue

        # Primeiro isola a linha textual principal mais próxima da borda
        # superior da lane. O título do eixo costuma estar em outra linha.
        hs = [w["h"] for w in words]
        ytol = max(4.0, 0.75 * float(np.median(hs)))
        ordered_y = sorted(words, key=lambda z: z["cy"])

        lines = [[ordered_y[0]]]
        for w in ordered_y[1:]:
            center = float(np.mean([z["cy"] for z in lines[-1]]))
            if abs(w["cy"] - center) <= ytol:
                lines[-1].append(w)
            else:
                lines.append([w])

        line_scored = []
        for line in lines:
            line = sorted(line, key=lambda z: z["bbox"][0])
            n = len(line)
            letters = sum(sum(ch.isalpha() for ch in z["text"]) for z in line)
            xspan = max(z["bbox"][2] for z in line) - min(z["bbox"][0] for z in line)
            coverage = xspan / max(1.0, b[2] - b[0])
            mean_conf = float(np.mean([z["conf"] for z in line]))

            # Linha de categorias: múltiplos tokens/labels, boa extensão
            # horizontal e proximidade ao topo da lane.
            topness = 1.0 - min(
                1.0,
                max(0.0, np.mean([z["cy"] for z in line]) - b[1])
                / max(1.0, b[3] - b[1])
            )
            score = (
                0.28 * min(1.0, n / 4.0)
                + 0.24 * min(1.0, coverage / 0.45)
                + 0.20 * min(1.0, letters / 18.0)
                + 0.18 * (mean_conf / 100.0)
                + 0.10 * topness
            )
            line_scored.append((score, line))

        if not line_scored:
            continue

        line_score, line = max(line_scored, key=lambda t: t[0])
        line = sorted(line, key=lambda z: z["bbox"][0])

        # Agrupamento lexical por gaps horizontais.
        gaps = []
        for a, c in zip(line[:-1], line[1:]):
            gaps.append(max(0.0, c["bbox"][0] - a["bbox"][2]))

        positive = [g for g in gaps if g > 0.5]
        if positive:
            sp = sorted(positive)
            # Gaps intrafrase tendem a ser pequenos; gaps intercategoria,
            # grandes. O percentil inferior é uma estimativa robusta do
            # espaçamento entre palavras de uma mesma frase.
            small = float(np.percentile(sp, 35))
            break_thr = max(12.0, 2.8 * small)
        else:
            break_thr = 14.0

        groups = [[line[0]]]
        for prev, cur in zip(line[:-1], line[1:]):
            gap = max(0.0, cur["bbox"][0] - prev["bbox"][2])
            if gap > break_thr:
                groups.append([cur])
            else:
                groups[-1].append(cur)

        out = []
        for g in groups:
            text = " ".join(z["text"] for z in g).strip()
            letters = sum(ch.isalpha() for ch in text)
            if letters < 2:
                continue

            x1 = min(z["bbox"][0] for z in g)
            y1 = min(z["bbox"][1] for z in g)
            x2 = max(z["bbox"][2] for z in g)
            y2 = max(z["bbox"][3] for z in g)

            conf = float(np.average(
                [z["conf"] for z in g],
                weights=[max(1.0, z["w"]) for z in g]
            ))

            out.append({
                "text": normalize_text(text),
                "confidence": conf,
                "bbox": sanitize_bbox((x1, y1, x2, y2), img.width, img.height),
                "center": float((x1 + x2) / 2.0),
                "n_words": len(g),
            })

        if len(out) >= 2:
            mean_conf = float(np.mean([g["confidence"] for g in out]))
            phrase_letters = sum(
                sum(ch.isalpha() for ch in g["text"]) for g in out
            )
            quality = (
                0.42 * min(1.0, len(out) / 4.0)
                + 0.30 * (mean_conf / 100.0)
                + 0.18 * min(1.0, phrase_letters / 28.0)
                + 0.10 * line_score
            )
            all_sets.append({
                "psm": psm,
                "groups": out,
                "quality": float(quality),
                "line_score": float(line_score),
                "mean_confidence": mean_conf,
            })

    if not all_sets:
        return {
            "groups": [],
            "quality": 0.0,
            "psm": None,
            "mean_confidence": 0.0,
        }

    return max(all_sets, key=lambda s: s["quality"])


def row_primary_slot_specs(row_info, lane_bbox, img: Image.Image):
    groups = row_info.get("groups", [])
    quality = float(row_info.get("quality", 0.0))
    mean_conf = float(row_info.get("mean_confidence", 0.0))

    if len(groups) < 2 or quality < 0.58 or mean_conf < 45.0:
        return []

    groups = sorted(groups, key=lambda g: g["center"])
    centers = [float(g["center"]) for g in groups]

    lx1, ly1, lx2, ly2 = sanitize_bbox(lane_bbox, img.width, img.height)

    specs = []
    for i, (g, c) in enumerate(zip(groups, centers)):
        prev_c = centers[i - 1] if i > 0 else None
        next_c = centers[i + 1] if i + 1 < len(centers) else None

        left = lx1 if prev_c is None else int(round((prev_c + c) / 2.0))
        right = lx2 if next_c is None else int(round((c + next_c) / 2.0))

        # O bbox da frase observada funciona como núcleo, mas permitimos
        # pequeno contexto lateral para OCR do slot.
        phrase = g["bbox"]
        pad = max(8, int(round(0.08 * max(2, right - left))))
        sx1 = max(left, phrase[0] - pad)
        sx2 = min(right, phrase[2] + pad)

        try:
            b = sanitize_bbox((sx1, ly1, sx2, ly2), img.width, img.height)
        except Exception:
            continue

        specs.append({
            "center": c,
            "bbox": b,
            "anchor_sources": "ROW_PHRASE",
            "anchor_strength": 6.0,
            "row_phrase_text": g["text"],
            "row_phrase_confidence": g["confidence"],
        })

    return specs


def reconcile_row_and_bar_specs(row_specs, bar_specs, lane_bbox, img: Image.Image):
    """
    Reconciliação sem conhecimento de F.

    - ROW_PRIMARY é preferido quando há frases categóricas coerentes.
    - Se barras revelam um centro sem frase correspondente, esse centro pode
      ser acrescentado somente quando está claramente afastado dos centros
      textuais.
    - Isso permite recuperar categoria ilegível, mas evita duplicar fragmentos.
    """
    if not row_specs:
        return bar_specs, "BAR_PRIMARY"
    if not bar_specs:
        return row_specs, "ROW_PRIMARY"

    rcenters = [float(s["center"]) for s in row_specs]
    bcenters = [float(s["center"]) for s in bar_specs]

    lane = sanitize_bbox(lane_bbox, img.width, img.height)
    span = max(1.0, lane[2] - lane[0])
    tol = max(18.0, 0.055 * span)

    extras = []
    for bs in bar_specs:
        c = float(bs["center"])
        if min(abs(c - r) for r in rcenters) > tol:
            extras.append(bs)

    if not extras:
        return row_specs, "ROW_PRIMARY"

    # Une os centros e refaz intervalos entre categorias.
    merged = []
    for rs in row_specs:
        x = dict(rs)
        x["anchor_sources"] = "ROW_PHRASE"
        merged.append(x)
    for bs in extras:
        x = dict(bs)
        x["anchor_sources"] = "BAR_EXTRA"
        x["row_phrase_text"] = ""
        x["row_phrase_confidence"] = -1.0
        merged.append(x)

    merged.sort(key=lambda s: float(s["center"]))
    centers = [float(s["center"]) for s in merged]
    lx1, ly1, lx2, ly2 = lane

    out = []
    for i, s in enumerate(merged):
        c = centers[i]
        prev_c = centers[i - 1] if i > 0 else None
        next_c = centers[i + 1] if i + 1 < len(centers) else None
        left = lx1 if prev_c is None else int(round((prev_c + c) / 2.0))
        right = lx2 if next_c is None else int(round((c + next_c) / 2.0))

        try:
            b = sanitize_bbox((left, ly1, right, ly2), img.width, img.height)
        except Exception:
            continue

        ss = dict(s)
        ss["bbox"] = b
        out.append(ss)

    return out, "ROW_PLUS_BAR_EXTRAS"


def slot_candidates_v5(
    full_img, slot_bbox, v7_boxes, lang, axislabel_boxes,
    row_phrase_text="", row_phrase_confidence=-1.0
):
    rows = slot_candidates_v4(
        full_img, slot_bbox, v7_boxes, lang, axislabel_boxes
    )

    # A frase detectada na linha completa é uma evidência independente
    # adicional da mesma imagem. Não é correção por dicionário.
    txt = normalize_text(row_phrase_text)
    if txt and float(row_phrase_confidence) >= 20:
        ar = alpha_ratio(txt)
        nr = numeric_ratio(txt)
        lexical = min(1.0, len([c for c in txt if c.isalpha()]) / 12.0)
        score = (
            max(0.0, float(row_phrase_confidence)) * 0.68
            + 19.0 * ar
            + 10.0 * lexical
            - 8.0 * nr
            + 8.0  # evidência de frase espacialmente agrupada
        )
        rows.append({
            "source": "ROW_PHRASE",
            "preprocess": "row_data",
            "rotation": 0,
            "psm": 6,
            "ocr_raw": txt,
            "ocr_normalized": txt,
            "ocr_confidence": float(row_phrase_confidence),
            "alpha_ratio": ar,
            "numeric_ratio": nr,
            "base_score": float(score),
            "tokens": txt,
        })

    return rows


def horizontal_pad_bbox(bbox, img: Image.Image, frac=0.045, max_pad=18):
    """
    Pequena folga horizontal para reduzir cortes de primeira/última letra.
    A altura permanece inalterada para não reintroduzir o título do eixo.
    """
    b = sanitize_bbox(bbox, img.width, img.height)
    w = max(2, b[2] - b[0])
    pad = max(4, min(max_pad, int(round(frac * w))))
    return sanitize_bbox(
        (
            max(0, b[0] - pad),
            b[1],
            min(img.width, b[2] + pad),
            b[3],
        ),
        img.width, img.height
    )


def _ocr_words_region(img: Image.Image, bbox, lang: str, psm=11):
    b = sanitize_bbox(bbox, img.width, img.height)
    crop = crop_pil(img, b)
    arr = np.asarray(crop.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    scale = 2.6
    up = cv2.resize(
        gray, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    d = pytesseract.image_to_data(
        up,
        lang=lang,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DATAFRAME
    )

    if d is None or len(d) == 0:
        return []

    d = d.copy()
    d["text"] = d["text"].fillna("").astype(str)
    d["conf"] = pd.to_numeric(
        d["conf"], errors="coerce"
    ).fillna(-1)

    out = []
    for _, r in d.iterrows():
        txt = normalize_text(r["text"])
        conf = float(r["conf"])
        if not txt or conf < 24:
            continue

        letters = sum(ch.isalpha() for ch in txt)
        if letters < 1:
            continue

        try:
            left = float(r["left"]) / scale + b[0]
            top = float(r["top"]) / scale + b[1]
            width = float(r["width"]) / scale
            height = float(r["height"]) / scale
        except Exception:
            continue

        if width <= 1 or height <= 1:
            continue

        out.append({
            "text": txt,
            "conf": conf,
            "bbox": (
                int(round(left)),
                int(round(top)),
                int(round(left + width)),
                int(round(top + height)),
            ),
            "cx": left + width / 2.0,
            "cy": top + height / 2.0,
            "w": width,
            "h": height,
        })
    return out


def _cluster_words_by_y(words):
    if not words:
        return []

    hs = [w["h"] for w in words]
    ytol = max(3.0, 0.50 * float(np.median(hs)))

    ordered = sorted(words, key=lambda z: z["cy"])
    lines = [[ordered[0]]]

    for w in ordered[1:]:
        center = float(np.mean(
            [z["cy"] for z in lines[-1]]
        ))
        if abs(w["cy"] - center) <= ytol:
            lines[-1].append(w)
        else:
            lines.append([w])

    return [
        sorted(line, key=lambda z: z["bbox"][0])
        for line in lines
    ]


def _phrase_groups_in_line(line):
    if not line:
        return []

    hs = [w["h"] for w in line]
    typical_h = float(np.median(hs)) if hs else 8.0

    gaps = [
        max(0.0, b["bbox"][0] - a["bbox"][2])
        for a, b in zip(line[:-1], line[1:])
    ]
    positive = [g for g in gaps if g > 0.5]

    if positive:
        small = float(np.percentile(
            sorted(positive), 35
        ))
        break_thr = max(
            10.0,
            min(34.0, max(2.5 * small, 1.45 * typical_h))
        )
    else:
        break_thr = max(11.0, 1.45 * typical_h)

    groups = [[line[0]]]
    for prev, cur in zip(line[:-1], line[1:]):
        gap = max(
            0.0,
            cur["bbox"][0] - prev["bbox"][2]
        )
        if gap > break_thr:
            groups.append([cur])
        else:
            groups[-1].append(cur)

    out = []
    for g in groups:
        text = normalize_text(
            " ".join(z["text"] for z in g)
        )
        if sum(ch.isalpha() for ch in text) < 2:
            continue

        x1 = min(z["bbox"][0] for z in g)
        y1 = min(z["bbox"][1] for z in g)
        x2 = max(z["bbox"][2] for z in g)
        y2 = max(z["bbox"][3] for z in g)

        conf = float(np.average(
            [z["conf"] for z in g],
            weights=[max(1.0, z["w"]) for z in g]
        ))

        out.append({
            "text": text,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "center": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
        })
    return out


def continuation_assignments(
    img: Image.Image,
    structural_lane_bbox,
    specs,
    lang: str,
    axislabel_boxes,
):
    """
    Procura SOMENTE linhas imediatamente inferiores à linha estrutural de
    tick labels.

    Regra de segurança contra título do eixo:
    - grupos sobrepostos a axislabel bbox são rejeitados;
    - uma linha inferior com um único grupo central é tratada como provável
      título do eixo e NÃO é incorporada automaticamente;
    - linhas com múltiplos grupos alinhados a slots distintos são aceitas;
    - um único grupo afastado do centro global pode ser aceito como continuação
      de um label extremo.

    Tudo é baseado em geometria/texto observado; F não participa.
    """
    if not specs:
        return {}, {
            "continuation_lines_considered": 0,
            "continuation_groups_accepted": 0,
            "probable_axis_title_groups_rejected": 0,
        }

    lane = sanitize_bbox(
        structural_lane_bbox,
        img.width,
        img.height
    )

    h = max(2, lane[3] - lane[1])
    search = sanitize_bbox(
        (
            lane[0],
            max(0, lane[1] - int(round(0.15 * h))),
            lane[2],
            min(
                img.height,
                lane[3] + max(22, int(round(1.55 * h)))
            ),
        ),
        img.width,
        img.height
    )

    # Usa duas segmentações; escolhe união de grupos depois da filtragem.
    words = []
    for psm in (6, 11):
        words.extend(
            _ocr_words_region(
                img, search, lang, psm=psm
            )
        )

    if not words:
        return {}, {
            "continuation_lines_considered": 0,
            "continuation_groups_accepted": 0,
            "probable_axis_title_groups_rejected": 0,
        }

    # Deduplicação aproximada de palavras oriundas dos dois PSMs.
    dedup = []
    for w in sorted(
        words,
        key=lambda z: (-z["conf"], z["cy"], z["cx"])
    ):
        duplicate = False
        for q in dedup:
            if (
                normalize_text(w["text"])
                == normalize_text(q["text"])
                and abs(w["cx"] - q["cx"]) <= 5
                and abs(w["cy"] - q["cy"]) <= 4
            ):
                duplicate = True
                break
        if not duplicate:
            dedup.append(w)

    lines = _cluster_words_by_y(dedup)

    slot_centers = [
        float(s.get("center"))
        for s in specs
        if s.get("center") is not None
    ]
    if not slot_centers:
        return {}, {
            "continuation_lines_considered": 0,
            "continuation_groups_accepted": 0,
            "probable_axis_title_groups_rejected": 0,
        }

    global_center = float(np.mean(slot_centers))
    span = max(
        1.0,
        max(slot_centers) - min(slot_centers)
    )

    # Tamanhos estruturais para tolerância de associação.
    slot_widths = []
    for s in specs:
        try:
            b = sanitize_bbox(
                s["bbox"], img.width, img.height
            )
            slot_widths.append(max(10.0, b[2] - b[0]))
        except Exception:
            slot_widths.append(60.0)

    assignments = {}
    line_count = 0
    rejected_titles = 0
    accepted_groups = 0

    for line in lines:
        if not line:
            continue

        line_cy = float(np.mean(
            [w["cy"] for w in line]
        ))

        # Linha estrutural principal ou acima dela não é "continuação".
        if line_cy <= lane[3] - 0.10 * h:
            continue

        # Limite conservador: não varre o rodapé inteiro.
        if line_cy > lane[3] + 1.50 * h:
            continue

        groups = _phrase_groups_in_line(line)
        if not groups:
            continue

        line_count += 1
        mapped = []

        for g in groups:
            # Rejeita sobreposição com provável axis-label V4.
            overlaps_axislabel = False
            for ab in axislabel_boxes:
                try:
                    if (
                        overlap_ratio(g["bbox"], ab) >= 0.25
                        or overlap_ratio(ab, g["bbox"]) >= 0.25
                    ):
                        overlaps_axislabel = True
                        break
                except Exception:
                    pass

            if overlaps_axislabel:
                rejected_titles += 1
                continue

            distances = [
                abs(g["center"] - c)
                for c in slot_centers
            ]
            j = int(np.argmin(distances))
            tol = max(
                22.0,
                0.48 * slot_widths[j]
            )

            if distances[j] <= tol:
                mapped.append((j, g))

        if not mapped:
            continue

        distinct_slots = sorted(
            set(j for j, _ in mapped)
        )

        # Uma única frase central inferior é provável título do eixo.
        if len(distinct_slots) == 1:
            j, g = mapped[0]
            central = (
                abs(g["center"] - global_center)
                <= max(45.0, 0.24 * span)
            )

            if central:
                rejected_titles += 1
                continue

        # V9: aplica o gate central a CADA grupo, mesmo quando existem
        # continuacoes laterais na mesma linha. Isso evita que um titulo
        # central ("axis title") seja incorporado ao slot central enquanto
        # continuacoes reais dos slots laterais sao aceitas.
        for j, g in mapped:
            central_group = (
                abs(g["center"] - global_center)
                <= max(45.0, 0.24 * span)
            )

            if central_group:
                rejected_titles += 1
                continue

            prev = assignments.get(j)
            if (
                prev is None
                or g["confidence"] > prev["confidence"]
            ):
                assignments[j] = g
            accepted_groups += 1

    return assignments, {
        "continuation_lines_considered": line_count,
        "continuation_groups_accepted": accepted_groups,
        "probable_axis_title_groups_rejected": rejected_titles,
    }


def slot_candidates_v8(
    full_img,
    structural_slot_bbox,
    continuation_group,
    v7_boxes,
    lang,
    axislabel_boxes,
    row_phrase_text="",
    row_phrase_confidence=-1.0,
):
    """
    Mantém o crop vertical seguro da V5.
    Só amplia verticalmente quando há uma continuação aprovada pelo gate
    geométrico global.
    """
    masked = mask_rectangles(
        full_img, axislabel_boxes
    )

    base_bbox = horizontal_pad_bbox(
        structural_slot_bbox,
        masked,
        frac=0.055,
        max_pad=18
    )

    rows = []
    rows.extend(
        run_ocr_variants(
            crop_pil(masked, base_bbox),
            lang,
            "SAFE_SLOT"
        )
    )

    # Somente se houve continuação aceita, cria um crop composto vertical.
    combined_bbox = base_bbox
    continuation_text = ""
    continuation_conf = -1.0

    if continuation_group is not None:
        cb = sanitize_bbox(
            continuation_group["bbox"],
            masked.width,
            masked.height
        )

        combined_bbox = sanitize_bbox(
            (
                min(base_bbox[0], cb[0] - 5),
                min(base_bbox[1], cb[1] - 3),
                max(base_bbox[2], cb[2] + 5),
                max(base_bbox[3], cb[3] + 3),
            ),
            masked.width,
            masked.height
        )

        continuation_text, continuation_conf = refine_continuation_phrase(
            full_img,
            continuation_group,
            structural_slot_bbox,
            lang,
            axislabel_boxes,
        )

        rows.extend(
            run_ocr_variants(
                crop_pil(masked, combined_bbox),
                lang,
                "CONTINUATION_SLOT"
            )
        )

    # Caixas V7 apenas como suporte, limitadas ao crop seguro/composto.
    for j, b in enumerate(v7_boxes, start=1):
        if (
            overlap_ratio(b, combined_bbox) < 0.18
            and overlap_ratio(combined_bbox, b) < 0.18
        ):
            continue

        ix1 = max(b[0], combined_bbox[0])
        iy1 = max(b[1], combined_bbox[1])
        ix2 = min(b[2], combined_bbox[2])
        iy2 = min(b[3], combined_bbox[3])

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        ib = sanitize_bbox(
            (ix1, iy1, ix2, iy2),
            masked.width,
            masked.height
        )

        rows.extend(
            run_ocr_variants(
                crop_pil(masked, ib),
                lang,
                f"TEXT_SUPPORT_{j}"
            )
        )

    # Evidência original da linha principal.
    row_text = normalize_text(
        row_phrase_text
    )
    if row_text and float(row_phrase_confidence) >= 20:
        ar = alpha_ratio(row_text)
        nr = numeric_ratio(row_text)
        lexical = min(
            1.0,
            len([c for c in row_text if c.isalpha()])
            / 12.0
        )
        score = (
            max(
                0.0,
                float(row_phrase_confidence)
            ) * 0.62
            + 18.0 * ar
            + 8.0 * lexical
            - 8.0 * nr
        )

        rows.append({
            "source": "ROW_PHRASE",
            "preprocess": "row_data",
            "rotation": 0,
            "psm": 6,
            "ocr_raw": row_text,
            "ocr_normalized": row_text,
            "ocr_confidence": float(
                row_phrase_confidence
            ),
            "alpha_ratio": ar,
            "numeric_ratio": nr,
            "base_score": float(score),
            "tokens": row_text,
        })

    # Composição explícita de duas linhas observadas do MESMO slot.
    # Ex.: "AGROPECUARIA" + "E PESCA".
    if row_text and continuation_text:
        composed = normalize_text(
            f"{row_text} {continuation_text}"
        )
        conf = float(np.mean([
            max(0.0, float(row_phrase_confidence)),
            max(0.0, continuation_conf),
        ]))
        ar = alpha_ratio(composed)
        nr = numeric_ratio(composed)
        lexical = min(
            1.0,
            len([c for c in composed if c.isalpha()])
            / 14.0
        )

        rows.append({
            "source": "ROW_PLUS_CONTINUATION",
            "preprocess": "spatial_composition",
            "rotation": 0,
            "psm": 6,
            "ocr_raw": composed,
            "ocr_normalized": composed,
            "ocr_confidence": conf,
            "alpha_ratio": ar,
            "numeric_ratio": nr,
            "base_score": float(
                conf * 0.70
                + 20.0 * ar
                + 10.0 * lexical
                - 7.0 * nr
                + 10.0
            ),
            "tokens": composed,
        })

    return rows, combined_bbox, continuation_text


def refine_continuation_phrase(
    img: Image.Image,
    continuation_group,
    structural_slot_bbox,
    lang: str,
    axislabel_boxes,
):
    """
    Depois que uma linha inferior já passou pelo gate geométrico,
    relê apenas a estreita faixa vertical daquela continuação dentro do slot.

    Exemplo esperado:
        linha principal: AGROPECUARIA
        continuação:     E PESCA

    Esta função não conhece F. Ela apenas melhora o OCR de uma região
    espacial já aprovada como continuação.
    """
    if continuation_group is None:
        return "", -1.0

    try:
        gb = sanitize_bbox(
            continuation_group["bbox"],
            img.width,
            img.height
        )
        sb = sanitize_bbox(
            structural_slot_bbox,
            img.width,
            img.height
        )
    except Exception:
        return (
            normalize_text(
                continuation_group.get("text", "")
            ),
            float(
                continuation_group.get("confidence", -1.0)
            ),
        )

    masked = mask_rectangles(
        img, axislabel_boxes
    )

    # Mantém o X do slot, mas usa apenas a altura da linha de continuação.
    pad_x = max(
        5,
        min(14, int(round(0.04 * (sb[2] - sb[0]))))
    )
    pad_y = max(
        3,
        min(8, int(round(0.35 * (gb[3] - gb[1]))))
    )

    rb = sanitize_bbox(
        (
            max(0, sb[0] - pad_x),
            max(0, gb[1] - pad_y),
            min(masked.width, sb[2] + pad_x),
            min(masked.height, gb[3] + pad_y),
        ),
        masked.width,
        masked.height
    )

    crop = crop_pil(
        masked, rb
    )

    candidates = run_ocr_variants(
        crop,
        lang,
        "CONTINUATION_REOCR"
    )

    # A linha já foi aceita geometricamente. Seleciona o melhor candidato
    # lexical observado, priorizando confiança + proporção alfabética.
    usable = []
    for c in candidates:
        text = normalize_text(
            c.get("ocr_normalized", "")
        )
        letters = sum(
            ch.isalpha() for ch in text
        )
        if letters < 1:
            continue

        score = (
            float(c.get("base_score", 0.0))
            + 5.0 * alpha_ratio(text)
            + min(6.0, 0.22 * letters)
        )
        usable.append(
            (score, text, float(c.get("ocr_confidence", -1.0)))
        )

    if usable:
        _, text, conf = max(
            usable,
            key=lambda z: z[0]
        )
        return text, conf

    return (
        normalize_text(
            continuation_group.get("text", "")
        ),
        float(
            continuation_group.get("confidence", -1.0)
        ),
    )


def suspicious_selected_text(text: str):
    """
    Gate de revisão, não autocorreção.

    Sinaliza padrões típicos de corte/bleed de borda:
    - pontuação isolada no início de label alfabético;
    - token alfabético isolado de 1 caractere no fim de uma frase longa.

    Não compara com F e não modifica o texto.
    """
    t = normalize_text(text)
    if not t:
        return False, ""

    raw = str(text).strip()

    # Ex.: -OMERCIO/SERVICOS
    if (
        raw
        and raw[0] in "-_/.,;:"
        and len(re.sub(r"[^A-Za-z]", "", raw)) >= 6
    ):
        return True, "LEADING_EDGE_PUNCTUATION"

    # Ex.: INFRA-ESTRUTURA C
    toks = [
        z for z in t.split()
        if any(ch.isalpha() for ch in z)
    ]
    if (
        len(toks) >= 2
        and len(toks[-1]) == 1
        and sum(ch.isalpha() for ch in t) >= 8
    ):
        return True, "TRAILING_SINGLE_LETTER"

    return False, ""

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
    vals = []
    for x in list(primary) + list(secondary):
        try:
            fx = float(x)
            if math.isfinite(fx):
                vals.append(fx)
        except Exception:
            continue
    vals = sorted(vals)
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

    x1, y1, x2, y2 = sanitize_bbox(lane_bbox)
    low, high = (x1, x2) if axis == "X" else (y1, y2)

    valid = []
    for c in centers:
        try:
            fc = float(c)
        except Exception:
            continue
        if math.isfinite(fc) and low <= fc <= high:
            valid.append(fc)

    if not valid:
        return []

    valid = sorted(valid)
    dedup = [valid[0]]
    for c in valid[1:]:
        if abs(c - dedup[-1]) >= 4.0:
            dedup.append(c)
        else:
            dedup[-1] = (dedup[-1] + c) / 2.0
    centers = dedup

    bounds = []
    for i, c in enumerate(centers):
        prev_c = centers[i - 1] if i > 0 else None
        next_c = centers[i + 1] if i + 1 < len(centers) else None

        if axis == "X":
            raw_left = x1 if prev_c is None else int(round((prev_c + c) / 2))
            raw_right = x2 if next_c is None else int(round((c + next_c) / 2))
            if raw_right <= raw_left:
                continue
            half = max(28, int(0.58 * (raw_right - raw_left)))
            left = max(x1, int(round(c - half)))
            right = min(x2, int(round(c + half)))
            if right <= left:
                continue
            bounds.append(sanitize_bbox((left, y1, right, y2)))
        else:
            raw_top = y1 if prev_c is None else int(round((prev_c + c) / 2))
            raw_bottom = y2 if next_c is None else int(round((c + next_c) / 2))
            if raw_bottom <= raw_top:
                continue
            half = max(22, int(0.58 * (raw_bottom - raw_top)))
            top = max(y1, int(round(c - half)))
            bottom = min(y2, int(round(c + half)))
            if bottom <= top:
                continue
            bounds.append(sanitize_bbox((x1, top, x2, bottom)))

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
    x1, y1, x2, y2 = sanitize_bbox(b, w, h)
    dx = int((x2 - x1) * frac)
    dy = int((y2 - y1) * frac)
    return sanitize_bbox((
        max(0, x1 - dx), max(0, y1 - dy),
        min(w, x2 + dx), min(h, y2 + dy)
    ), w, h)


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
                def phrase_rank(x):
                    t = x["ocr_normalized"]
                    letters = sum(ch.isalpha() for ch in t)
                    words = len([w for w in t.split() if any(ch.isalpha() for ch in w)])
                    completeness_bonus = min(8.0, 0.12 * letters + 0.9 * max(0, words - 1))
                    return x["base_score"] + completeness_bonus

                best = max(c["members"], key=phrase_rank)
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
        # V9: recurrence across independent OCR variants is stronger
        # evidence than mere phrase length. The previous cap at 22 made
        # support=5 and support=18 effectively equivalent.
        consensus_bonus = min(
            34.0,
            2.0 * support + 4.0 * source_support
        )
        c["support"] = support
        c["source_support"] = source_support
        c["mean_confidence"] = float(np.mean(confs)) if confs else 0.0
        c["max_confidence"] = float(np.max(confs)) if confs else 0.0
        letters = sum(ch.isalpha() for ch in rep)
        words = len([w for w in rep.split() if any(ch.isalpha() for ch in w)])
        phrase_bonus = min(8.0, 0.12 * letters + 0.9 * max(0, words - 1))
        c["score"] = float(np.max(base) + consensus_bonus + phrase_bonus)
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
        high_consensus_override = (
            best["support"] >= 8
            and best["max_confidence"] >= 90.0
            and (
                second is None
                or best["support"] >= 1.5 * max(
                    1, second.get("support", 0)
                )
            )
        )

        if (
            second is not None
            and margin < AMBIGUITY_SCORE_MARGIN
            and not high_consensus_override
        ):
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

    pb = sanitize_bbox(plot_bbox, canvas.width, canvas.height)
    lb = sanitize_bbox(lane_bbox, canvas.width, canvas.height)

    d.rectangle(pb, outline=(40, 40, 40), width=2)
    d.rectangle(lb, outline=(110, 110, 110), width=2)

    for r in slot_results:
        try:
            b = sanitize_bbox(r["slot_bbox"], canvas.width, canvas.height)
        except Exception:
            continue
        d.rectangle(b, outline=(0, 0, 0), width=2)
        label = f"{r['slot_index']}:{r['selection_status']}"
        d.text((int(b[0] + 2), int(b[1] + 2)), label, fill=(0, 0, 0))

    d.text((8, 8), str(title), fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_cat_contact_sheet(images_df, root, out_path, cols=3):
    """
    Prancha própria da B-A-CAT V9.
    Não reutiliza make_contact_sheet da B-A V7 porque aquele renderizador
    exige colunas específicas da estrutura V7.
    """
    if images_df is None or len(images_df) == 0:
        return

    g = images_df.sort_values(["profile", "filename"]).reset_index(drop=True)
    cols = max(1, int(cols))
    rows = int(math.ceil(len(g) / cols))

    cell_w = 460
    image_h = 285
    text_h = 105
    cell_h = image_h + text_h
    margin = 10

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    font = ImageFont.load_default()

    for i, row in g.iterrows():
        rr = i // cols
        cc = i % cols
        ox = cc * cell_w
        oy = rr * cell_h

        path = root / str(row["filename"])
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((cell_w - 2 * margin, image_h - 2 * margin))
            px = ox + (cell_w - im.width) // 2
            py = oy + margin + (image_h - 2 * margin - im.height) // 2
            sheet.paste(im, (int(px), int(py)))
        except Exception as exc:
            d = ImageDraw.Draw(sheet)
            d.text((ox + margin, oy + margin), f"ERRO imagem: {exc}", fill="black", font=font)

        d = ImageDraw.Draw(sheet)
        labels = str(row.get("categorical_selected_labels_spatial", ""))
        if len(labels) > 90:
            labels = labels[:87] + "..."

        lines = [
            str(row.get("filename", "")),
            (
                f"orient={row.get('bar_orientation_observed', '')} | "
                f"cat={row.get('categorical_axis_observed', '')} | "
                f"status={row.get('categorical_axis_status', '')}"
            ),
            f"slots={row.get('categorical_n_slots', '')} | {labels}",
        ]

        ty = oy + image_h + 4
        for line in lines:
            d.text((ox + margin, int(ty)), line, fill="black", font=font)
            ty += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def process_image(path, profile, lang, outdir):
    img = Image.open(path).convert("RGB")
    pres = ba_v7.v4.analyze(path)
    axes = ba_v7.v4.axis_v3.analyze(path)

    plot_bbox, plot_source = extract_plot_bbox(
        pres, axes, img.width, img.height
    )

    v7_crop_dir = outdir / "v7_aux_crops"
    ar_x, lrs_x = run_v7_axis_boxes(
        img, pres, axes, profile, "X",
        lang, v7_crop_dir, path.name
    )
    ar_y, lrs_y = run_v7_axis_boxes(
        img, pres, axes, profile, "Y",
        lang, v7_crop_dir, path.name
    )

    orientation = infer_orientation_from_tick_boxes(
        img, ar_x, lrs_x, ar_y, lrs_y,
        plot_bbox, lang
    )
    cat_axis = orientation["categorical_axis"]

    image_row = {
        "filename": path.name,
        "profile": profile,
        "plot_bbox": bbox_to_text(plot_bbox),
        "plot_bbox_source": plot_source,
        "orientation_evidence_source":
            orientation["orientation_evidence_source"],
        "bar_orientation_observed":
            orientation["orientation"],
        "categorical_axis_observed": cat_axis,
        "orientation_vertical_score":
            orientation["vertical_score"],
        "orientation_horizontal_score":
            orientation["horizontal_score"],
        "observed_x_alpha_score":
            orientation["x_alpha_score"],
        "observed_x_numeric_score":
            orientation["x_numeric_score"],
        "observed_y_alpha_score":
            orientation["y_alpha_score"],
        "observed_y_numeric_score":
            orientation["y_numeric_score"],
        "x_lane_best_text":
            orientation["x_lane_best_text"],
        "y_lane_best_text":
            orientation["y_lane_best_text"],
    }

    if not cat_axis:
        image_row.update({
            "categorical_axis_status":
                "ORIENTATION_AMBIGUOUS",
            "categorical_n_slots": 0,
            "categorical_selected_labels_spatial": "",
        })
        return image_row, [], []

    cat_rows = (
        lrs_x if cat_axis == "X" else lrs_y
    )
    _, raw_v7_boxes = centers_from_v7_labels(
        cat_rows, cat_axis
    )
    v7_boxes = robust_filter_text_boxes(
        raw_v7_boxes, cat_axis
    )

    lane_bbox = v4_layer_bbox(
        pres, cat_axis, img
    )
    if lane_bbox is None and cat_axis == "X":
        lane_bbox = v7_category_band_bbox(
            ar_x, img
        )
    if lane_bbox is None:
        lane_bbox = tight_lane_from_boxes(
            v7_boxes, img, cat_axis
        )

    if lane_bbox is None:
        image_row.update({
            "categorical_axis_status":
                "CATEGORICAL_FAILED",
            "categorical_n_slots": 0,
            "categorical_selected_labels_spatial": "",
        })
        return image_row, [], []

    # Estrutura-base exatamente na linha da V5.
    lane_bbox = widen_x_tick_lane(
        lane_bbox, img, cat_axis
    )
    lane_bbox, lane_refinement = (
        refine_categorical_lane_row(
            img, lane_bbox, plot_bbox,
            cat_axis, lang
        )
    )

    axislabel_boxes = v4_axislabel_bboxes(
        pres, cat_axis, img
    )

    # Slots preservam o mecanismo da V5.
    row_info = row_word_groups(
        img, lane_bbox, lang
    )
    row_specs = row_primary_slot_specs(
        row_info, lane_bbox, img
    )

    bar_plot_bbox = plot_bbox
    if (
        cat_axis == "X"
        and orientation["orientation"] == "VERTICAL"
    ):
        px1, py1, px2, py2 = plot_bbox
        bar_plot_bbox = sanitize_bbox(
            (
                max(
                    0,
                    int(round(0.02 * img.width))
                ),
                py1,
                min(
                    img.width,
                    int(round(0.98 * img.width))
                ),
                py2,
            ),
            img.width,
            img.height
        )

    bar_info = bar_category_centers_v4(
        img,
        bar_plot_bbox,
        orientation["orientation"]
    )
    bar_specs = bar_primary_slot_specs(
        bar_info,
        lane_bbox,
        bar_plot_bbox,
        cat_axis,
        img
    )

    specs, slot_strategy = (
        reconcile_row_and_bar_specs(
            row_specs,
            bar_specs,
            lane_bbox,
            img
        )
    )

    if not specs:
        specs = category_anchor_specs(
            pres,
            v7_boxes,
            orientation,
            lane_bbox,
            cat_axis,
            img,
            plot_bbox
        )
        slot_strategy = "MULTI_EVIDENCE_FALLBACK"

    # V7: expansão vertical SOMENTE para grupos de continuação aprovados.
    continuation_map, continuation_diag = (
        continuation_assignments(
            img,
            lane_bbox,
            specs,
            lang,
            axislabel_boxes
        )
    )

    slot_rows = []
    cand_rows = []

    for idx, spec in enumerate(
        specs, start=1
    ):
        structural_bbox = spec["bbox"]
        row_text = spec.get(
            "row_phrase_text", ""
        )
        row_conf = spec.get(
            "row_phrase_confidence", -1.0
        )

        continuation = continuation_map.get(
            idx - 1
        )

        rows, ocr_bbox, continuation_text = (
            slot_candidates_v8(
                img,
                structural_bbox,
                continuation,
                v7_boxes,
                lang,
                axislabel_boxes,
                row_phrase_text=row_text,
                row_phrase_confidence=row_conf,
            )
        )

        sel = select_slot(rows)

        for c in rows:
            cand_rows.append({
                "filename": path.name,
                "profile": profile,
                "categorical_axis_observed":
                    cat_axis,
                "slot_strategy":
                    slot_strategy,
                "lane_refinement":
                    lane_refinement,
                "slot_index": idx,
                "structural_slot_bbox":
                    bbox_to_text(structural_bbox),
                "ocr_slot_bbox":
                    bbox_to_text(ocr_bbox),
                "continuation_text":
                    continuation_text,
                "anchor_sources": spec.get(
                    "anchor_sources", ""
                ),
                "anchor_strength": spec.get(
                    "anchor_strength", 0.0
                ),
                **{
                    k: v
                    for k, v in c.items()
                    if k != "tokens"
                },
                "tokens": c.get(
                    "tokens", ""
                ),
            })

        slot_rows.append({
            "filename": path.name,
            "profile": profile,
            "categorical_axis_observed":
                cat_axis,
            "slot_strategy":
                slot_strategy,
            "lane_refinement":
                lane_refinement,
            "slot_index": idx,
            "slot_center":
                spec.get("center"),
            "structural_slot_bbox":
                bbox_to_text(structural_bbox),
            "ocr_slot_bbox":
                bbox_to_text(ocr_bbox),
            "continuation_text":
                continuation_text,
            "anchor_sources":
                spec.get(
                    "anchor_sources", ""
                ),
            "anchor_strength":
                spec.get(
                    "anchor_strength", 0.0
                ),
            "row_phrase_text":
                row_text,
            "row_phrase_confidence":
                row_conf,
            "selection_status":
                sel["status"],
            "categorical_selected_text":
                sel["selected_text"],
            "categorical_selection_score":
                sel["score"],
            "categorical_selection_support":
                sel["support"],
            "categorical_selection_confidence":
                sel["confidence"],
            "n_raw_candidates":
                len(rows),
        })

    status = axis_status(
        slot_rows
    )

    # Gate A: fallback estrutural não pode declarar COMPLETE sozinho.
    # BI_006 mostrou que duas caixas OCR podem ser "completas" internamente
    # e ainda não representar toda a estrutura observada.
    if (
        status == "CATEGORICAL_COMPLETE"
        and slot_strategy == "MULTI_EVIDENCE_FALLBACK"
    ):
        status = "CATEGORICAL_AMBIGUOUS"

    # Gate B: conflito entre uma linha textual parcialmente detectada e
    # BAR_PRIMARY. Ex.: BC_048 tinha 3 slots geométricos, mas apenas 2 grupos
    # textuais razoáveis; o resultado não deve ser COMPLETE silenciosamente.
    row_n = len(
        row_info.get("groups", [])
    )
    row_q = float(
        row_info.get("quality", 0.0)
    )
    if (
        status == "CATEGORICAL_COMPLETE"
        and slot_strategy == "BAR_PRIMARY"
        and row_q >= 0.45
        and 0 < row_n < len(slot_rows)
    ):
        status = "CATEGORICAL_AMBIGUOUS"

    # Gate C: sinais de bleed/corte nas bordas viram revisão.
    edge_flags = []
    for r in slot_rows:
        bad, reason = suspicious_selected_text(
            r.get("categorical_selected_text", "")
        )
        r["suspicious_edge_text"] = int(bad)
        r["suspicious_edge_reason"] = reason
        if bad:
            edge_flags.append(
                f"{r.get('slot_index')}:{reason}"
            )

    if (
        status == "CATEGORICAL_COMPLETE"
        and edge_flags
    ):
        status = "CATEGORICAL_AMBIGUOUS"

    selected_spatial = " | ".join(
        (
            r["categorical_selected_text"]
            if r["categorical_selected_text"]
            else f"<{r['selection_status']}>"
        )
        for r in slot_rows
    )

    image_row.update({
        "categorical_lane_bbox":
            bbox_to_text(lane_bbox),
        "categorical_lane_refinement":
            lane_refinement,
        "categorical_slot_strategy":
            slot_strategy,
        "row_group_quality":
            row_info.get("quality", 0.0),
        "row_group_psm":
            row_info.get("psm"),
        "row_groups_detected":
            len(row_info.get("groups", [])),
        "row_groups_text":
            " | ".join(
                g["text"]
                for g in row_info.get(
                    "groups", []
                )
            ),
        "bar_rectangles_detected":
            bar_info["n_rects"],
        "bar_category_groups_detected":
            len(bar_info["group_centers"]),
        "bar_group_confidence":
            bar_info["confidence"],
        "bar_group_centers":
            "|".join(
                f"{v:.1f}"
                for v in bar_info[
                    "group_centers"
                ]
            ),
        "continuation_lines_considered":
            continuation_diag[
                "continuation_lines_considered"
            ],
        "continuation_groups_accepted":
            continuation_diag[
                "continuation_groups_accepted"
            ],
        "probable_axis_title_groups_rejected":
            continuation_diag[
                "probable_axis_title_groups_rejected"
            ],
        "categorical_axis_status":
            status,
        "categorical_n_slots":
            len(slot_rows),
        "categorical_selected_labels_spatial":
            selected_spatial,
        "categorical_n_selected":
            sum(
                r["selection_status"]
                in (
                    "CONSENSUS",
                    "SINGLE_HIGH_CONF",
                )
                for r in slot_rows
            ),
        "categorical_n_ambiguous":
            sum(
                r["selection_status"]
                == "AMBIGUOUS"
                for r in slot_rows
            ),
        "categorical_n_unreadable":
            sum(
                r["selection_status"]
                == "UNREADABLE"
                for r in slot_rows
            ),
        "categorical_n_axislabel_masked":
            len(axislabel_boxes),
        "categorical_n_suspicious_edge_slots":
            len(edge_flags),
        "categorical_suspicious_edge_flags":
            " | ".join(edge_flags),
    })

    overlay_path = (
        outdir / "overlays" / profile /
        f"{path.stem}_BA_CAT_V9.png"
    )
    draw_overlay(
        img,
        plot_bbox,
        lane_bbox,
        slot_rows,
        overlay_path,
        (
            f"{path.name} | "
            f"{orientation['orientation']} | "
            f"{status} | {slot_strategy}"
        ),
    )
    image_row["overlay_path"] = str(
        overlay_path
    )

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
        outdir / "ba_cat_v9_manifest.csv",
        index=False, encoding="utf-8-sig"
    )

    print("=" * 80)
    print("B-A-CAT V9 — CALIBRAÇÃO — CONSENSUS + CENTRAL TITLE GATE")
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
        outdir / "ba_cat_v9_images.csv",
        index=False, encoding="utf-8-sig"
    )
    slots_df.to_csv(
        outdir / "ba_cat_v9_slots.csv",
        index=False, encoding="utf-8-sig"
    )
    candidates_df.to_csv(
        outdir / "ba_cat_v9_candidates.csv",
        index=False, encoding="utf-8-sig"
    )

    if errors_df.empty:
        write_empty_csv(
            outdir / "ba_cat_v9_errors.csv",
            ["filename", "profile", "error_type", "error_message"]
        )
    else:
        errors_df.to_csv(
            outdir / "ba_cat_v9_errors.csv",
            index=False, encoding="utf-8-sig"
        )

    audit = make_audit_all(images_df)
    audit.to_csv(
        outdir / "ba_cat_v9_audit_all.csv",
        index=False, encoding="utf-8-sig"
    )

    priority = make_priority_review(audit)
    priority.to_csv(
        outdir / "ba_cat_v9_priority_review.csv",
        index=False, encoding="utf-8-sig"
    )

    # Contatos próprios da B-A-CAT V9.
    if not images_df.empty:
        for profile in PROFILES:
            g = images_df[images_df["profile"] == profile]
            if len(g):
                try:
                    make_cat_contact_sheet(
                        g, root,
                        outdir / f"contact_BA_CAT_V9_{profile}.png"
                    )
                except Exception as exc:
                    print(
                        f"AVISO: não foi possível gerar contact sheet {profile}: "
                        f"{type(exc).__name__}: {exc}"
                    )

    # Resumo automático — diagnóstico, não acurácia.
    lines = [
        "B-A-CAT V9 — CALIBRAÇÃO",
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
        "- V9: preserva a estrutura da V8 e fortalece consenso entre OCR independentes;",
        "- V9: mantém re-OCR localizado das continuações já aprovadas;",
        "- V9: qualquer continuação secundária central é bloqueada como provável axis title;",
        "- V9: mantém gates de fallback, conflito BAR/ROW e artefatos de borda;",
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

    summary_path = outdir / "ba_cat_v9_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Concluído.")
    print("Resumo:", summary_path)
    print("Auditoria:", outdir / "ba_cat_v9_audit_all.csv")
    print("Slots:", outdir / "ba_cat_v9_slots.csv")
    print("Candidatos:", outdir / "ba_cat_v9_candidates.csv")
    print("Prioridade:", outdir / "ba_cat_v9_priority_review.csv")
    print("Envie também contact_BA_CAT_V9_BI.png e contact_BA_CAT_V9_BC.png.")


if __name__ == "__main__":
    main()
