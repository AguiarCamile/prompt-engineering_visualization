# -*- coding: utf-8 -*-
r"""
FASE B-A V7 — EXTRAÇÃO TEXTUAL DOS TICK LABELS
==============================================

Objetivo
--------
Refinar a extração textual dos tick labels na MESMA amostra de calibração
utilizada na B-A V1/V1.1. Esta etapa continua sendo de EXTRAÇÃO, e NÃO de
conformidade de conteúdo.

Correções principais da V7
--------------------------
A V7 preserva integralmente a arquitetura conservadora da V6 e modifica
somente dois pontos demonstrados pela auditoria de calibração:

1. CLIPPING LONGITUDINAL DOS CANDIDATOS NUMÉRICOS
   - Y: além da proximidade horizontal ao eixo, o centro da caixa deve estar
     dentro da extensão vertical do plot, com pequena tolerância geométrica.
   - X: além da proximidade vertical ao eixo, o centro da caixa deve estar
     dentro da extensão horizontal do plot, com pequena tolerância.
   - caixas rejeitadas longitudinalmente NÃO retornam por fallback.

2. NOTAÇÃO CIENTÍFICA
   - duas micro-ROIs por eixo: uma externa e outra atravessando levemente
     a borda do plot;
   - parsing científico estrito e normalização OCR conservadora;
   - mantém o consenso ponderado da V6;
   - acrescenta uma rota SINGLE_HIGH_CONF para uma leitura científica única,
     exata, de alta confiança e sem concorrente forte incompatível.

Nenhuma outra parte do extrator é alterada.

Princípio metodológico
----------------------
O OCR/extrator NÃO recebe:
- categorias esperadas;
- anos esperados;
- valores esperados;
- sequência de ticks da referência;
- especificação F como vocabulário de OCR.

Ele recebe apenas a NATUREZA do eixo:
categorical / temporal / numeric.

Portanto, esta fase mede a capacidade de RECUPERAR o que está visível, sem
"forçar" a resposta correta.

A V4 de presença permanece congelada e é verificada por SHA-256.

Amostra
-------
A V2 tenta reutilizar exatamente o manifesto da calibração anterior:
1. _ticklabel_content_extraction_ba_v1_1/ticklabel_ba_v1_1_manifest.csv
2. _ticklabel_content_extraction_ba_v1/ticklabel_ba_v1_manifest.csv

Se nenhum manifesto existir, recria a amostra balanceada com a mesma semente:
3 unidades × 4 condições × 6 perfis = 72 imagens.

Saída padrão
------------
C:\Users\Labvis\Downloads\imagens3120\imagens\
_ticklabel_content_extraction_ba_v2
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
import shutil
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import pytesseract
    from pytesseract import Output
except Exception as exc:
    raise RuntimeError(
        "pytesseract não está instalado. Execute: "
        "python -m pip install pytesseract"
    ) from exc

import ticklabel_presence_calibration_v4 as v4


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
OUT_DIRNAME = "_ticklabel_content_extraction_ba_v7"

EXPECTED_V4_SHA256 = "d15c325290c7596632f9f4c62d907e7a78d0a794862b690c4c4691ed87ea86b2"

PROFILES = ("BI","BC","LI","LC","SI","SC")
CONDITIONS = ("00","01","10","11")
N_UNITS_PER_PROFILE_CONDITION = 3
N_UNITS_PER_PROFILE_FALLBACK = 12
SEED = 20260831

NAME_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)

AXIS_KIND = {
    "BI": {"X":"categorical", "Y":"numeric"},
    "BC": {"X":"categorical", "Y":"numeric"},
    "LI": {"X":"temporal",    "Y":"numeric"},
    "LC": {"X":"temporal",    "Y":"numeric"},
    "SI": {"X":"numeric",     "Y":"numeric"},
    "SC": {"X":"numeric",     "Y":"numeric"},
}

# Diagnóstico da extração, NÃO conformidade.
MIN_INDIVIDUAL_SUCCESS_FRAC = 0.50
EARLY_STOP_CONF = 60.0

# Regex para fatores científicos.
SUPERSCRIPT_MAP = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻",
    "0123456789+-"
)

SCALE_PATTERNS = [
    re.compile(r"(?i)\b1\s*[eE]\s*([+-]?\d{1,3})\b"),
    re.compile(r"(?i)(?:×|x|\*)\s*10\s*\^?\s*([+-]?\d{1,3})"),
    re.compile(r"(?i)\b10\s*\^\s*([+-]?\d{1,3})\b"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--crosswalk", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tesseract", type=Path, default=None)
    return p.parse_args()


def check_v4_hash():
    path = Path(v4.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_V4_SHA256:
        raise RuntimeError(
            "ticklabel_presence_calibration_v4.py foi alterado.\n"
            "A Fase B-A V2 deve reutilizar exatamente a V4 congelada.\n"
            f"SHA esperado: {EXPECTED_V4_SHA256}\n"
            f"SHA encontrado: {actual}"
        )
    return actual


def configure_tesseract(explicit=None):
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())

    which = shutil.which("tesseract")
    if which:
        candidates.append(Path(which))

    candidates.extend([
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ])

    for p in candidates:
        if p.exists():
            pytesseract.pytesseract.tesseract_cmd = str(p)
            break

    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR não encontrado. Se necessário use "
            "--tesseract CAMINHO\\tesseract.exe"
        ) from exc

    try:
        langs = set(pytesseract.get_languages(config=""))
    except Exception:
        langs = {"eng"}

    if "por" in langs and "eng" in langs:
        lang = "por+eng"
    elif "por" in langs:
        lang = "por"
    elif "eng" in langs:
        lang = "eng"
    else:
        lang = next(iter(langs)) if langs else "eng"

    return version, lang, sorted(langs)


def inventory(root):
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        m = NAME_RE.match(p.name)
        if not m:
            continue
        profile = m.group(1).upper()
        unit = int(m.group(2))
        repeat = int(m.group(3))
        rows.append({
            "filename":p.name,
            "profile":profile,
            "unit_id":f"{profile}_{unit:03d}",
            "unit_number":unit,
            "repeat":repeat,
        })
    return pd.DataFrame(rows)


def normalize_crosswalk_columns(cw):
    low = {str(c).lower():c for c in cw.columns}
    rename = {}

    opts = {
        "condition_code":[
            "condition_code","condition","prompt_condition_code",
            "condicao_codigo","condition_id"
        ],
        "condition_label":[
            "condition_label","condition_name","prompt_condition",
            "condicao"
        ],
        "participant_id":[
            "participant_id","participant","participant_number",
            "participante","p"
        ],
    }

    for target,names in opts.items():
        if target in cw.columns:
            continue
        for n in names:
            if n.lower() in low:
                rename[low[n.lower()]] = target
                break

    return cw.rename(columns=rename)


def find_crosswalk(root, explicit=None):
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())

    candidates.extend([
        root/"_experimental_crosswalk_v3"/"experimental_unit_crosswalk_v3.csv",
        root.parent/"_experimental_crosswalk_v3"/"experimental_unit_crosswalk_v3.csv",
    ])

    for p in candidates:
        if p.exists():
            return p.resolve()

    return None


def load_unit_crosswalk(path):
    cw = pd.read_csv(path,dtype=str)
    cw = normalize_crosswalk_columns(cw)

    if "unit_id" not in cw.columns or "condition_code" not in cw.columns:
        raise RuntimeError(
            "Crosswalk encontrado, mas sem unit_id/condition_code."
        )

    cw["condition_code"] = (
        cw["condition_code"].astype(str)
        .str.replace(r"\.0$","",regex=True)
        .str.zfill(2)
    )

    keep = ["unit_id","condition_code"]
    for c in ("condition_label","participant_id"):
        if c in cw.columns:
            keep.append(c)

    return cw[keep].drop_duplicates("unit_id")


def find_previous_manifest(root, explicit=None):
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Manifesto informado não encontrado: {p}")
        return p

    candidates = [
        root/"_ticklabel_content_extraction_ba_v6"/
        "ticklabel_ba_v6_manifest.csv",

        root/"_ticklabel_content_extraction_ba_v5"/
        "ticklabel_ba_v5_manifest.csv",

        root/"_ticklabel_content_extraction_ba_v4"/
        "ticklabel_ba_v4_manifest.csv",

        root/"_ticklabel_content_extraction_ba_v3"/
        "ticklabel_ba_v3_manifest.csv",

        root/"_ticklabel_content_extraction_ba_v2"/
        "ticklabel_ba_v2_manifest.csv",

        root/"_ticklabel_content_extraction_ba_v1_1"/
        "ticklabel_ba_v1_1_manifest.csv",

        root/"_ticklabel_content_extraction_ba_v1"/
        "ticklabel_ba_v1_manifest.csv",
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    return None


def sample_from_manifest(manifest_path, inv, crosswalk_path):
    m = pd.read_csv(manifest_path)

    if "filename" not in m.columns:
        raise RuntimeError("Manifesto anterior sem coluna filename.")

    sample = m[["filename"]].drop_duplicates().copy()
    sample = sample.merge(
        inv,
        on="filename",
        how="left",
        validate="one_to_one"
    )

    if sample["profile"].isna().any():
        missing = sample.loc[sample["profile"].isna(),"filename"].tolist()
        raise RuntimeError(
            "Há imagens do manifesto que não existem no inventário atual: "
            + ", ".join(missing[:20])
        )

    if crosswalk_path is not None:
        cw = load_unit_crosswalk(crosswalk_path)
        sample = sample.merge(cw,on="unit_id",how="left")

    if len(sample) != 72:
        raise RuntimeError(
            f"Manifesto anterior tem {len(sample)} imagens; esperado=72."
        )

    return sample.sort_values(
        ["profile","unit_id","repeat"]
    ).reset_index(drop=True)


def select_new_sample(inv, crosswalk_path, seed):
    rng = random.Random(seed)
    selected_units = []
    mode = "FALLBACK_PROFILE_RANDOM"

    if crosswalk_path is not None:
        cw = load_unit_crosswalk(crosswalk_path)
        u = inv[["profile","unit_id"]].drop_duplicates().merge(
            cw,on="unit_id",how="left"
        )

        ok = True
        for profile in PROFILES:
            for cond in CONDITIONS:
                candidates = sorted(
                    u.loc[
                        (u["profile"]==profile) &
                        (u["condition_code"]==cond),
                        "unit_id"
                    ].dropna().unique().tolist()
                )
                if len(candidates) < N_UNITS_PER_PROFILE_CONDITION:
                    ok = False
                    break

                rng.shuffle(candidates)
                selected_units.extend(
                    candidates[:N_UNITS_PER_PROFILE_CONDITION]
                )

            if not ok:
                break

        if ok and len(selected_units)==72:
            mode = "BALANCED_PROFILE_X_CONDITION"
        else:
            selected_units = []

    if not selected_units:
        for profile in PROFILES:
            candidates = sorted(
                inv.loc[
                    inv["profile"]==profile,
                    "unit_id"
                ].unique().tolist()
            )
            rng.shuffle(candidates)
            selected_units.extend(
                candidates[:N_UNITS_PER_PROFILE_FALLBACK]
            )

    rows = []
    for unit_id in selected_units:
        rr = inv[inv["unit_id"]==unit_id].to_dict("records")
        rng.shuffle(rr)
        rows.append(rr[0])

    sample = pd.DataFrame(rows)

    if crosswalk_path is not None:
        cw = load_unit_crosswalk(crosswalk_path)
        sample = sample.merge(cw,on="unit_id",how="left")

    return (
        sample.sort_values(
            ["profile","unit_id","repeat"]
        ).reset_index(drop=True),
        mode
    )


# ============================================================
# Geometria
# ============================================================

def rect_from_box(b):
    if isinstance(b,dict):
        return (
            int(b["x1"]),int(b["y1"]),
            int(b["x2"]),int(b["y2"])
        )
    return tuple(map(int,b))


def clip_rect(rect,W,H):
    x1,y1,x2,y2 = rect
    return (
        max(0,min(W-1,int(x1))),
        max(0,min(H-1,int(y1))),
        max(0,min(W-1,int(x2))),
        max(0,min(H-1,int(y2))),
    )


def intersect_rect(a,b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b

    x1=max(ax1,bx1)
    y1=max(ay1,by1)
    x2=min(ax2,bx2)
    y2=min(ay2,by2)

    if x2<=x1 or y2<=y1:
        return None

    return (x1,y1,x2,y2)


def rect_area(a):
    if a is None:
        return 0
    x1,y1,x2,y2 = a
    return max(0,x2-x1+1)*max(0,y2-y1+1)


def overlap_frac(a,b):
    inter=intersect_rect(a,b)
    return rect_area(inter)/max(1,rect_area(a))


def crop_rect(img,rect,pad=4):
    W,H=img.size
    x1,y1,x2,y2=rect
    x1=max(0,x1-pad)
    y1=max(0,y1-pad)
    x2=min(W-1,x2+pad)
    y2=min(H-1,y2+pad)
    return img.crop((x1,y1,x2+1,y2+1)),(x1,y1,x2,y2)




def filter_numeric_tick_lane(boxes,axes,axis):
    """
    V7: filtro geométrico em DUAS dimensões.

    Primeiro aplica clipping longitudinal pela extensão observada do plot.
    Depois aplica a proximidade transversal ao eixo.

    Y:
      - centro vertical dentro de [plot_top-eps, plot_bottom+eps];
      - borda direita próxima da linha do eixo Y.

    X:
      - centro horizontal dentro de [plot_left-eps, plot_right+eps];
      - topo próximo da linha do eixo X.

    O filtro não conhece valores esperados, F, domínio dos dados ou referência.

    IMPORTANTE:
    caixas rejeitadas longitudinalmente NÃO retornam por fallback.
    """
    if not boxes:
        return boxes,{
            "status":"EMPTY",
            "n_before":0,
            "n_longitudinal_valid":0,
            "n_longitudinal_rejected":0,
            "n_after":0,
            "reference_distance":"",
            "tolerance":"",
            "longitudinal_tolerance":"",
        }

    xl=axes.get("plot_x_left")
    xr=axes.get("plot_x_right")
    yt=axes.get("plot_y_top")
    yb=axes.get("plot_y_bottom")

    if any(v is None or v=="" for v in (xl,xr,yt,yb)):
        return boxes,{
            "status":"UNAVAILABLE",
            "n_before":len(boxes),
            "n_longitudinal_valid":len(boxes),
            "n_longitudinal_rejected":0,
            "n_after":len(boxes),
            "reference_distance":"",
            "tolerance":"",
            "longitudinal_tolerance":"",
        }

    xl,xr,yt,yb=map(float,(xl,xr,yt,yb))
    pw=max(1.0,xr-xl)
    ph=max(1.0,yb-yt)

    # Pequena margem apenas para anti-aliasing/variação da segmentação.
    if axis=="Y":
        long_tol=max(3.0,.035*ph)

        longitudinal=[]

        for r in boxes:
            x1,y1,x2,y2=r
            cy=(float(y1)+float(y2))/2.0

            if (yt-long_tol) <= cy <= (yb+long_tol):
                longitudinal.append(r)

    else:
        long_tol=max(3.0,.035*pw)

        longitudinal=[]

        for r in boxes:
            x1,y1,x2,y2=r
            cx=(float(x1)+float(x2))/2.0

            if (xl-long_tol) <= cx <= (xr+long_tol):
                longitudinal.append(r)

    n_rejected=len(boxes)-len(longitudinal)

    if not longitudinal:
        return [],{
            "status":"LONGITUDINAL_EMPTY",
            "n_before":len(boxes),
            "n_longitudinal_valid":0,
            "n_longitudinal_rejected":n_rejected,
            "n_after":0,
            "reference_distance":"",
            "tolerance":"",
            "longitudinal_tolerance":long_tol,
        }

    scored=[]

    if axis=="Y":
        for r in longitudinal:
            x1,y1,x2,y2=r

            # borda direita do label em relação ao eixo vertical
            d=xl-float(x2)

            if d>=-4:
                scored.append((abs(d),r))

        tol=max(8.0,.022*pw)

    else:
        for r in longitudinal:
            x1,y1,x2,y2=r

            # topo do label em relação ao eixo horizontal
            d=float(y1)-yb

            if d>=-4:
                scored.append((abs(d),r))

        tol=max(8.0,.020*ph)

    if not scored:
        # Conservador: preserva somente as caixas longitudinalmente válidas.
        return longitudinal,{
            "status":"LONGITUDINAL_ONLY",
            "n_before":len(boxes),
            "n_longitudinal_valid":len(longitudinal),
            "n_longitudinal_rejected":n_rejected,
            "n_after":len(longitudinal),
            "reference_distance":"",
            "tolerance":tol,
            "longitudinal_tolerance":long_tol,
        }

    ref=min(d for d,_ in scored)

    kept=[
        r for d,r in scored
        if d<=ref+tol
    ]

    if not kept:
        kept=longitudinal
        status="LONGITUDINAL_ONLY"
    elif len(kept)<len(longitudinal):
        status="FILTERED_2D"
    elif n_rejected>0:
        status="LONGITUDINAL_FILTERED"
    else:
        status="UNCHANGED"

    return kept,{
        "status":status,
        "n_before":len(boxes),
        "n_longitudinal_valid":len(longitudinal),
        "n_longitudinal_rejected":n_rejected,
        "n_after":len(kept),
        "reference_distance":ref,
        "tolerance":tol,
        "longitudinal_tolerance":long_tol,
    }


def crop_numeric_tick_directional(full_img,rect,axes,axis):
    """
    Crop numérico direcional da V4.

    Motivação:
    na V3, o padding simétrico podia reintroduzir a própria linha do eixo
    dentro do crop. Em SI/SC isso fazia o Tesseract ler, por exemplo,
    4 como 44, 6 como 64, 8 como 84.

    O método usa apenas a geometria observada do plot.
    Não usa valor esperado, domínio de dados ou conteúdo de F.
    """
    W,H=full_img.size
    x1,y1,x2,y2=map(int,rect)

    xl=axes.get("plot_x_left")
    yb=axes.get("plot_y_bottom")

    # Padding para o lado externo ao eixo e para as extremidades do texto.
    pad_outer=4
    pad_cross=3
    axis_gap=2

    if axis=="Y" and xl not in (None,""):
        xl=int(xl)

        # Texto dos ticks Y fica à esquerda da linha vertical.
        # Nunca deixa o crop alcançar a linha do eixo.
        safe_right=min(x2, xl-axis_gap)

        if safe_right<=x1:
            safe_right=x2

        ax1=max(0,x1-pad_outer)
        ay1=max(0,y1-pad_cross)
        ax2=min(W-1,safe_right)
        ay2=min(H-1,y2+pad_cross)

        return (
            full_img.crop((ax1,ay1,ax2+1,ay2+1)),
            (ax1,ay1,ax2,ay2),
            "Y_LEFT_OF_AXIS"
        )

    if axis=="X" and yb not in (None,""):
        yb=int(yb)

        # Texto dos ticks X fica abaixo da linha horizontal.
        # Nunca deixa o crop subir sobre a linha do eixo.
        safe_top=max(y1, yb+axis_gap)

        if safe_top>=y2:
            safe_top=y1

        ax1=max(0,x1-pad_cross)
        ay1=max(0,safe_top)
        ax2=min(W-1,x2+pad_cross)
        ay2=min(H-1,y2+pad_outer)

        return (
            full_img.crop((ax1,ay1,ax2+1,ay2+1)),
            (ax1,ay1,ax2,ay2),
            "X_BELOW_AXIS"
        )

    crop,absbox=crop_rect(full_img,rect,pad=4)
    return crop,absbox,"FALLBACK_SYMMETRIC"


def tick_boxes_clipped_to_layer(res,axis,W,H,restrict_rect=None):
    """
    Intersecta as caixas V4 com a camada de texto próxima ao eixo.

    V3:
    - aceita restrict_rect opcional;
    - para X categórico, restrict_rect é a banda de tick labels detectada
      por distribuição espacial de texto, sem usar conteúdo esperado.
    """
    if axis=="X":
        boxes=res.get("_x_boxes",[])
        layer=res.get("_x_layer_rect")
        axislabels=res.get("_x_axislabel",[])
    else:
        boxes=res.get("_y_boxes",[])
        layer=res.get("_y_layer_rect")
        axislabels=res.get("_y_axislabel",[])

    if layer is None:
        return []

    layer=clip_rect(rect_from_box(layer),W,H)

    if restrict_rect is not None:
        rr=clip_rect(restrict_rect,W,H)
        layer2=intersect_rect(layer,rr)
        if layer2 is not None:
            layer=layer2

    axislabel_rects=[
        clip_rect(rect_from_box(a),W,H)
        for a in axislabels
    ]

    out=[]

    for b in boxes:
        br=clip_rect(rect_from_box(b),W,H)
        inter=intersect_rect(br,layer)

        if inter is None or rect_area(inter)<8:
            continue

        is_axislabel = any(
            overlap_frac(br,a)>=0.55
            for a in axislabel_rects
        )

        if is_axislabel and overlap_frac(br,layer)<0.35:
            continue

        out.append(inter)

    if axis=="X":
        out.sort(key=lambda r:(r[0]+r[2])/2)
    else:
        out.sort(key=lambda r:(r[1]+r[3])/2, reverse=True)

    return out


def mask_axis_label_candidates(img,res,axis):
    out=img.copy()
    d=ImageDraw.Draw(out)

    boxes=(
        res.get("_x_axislabel",[])
        if axis=="X"
        else res.get("_y_axislabel",[])
    )

    for b in boxes:
        r=rect_from_box(b)
        d.rectangle(r,fill="white")

    return out


def extract_layer_crop(full_img,res,axis):
    rect=(
        res.get("_x_layer_rect")
        if axis=="X"
        else res.get("_y_layer_rect")
    )

    if rect is None:
        return None,None

    masked=mask_axis_label_candidates(full_img,res,axis)
    W,H=masked.size
    r=clip_rect(rect_from_box(rect),W,H)
    crop,absbox=crop_rect(masked,r,pad=3)
    return crop,absbox


def detect_categorical_tick_band(full_img,axes):
    """
    Detecta a banda de labels categóricos X exclusivamente por geometria.

    Ideia:
    - examina a região logo abaixo da área de plot;
    - agrupa linhas de pixels escuros;
    - favorece grupos de texto distribuídos horizontalmente em vários bins;
    - texto central estreito, típico de axis label, tende a ser rejeitado.

    Não usa palavras esperadas nem valores de F.
    """
    W,H=full_img.size

    vals=[
        axes.get("plot_x_left"),
        axes.get("plot_x_right"),
        axes.get("plot_y_bottom"),
    ]

    if any(v is None or v=="" for v in vals):
        return {
            "status":"UNAVAILABLE",
            "bbox":None,
            "n_groups":0,
            "n_selected_groups":0,
        }

    xl=int(axes.get("plot_x_left"))
    xr=int(axes.get("plot_x_right"))
    yb=int(axes.get("plot_y_bottom"))

    pw=max(1,xr-xl)

    x1=max(0,xl-int(.02*pw))
    x2=min(W-1,xr+int(.02*pw))
    y1=max(0,yb+1)
    y2=min(H-1,yb+int(.18*H))

    if y2<=y1 or x2<=x1:
        return {
            "status":"UNAVAILABLE",
            "bbox":None,
            "n_groups":0,
            "n_selected_groups":0,
        }

    crop=full_img.crop((x1,y1,x2+1,y2+1)).convert("L")
    arr=np.asarray(crop)

    # Texto escuro; evita depender da cor formal do texto.
    ink=(arr<205)

    row_counts=ink.sum(axis=1)
    min_row=max(3,int(.0015*ink.shape[1]))
    active=row_counts>=min_row

    # Grupos de linhas, tolerando pequenos vazios internos.
    groups=[]
    start=None
    last=None

    for i,on in enumerate(active):
        if on:
            if start is None:
                start=i
            last=i
        else:
            if start is not None and last is not None:
                if i-last<=2:
                    continue
                groups.append((start,last))
                start=None
                last=None

    if start is not None:
        groups.append((start,last))

    selected=[]
    diagnostics=[]

    n_bins=24
    bw=max(1,int(np.ceil(ink.shape[1]/n_bins)))

    for g1,g2 in groups:
        sub=ink[g1:g2+1,:]
        col_counts=sub.sum(axis=0)
        cols=np.where(col_counts>0)[0]

        if len(cols)==0:
            continue

        extent=(cols[-1]-cols[0]+1)/max(1,ink.shape[1])

        occupied=0
        for b in range(n_bins):
            a=b*bw
            z=min(ink.shape[1],(b+1)*bw)
            if a>=z:
                continue
            if sub[:,a:z].sum()>=2:
                occupied+=1

        diagnostics.append((g1,g2,occupied,extent))

        # Tick labels categóricos aparecem em várias posições ao longo de X.
        if occupied>=2 and extent>=.15:
            selected.append((g1,g2))

    if not selected:
        return {
            "status":"NOT_FOUND",
            "bbox":None,
            "n_groups":len(groups),
            "n_selected_groups":0,
        }

    # Mantém grupos próximos entre si e próximos ao eixo.
    selected=sorted(selected)
    kept=[selected[0]]

    for g in selected[1:]:
        prev=kept[-1]
        gap=g[0]-prev[1]-1
        if gap<=12:
            kept.append(g)
        else:
            break

    sy1=min(g[0] for g in kept)
    sy2=max(g[1] for g in kept)

    # Margem pequena para antialiasing/descendentes.
    sy1=max(0,sy1-3)
    sy2=min(ink.shape[0]-1,sy2+3)

    bbox=(x1,y1+sy1,x2,y1+sy2)

    return {
        "status":"FOUND",
        "bbox":bbox,
        "n_groups":len(groups),
        "n_selected_groups":len(kept),
    }


def extract_categorical_band_crop(full_img,axes):
    info=detect_categorical_tick_band(full_img,axes)
    if info["bbox"] is None:
        return None,None,info

    crop,absbox=crop_rect(full_img,info["bbox"],pad=2)
    return crop,absbox,info


# ============================================================
# OCR
# ============================================================

def preprocess_variants(pil_img,upscale=3):
    im=pil_img.convert("L")
    im=ImageOps.autocontrast(im)
    im=im.resize(
        (
            max(1,im.width*upscale),
            max(1,im.height*upscale)
        ),
        Image.Resampling.LANCZOS
    )

    arr=np.asarray(im)

    _,otsu=cv2.threshold(
        arr,0,255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )

    return {
        "gray":im,
        "otsu":Image.fromarray(otsu),
    }


def rotate_expand(im,angle):
    if angle==0:
        return im
    return im.rotate(angle,expand=True,fillcolor="white")


def ocr_token_rows(pil_img,lang,psm,whitelist=None):
    """
    Retorna tokens OCR individualmente.

    Esta função é central na V3: números de tokens diferentes não são
    concatenados antes do parsing.
    """
    cfg=f"--oem 3 --psm {psm}"

    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"

    data=pytesseract.image_to_data(
        pil_img,
        lang=lang,
        config=cfg,
        output_type=Output.DATAFRAME,
    )

    if data is None or len(data)==0:
        return []

    data=data.copy()
    data["text"]=data["text"].fillna("").astype(str)
    data["conf"]=pd.to_numeric(data["conf"],errors="coerce")

    good=data[
        (data["text"].str.strip()!="") &
        data["conf"].notna() &
        (data["conf"]>=0)
    ].copy()

    rows=[]

    for _,r in good.iterrows():
        rows.append({
            "text":str(r["text"]).strip(),
            "conf":float(r["conf"]),
            "left":int(r.get("left",0)),
            "top":int(r.get("top",0)),
            "width":int(r.get("width",0)),
            "height":int(r.get("height",0)),
        })

    return rows


def ocr_data(pil_img,lang,psm,whitelist=None):
    rows=ocr_token_rows(
        pil_img,lang,psm,whitelist
    )

    if not rows:
        return "",-1.0

    text=" ".join(r["text"] for r in rows).strip()
    conf=float(np.mean([r["conf"] for r in rows]))

    return text,conf


def strip_accents(s):
    s=unicodedata.normalize("NFKD",str(s))
    return "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )


def normalize_basic(s):
    s=strip_accents(str(s)).upper()
    s=s.replace("–","-").replace("—","-").replace("−","-")
    s=re.sub(r"\s+"," ",s).strip()
    return s


def normalize_categorical(s):
    s=normalize_basic(s)
    s=re.sub(r"[^A-Z0-9/&+\- ]+"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s


def normalize_numeric_text(s):
    s=normalize_basic(s)
    s=s.replace("R$","").replace("$","")
    s=re.sub(r"\s+","",s)
    return s


YEAR_RE=re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")


def parse_years(s):
    return YEAR_RE.findall(normalize_basic(s))


def numeric_token_parse(s):
    """
    Parser numérico ESTRITO da V5.

    Diferença principal:
    - usa fullmatch;
    - não aceita qualquer número escondido dentro de uma palavra;
    - portanto strings como ER0BEE deixam de virar 0.

    Ainda permite correções OCR conservadoras O->0 e I/L->1 quando o token
    já tem estrutura predominantemente numérica.
    """
    raw=str(s).strip()

    if not raw:
        return None

    t=normalize_basic(raw).strip()

    currency=int("R$" in t or "$" in t)
    percent=int("%" in t)

    compact=t
    compact=compact.replace("R$","").replace("$","")
    compact=compact.strip("()[]{}:;")
    compact=re.sub(r"\s+","",compact)

    if not compact:
        return None

    # Antes de qualquer correção, rejeita letras incompatíveis com um token
    # numérico, sufixos usuais ou confusões OCR O/I/L.
    letters="".join(re.findall(r"[A-Z]+",compact))
    allowed_letter_chars=set("OILKMB")
    if letters and any(ch not in allowed_letter_chars for ch in letters):
        return None

    if re.search(r"\d",compact):
        compact=compact.replace("O","0")
        compact=compact.replace("I","1").replace("L","1")

    m=re.fullmatch(
        r"([-+]?\d+(?:[.,]\d+)?)"
        r"(MIL|MI|BI|K|M|B)?"
        r"(%?)",
        compact,
        flags=re.I
    )

    if not m:
        return None

    num_text=m.group(1)
    suffix=(m.group(2) or "").upper()

    q=num_text

    if "," in q and "." in q:
        if q.rfind(",") > q.rfind("."):
            q=q.replace(".","").replace(",",".")
        else:
            q=q.replace(",","")
    elif "," in q:
        parts=q.split(",")
        if len(parts)==2 and len(parts[1])<=2:
            q=q.replace(",",".")
        else:
            q=q.replace(",","")

    try:
        value=float(q)
    except Exception:
        return None

    suffix_factor={
        "":1.0,
        "K":1e3,
        "MIL":1e3,
        "M":1e6,
        "MI":1e6,
        "B":1e9,
        "BI":1e9,
    }.get(suffix,1.0)

    return {
        "raw_text":raw,
        "normalized_text":compact,
        "numeric_value_base":value,
        "suffix":suffix,
        "suffix_factor":suffix_factor,
        "value_with_suffix":value*suffix_factor,
        "currency_flag":currency,
        "percent_flag":percent,
    }


def normalize_scale_text_v7(text):
    """
    Normalização conservadora específica para notação científica.

    Correções permitidas somente em posições estruturalmente compatíveis:
    - I/L antes de E -> 1
    - O no expoente -> 0
    - B no expoente -> 8
    - E/e uniformizado
    - × e * uniformizados como X

    Não transforma texto arbitrário em número.
    """
    raw=str(text)

    s=strip_accents(raw).upper()
    s=s.translate(SUPERSCRIPT_MAP)
    s=s.replace("×","X").replace("*","X")
    s=s.replace("−","-").replace("–","-").replace("—","-")
    s=re.sub(r"\s+","",s)

    # Forma semelhante a Ie8 / le8.
    s=re.sub(r"^[IL](?=E)", "1", s)

    # Correções somente após E no expoente.
    m=re.fullmatch(r"(1E[+-]?)([0-9OB]{1,3})",s)
    if m:
        expo=m.group(2).replace("O","0").replace("B","8")
        s=m.group(1)+expo

    # Também permite 10^8 / X10^8 com correções no expoente.
    m=re.fullmatch(r"(X?10\^?)([+-]?)([0-9OB]{1,3})",s)
    if m:
        expo=m.group(3).replace("O","0").replace("B","8")
        s=m.group(1)+m.group(2)+expo

    return s


def parse_scale_text(text):
    """
    V7: parser científico estrito.

    Aceita somente a string inteira como uma forma científica reconhecida.
    Não procura uma substring científica dentro de texto maior.
    """
    raw=str(text)

    if not raw.strip():
        return None

    s=normalize_scale_text_v7(raw)

    patterns=[
        re.compile(r"^1E([+-]?\d{1,3})$"),
        re.compile(r"^X10\^?([+-]?\d{1,3})$"),
        re.compile(r"^10\^([+-]?\d{1,3})$"),
    ]

    for pat in patterns:
        m=pat.fullmatch(s)

        if not m:
            continue

        try:
            exp=int(m.group(1))
            return {
                "scale_text_raw":raw,
                "scale_text_normalized":s,
                "scale_exponent":exp,
                "scale_factor":10.0**exp,
            }
        except Exception:
            return None

    return None


def token_usable(text,kind):
    if not text or not str(text).strip():
        return False

    if kind=="categorical":
        n=normalize_categorical(text)
        return sum(ch.isalpha() for ch in n)>=2

    if kind=="temporal":
        return len(parse_years(text))>0

    if kind=="numeric":
        return numeric_token_parse(text) is not None

    return True


def semantic_score(text,conf,kind):
    if not text:
        return -999.0

    score=max(-1.0,conf)

    if kind=="categorical":
        n=normalize_categorical(text)
        alpha=sum(ch.isalpha() for ch in n)
        if alpha>=3:
            score+=25
        if len(n)>=5:
            score+=10

    elif kind=="temporal":
        years=parse_years(text)
        if years:
            score+=45+4*len(years)

    elif kind=="numeric":
        if numeric_token_parse(text) is not None:
            score+=35

    return score


def whitelist_for_kind(kind):
    if kind=="temporal":
        return "0123456789"
    if kind=="numeric":
        return "0123456789.,-+eExXkKmMbBiIRr$%()"
    return None


def _token_center_score(tok,im_w,im_h):
    cx=tok["left"]+tok["width"]/2
    cy=tok["top"]+tok["height"]/2
    dx=abs(cx-im_w/2)/max(1,im_w/2)
    dy=abs(cy-im_h/2)/max(1,im_h/2)
    center=max(0.0,1.0-(dx+dy)/2)
    area=max(1,tok["width"]*tok["height"])
    return center,area


def numeric_candidates_for_crop(crop,lang):
    """
    Gera múltiplos candidatos numéricos SEM concatenar tokens distintos.

    Retorna candidatos deduplicados por (valor base, fator de sufixo),
    preservando a leitura de maior qualidade para cada alternativa.
    """
    variants=preprocess_variants(crop,upscale=3)

    attempts=[
        ("gray",7),
        ("otsu",7),
        ("gray",8),
        ("otsu",8),
        ("gray",13),
        ("otsu",13),
    ]

    by_key={}

    for prep_name,psm in attempts:
        rim=variants[prep_name]

        toks=ocr_token_rows(
            rim,
            lang=lang,
            psm=psm,
            whitelist=whitelist_for_kind("numeric")
        )

        for tok in toks:
            parsed=numeric_token_parse(tok["text"])

            if parsed is None:
                continue

            center,area=_token_center_score(
                tok,rim.width,rim.height
            )

            score=(
                float(tok["conf"])
                + 10.0*center
                + 0.5*min(10.0,np.log10(area+1))
            )

            key=(
                round(float(parsed["numeric_value_base"]),10),
                float(parsed["suffix_factor"])
            )

            row={
                "candidate_source":"INDIVIDUAL",
                "raw_text":tok["text"],
                "confidence":float(tok["conf"]),
                "score":float(score),
                "preprocess":prep_name,
                "psm":psm,
                "numeric_value_base":float(parsed["numeric_value_base"]),
                "suffix":parsed["suffix"],
                "suffix_factor":float(parsed["suffix_factor"]),
                "value_with_suffix":float(parsed["value_with_suffix"]),
                "currency_flag":parsed["currency_flag"],
                "percent_flag":parsed["percent_flag"],
            }

            if (
                key not in by_key
                or row["score"]>by_key[key]["score"]
            ):
                by_key[key]=row

    candidates=sorted(
        by_key.values(),
        key=lambda r:(r["score"],r["confidence"]),
        reverse=True
    )

    return candidates


def best_numeric_or_temporal_token(crop,kind,lang):
    if kind=="numeric":
        candidates=numeric_candidates_for_crop(
            crop,lang
        )

        if not candidates:
            return {
                "text":"",
                "confidence":-1.0,
                "score":-999.0,
                "rotation":0,
                "preprocess":"",
                "psm":None,
                "all_tokens":"",
                "numeric_candidates":[],
            }

        best=candidates[0]

        return {
            "text":best["raw_text"],
            "confidence":best["confidence"],
            "score":best["score"],
            "rotation":0,
            "preprocess":best["preprocess"],
            "psm":best["psm"],
            "all_tokens":" | ".join(
                c["raw_text"] for c in candidates
            ),
            "numeric_candidates":candidates,
        }

    # Temporal: preserva a estratégia anterior.
    best={
        "text":"",
        "confidence":-1.0,
        "score":-999.0,
        "rotation":0,
        "preprocess":"",
        "psm":None,
        "all_tokens":"",
        "numeric_candidates":[],
    }

    variants=preprocess_variants(crop,upscale=3)

    attempts=[
        ("gray",0,7),
        ("otsu",0,7),
        ("gray",0,8),
    ]

    for prep_name,angle,psm in attempts:
        rim=rotate_expand(variants[prep_name],angle)

        toks=ocr_token_rows(
            rim,
            lang=lang,
            psm=psm,
            whitelist=whitelist_for_kind("temporal")
        )

        for tok in toks:
            years=parse_years(tok["text"])

            if not years:
                continue

            center,area=_token_center_score(
                tok,rim.width,rim.height
            )

            score=(
                tok["conf"]
                + 10.0*center
                + 0.5*min(10.0,np.log10(area+1))
            )

            if re.fullmatch(
                r"\d{4}",
                normalize_basic(tok["text"])
            ):
                score+=8.0

            if score>best["score"]:
                best={
                    "text":tok["text"],
                    "confidence":tok["conf"],
                    "score":score,
                    "rotation":angle,
                    "preprocess":prep_name,
                    "psm":psm,
                    "all_tokens":" | ".join(
                        t["text"] for t in toks
                    ),
                    "numeric_candidates":[],
                }

        if (
            best["text"]
            and best["confidence"]>=EARLY_STOP_CONF
        ):
            break

    return best


def best_ocr_for_crop(crop,kind,lang):
    if kind in ("numeric","temporal"):
        return best_numeric_or_temporal_token(
            crop,kind,lang
        )

    best={
        "text":"",
        "confidence":-1.0,
        "score":-999.0,
        "rotation":0,
        "preprocess":"",
        "psm":None,
        "all_tokens":"",
    }

    variants=preprocess_variants(crop,upscale=3)

    w,h=crop.size
    if h>1.35*w:
        rot_order=[-90,90,0]
    elif w>1.8*h:
        rot_order=[0,-45,45]
    else:
        rot_order=[0,-45,45,-90,90]

    attempts=[]

    for ang in rot_order:
        attempts.append(("gray",ang,7))

    for ang in rot_order[:3]:
        attempts.append(("otsu",ang,7))

    for prep_name,angle,psm in attempts:
        rim=rotate_expand(
            variants[prep_name],angle
        )

        text,conf=ocr_data(
            rim,lang=lang,psm=psm,
            whitelist=None
        )

        score=semantic_score(
            text,conf,kind
        )

        if score>best["score"]:
            best={
                "text":text,
                "confidence":conf,
                "score":score,
                "rotation":angle,
                "preprocess":prep_name,
                "psm":psm,
                "all_tokens":text,
            }

        if token_usable(text,kind) and conf>=EARLY_STOP_CONF:
            break

    return best


def best_band_ocr(crop,kind,lang):
    """
    Faixa = evidência COMPLEMENTAR.
    Na V2, categóricos sempre recebem uma leitura de faixa.
    """
    best={
        "text":"",
        "confidence":-1.0,
        "score":-999.0,
        "preprocess":"",
        "psm":None,
    }

    whitelist=whitelist_for_kind(kind)
    variants=preprocess_variants(crop,upscale=2)

    attempts=[
        ("gray",6),
        ("gray",11),
        ("otsu",6),
        ("otsu",11),
    ]

    for prep_name,psm in attempts:
        text,conf=ocr_data(
            variants[prep_name],
            lang=lang,
            psm=psm,
            whitelist=whitelist
        )

        score=semantic_score(text,conf,kind)

        if score>best["score"]:
            best={
                "text":text,
                "confidence":conf,
                "score":score,
                "preprocess":prep_name,
                "psm":psm,
            }

        if token_usable(text,kind) and conf>=65:
            break

    return best



def numeric_lane_rect(full_img,axes,axis):
    """
    ROI da faixa inteira dos tick labels numéricos.

    Y: faixa à esquerda do eixo vertical, cobrindo toda a altura do plot.
    X: faixa abaixo do eixo horizontal, cobrindo toda a largura do plot.

    A ROI não usa valores esperados.
    """
    W,H=full_img.size

    vals=[
        axes.get("plot_x_left"),
        axes.get("plot_x_right"),
        axes.get("plot_y_top"),
        axes.get("plot_y_bottom"),
    ]

    if any(v is None or v=="" for v in vals):
        return None

    xl,xr,yt,yb=map(int,vals)
    pw=max(1,xr-xl)
    ph=max(1,yb-yt)

    if axis=="Y":
        x1=max(0,xl-int(.20*pw))
        x2=max(0,xl-2)
        y1=max(0,yt-int(.02*ph))
        y2=min(H-1,yb+int(.02*ph))
    else:
        x1=max(0,xl-int(.02*pw))
        x2=min(W-1,xr+int(.02*pw))
        y1=min(H-1,yb+2)
        y2=min(H-1,yb+int(.13*H))

    if x2<=x1 or y2<=y1:
        return None

    return (x1,y1,x2,y2)


def numeric_lane_ocr_candidates(
    full_img,axes,axis,lang,boxes,axis_crop_dir
):
    """
    Segunda leitura do eixo numérico inteiro.

    Retorna candidatos OCR REAIS mapeados às caixas por posição.
    Não interpola e não cria valores ausentes.
    """
    rect=numeric_lane_rect(
        full_img,axes,axis
    )

    if rect is None or not boxes:
        return [],{
            "status":"UNAVAILABLE",
            "raw_tokens":"",
            "n_tokens":0,
            "n_mapped":0,
            "crop_path":"",
        }

    crop,absbox=crop_rect(
        full_img,rect,pad=0
    )

    lane_path=axis_crop_dir/f"{axis}_NUMERIC_LANE.png"
    crop.save(lane_path)

    scale_up=3
    base=crop.convert("L")
    base=ImageOps.autocontrast(base)
    base=base.resize(
        (
            max(1,base.width*scale_up),
            max(1,base.height*scale_up)
        ),
        Image.Resampling.LANCZOS
    )

    arr=np.asarray(base)
    _,otsu=cv2.threshold(
        arr,0,255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU
    )

    variants={
        "gray":base,
        "otsu":Image.fromarray(otsu),
    }

    all_tokens=[]
    dedup={}

    for prep_name,im in variants.items():
        for psm in (6,11,12):
            toks=ocr_token_rows(
                im,
                lang=lang,
                psm=psm,
                whitelist=whitelist_for_kind("numeric")
            )

            for tok in toks:
                parsed=numeric_token_parse(
                    tok["text"]
                )

                if parsed is None:
                    continue

                # coordenadas no espaço da imagem original
                cx=(
                    absbox[0]
                    + (tok["left"]+tok["width"]/2)/scale_up
                )
                cy=(
                    absbox[1]
                    + (tok["top"]+tok["height"]/2)/scale_up
                )

                all_tokens.append(tok["text"])

                if axis=="Y":
                    centers=[
                        (float(b[1])+float(b[3]))/2
                        for b in boxes
                    ]
                    dists=[
                        abs(cy-c) for c in centers
                    ]
                else:
                    centers=[
                        (float(b[0])+float(b[2]))/2
                        for b in boxes
                    ]
                    dists=[
                        abs(cx-c) for c in centers
                    ]

                if not dists:
                    continue

                idx=int(np.argmin(dists))

                # tolerância baseada no espaçamento entre boxes
                if len(centers)>=2:
                    sc=sorted(centers)
                    diffs=[
                        abs(sc[i+1]-sc[i])
                        for i in range(len(sc)-1)
                        if abs(sc[i+1]-sc[i])>1
                    ]
                    spacing=float(np.median(diffs)) if diffs else 20.0
                else:
                    spacing=20.0

                tol=max(8.0,.48*spacing)

                if dists[idx]>tol:
                    continue

                score=(
                    float(tok["conf"])
                    + 4.0
                )

                row={
                    "candidate_source":"LANE",
                    "raw_text":tok["text"],
                    "confidence":float(tok["conf"]),
                    "score":float(score),
                    "preprocess":prep_name,
                    "psm":psm,
                    "numeric_value_base":float(
                        parsed["numeric_value_base"]
                    ),
                    "suffix":parsed["suffix"],
                    "suffix_factor":float(
                        parsed["suffix_factor"]
                    ),
                    "value_with_suffix":float(
                        parsed["value_with_suffix"]
                    ),
                    "currency_flag":parsed["currency_flag"],
                    "percent_flag":parsed["percent_flag"],
                }

                key=(
                    idx,
                    round(row["numeric_value_base"],10),
                    row["suffix_factor"]
                )

                if (
                    key not in dedup
                    or row["score"]>dedup[key]["score"]
                ):
                    dedup[key]=row

    mapped=[]

    for (idx,_,_),cand in dedup.items():
        mapped.append({
            "label_index_zero":idx,
            **cand,
        })

    return mapped,{
        "status":"OK",
        "raw_tokens":" | ".join(all_tokens),
        "n_tokens":len(all_tokens),
        "n_mapped":len(mapped),
        "crop_path":str(lane_path),
    }


def _candidate_conf_weight(conf):
    try:
        c=float(conf)
    except Exception:
        return 0.0
    if not np.isfinite(c):
        return 0.0
    return max(0.0,min(1.0,c/100.0))


def _round_axis_value(v,step):
    if not np.isfinite(v):
        return v

    s=abs(float(step))

    if s>=10:
        return float(round(v))
    if s>=1:
        return float(round(v,1))
    if s>=.1:
        return float(round(v,2))
    return float(round(v,3))


def select_numeric_axis_candidates(label_rows,axis):
    """
    Seleção CONSERVADORA dos valores numéricos do eixo.

    A coerência espacial pode:
    - escolher entre candidatos OCR existentes;
    - marcar candidatos incompatíveis como não selecionados.

    Ela NÃO pode:
    - interpolar um valor ausente;
    - completar dígitos;
    - criar 100 a partir de 00;
    - criar qualquer conteúdo que não exista em algum candidato OCR.

    Estados:
    COHERENT_DIRECT
        os melhores candidatos diretos já formam um eixo coerente.

    SELECTED_BY_COHERENCE
        a coerência espacial escolheu, para pelo menos um tick, um candidato
        OCR alternativo já existente.

    PARTIAL_UNREADABLE
        existe um modelo coerente, mas uma ou mais posições não possuem
        candidato OCR compatível. Ficam explicitamente vazias.

    AMBIGUOUS
        há candidatos, mas nenhum modelo global suficientemente sustentado.

    INSUFFICIENT
        menos de dois ticks têm candidatos válidos.
    """
    if not label_rows:
        return {
            "status":"INSUFFICIENT",
            "model_slope":"",
            "model_intercept":"",
            "model_step":"",
            "n_inliers":0,
            "n_positions":0,
            "n_positions_with_candidates":0,
            "n_selected":0,
            "n_unreadable":0,
            "anchor_span":0.0,
            "selected_values":[],
            "selected_sources":[],
            "selected_candidate_sources":[],
        }

    rows=label_rows
    n=len(rows)

    if axis=="X":
        pos=np.array([
            (float(r["bbox_x1"])+float(r["bbox_x2"]))/2
            for r in rows
        ],dtype=float)
    else:
        pos=np.array([
            -(float(r["bbox_y1"])+float(r["bbox_y2"]))/2
            for r in rows
        ],dtype=float)

    pmin=float(np.min(pos))
    pmax=float(np.max(pos))
    pspan=max(1e-9,pmax-pmin)
    t=(pos-pmin)/pspan

    candsets=[
        list(r.get("_numeric_candidates",[]))
        for r in rows
    ]

    valid=[
        i for i,c in enumerate(candsets)
        if len(c)>0
    ]

    if len(valid)<2:
        values=[]
        sources=[]
        csrc=[]

        for cands in candsets:
            if cands:
                values.append(
                    float(cands[0]["numeric_value_base"])
                )
                sources.append("DIRECT_UNVERIFIED")
                csrc.append(cands[0].get(
                    "candidate_source","INDIVIDUAL"
                ))
            else:
                values.append(np.nan)
                sources.append("UNREADABLE")
                csrc.append("")

        return {
            "status":"INSUFFICIENT",
            "model_slope":"",
            "model_intercept":"",
            "model_step":"",
            "n_inliers":len(valid),
            "n_positions":n,
            "n_positions_with_candidates":len(valid),
            "n_selected":len(valid),
            "n_unreadable":n-len(valid),
            "anchor_span":0.0,
            "selected_values":values,
            "selected_sources":sources,
            "selected_candidate_sources":csrc,
        }

    models=[]

    for ai in range(len(valid)):
        i=valid[ai]

        for aj in range(ai+1,len(valid)):
            j=valid[aj]

            span=abs(float(t[j]-t[i]))

            if span<.20:
                continue

            for ci in candsets[i][:5]:
                for cj in candsets[j][:5]:
                    vi=float(ci["numeric_value_base"])
                    vj=float(cj["numeric_value_base"])

                    if abs(vj-vi)<1e-12:
                        continue

                    slope=(vj-vi)/(t[j]-t[i])
                    intercept=vi-slope*t[i]

                    if n>=2:
                        med_dt=float(
                            np.median(
                                np.diff(np.sort(t))
                            )
                        )
                    else:
                        med_dt=1.0

                    step=max(
                        abs(slope)*max(med_dt,1e-6),
                        1e-6
                    )

                    tol=max(
                        .26*step,
                        .04
                    )

                    n_in=0
                    conf_sum=0.0
                    residual_sum=0.0

                    for k,cands in enumerate(candsets):
                        if not cands:
                            continue

                        pred=slope*t[k]+intercept
                        best_match=None

                        for rank,c in enumerate(cands):
                            val=float(c["numeric_value_base"])
                            resid=abs(val-pred)

                            item=(
                                resid,
                                -float(c["score"]),
                                rank,
                                c
                            )

                            if best_match is None or item<best_match:
                                best_match=item

                        resid,negscore,rank,bestc=best_match

                        if resid<=tol:
                            n_in+=1
                            conf_sum+=_candidate_conf_weight(
                                bestc["confidence"]
                            )
                            residual_sum+=resid/max(
                                step,1e-9
                            )

                    score=(
                        14.0*n_in
                        + 3.0*conf_sum
                        + 5.0*span
                        - residual_sum
                    )

                    models.append({
                        "slope":slope,
                        "intercept":intercept,
                        "step":step,
                        "tol":tol,
                        "n_inliers":n_in,
                        "anchor_span":span,
                        "score":score,
                    })

    if not models:
        values=[]
        sources=[]
        csrc=[]

        for cands in candsets:
            if cands:
                values.append(
                    float(cands[0]["numeric_value_base"])
                )
                sources.append("DIRECT_UNVERIFIED")
                csrc.append(cands[0].get(
                    "candidate_source","INDIVIDUAL"
                ))
            else:
                values.append(np.nan)
                sources.append("UNREADABLE")
                csrc.append("")

        return {
            "status":"AMBIGUOUS",
            "model_slope":"",
            "model_intercept":"",
            "model_step":"",
            "n_inliers":0,
            "n_positions":n,
            "n_positions_with_candidates":len(valid),
            "n_selected":len(valid),
            "n_unreadable":n-len(valid),
            "anchor_span":0.0,
            "selected_values":values,
            "selected_sources":sources,
            "selected_candidate_sources":csrc,
        }

    best=max(
        models,
        key=lambda m:m["score"]
    )

    required=max(
        3,
        int(math.ceil(.60*len(valid)))
    )

    accepted=(
        best["n_inliers"]>=required
        and best["anchor_span"]>=.35
    )

    if not accepted:
        values=[]
        sources=[]
        csrc=[]

        for cands in candsets:
            if cands:
                values.append(
                    float(cands[0]["numeric_value_base"])
                )
                sources.append("DIRECT_UNVERIFIED")
                csrc.append(cands[0].get(
                    "candidate_source","INDIVIDUAL"
                ))
            else:
                values.append(np.nan)
                sources.append("UNREADABLE")
                csrc.append("")

        return {
            "status":"AMBIGUOUS",
            "model_slope":best["slope"],
            "model_intercept":best["intercept"],
            "model_step":best["step"],
            "n_inliers":best["n_inliers"],
            "n_positions":n,
            "n_positions_with_candidates":len(valid),
            "n_selected":len(valid),
            "n_unreadable":n-len(valid),
            "anchor_span":best["anchor_span"],
            "selected_values":values,
            "selected_sources":sources,
            "selected_candidate_sources":csrc,
        }

    values=[]
    sources=[]
    csrc=[]
    any_alternative=False
    n_selected=0
    n_unreadable=0

    for k,cands in enumerate(candsets):
        pred=best["slope"]*t[k]+best["intercept"]

        if not cands:
            values.append(np.nan)
            sources.append("UNREADABLE")
            csrc.append("")
            n_unreadable+=1
            continue

        matches=[]

        for rank,c in enumerate(cands):
            val=float(c["numeric_value_base"])
            resid=abs(val-pred)

            if resid<=best["tol"]:
                matches.append((
                    resid,
                    -float(c["score"]),
                    rank,
                    c
                ))

        if not matches:
            # NÃO inventa valor: fica unreadable nesta posição.
            values.append(np.nan)
            sources.append("UNREADABLE")
            csrc.append("")
            n_unreadable+=1
            continue

        resid,negscore,rank,chosen=min(
            matches,
            key=lambda z:(z[0],z[1],z[2])
        )

        values.append(
            float(chosen["numeric_value_base"])
        )
        csrc.append(
            chosen.get("candidate_source","INDIVIDUAL")
        )

        if rank==0:
            sources.append("DIRECT_MODEL_MATCH")
        else:
            sources.append("ALTERNATIVE_OCR_SELECTED")
            any_alternative=True

        n_selected+=1

        # guarda o candidato efetivamente selecionado na própria linha
        rows[k]["_selected_numeric_candidate"]=chosen

    if n_unreadable>0:
        status="PARTIAL_UNREADABLE"
    elif any_alternative:
        status="SELECTED_BY_COHERENCE"
    else:
        status="COHERENT_DIRECT"

    return {
        "status":status,
        "model_slope":best["slope"],
        "model_intercept":best["intercept"],
        "model_step":best["step"],
        "n_inliers":best["n_inliers"],
        "n_positions":n,
        "n_positions_with_candidates":len(valid),
        "n_selected":n_selected,
        "n_unreadable":n_unreadable,
        "anchor_span":best["anchor_span"],
        "selected_values":values,
        "selected_sources":sources,
        "selected_candidate_sources":csrc,
    }


# ============================================================
# OCR de fator científico
# ============================================================

def scale_search_regions(full_img,axes,axis):
    """
    V7: duas micro-ROIs para o offset científico.

    Y:
      ROI 1 externa, imediatamente acima do canto superior esquerdo;
      ROI 2 cruza levemente a borda superior do plot.

    X:
      ROI 1 externa, imediatamente abaixo do canto inferior direito;
      ROI 2 cruza levemente a borda inferior.

    A ROI 2 existe porque algumas renderizações posicionam o offset
    praticamente sobre a borda do plot.

    Nenhum conteúdo esperado é usado.
    """
    W,H=full_img.size

    xl=axes.get("plot_x_left")
    xr=axes.get("plot_x_right")
    yt=axes.get("plot_y_top")
    yb=axes.get("plot_y_bottom")

    if any(v is None or v=="" for v in (xl,xr,yt,yb)):
        return []

    xl,xr,yt,yb=map(int,(xl,xr,yt,yb))
    pw=max(1,xr-xl)
    ph=max(1,yb-yt)

    regs=[]

    if axis=="Y":
        # ROI 1: externa.
        regs.append((
            max(0,xl-int(.035*pw)),
            max(0,yt-int(.085*ph)),
            min(W-1,xl+int(.145*pw)),
            max(0,yt-1),
        ))

        # ROI 2: atravessa discretamente a borda superior.
        regs.append((
            max(0,xl-int(.025*pw)),
            max(0,yt-int(.055*ph)),
            min(W-1,xl+int(.125*pw)),
            min(H-1,yt+int(.035*ph)),
        ))

    else:
        # ROI 1: externa.
        regs.append((
            max(0,xr-int(.155*pw)),
            min(H-1,yb+1),
            min(W-1,xr+int(.035*pw)),
            min(H-1,yb+int(.085*ph)),
        ))

        # ROI 2: atravessa discretamente a borda inferior.
        regs.append((
            max(0,xr-int(.135*pw)),
            max(0,yb-int(.025*ph)),
            min(W-1,xr+int(.030*pw)),
            min(H-1,yb+int(.055*ph)),
        ))

    clean=[]
    seen=set()

    for r in regs:
        r=clip_rect(r,W,H)

        if rect_area(r)<=20:
            continue

        if r in seen:
            continue

        clean.append(r)
        seen.add(r)

    return clean


def scale_candidate_strings(tokens):
    """
    Gera candidatos a partir de tokens individuais e pequenas combinações
    adjacentes, sem juntar indiscriminadamente toda a região.
    """
    if not tokens:
        return []

    out=[]

    # Cada token isolado.
    for i,t in enumerate(tokens):
        out.append((
            t["text"],
            t["conf"],
            f"TOKEN_{i+1}"
        ))

    # Combinações locais de 2 e 3 tokens para casos como "1 e 8" ou "1e 8".
    for n in (2,3):
        for i in range(0,len(tokens)-n+1):
            chunk=tokens[i:i+n]
            joined="".join(t["text"] for t in chunk)
            conf=float(np.mean([t["conf"] for t in chunk]))
            out.append((
                joined,
                conf,
                f"JOIN_{n}_{i+1}"
            ))

    return out



def scale_vote_weight(confidence):
    """
    Peso de qualidade OCR.

    Confianças muito baixas não devem transformar uma leitura espúria
    em consenso. O peso cresce continuamente acima de 20.

    Esse limiar é um critério de qualidade do OCR e não depende do
    expoente esperado.
    """
    try:
        c=float(confidence)
    except Exception:
        return 0.0

    if not np.isfinite(c) or c<=20.0:
        return 0.0

    return min(1.0,(c-20.0)/70.0)


def best_scale_ocr(full_img,axes,axis,lang,crop_dir,filename):
    """
    V7: leitura conservadora do fator científico.

    Estados:
    SCALE_NOT_DETECTED
    SCALE_AMBIGUOUS
    SCALE_READABLE_CONFIDENT

    Rotas para CONFIDENT:
    A) CONSENSUS:
       - >=2 votos qualificados para o expoente vencedor;
       - >=65% do peso qualificado;
       - pelo menos uma confiança >=45.

    B) SINGLE_HIGH_CONF:
       - leitura científica exata;
       - confiança >=82;
       - micro-ROI geometricamente válida;
       - nenhum expoente concorrente com confiança >=45.

    Uma leitura de alta confiança não é inferência:
    ela continua sendo conteúdo OCR efetivamente observado.
    """
    regs=scale_search_regions(
        full_img,axes,axis
    )

    raw_votes=[]

    for idx,r in enumerate(regs,start=1):
        crop,absbox=crop_rect(
            full_img,r,pad=0
        )

        path=(
            crop_dir /
            filename.rsplit(".",1)[0] /
            f"{axis}_SCALE_{idx}.png"
        )
        path.parent.mkdir(
            parents=True,exist_ok=True
        )
        crop.save(path)

        variants=preprocess_variants(
            crop,upscale=5
        )

        # Além de gray/otsu, cria uma variante binária levemente dilatada.
        arr=np.asarray(variants["otsu"])
        inv=255-arr

        kernel=np.ones((2,2),np.uint8)
        dil=cv2.dilate(inv,kernel,iterations=1)
        dil=255-dil

        variants["dilate"]=Image.fromarray(dil)

        for prep_name in ("gray","otsu","dilate"):
            pim=variants[prep_name]

            # --------------------------------------------------
            # 1. OCR por tokens
            # --------------------------------------------------
            for psm in (6,7,11,13):
                toks=ocr_token_rows(
                    pim,
                    lang=lang,
                    psm=psm,
                    whitelist="0123456789eExX^+-ILOB"
                )

                for raw,conf,source in scale_candidate_strings(toks):
                    parsed=parse_scale_text(raw)

                    if parsed is None:
                        continue

                    raw_votes.append({
                        "axis":axis,
                        "region_index":idx,
                        "preprocess":prep_name,
                        "psm":psm,
                        "source":source,
                        "candidate_raw":raw,
                        "candidate_normalized":parsed[
                            "scale_text_normalized"
                        ],
                        "confidence":float(conf),
                        "vote_weight":scale_vote_weight(conf),
                        "scale_exponent":int(
                            parsed["scale_exponent"]
                        ),
                        "scale_factor":float(
                            parsed["scale_factor"]
                        ),
                        "crop_path":str(path),
                    })

            # --------------------------------------------------
            # 2. OCR da micro-ROI como string completa
            # --------------------------------------------------
            for psm in (7,13):
                cfg=(
                    f"--oem 3 --psm {psm} "
                    "-c tessedit_char_whitelist=0123456789eExX^+-ILOB"
                )

                try:
                    raw_line=pytesseract.image_to_string(
                        pim,
                        lang=lang,
                        config=cfg
                    ).strip()
                except Exception:
                    raw_line=""

                parsed=parse_scale_text(raw_line)

                if parsed is None:
                    continue

                # Obtém confiança média da mesma execução para não tratar
                # image_to_string como evidência sem qualidade.
                toks=ocr_token_rows(
                    pim,
                    lang=lang,
                    psm=psm,
                    whitelist="0123456789eExX^+-ILOB"
                )

                confs=[
                    float(t["conf"])
                    for t in toks
                    if str(t["text"]).strip()
                ]

                conf=float(np.mean(confs)) if confs else -1.0

                raw_votes.append({
                    "axis":axis,
                    "region_index":idx,
                    "preprocess":prep_name,
                    "psm":psm,
                    "source":"WHOLE_ROI",
                    "candidate_raw":raw_line,
                    "candidate_normalized":parsed[
                        "scale_text_normalized"
                    ],
                    "confidence":conf,
                    "vote_weight":scale_vote_weight(conf),
                    "scale_exponent":int(
                        parsed["scale_exponent"]
                    ),
                    "scale_factor":float(
                        parsed["scale_factor"]
                    ),
                    "crop_path":str(path),
                })

    if not raw_votes:
        return {
            "scale_status":"NONE",
            "scale_interpretation_status":"SCALE_NOT_DETECTED",
            "scale_acceptance_route":"",
            "scale_text_raw":"",
            "scale_text_normalized":"",
            "scale_exponent":"",
            "scale_factor":1.0,
            "confidence":-1.0,
            "n_votes":0,
            "n_eligible_votes":0,
            "winning_votes":0,
            "winning_eligible_votes":0,
            "weighted_vote_fraction":0.0,
            "winner_weight":0.0,
            "total_weight":0.0,
            "max_competing_confidence":-1.0,
            "crop_path":"",
            "votes":[],
        }

    # Deduplicação: uma mesma configuração OCR/ROI/expoente conta uma vez.
    dedup={}

    for v in raw_votes:
        key=(
            v["region_index"],
            v["preprocess"],
            v["psm"],
            v["scale_exponent"],
        )

        if (
            key not in dedup
            or float(v["confidence"])>float(dedup[key]["confidence"])
        ):
            dedup[key]=v

    votes=list(dedup.values())

    for v in votes:
        v["eligible_vote"]=int(
            float(v["vote_weight"])>0
        )

    by_exp={}

    for v in votes:
        e=int(v["scale_exponent"])

        if e not in by_exp:
            by_exp[e]={
                "all_votes":0,
                "eligible_votes":0,
                "weight":0.0,
                "max_conf":-1.0,
            }

        d=by_exp[e]
        d["all_votes"]+=1
        d["weight"]+=float(v["vote_weight"])
        d["max_conf"]=max(
            d["max_conf"],
            float(v["confidence"])
        )

        if int(v["eligible_vote"])==1:
            d["eligible_votes"]+=1

    total_weight=sum(
        d["weight"] for d in by_exp.values()
    )

    ranked=sorted(
        by_exp.items(),
        key=lambda kv:(
            kv[1]["weight"],
            kv[1]["eligible_votes"],
            kv[1]["max_conf"]
        ),
        reverse=True
    )

    winner_exp,wdata=ranked[0]

    winner_votes=[
        v for v in votes
        if int(v["scale_exponent"])==winner_exp
    ]

    best_vote=max(
        winner_votes,
        key=lambda v:(
            float(v["vote_weight"]),
            float(v["confidence"])
        )
    )

    weighted_frac=(
        wdata["weight"]/total_weight
        if total_weight>0
        else 0.0
    )

    # Maior confiança de um expoente concorrente.
    competing=[
        float(v["confidence"])
        for v in votes
        if int(v["scale_exponent"])!=winner_exp
    ]
    max_competing_conf=max(competing) if competing else -1.0

    consensus_confident=(
        wdata["eligible_votes"]>=2
        and weighted_frac>=.65
        and wdata["max_conf"]>=45.0
    )

    single_high_confident=(
        wdata["max_conf"]>=82.0
        and max_competing_conf<45.0
    )

    if consensus_confident:
        interpretation="SCALE_READABLE_CONFIDENT"
        status="CONFIDENT"
        acceptance_route="CONSENSUS"
        factor=10.0**winner_exp

    elif single_high_confident:
        interpretation="SCALE_READABLE_CONFIDENT"
        status="CONFIDENT"
        acceptance_route="SINGLE_HIGH_CONF"
        factor=10.0**winner_exp

    else:
        interpretation="SCALE_AMBIGUOUS"
        status="AMBIGUOUS"
        acceptance_route=""
        factor=1.0

    return {
        "scale_status":status,
        "scale_interpretation_status":interpretation,
        "scale_acceptance_route":acceptance_route,
        "scale_text_raw":best_vote["candidate_raw"],
        "scale_text_normalized":best_vote.get(
            "candidate_normalized",
            normalize_scale_text_v7(best_vote["candidate_raw"])
        ),
        "scale_exponent":winner_exp,
        "scale_factor":factor,
        "confidence":best_vote["confidence"],
        "n_votes":len(votes),
        "n_eligible_votes":sum(
            d["eligible_votes"]
            for d in by_exp.values()
        ),
        "winning_votes":wdata["all_votes"],
        "winning_eligible_votes":wdata["eligible_votes"],
        "weighted_vote_fraction":weighted_frac,
        "winner_weight":wdata["weight"],
        "total_weight":total_weight,
        "max_competing_confidence":max_competing_conf,
        "crop_path":best_vote["crop_path"],
        "votes":votes,
    }


# ============================================================
# Extração por eixo
# ============================================================

def normalize_for_kind(text,kind):
    if kind=="categorical":
        return normalize_categorical(text)

    if kind=="temporal":
        years=parse_years(text)
        return "|".join(years) if years else normalize_numeric_text(text)

    if kind=="numeric":
        return normalize_numeric_text(text)

    return normalize_basic(text)


def axis_extraction(
    full_img,res,axes,profile,axis,lang,crop_dir,filename
):
    kind=AXIS_KIND[profile][axis]
    W,H=full_img.size

    category_band_info={
        "status":"",
        "bbox":None,
        "n_groups":0,
        "n_selected_groups":0,
    }

    restrict_rect=None

    if kind=="categorical" and axis=="X":
        category_band_info=detect_categorical_tick_band(
            full_img,axes
        )
        restrict_rect=category_band_info.get("bbox")

    boxes=tick_boxes_clipped_to_layer(
        res,axis,W,H,
        restrict_rect=restrict_rect
    )

    lane_info={
        "status":"",
        "n_before":len(boxes),
        "n_after":len(boxes),
        "reference_distance":"",
        "tolerance":"",
    }

    if kind=="numeric":
        boxes,lane_info=filter_numeric_tick_lane(
            boxes,axes,axis
        )

    seg_ok=int(
        res.get(
            "tlx_segmentation_ok" if axis=="X"
            else "tly_segmentation_ok",
            0
        ) or 0
    )

    present=res.get(
        "tlx_present_candidate" if axis=="X"
        else "tly_present_candidate"
    )

    label_rows=[]
    usable=0

    axis_crop_dir=(
        crop_dir /
        profile /
        filename.rsplit(".",1)[0] /
        axis
    )
    axis_crop_dir.mkdir(
        parents=True,exist_ok=True
    )

    # ----------------------------------------------------------
    # OCR individual
    # ----------------------------------------------------------
    for idx,r in enumerate(boxes,start=1):
        if kind=="numeric":
            crop,absbox,crop_mode=crop_numeric_tick_directional(
                full_img,r,axes,axis
            )
        else:
            crop,absbox=crop_rect(
                full_img,r,pad=4
            )
            crop_mode="SYMMETRIC_NONNUMERIC"

        best=best_ocr_for_crop(
            crop,kind,lang
        )

        norm=normalize_for_kind(
            best["text"],kind
        )

        ok=token_usable(
            best["text"],kind
        )

        usable+=int(ok)

        crop_path=axis_crop_dir/f"{axis}_{idx:02d}.png"
        crop.save(crop_path)

        parsed_num=(
            numeric_token_parse(best["text"])
            if kind=="numeric"
            else None
        )

        numeric_candidates=(
            list(best.get("numeric_candidates",[]))
            if kind=="numeric"
            else []
        )

        parsed_years=(
            parse_years(best["text"])
            if kind=="temporal"
            else []
        )

        label_rows.append({
            "axis":axis,
            "kind":kind,
            "label_index":idx,

            "bbox_x1":absbox[0],
            "bbox_y1":absbox[1],
            "bbox_x2":absbox[2],
            "bbox_y2":absbox[3],

            "ocr_raw":best["text"],
            "ocr_all_tokens":best.get("all_tokens",""),
            "ocr_normalized":norm,
            "ocr_confidence":best["confidence"],
            "ocr_score":best["score"],
            "ocr_rotation":best["rotation"],
            "ocr_preprocess":best["preprocess"],
            "ocr_psm":best["psm"],
            "token_usable":int(ok),
            "crop_mode":crop_mode,

            "parsed_years":"|".join(parsed_years),

            "numeric_value_base":(
                parsed_num["numeric_value_base"]
                if parsed_num else ""
            ),
            "numeric_suffix":(
                parsed_num["suffix"]
                if parsed_num else ""
            ),
            "numeric_suffix_factor":(
                parsed_num["suffix_factor"]
                if parsed_num else ""
            ),
            "numeric_value_with_suffix":(
                parsed_num["value_with_suffix"]
                if parsed_num else ""
            ),
            "currency_flag":(
                parsed_num["currency_flag"]
                if parsed_num else ""
            ),
            "percent_flag":(
                parsed_num["percent_flag"]
                if parsed_num else ""
            ),

            "_numeric_candidates":numeric_candidates,

            "crop_path":str(crop_path),
        })

    # ----------------------------------------------------------
    # Segunda leitura da faixa numérica inteira
    # ----------------------------------------------------------
    numeric_lane_ocr={
        "status":"",
        "raw_tokens":"",
        "n_tokens":0,
        "n_mapped":0,
        "crop_path":"",
    }

    if kind=="numeric" and label_rows:
        mapped,lane_ocr_info=numeric_lane_ocr_candidates(
            full_img,axes,axis,lang,
            boxes,axis_crop_dir
        )

        numeric_lane_ocr=lane_ocr_info

        for item in mapped:
            k=int(item["label_index_zero"])

            if k<0 or k>=len(label_rows):
                continue

            cand={
                kk:vv
                for kk,vv in item.items()
                if kk!="label_index_zero"
            }

            existing=label_rows[k]["_numeric_candidates"]

            key=(
                round(float(cand["numeric_value_base"]),10),
                float(cand["suffix_factor"])
            )

            found=None

            for j,e in enumerate(existing):
                ekey=(
                    round(float(e["numeric_value_base"]),10),
                    float(e["suffix_factor"])
                )
                if ekey==key:
                    found=j
                    break

            if found is None:
                existing.append(cand)
            else:
                if cand["score"]>existing[found]["score"]:
                    existing[found]=cand

            existing.sort(
                key=lambda c:(
                    c["score"],
                    c["confidence"]
                ),
                reverse=True
            )

    # Atualiza diagnóstico dos candidatos após fusão individual+lane.
    for lr in label_rows:
        cands=lr.get("_numeric_candidates",[])

        lr["numeric_candidate_count"]=len(cands)
        lr["numeric_candidates_text"]=" | ".join(
            f"{c.get('candidate_source','?')}:{c['raw_text']}@{c['confidence']:.1f}"
            for c in cands
        )

    individual_frac=usable/max(
        1,len(boxes)
    )

    # ----------------------------------------------------------
    # Leitura de faixa categórica / fallback
    # ----------------------------------------------------------
    need_band=(
        kind=="categorical"
        or not(
            len(boxes)>0 and
            individual_frac>=MIN_INDIVIDUAL_SUCCESS_FRAC
        )
    )

    band={
        "text":"",
        "confidence":-1.0,
        "score":-999.0,
        "preprocess":"",
        "psm":None,
    }

    band_crop_path=""

    if need_band:
        if kind=="categorical" and axis=="X":
            band_crop,band_bbox,category_band_info=extract_categorical_band_crop(
                full_img,axes
            )
        else:
            band_crop,band_bbox=extract_layer_crop(
                full_img,res,axis
            )

        if band_crop is not None:
            band=best_band_ocr(
                band_crop,kind,lang
            )

            band_crop_path=axis_crop_dir/f"{axis}_BAND.png"
            band_crop.save(
                band_crop_path
            )

    band_norm=normalize_for_kind(
        band["text"],kind
    )

    band_usable=token_usable(
        band["text"],kind
    )

    # ----------------------------------------------------------
    # Fator científico conservador
    # ----------------------------------------------------------
    scale={
        "scale_status":"NONE",
        "scale_interpretation_status":"SCALE_NOT_DETECTED",
        "scale_acceptance_route":"",
        "scale_text_raw":"",
        "scale_text_normalized":"",
        "scale_exponent":"",
        "scale_factor":1.0,
        "confidence":-1.0,
        "n_votes":0,
        "n_eligible_votes":0,
        "winning_votes":0,
        "winning_eligible_votes":0,
        "weighted_vote_fraction":0.0,
        "winner_weight":0.0,
        "total_weight":0.0,
        "crop_path":"",
        "votes":[],
    }

    if kind=="numeric":
        scale=best_scale_ocr(
            full_img,axes,axis,lang,
            crop_dir/profile,filename
        )

    scale_status=scale.get(
        "scale_status","NONE"
    )

    scale_factor=float(
        scale.get("scale_factor",1.0) or 1.0
    )

    # ----------------------------------------------------------
    # Seleção conservadora entre candidatos observados
    # ----------------------------------------------------------
    numeric_selection={
        "status":"",
        "model_slope":"",
        "model_intercept":"",
        "model_step":"",
        "n_inliers":0,
        "n_positions":0,
        "n_positions_with_candidates":0,
        "n_selected":0,
        "n_unreadable":0,
        "anchor_span":0.0,
        "selected_values":[],
        "selected_sources":[],
        "selected_candidate_sources":[],
    }

    if kind=="numeric":
        numeric_selection=select_numeric_axis_candidates(
            label_rows,axis
        )

        for idx,lr in enumerate(label_rows):
            vals=numeric_selection.get(
                "selected_values",[]
            )
            srcs=numeric_selection.get(
                "selected_sources",[]
            )
            csrcs=numeric_selection.get(
                "selected_candidate_sources",[]
            )

            value=(
                vals[idx]
                if idx<len(vals)
                else np.nan
            )

            lr["numeric_selected_value_base"]=(
                "" if not np.isfinite(value)
                else float(value)
            )
            lr["numeric_selection_source"]=(
                srcs[idx]
                if idx<len(srcs)
                else ""
            )
            lr["numeric_selected_candidate_source"]=(
                csrcs[idx]
                if idx<len(csrcs)
                else ""
            )

            chosen=lr.get(
                "_selected_numeric_candidate"
            )

            if chosen is None and np.isfinite(value):
                # Para AMBIGUOUS/INSUFFICIENT: identifica o candidato
                # observado que corresponde ao valor direto mostrado.
                for c in lr.get("_numeric_candidates",[]):
                    if abs(
                        float(c["numeric_value_base"])
                        - float(value)
                    )<1e-9:
                        chosen=c
                        break

            if chosen is not None:
                lr["numeric_selected_raw_text"]=chosen["raw_text"]
                lr["numeric_selected_confidence"]=chosen["confidence"]
                lr["numeric_selected_suffix"]=chosen["suffix"]
                lr["numeric_selected_suffix_factor"]=chosen["suffix_factor"]
                own_value_with_suffix=chosen["value_with_suffix"]
            else:
                lr["numeric_selected_raw_text"]=""
                lr["numeric_selected_confidence"]=""
                lr["numeric_selected_suffix"]=""
                lr["numeric_selected_suffix_factor"]=""
                own_value_with_suffix=""

            lr["scale_status"]=scale_status
            lr["scale_factor"]=scale_factor

            # Sem inferência de sufixo. Usa apenas o sufixo do candidato
            # efetivamente observado/selecionado.
            if own_value_with_suffix=="":
                lr["effective_value"]=""
            elif scale_status=="AMBIGUOUS":
                lr["effective_value"]=""
            else:
                lr["effective_value"]=(
                    float(own_value_with_suffix)
                    * scale_factor
                )

    else:
        for lr in label_rows:
            lr["numeric_selected_value_base"]=""
            lr["numeric_selection_source"]=""
            lr["numeric_selected_candidate_source"]=""
            lr["numeric_selected_raw_text"]=""
            lr["numeric_selected_confidence"]=""
            lr["numeric_selected_suffix"]=""
            lr["numeric_selected_suffix_factor"]=""
            lr["scale_status"]=""
            lr["scale_factor"]=""
            lr["effective_value"]=""

    # ----------------------------------------------------------
    # Texto candidato
    # ----------------------------------------------------------
    ind_tokens=[
        r["ocr_normalized"]
        for r in label_rows
        if r["token_usable"]==1
    ]

    individual_text=" | ".join(
        ind_tokens
    )

    if kind=="categorical":
        if individual_text and band_usable:
            mode="HYBRID_EVIDENCE"
            candidate_text=individual_text
        elif individual_text:
            mode="INDIVIDUAL_ONLY"
            candidate_text=individual_text
        elif band_usable:
            mode="BAND_ONLY"
            candidate_text=band_norm
        else:
            mode="FAILED"
            candidate_text=""
    else:
        if (
            len(boxes)>0 and
            individual_frac>=MIN_INDIVIDUAL_SUCCESS_FRAC
        ):
            mode="INDIVIDUAL"
            candidate_text=individual_text
        elif band_usable:
            mode="BAND_FALLBACK"
            candidate_text=band_norm
        elif individual_text:
            mode="INDIVIDUAL_PARTIAL"
            candidate_text=individual_text
        else:
            mode="FAILED"
            candidate_text=""

    # ----------------------------------------------------------
    # Controles estruturais
    # ----------------------------------------------------------
    temporal_years=[]
    temporal_monotonic=""

    if kind=="temporal":
        for lr in label_rows:
            temporal_years.extend(
                parse_years(lr["ocr_raw"])
            )

        dedup=[]
        for y in temporal_years:
            if not dedup or y!=dedup[-1]:
                dedup.append(y)

        temporal_years=dedup

        if len(temporal_years)>=2:
            vals=list(map(int,temporal_years))
            temporal_monotonic=int(
                all(
                    vals[i]<=vals[i+1]
                    for i in range(len(vals)-1)
                )
            )

    numeric_values=[]

    if kind=="numeric":
        for lr in label_rows:
            if lr.get("effective_value","")!="":
                try:
                    numeric_values.append(
                        float(lr["effective_value"])
                    )
                except Exception:
                    pass

    # ----------------------------------------------------------
    # Suporte automático
    # ----------------------------------------------------------
    if mode=="FAILED":
        support="FAIL_CANDIDATE"

    elif kind=="categorical":
        if (
            individual_text
            and band_usable
            and category_band_info.get("status")=="FOUND"
        ):
            support="MULTI_EVIDENCE"
        else:
            support="SINGLE_EVIDENCE"

    elif kind=="temporal":
        if (
            individual_frac>=0.75
            and temporal_monotonic in ("",1)
        ):
            support="STRONG_CANDIDATE"
        elif individual_text or band_usable:
            support="PARTIAL_CANDIDATE"
        else:
            support="FAIL_CANDIDATE"

    else:
        selection_status=numeric_selection.get(
            "status",""
        )

        if (
            individual_frac>=0.75
            and scale_status!="AMBIGUOUS"
            and selection_status in (
                "COHERENT_DIRECT",
                "SELECTED_BY_COHERENCE"
            )
        ):
            support="STRONG_CANDIDATE"
        elif individual_text or band_usable:
            support="PARTIAL_CANDIDATE"
        else:
            support="FAIL_CANDIDATE"

    cbbox=category_band_info.get("bbox")

    sel_values=numeric_selection.get(
        "selected_values",[]
    )

    axis_row={
        "axis":axis,
        "kind":kind,

        "presence_candidate":present,
        "segmentation_ok_v4":seg_ok,

        "category_band_status":category_band_info.get("status",""),
        "category_band_bbox":(
            "|".join(map(str,cbbox))
            if cbbox is not None
            else ""
        ),
        "category_band_n_groups":category_band_info.get("n_groups",0),
        "category_band_n_selected_groups":category_band_info.get(
            "n_selected_groups",0
        ),

        "numeric_lane_status":lane_info.get("status",""),
        "numeric_lane_n_before":lane_info.get("n_before",len(boxes)),
        "numeric_lane_n_longitudinal_valid":lane_info.get(
            "n_longitudinal_valid",len(boxes)
        ),
        "numeric_lane_n_longitudinal_rejected":lane_info.get(
            "n_longitudinal_rejected",0
        ),
        "numeric_lane_n_after":lane_info.get("n_after",len(boxes)),
        "numeric_lane_reference_distance":lane_info.get("reference_distance",""),
        "numeric_lane_tolerance":lane_info.get("tolerance",""),
        "numeric_lane_longitudinal_tolerance":lane_info.get(
            "longitudinal_tolerance",""
        ),

        "numeric_lane_ocr_status":numeric_lane_ocr.get("status",""),
        "numeric_lane_ocr_raw_tokens":numeric_lane_ocr.get("raw_tokens",""),
        "numeric_lane_ocr_n_tokens":numeric_lane_ocr.get("n_tokens",0),
        "numeric_lane_ocr_n_mapped":numeric_lane_ocr.get("n_mapped",0),
        "numeric_lane_ocr_crop_path":numeric_lane_ocr.get("crop_path",""),

        "n_tick_boxes_after_layer_clip":len(boxes),
        "n_usable_individual":usable,
        "individual_usable_frac":individual_frac,

        "individual_text_candidate":individual_text,

        "band_ocr_raw":band["text"],
        "band_ocr_normalized":band_norm,
        "band_ocr_confidence":band["confidence"],
        "band_ocr_score":band["score"],
        "band_ocr_preprocess":band["preprocess"],
        "band_ocr_psm":band["psm"],
        "band_usable":int(band_usable),
        "band_crop_path":str(
            band_crop_path
        ) if band_crop_path else "",

        "scale_status":scale_status,
        "scale_interpretation_status":scale.get(
            "scale_interpretation_status",
            "SCALE_NOT_DETECTED"
        ),
        "scale_text_raw":scale.get("scale_text_raw",""),
        "scale_text_normalized":scale.get(
            "scale_text_normalized",""
        ),
        "scale_exponent":scale.get("scale_exponent",""),
        "scale_factor":scale_factor,
        "scale_ocr_confidence":scale.get("confidence",-1.0),
        "scale_n_votes":scale.get("n_votes",0),
        "scale_n_eligible_votes":scale.get("n_eligible_votes",0),
        "scale_winning_votes":scale.get("winning_votes",0),
        "scale_winning_eligible_votes":scale.get("winning_eligible_votes",0),
        "scale_weighted_vote_fraction":scale.get(
            "weighted_vote_fraction",0.0
        ),
        "scale_winner_weight":scale.get("winner_weight",0.0),
        "scale_total_weight":scale.get("total_weight",0.0),
        "scale_acceptance_route":scale.get(
            "scale_acceptance_route",""
        ),
        "scale_max_competing_confidence":scale.get(
            "max_competing_confidence",-1.0
        ),
        "scale_crop_path":scale.get("crop_path",""),

        "temporal_years_spatial":"|".join(
            temporal_years
        ),
        "temporal_monotonic_candidate":temporal_monotonic,

        "numeric_selection_status":numeric_selection.get("status",""),
        "numeric_selection_model_slope":numeric_selection.get(
            "model_slope",""
        ),
        "numeric_selection_model_intercept":numeric_selection.get(
            "model_intercept",""
        ),
        "numeric_selection_model_step":numeric_selection.get(
            "model_step",""
        ),
        "numeric_selection_n_inliers":numeric_selection.get(
            "n_inliers",0
        ),
        "numeric_selection_n_positions":numeric_selection.get(
            "n_positions",0
        ),
        "numeric_selection_n_positions_with_candidates":numeric_selection.get(
            "n_positions_with_candidates",0
        ),
        "numeric_selection_n_selected":numeric_selection.get(
            "n_selected",0
        ),
        "numeric_selection_n_unreadable":numeric_selection.get(
            "n_unreadable",0
        ),
        "numeric_selection_anchor_span":numeric_selection.get(
            "anchor_span",0.0
        ),
        "numeric_selected_values_spatial":"|".join(
            "" if not np.isfinite(v) else f"{v:g}"
            for v in sel_values
        ),
        "numeric_selection_sources":"|".join(
            numeric_selection.get(
                "selected_sources",[]
            )
        ),
        "numeric_selected_candidate_sources":"|".join(
            numeric_selection.get(
                "selected_candidate_sources",[]
            )
        ),

        "numeric_effective_values_spatial":"|".join(
            f"{v:g}" for v in numeric_values
        ),

        "extraction_mode_candidate":mode,
        "extraction_support_candidate":support,
        "extracted_text_candidate":candidate_text,
    }

    numeric_candidate_rows=[]

    if kind=="numeric":
        for lr in label_rows:
            for rank,c in enumerate(
                lr.get("_numeric_candidates",[]),
                start=1
            ):
                numeric_candidate_rows.append({
                    "filename":filename,
                    "profile":profile,
                    "axis":axis,
                    "label_index":lr["label_index"],
                    "candidate_rank":rank,
                    "candidate_source":c.get(
                        "candidate_source","INDIVIDUAL"
                    ),
                    "raw_text":c["raw_text"],
                    "confidence":c["confidence"],
                    "score":c["score"],
                    "preprocess":c["preprocess"],
                    "psm":c["psm"],
                    "numeric_value_base":c["numeric_value_base"],
                    "suffix":c["suffix"],
                    "suffix_factor":c["suffix_factor"],
                    "value_with_suffix":c["value_with_suffix"],
                })

    for lr in label_rows:
        lr.pop("_numeric_candidates",None)
        lr.pop("_selected_numeric_candidate",None)

    scale_votes=[
        {
            "filename":filename,
            "profile":profile,
            "axis":axis,
            **v,
        }
        for v in scale.get("votes",[])
    ]

    return axis_row,label_rows,scale_votes,numeric_candidate_rows


# ============================================================
# Auditoria visual
# ============================================================

def font_default(size=10):
    try:
        return ImageFont.truetype(
            "DejaVuSans.ttf",size
        )
    except Exception:
        return ImageFont.load_default()


def short(s,n=95):
    s=str(s)
    return s if len(s)<=n else s[:n-3]+"..."


def make_contact_sheet(g,root,out_path):
    if g.empty:
        return

    cols=3
    pw=650
    ph=450
    ih=285
    rows=math.ceil(len(g)/cols)

    sheet=Image.new(
        "RGB",
        (cols*pw,rows*ph),
        "white"
    )

    d=ImageDraw.Draw(sheet)
    f=font_default(10)

    for idx,row in g.reset_index(drop=True).iterrows():
        c=idx%cols
        r=idx//cols

        x0=c*pw
        y0=r*ph

        src=root/row["filename"]
        im=Image.open(src).convert("RGB")
        im.thumbnail((pw-16,ih-8))

        sheet.paste(
            im,
            (x0+(pw-im.width)//2,y0+4)
        )

        xind=short(row.get("x_individual_text_candidate",""))
        xband=short(row.get("x_band_ocr_normalized",""))
        yind=short(row.get("y_individual_text_candidate",""))
        yscale=short(row.get("y_scale_text_normalized",""))
        xscale=short(row.get("x_scale_text_normalized",""))

        txt=(
            f"{row['filename']} | {row['profile']}\n"
            f"X {row['x_kind']} | {row['x_extraction_mode_candidate']} | "
            f"{row['x_extraction_support_candidate']} | "
            f"boxes={row['x_n_tick_boxes_after_layer_clip']} "
            f"usable={row['x_n_usable_individual']}\n"
            f"X IND: {xind}\n"
            f"X BAND: {xband}\n"
            f"X SELECT: {row.get('x_numeric_selection_status','-')} "
            f"vals={short(row.get('x_numeric_selected_values_spatial',''),65)}\n"
            f"X SCALE: {row.get('x_scale_interpretation_status','-')} "
            f"{xscale or '-'} factor={row.get('x_scale_factor',1)}\n"
            f"Y {row['y_kind']} | {row['y_extraction_mode_candidate']} | "
            f"{row['y_extraction_support_candidate']} | "
            f"boxes={row['y_n_tick_boxes_after_layer_clip']} "
            f"usable={row['y_n_usable_individual']}\n"
            f"Y IND: {yind}\n"
            f"Y SELECT: {row.get('y_numeric_selection_status','-')} "
            f"vals={short(row.get('y_numeric_selected_values_spatial',''),65)}\n"
            f"Y SCALE: {row.get('y_scale_interpretation_status','-')} "
            f"{yscale or '-'} factor={row.get('y_scale_factor',1)}"
        )

        d.multiline_text(
            (x0+8,y0+ih+4),
            txt,
            fill="black",
            font=f,
            spacing=2
        )

    sheet.save(out_path)


def main():
    args=parse_args()

    root=args.root.expanduser().resolve()

    outdir=(
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root/OUT_DIRNAME
    )

    crop_dir=outdir/"crops"
    crop_dir.mkdir(parents=True,exist_ok=True)

    v4_sha=check_v4_hash()
    t_version,lang,langs=configure_tesseract(
        args.tesseract
    )

    inv=inventory(root)

    if len(inv)!=3120:
        raise RuntimeError(
            f"Inventário canônico na raiz: {len(inv)}; "
            "esperado=3120."
        )

    crosswalk_path=find_crosswalk(
        root,args.crosswalk
    )

    previous_manifest=find_previous_manifest(
        root,args.manifest
    )

    if previous_manifest is not None:
        sample=sample_from_manifest(
            previous_manifest,
            inv,
            crosswalk_path
        )
        sample_mode=f"REUSED_MANIFEST: {previous_manifest}"
    else:
        sample,sample_mode=select_new_sample(
            inv,crosswalk_path,args.seed
        )

    sample.to_csv(
        outdir/"ticklabel_ba_v7_manifest.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("="*80)
    print("FASE B-A V7 — EXTRAÇÃO TEXTUAL DOS TICK LABELS")
    print("="*80)
    print(f"Detector V4 SHA-256: {v4_sha}")
    print(f"Tesseract: {t_version}")
    print(f"Idioma OCR: {lang}")
    print(f"Amostragem: {sample_mode}")
    print(f"Imagens: {len(sample)}")
    print("A V7 continua sendo calibração de EXTRAÇÃO; não avalia conformidade.")
    print()

    image_rows=[]
    axis_rows=[]
    label_rows=[]
    scale_vote_rows=[]
    numeric_candidate_rows=[]
    errors=[]

    total=len(sample)

    for i,r in sample.iterrows():
        path=root/r["filename"]

        print(
            f"[{i+1:02d}/{total}] {r['filename']} ...",
            end=" ",
            flush=True
        )

        try:
            full_img=Image.open(path).convert("RGB")

            pres=v4.analyze(path)
            axes=v4.axis_v3.analyze(path)

            image_row={
                "filename":r["filename"],
                "profile":r["profile"],
                "unit_id":r["unit_id"],
                "unit_number":r["unit_number"],
                "repeat":r["repeat"],
            }

            for c in (
                "condition_code",
                "condition_label",
                "participant_id"
            ):
                if c in r.index:
                    image_row[c]=r[c]

            for axis in ("X","Y"):
                ar,lrs,svrs,ncrs=axis_extraction(
                    full_img,
                    pres,
                    axes,
                    r["profile"],
                    axis,
                    lang,
                    crop_dir,
                    r["filename"],
                )

                prefix="x_" if axis=="X" else "y_"

                for k,v in ar.items():
                    if k!="axis":
                        image_row[prefix+k]=v

                axis_rows.append({
                    "filename":r["filename"],
                    "profile":r["profile"],
                    "unit_id":r["unit_id"],
                    **{
                        c:r[c]
                        for c in (
                            "condition_code",
                            "condition_label",
                            "participant_id"
                        )
                        if c in r.index
                    },
                    **ar,
                })

                for lr in lrs:
                    label_rows.append({
                        "filename":r["filename"],
                        "profile":r["profile"],
                        "unit_id":r["unit_id"],
                        **{
                            c:r[c]
                            for c in (
                                "condition_code",
                                "condition_label",
                                "participant_id"
                            )
                            if c in r.index
                        },
                        **lr,
                    })

                for nc in ncrs:
                    numeric_candidate_rows.append({
                        "unit_id":r["unit_id"],
                        **{
                            c:r[c]
                            for c in (
                                "condition_code",
                                "condition_label",
                                "participant_id"
                            )
                            if c in r.index
                        },
                        **nc,
                    })

                for sv in svrs:
                    scale_vote_rows.append({
                        "unit_id":r["unit_id"],
                        **{
                            c:r[c]
                            for c in (
                                "condition_code",
                                "condition_label",
                                "participant_id"
                            )
                            if c in r.index
                        },
                        **sv,
                    })

            # Auditoria humana da EXTRAÇÃO.
            image_row["manual_x_extraction_review"]=""
            image_row["manual_y_extraction_review"]=""
            image_row["manual_x_visible_text"]=""
            image_row["manual_y_visible_text"]=""
            image_row["manual_notes"]=""

            image_rows.append(image_row)

            print("OK",flush=True)

        except Exception as exc:
            errors.append({
                "filename":r["filename"],
                "profile":r["profile"],
                "unit_id":r["unit_id"],
                "error_type":type(exc).__name__,
                "error_message":str(exc),
            })

            print(
                f"ERRO: {type(exc).__name__}: {exc}",
                flush=True
            )

    images_df=pd.DataFrame(image_rows)
    axes_df=pd.DataFrame(axis_rows)
    labels_df=pd.DataFrame(label_rows)
    scale_votes_df=pd.DataFrame(scale_vote_rows)
    numeric_candidates_df=pd.DataFrame(numeric_candidate_rows)
    err_df=pd.DataFrame(errors)

    images_df.to_csv(
        outdir/"ticklabel_ba_v7_images.csv",
        index=False,
        encoding="utf-8-sig"
    )

    axes_df.to_csv(
        outdir/"ticklabel_ba_v7_axes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    labels_df.to_csv(
        outdir/"ticklabel_ba_v7_labels.csv",
        index=False,
        encoding="utf-8-sig"
    )

    scale_votes_df.to_csv(
        outdir/"ticklabel_ba_v7_scale_votes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    numeric_candidates_df.to_csv(
        outdir/"ticklabel_ba_v7_numeric_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    err_df.to_csv(
        outdir/"ticklabel_ba_v7_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if not images_df.empty:
        for profile in PROFILES:
            g=images_df[
                images_df["profile"]==profile
            ]

            if len(g):
                make_contact_sheet(
                    g,
                    root,
                    outdir/f"contact_BA_V7_{profile}.png"
                )

    # ----------------------------
    # Resumo de suporte da extração
    # ----------------------------
    rows=[]

    if not axes_df.empty:
        for (profile,axis),g in axes_df.groupby(
            ["profile","axis"]
        ):
            support=g[
                "extraction_support_candidate"
            ].value_counts().to_dict()

            modes=g[
                "extraction_mode_candidate"
            ].value_counts().to_dict()

            scale_confident=int(
                (g["scale_interpretation_status"]=="SCALE_READABLE_CONFIDENT").sum()
            )
            scale_ambiguous=int(
                (g["scale_interpretation_status"]=="SCALE_AMBIGUOUS").sum()
            )
            scale_not_detected=int(
                (g["scale_interpretation_status"]=="SCALE_NOT_DETECTED").sum()
            )

            select_direct=int(
                (g["numeric_selection_status"]=="COHERENT_DIRECT").sum()
            )
            select_coherence=int(
                (g["numeric_selection_status"]=="SELECTED_BY_COHERENCE").sum()
            )
            select_partial=int(
                (g["numeric_selection_status"]=="PARTIAL_UNREADABLE").sum()
            )
            select_ambiguous=int(
                (g["numeric_selection_status"]=="AMBIGUOUS").sum()
            )
            select_insufficient=int(
                (g["numeric_selection_status"]=="INSUFFICIENT").sum()
            )

            rows.append({
                "profile":profile,
                "axis":axis,
                "kind":g["kind"].iloc[0],
                "n":len(g),

                "multi_evidence":int(
                    support.get("MULTI_EVIDENCE",0)
                ),
                "strong_candidate":int(
                    support.get("STRONG_CANDIDATE",0)
                ),
                "single_evidence":int(
                    support.get("SINGLE_EVIDENCE",0)
                ),
                "partial_candidate":int(
                    support.get("PARTIAL_CANDIDATE",0)
                ),
                "fail_candidate":int(
                    support.get("FAIL_CANDIDATE",0)
                ),

                "hybrid_evidence":int(
                    modes.get("HYBRID_EVIDENCE",0)
                ),
                "band_only":int(
                    modes.get("BAND_ONLY",0)
                ),

                "mean_n_tick_boxes":float(
                    pd.to_numeric(
                        g["n_tick_boxes_after_layer_clip"],
                        errors="coerce"
                    ).mean()
                ),

                "mean_usable_frac":float(
                    pd.to_numeric(
                        g["individual_usable_frac"],
                        errors="coerce"
                    ).mean()
                ),

                "n_scale_confident":scale_confident,
                "n_scale_ambiguous":scale_ambiguous,
                "n_scale_not_detected":scale_not_detected,
                "n_select_direct":select_direct,
                "n_select_coherence":select_coherence,
                "n_select_partial_unreadable":select_partial,
                "n_select_ambiguous":select_ambiguous,
                "n_select_insufficient":select_insufficient,
            })

    summary_df=pd.DataFrame(rows)

    summary_df.to_csv(
        outdir/"ticklabel_ba_v7_summary_by_profile_axis.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Lista dirigida para revisão:
    # - suporte parcial/falha
    # - categórico sempre, porque queremos validar cobertura real
    # - numérico com fator científico detectado, para validar o parser
    review=axes_df[
        (
            axes_df["extraction_support_candidate"]
            .isin(["PARTIAL_CANDIDATE","FAIL_CANDIDATE"])
        ) |
        (
            axes_df["scale_interpretation_status"].isin(
                ["SCALE_READABLE_CONFIDENT","SCALE_AMBIGUOUS"]
            )
        ) |
        (
            axes_df["numeric_selection_status"].isin(
                [
                    "SELECTED_BY_COHERENCE",
                    "PARTIAL_UNREADABLE",
                    "AMBIGUOUS",
                    "INSUFFICIENT",
                ]
            )
        ) |
        (
            pd.to_numeric(
                axes_df["numeric_lane_n_longitudinal_rejected"],
                errors="coerce"
            ).fillna(0)>0
        )
    ].copy()

    review["manual_extraction_review"]=""
    review["manual_visible_text"]=""
    review["manual_scale_text"]=""
    review["manual_notes"]=""

    review.to_csv(
        outdir/"ticklabel_ba_v7_priority_review.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Auditorias de calibração separadas para não inflar a lista de exceções.
    categorical_audit=axes_df[
        axes_df["kind"]=="categorical"
    ].copy()
    categorical_audit.to_csv(
        outdir/"ticklabel_ba_v7_categorical_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    critical_numeric_audit=axes_df[
        (
            axes_df["profile"].isin(["SI","SC"]) &
            (axes_df["axis"]=="Y")
        ) |
        (
            axes_df["scale_interpretation_status"].isin(
                ["SCALE_READABLE_CONFIDENT","SCALE_AMBIGUOUS"]
            )
        )
    ].copy()
    critical_numeric_audit.to_csv(
        outdir/"ticklabel_ba_v7_critical_numeric_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    scale_audit=axes_df[
        axes_df["kind"]=="numeric"
    ][[
        c for c in [
            "filename","profile","unit_id","axis","kind",
            "scale_interpretation_status",
            "scale_acceptance_route",
            "scale_text_raw",
            "scale_text_normalized",
            "scale_exponent",
            "scale_factor",
            "scale_ocr_confidence",
            "scale_n_votes",
            "scale_n_eligible_votes",
            "scale_winning_votes",
            "scale_winning_eligible_votes",
            "scale_weighted_vote_fraction",
            "scale_max_competing_confidence",
            "scale_crop_path",
        ]
        if c in axes_df.columns
    ]].copy()

    scale_audit.to_csv(
        outdir/"ticklabel_ba_v7_scale_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    longitudinal_audit=axes_df[
        pd.to_numeric(
            axes_df["numeric_lane_n_longitudinal_rejected"],
            errors="coerce"
        ).fillna(0)>0
    ].copy()

    longitudinal_audit.to_csv(
        outdir/"ticklabel_ba_v7_longitudinal_audit.csv",
        index=False,
        encoding="utf-8-sig"
    )

    lines=[
        "FASE B-A V7 — EXTRAÇÃO TEXTUAL DOS TICK LABELS",
        "="*78,
        f"Detector V4 congelado SHA-256: {v4_sha}",
        f"Tesseract: {t_version}",
        f"Idioma OCR: {lang}",
        f"Amostragem: {sample_mode}",
        f"Imagens processadas: {len(images_df)}",
        f"Erros: {len(err_df)}",
        "",
        "ESCOPO:",
        "- calibração da EXTRAÇÃO textual;",
        "- SEM comparação com valores esperados;",
        "- SEM decisão de conformidade;",
        "- SEM avaliação tipográfica.",
        "",
        "MUDANÇAS V7:",
        "- preserva integralmente a arquitetura conservadora da V6;",
        "- clipping longitudinal das caixas numéricas pelo domínio observado do plot;",
        "- caixas rejeitadas longitudinalmente não retornam por fallback;",
        "- duas micro-ROIs para notação científica, incluindo leve sobreposição no plot;",
        "- parser científico estrito e normalização OCR conservadora;",
        "- escala CONFIDENT por consenso ou SINGLE_HIGH_CONF >=82 sem concorrente forte;",
        "- nenhuma alteração em temporal, categórico ou política de não inferência;",
        "",
        "INTERPRETAÇÃO:",
        "- MULTI_EVIDENCE / STRONG_CANDIDATE etc. são suporte automático;",
        "- NÃO equivalem a acurácia;",
        "- a cobertura real será estabelecida por auditoria visual da calibração.",
        "",
        f"Linhas prioritárias para auditoria: {len(review)}",
    ]

    if not summary_df.empty:
        lines.extend([
            "",
            "RESUMO POR PERFIL × EIXO:"
        ])

        for _,r in summary_df.iterrows():
            lines.append(
                f"{r['profile']} {r['axis']} ({r['kind']}): "
                f"n={int(r['n'])}; "
                f"MULTI={int(r['multi_evidence'])}; "
                f"STRONG={int(r['strong_candidate'])}; "
                f"SINGLE={int(r['single_evidence'])}; "
                f"PARTIAL={int(r['partial_candidate'])}; "
                f"FAIL={int(r['fail_candidate'])}; "
                f"SCALE_CONF={int(r['n_scale_confident'])}; "
                f"SCALE_AMB={int(r['n_scale_ambiguous'])}; "
                f"SCALE_NONE={int(r['n_scale_not_detected'])}; "
                f"SEL_DIR={int(r['n_select_direct'])}; "
                f"SEL_COH={int(r['n_select_coherence'])}; "
                f"SEL_PART={int(r['n_select_partial_unreadable'])}; "
                f"SEL_AMB={int(r['n_select_ambiguous'])}; "
                f"SEL_INSUF={int(r['n_select_insufficient'])}"
            )

    (outdir/"ticklabel_ba_v7_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print()
    print("Concluído.")
    print("Saída:",outdir)
    print("Envie principalmente:")
    print("- ticklabel_ba_v7_summary.txt")
    print("- ticklabel_ba_v7_images.csv")
    print("- ticklabel_ba_v7_labels.csv")
    print("- ticklabel_ba_v7_scale_votes.csv")
    print("- ticklabel_ba_v7_numeric_candidates.csv")
    print("- ticklabel_ba_v7_priority_review.csv")
    print("- ticklabel_ba_v7_categorical_audit.csv")
    print("- ticklabel_ba_v7_critical_numeric_audit.csv")
    print("- ticklabel_ba_v7_scale_audit.csv")
    print("- ticklabel_ba_v7_longitudinal_audit.csv")
    print("- contact_BA_V7_BI.png ... contact_BA_V7_SC.png")


if __name__=="__main__":
    main()
