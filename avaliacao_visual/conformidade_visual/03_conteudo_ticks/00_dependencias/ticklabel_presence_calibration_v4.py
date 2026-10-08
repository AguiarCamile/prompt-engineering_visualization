# -*- coding: utf-8 -*-
r"""
ticklabel_presence_calibration_v4.py

FASE A — CALIBRAÇÃO V4
PRESENÇA E LOCALIZAÇÃO DOS TICK LABELS X/Y
==========================================

Mudança central da V3
---------------------
A V3 separa duas perguntas que na V2 ainda estavam parcialmente acopladas:

1. PRESENÇA DA CAMADA TEXTUAL:
   Há evidência visual distribuída ao longo do eixo compatível com tick labels?

2. SEGMENTAÇÃO INDIVIDUAL:
   Foi possível separar essa camada em rótulos individuais?

Assim, uma imagem pode ter:
    TLX_PRESENT = 1
    TLX_SEGMENTATION_OK = 0
quando os rótulos estão claramente presentes, mas muito sobrepostos/congestionados
para serem individualizados de forma confiável.

Isso evita transformar falha de segmentação em ausência do elemento.

Esta fase continua:
- SEM OCR;
- SEM usar F para localizar tick labels;
- SEM avaliar conteúdo;
- SEM avaliar tipografia;
- SEM avaliar as pequenas marcas físicas dos ticks.

Detector de eixos congelado
---------------------------
axis_conformity_calibration_v3.py
SHA-256 esperado:
1582555adf23d8ed2a6b23e89eaefdcd317d0e11a2f7d361fbd69c6258866bfe

Amostra
-------
Reutiliza exatamente as mesmas 60 imagens das calibrações V1/V2, preferindo:
_ticklabel_presence_calibration_v2\ticklabel_calibration_manifest_v2.csv

Saídas
------
_ticklabel_presence_calibration_v4

- ticklabel_calibration_v4.csv
- ticklabel_calibration_errors_v4.csv
- ticklabel_calibration_manifest_v4.csv
- ticklabel_calibration_summary_v4.txt
- contact_BI.png ... contact_SC.png
- overlays/*.png
- crops_x/*.png
- crops_y/*.png

Overlay
-------
- verde: labels X individualizados
- magenta: labels Y individualizados
- verde-escuro: grupos da camada textual X usados para PRESENÇA
- roxo-escuro: grupos da camada textual Y usados para PRESENÇA
- laranja: camada secundária X compatível com axis label
- ciano: camada secundária Y compatível com axis label
- azul-claro: banda ampla X
- amarelo-claro: banda ampla Y
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import axis_conformity_calibration_v3 as axis_v3


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
OUT_DIRNAME = "_ticklabel_presence_calibration_v4"

PROFILES = ("BI", "BC", "LI", "LC", "SI", "SC")
EXPECTED_AXIS_V3_SHA256 = "1582555adf23d8ed2a6b23e89eaefdcd317d0e11a2f7d361fbd69c6258866bfe"
SEED = 20260830
N_UNITS_PER_PROFILE = 10

NAME_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)

# ============================================================
# Parâmetros independentes de F
# ============================================================

MIN_BG_DIFF = 18.0

# Bandas geométricas amplas.
X_BAND_MIN_PX = 46
X_BAND_FRAC_H = 0.125
Y_BAND_MIN_PX = 92
Y_BAND_FRAC_W = 0.130
ALONG_AXIS_PAD_PX = 12

# Componentes visuais com porte compatível com glifos.
MIN_CC_AREA = 3
MAX_CC_AREA_FRAC = 0.0035
MIN_CC_H = 2
MAX_CC_H_FRAC = 0.052
MIN_CC_W = 1
MAX_CC_W_FRAC = 0.13

# Camada textual próxima ao eixo.
NEAR_LAYER_MIN_PX = 13
NEAR_LAYER_HEIGHT_MULT = 3.0

# Para labels X inclinados: a camada de PRESENÇA só precisa capturar
# a parte próxima ao eixo; a segmentação pode se expandir mais.
X_SEG_CHAIN_X_GAP_MULT = 2.5
X_SEG_CHAIN_Y_GAP_MULT = 1.9
X_SEG_CHAIN_MAX_DEPTH_MULT = 6.0

# Agrupamento individual.
X_ROW_TOL_MULT = 0.90
X_WORD_GAP_MULT = 2.2
Y_ROW_TOL_MULT = 0.80
Y_WORD_GAP_MULT = 4.0

# Segmentação plausível.
MIN_BOX_AREA = 10
MAX_BOX_W_FRAC_X = 0.36
MAX_BOX_W_FRAC_Y = 0.30
MAX_BOX_H_FRAC = 0.10

# PRESENÇA pela camada textual.
# Não depende do número de labels individualizados.
LAYER_BIN_MULT = 2.2
MIN_LAYER_GROUPS = 2
MIN_LAYER_SPAN = 0.12
MIN_LAYER_FOREGROUND_PIXELS = 10

# V4 — evidência de faixa textual congestionada.
# Não exige separação em múltiplos grupos; usa a extensão ocupada ao longo do eixo.
MIN_CONGESTED_OCCUPIED_BINS = 8
MIN_CONGESTED_OCCUPIED_EXTENT = 0.28
MIN_CONGESTED_OCCUPIED_RATIO = 0.16

# Guarda de especificidade: a primeira camada textual deve começar
# razoavelmente próxima ao eixo dentro da banda ampla.
MAX_NEAR_START_FRAC = 0.45

# Controles negativos sintéticos (somente calibração, nunca resultados experimentais).
NEGATIVE_BASES_PER_PROFILE = 2
NEGATIVE_MASK_NEAR_FRAC = 0.62

# Segmentação OK.
MIN_SEGMENTED_LABELS = 2
MIN_SEGMENTATION_SPAN = 0.12

# Camada secundária compatível com axis label.
AXISLABEL_MIN_GAP_MULT = 1.2
AXISLABEL_CENTER_TOL_FRAC = 0.25

# Contact sheet
COLS = 3
PANEL_W = 585
PANEL_H = 480
MAIN_H = 260
BAND_H = 86


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--units-per-profile", type=int, default=N_UNITS_PER_PROFILE)
    return p.parse_args()


def check_axis_hash():
    p = Path(axis_v3.__file__).resolve()
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != EXPECTED_AXIS_V3_SHA256:
        raise RuntimeError(
            "axis_conformity_calibration_v3.py foi alterado.\n"
            f"SHA esperado: {EXPECTED_AXIS_V3_SHA256}\n"
            f"SHA encontrado: {actual}"
        )
    return actual


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
            "filename": p.name,
            "profile": profile,
            "unit_id": f"{profile}_{unit:03d}",
            "repeat": repeat,
        })
    return pd.DataFrame(rows)


def load_manifest(root, inv, n_per_profile, seed):
    candidates = [
        root / "_ticklabel_presence_calibration_v3" / "ticklabel_calibration_manifest_v3.csv",
        root / "_ticklabel_presence_calibration_v2" / "ticklabel_calibration_manifest_v2.csv",
        root / "_ticklabel_presence_calibration_v1" / "ticklabel_calibration_manifest_v1.csv",
        root / "_tick_presence_calibration_v1" / "tick_calibration_manifest_v1.csv",
    ]
    needed = ["filename", "profile", "unit_id", "repeat"]

    for p in candidates:
        if not p.exists():
            continue
        m = pd.read_csv(p)
        if set(needed).issubset(m.columns) and len(m) == 60:
            return m[needed].copy(), str(p)

    # Fallback somente se os manifestos anteriores não existirem.
    rng = random.Random(seed)
    out = []
    for profile in PROFILES:
        g = inv[inv["profile"] == profile]
        units = sorted(g["unit_id"].unique().tolist())
        if len(units) < n_per_profile:
            raise RuntimeError(f"{profile}: unidades insuficientes.")
        rng.shuffle(units)
        for unit_id in units[:n_per_profile]:
            rr = g[g["unit_id"] == unit_id].to_dict("records")
            rng.shuffle(rr)
            out.append(rr[0])

    m = pd.DataFrame(out).sort_values(["profile", "unit_id", "repeat"])
    return m[needed].reset_index(drop=True), "new_sample"


def estimate_background(arr):
    h, w, _ = arr.shape
    bw = max(2, min(6, max(2, w//100), max(2, h//100)))
    parts = [
        arr[:bw].reshape(-1,3),
        arr[-bw:].reshape(-1,3),
        arr[:,:bw].reshape(-1,3),
        arr[:,-bw:].reshape(-1,3),
    ]
    return np.median(np.concatenate(parts,axis=0),axis=0)


def foreground_mask(arr,bg):
    a = arr.astype(np.float32)
    d = np.sqrt(((a-bg.reshape(1,1,3))**2).sum(axis=2))
    return (d >= MIN_BG_DIFF).astype(np.uint8)


def clip_rect(rect,w,h):
    x1,y1,x2,y2 = rect
    x1=max(0,min(w-1,int(round(x1))))
    x2=max(0,min(w-1,int(round(x2))))
    y1=max(0,min(h-1,int(round(y1))))
    y2=max(0,min(h-1,int(round(y2))))
    if x2<=x1 or y2<=y1:
        return None
    return (x1,y1,x2,y2)


def infer_bands(img,axes):
    w,h=img.size
    pl=axes.get("plot_x_left")
    pr=axes.get("plot_x_right")
    pt=axes.get("plot_y_top")
    pb=axes.get("plot_y_bottom")

    if any(v in ("",None) or pd.isna(v) for v in (pl,pr,pt,pb)):
        return None,None,None,None

    x_axis_y=axes.get("axis_x_coord")
    if x_axis_y in ("",None) or pd.isna(x_axis_y):
        x_axis_y=pb

    y_axis_x=axes.get("axis_y_coord")
    if y_axis_x in ("",None) or pd.isna(y_axis_x):
        y_axis_x=pl

    x_depth=max(X_BAND_MIN_PX,int(round(X_BAND_FRAC_H*h)))
    y_depth=max(Y_BAND_MIN_PX,int(round(Y_BAND_FRAC_W*w)))

    xr=clip_rect(
        (pl-ALONG_AXIS_PAD_PX,x_axis_y+2,
         pr+ALONG_AXIS_PAD_PX,x_axis_y+x_depth),
        w,h
    )
    yr=clip_rect(
        (y_axis_x-y_depth,pt-ALONG_AXIS_PAD_PX,
         y_axis_x-2,pb+ALONG_AXIS_PAD_PX),
        w,h
    )
    return xr,yr,float(x_axis_y),float(y_axis_x)


def components(mask,rect):
    x1,y1,x2,y2=rect
    crop=mask[y1:y2+1,x1:x2+1]
    n,lab,stats,cents=cv2.connectedComponentsWithStats(
        crop.astype(np.uint8),connectivity=8
    )

    H,W=mask.shape
    max_area=max(20,int(round(MAX_CC_AREA_FRAC*H*W)))
    max_h=max(8,int(round(MAX_CC_H_FRAC*H)))
    max_w=max(18,int(round(MAX_CC_W_FRAC*W)))

    out=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i].tolist()
        if area<MIN_CC_AREA or area>max_area: continue
        if h<MIN_CC_H or h>max_h: continue
        if w<MIN_CC_W or w>max_w: continue
        gx1=x1+x; gy1=y1+y
        gx2=gx1+w-1; gy2=gy1+h-1
        out.append({
            "x1":gx1,"y1":gy1,"x2":gx2,"y2":gy2,
            "w":w,"h":h,"area":area,
            "cx":(gx1+gx2)/2,"cy":(gy1+gy2)/2,
        })
    return out


def median_glyph_h(comps):
    vals=[c["h"] for c in comps if c["h"]>=2]
    return float(np.median(vals)) if vals else 7.0


def merge_boxes(bb):
    if not bb: return None
    x1=min(b["x1"] for b in bb); y1=min(b["y1"] for b in bb)
    x2=max(b["x2"] for b in bb); y2=max(b["y2"] for b in bb)
    return {
        "x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),
        "w":int(x2-x1+1),"h":int(y2-y1+1),
        "cx":float((x1+x2)/2),"cy":float((y1+y2)/2),
        "n_components":len(bb),
    }


# ============================================================
# 1. CAMADA DE PRESENÇA — independente da segmentação
# ============================================================

def near_layer_x(mask,comps,rect,axis_y):
    """
    Seleciona uma faixa textual próxima ao eixo X.
    Também retorna a distância relativa da primeira camada ao eixo,
    usada na V4 como guarda de especificidade.
    """
    if not comps:
        return [],None,0.0,float("nan")

    gh=median_glyph_h(comps)
    ds=[max(0,c["y1"]-axis_y) for c in comps]
    d0=float(np.percentile(ds,10)) if ds else 0.0
    depth=max(NEAR_LAYER_MIN_PX,NEAR_LAYER_HEIGHT_MULT*gh)

    selected=[
        c for c in comps
        if max(0,c["y1"]-axis_y) <= d0+depth
    ]

    y2=int(round(axis_y+d0+depth+max(2,0.5*gh)))
    layer_rect=clip_rect(
        (rect[0],axis_y+1,rect[2],y2),
        mask.shape[1],mask.shape[0]
    )

    band_depth=max(1.0,rect[3]-axis_y)
    near_start_frac=float(d0/band_depth)

    return selected,layer_rect,gh,near_start_frac


def near_layer_y(mask,comps,rect,axis_x):
    """
    Seleciona a faixa textual próxima ao eixo Y e retorna a distância
    relativa da primeira camada ao eixo.
    """
    if not comps:
        return [],None,0.0,float("nan")

    gh=median_glyph_h(comps)
    ds=[max(0,axis_x-c["x2"]) for c in comps]
    d0=float(np.percentile(ds,10)) if ds else 0.0
    depth=max(NEAR_LAYER_MIN_PX,NEAR_LAYER_HEIGHT_MULT*gh)

    selected=[
        c for c in comps
        if max(0,axis_x-c["x2"]) <= d0+depth
    ]

    x1=int(round(axis_x-d0-depth-max(2,0.5*gh)))
    layer_rect=clip_rect(
        (x1,rect[1],axis_x-1,rect[3]),
        mask.shape[1],mask.shape[0]
    )

    band_depth=max(1.0,axis_x-rect[0])
    near_start_frac=float(d0/band_depth)

    return selected,layer_rect,gh,near_start_frac


def projection_groups(mask,layer_rect,axis,gh):
    """
    Evidência coarse da camada textual.

    Retorna:
    - grupos coarse;
    - span entre grupos quando há >=2 grupos;
    - número de bins ocupados;
    - extensão do primeiro ao último bin ocupado (independe de agrupamento);
    - proporção de bins ocupados;
    - número total de bins.

    A extensão ocupada é a principal novidade para detectar labels
    congestionados que viram um único grupo.
    """
    if layer_rect is None:
        return [],0.0,0,0.0,0.0,0

    x1,y1,x2,y2=layer_rect
    crop=mask[y1:y2+1,x1:x2+1]

    if axis=="x":
        proj=(crop>0).sum(axis=0)
        length=crop.shape[1]
    else:
        proj=(crop>0).sum(axis=1)
        length=crop.shape[0]

    binw=max(4,int(round(LAYER_BIN_MULT*max(2.0,gh))))

    occupied=[]
    groups=[]
    n_bins=int(math.ceil(length/binw))

    for i in range(n_bins):
        a=i*binw
        b=min(length,(i+1)*binw)
        pix=int(proj[a:b].sum())
        occupied.append(pix >= MIN_LAYER_FOREGROUND_PIXELS)

    i=0
    while i<n_bins:
        if not occupied[i]:
            i+=1
            continue
        start=i
        end=i
        gap_used=False
        i+=1
        while i<n_bins:
            if occupied[i]:
                end=i
                gap_used=False
                i+=1
            elif not gap_used and i+1<n_bins and occupied[i+1]:
                gap_used=True
                i+=1
            else:
                break

        p1=start*binw
        p2=min(length-1,(end+1)*binw-1)

        if axis=="x":
            groups.append({
                "x1":x1+p1,"y1":y1,"x2":x1+p2,"y2":y2,
                "cx":x1+(p1+p2)/2,"cy":(y1+y2)/2
            })
        else:
            groups.append({
                "x1":x1,"y1":y1+p1,"x2":x2,"y2":y1+p2,
                "cx":(x1+x2)/2,"cy":y1+(p1+p2)/2
            })
        i=max(i,end+1)

    if len(groups)>=2:
        if axis=="x":
            vals=[g["cx"] for g in groups]
            span=(max(vals)-min(vals))/max(1,layer_rect[2]-layer_rect[0])
        else:
            vals=[g["cy"] for g in groups]
            span=(max(vals)-min(vals))/max(1,layer_rect[3]-layer_rect[1])
    else:
        span=0.0

    occ_idx=[i for i,v in enumerate(occupied) if v]
    if occ_idx:
        occupied_extent=(occ_idx[-1]-occ_idx[0]+1)/max(1,n_bins)
    else:
        occupied_extent=0.0

    occupied_count=int(sum(occupied))
    occupied_ratio=occupied_count/max(1,n_bins)

    return (
        groups,
        float(span),
        occupied_count,
        float(occupied_extent),
        float(occupied_ratio),
        int(n_bins),
    )


def discrete_layer_presence(groups,span,near_start_frac):
    if not np.isfinite(near_start_frac):
        return 0
    return int(
        near_start_frac <= MAX_NEAR_START_FRAC
        and len(groups)>=MIN_LAYER_GROUPS
        and span>=MIN_LAYER_SPAN
    )


def congested_layer_presence(
    occupied_bins,
    occupied_extent,
    occupied_ratio,
    near_start_frac,
):
    """
    Evidência para uma faixa textual extensa/congestionada mesmo quando
    toda a camada vira um único grupo.
    """
    if not np.isfinite(near_start_frac):
        return 0
    return int(
        near_start_frac <= MAX_NEAR_START_FRAC
        and occupied_bins >= MIN_CONGESTED_OCCUPIED_BINS
        and occupied_extent >= MIN_CONGESTED_OCCUPIED_EXTENT
        and occupied_ratio >= MIN_CONGESTED_OCCUPIED_RATIO
    )


def combine_presence(seg_ok, discrete_ok, congested_ok, near_start_frac):
    if not np.isfinite(near_start_frac) or near_start_frac > MAX_NEAR_START_FRAC:
        return 0
    return int(bool(seg_ok) or bool(discrete_ok) or bool(congested_ok))


def evidence_label(seg_ok, discrete_ok, congested_ok):
    parts=[]
    if seg_ok:
        parts.append("SEG")
    if discrete_ok:
        parts.append("DISCRETE")
    if congested_ok:
        parts.append("CONGESTED")
    return "+".join(parts) if parts else "NONE"


# ============================================================
# 2. SEGMENTAÇÃO INDIVIDUAL — diagnóstico independente
# ============================================================

def expand_x_for_segmentation(comps,seeds,axis_y,gh):
    if not seeds:
        return []

    selected=list(seeds)
    ids={id(c) for c in selected}
    max_depth=X_SEG_CHAIN_MAX_DEPTH_MULT*gh

    changed=True
    while changed:
        changed=False
        for c in comps:
            if id(c) in ids:
                continue
            if max(0,c["y1"]-axis_y)>max_depth:
                continue

            for s in selected:
                xgap=max(0,max(c["x1"]-s["x2"],s["x1"]-c["x2"]))
                ygap=max(0,max(c["y1"]-s["y2"],s["y1"]-c["y2"]))
                if (
                    xgap<=X_SEG_CHAIN_X_GAP_MULT*gh
                    and ygap<=X_SEG_CHAIN_Y_GAP_MULT*gh
                ):
                    selected.append(c)
                    ids.add(id(c))
                    changed=True
                    break
    return selected


def expand_y_for_segmentation(comps,seeds,axis_x,gh):
    if not seeds:
        return []

    selected=list(seeds)
    ids={id(c) for c in selected}

    changed=True
    while changed:
        changed=False
        for c in comps:
            if id(c) in ids:
                continue
            for s in selected:
                y_overlap=max(
                    0,
                    min(c["y2"],s["y2"])-max(c["y1"],s["y1"])+1
                )
                min_h=max(1,min(c["h"],s["h"]))
                xgap=max(0,max(c["x1"]-s["x2"],s["x1"]-c["x2"]))
                if (
                    y_overlap/min_h>=0.35
                    and xgap<=Y_WORD_GAP_MULT*gh
                ):
                    selected.append(c)
                    ids.add(id(c))
                    changed=True
                    break
    return selected


def group_x(comps,canvas_w,canvas_h):
    if not comps: return []

    gh=median_glyph_h(comps)
    row_tol=max(3,X_ROW_TOL_MULT*gh)
    word_gap=max(5,X_WORD_GAP_MULT*gh)

    ordered=sorted(comps,key=lambda c:(c["cy"],c["x1"]))
    rows=[]

    for c in ordered:
        placed=False
        for row in rows:
            med=float(np.median([z["cy"] for z in row]))
            if abs(c["cy"]-med)<=row_tol:
                row.append(c)
                placed=True
                break
        if not placed:
            rows.append([c])

    tokens=[]
    for row in rows:
        row=sorted(row,key=lambda c:c["x1"])
        cur=[row[0]]
        for c in row[1:]:
            gap=c["x1"]-cur[-1]["x2"]-1
            if gap<=word_gap:
                cur.append(c)
            else:
                tokens.append(merge_boxes(cur))
                cur=[c]
        tokens.append(merge_boxes(cur))

    # União local em diagonal/empilhamento.
    tokens=[t for t in tokens if t]
    changed=True
    while changed:
        changed=False
        out=[]
        used=[False]*len(tokens)
        for i,a in enumerate(tokens):
            if used[i]: continue
            cl=[a]; used[i]=True
            grew=True
            while grew:
                grew=False
                A=merge_boxes(cl)
                for j,b in enumerate(tokens):
                    if used[j]: continue
                    xgap=max(0,max(b["x1"]-A["x2"],A["x1"]-b["x2"]))
                    ygap=max(0,max(b["y1"]-A["y2"],A["y1"]-b["y2"]))
                    if xgap<=1.6*gh and ygap<=1.5*gh:
                        cl.append(b); used[j]=True; grew=True
            out.append(merge_boxes(cl))
            if len(cl)>1:
                changed=True
        tokens=out

    maxw=max(35,int(MAX_BOX_W_FRAC_X*canvas_w))
    maxh=max(18,int(MAX_BOX_H_FRAC*canvas_h))

    out=[]
    for t in tokens:
        if t["w"]*t["h"]<MIN_BOX_AREA: continue
        if t["w"]>maxw or t["h"]>maxh: continue
        if t["h"]<=2 and t["w"]>20: continue
        out.append(t)
    return out


def group_y(comps,canvas_w,canvas_h):
    if not comps: return []

    gh=median_glyph_h(comps)
    row_tol=max(3,Y_ROW_TOL_MULT*gh)
    word_gap=max(6,Y_WORD_GAP_MULT*gh)

    ordered=sorted(comps,key=lambda c:(c["cy"],c["x1"]))
    rows=[]

    for c in ordered:
        placed=False
        for row in rows:
            med=float(np.median([z["cy"] for z in row]))
            if abs(c["cy"]-med)<=row_tol:
                row.append(c)
                placed=True
                break
        if not placed:
            rows.append([c])

    tokens=[]
    for row in rows:
        row=sorted(row,key=lambda c:c["x1"])
        cur=[row[0]]
        for c in row[1:]:
            gap=c["x1"]-cur[-1]["x2"]-1
            if gap<=word_gap:
                cur.append(c)
            else:
                tokens.append(merge_boxes(cur))
                cur=[c]
        tokens.append(merge_boxes(cur))

    maxw=max(35,int(MAX_BOX_W_FRAC_Y*canvas_w))
    maxh=max(18,int(MAX_BOX_H_FRAC*canvas_h))

    out=[]
    for t in tokens:
        if t["w"]*t["h"]<MIN_BOX_AREA: continue
        if t["w"]>maxw or t["h"]>maxh: continue
        if t["w"]<=2 and t["h"]>20: continue
        out.append(t)
    return out


def segmentation_span(boxes,rect,axis):
    if len(boxes)<2:
        return 0.0
    if axis=="x":
        vals=[b["cx"] for b in boxes]
        return float((max(vals)-min(vals))/max(1,rect[2]-rect[0]))
    vals=[b["cy"] for b in boxes]
    return float((max(vals)-min(vals))/max(1,rect[3]-rect[1]))


def segmentation_ok(boxes,span):
    return int(
        len(boxes)>=MIN_SEGMENTED_LABELS
        and span>=MIN_SEGMENTATION_SPAN
    )


# ============================================================
# 3. CAMADA SECUNDÁRIA — provável axis label, sem OCR
# ============================================================

def probable_axislabel_x(comps,near_ids,rect,axis_y,gh):
    far=[c for c in comps if id(c) not in near_ids]
    if not far:
        return []

    # Agrupa o texto distante e procura um bloco aproximadamente central.
    boxes=group_x(far,rect[2]-rect[0]+1,rect[3]-rect[1]+1)
    center=(rect[0]+rect[2])/2
    width=max(1,rect[2]-rect[0])

    cand=[]
    for b in boxes:
        dist=b["cy"]-axis_y
        centered=abs(b["cx"]-center)/width<=AXISLABEL_CENTER_TOL_FRAC
        if dist>=AXISLABEL_MIN_GAP_MULT*gh and centered:
            cand.append(b)
    return cand


def probable_axislabel_y(comps,near_ids,rect,axis_x,gh):
    far=[c for c in comps if id(c) not in near_ids]
    if not far:
        return []

    # Em Y o axis label costuma ser um texto vertical/alongado mais distante.
    boxes=group_y(far,rect[2]-rect[0]+1,rect[3]-rect[1]+1)
    cand=[]
    for b in boxes:
        dist=axis_x-b["cx"]
        vertical=b["h"]>1.8*max(1,b["w"])
        if dist>=AXISLABEL_MIN_GAP_MULT*gh and vertical:
            cand.append(b)
    return cand


def analyze(path):
    img=Image.open(path).convert("RGB")
    arr=np.asarray(img)
    H,W,_=arr.shape

    bg=estimate_background(arr)
    mask=foreground_mask(arr,bg)

    axes=axis_v3.analyze(path)
    xr,yr,x_axis_y,y_axis_x=infer_bands(img,axes)

    if xr is None or yr is None:
        return {
            "plot_status":axes.get("plot_status",""),
            "tlx_evaluable":0,"tly_evaluable":0,
            "tlx_present_candidate":None,"tly_present_candidate":None,
            "tlx_segmentation_ok":0,"tly_segmentation_ok":0,
            "tlx_n_segmented":0,"tly_n_segmented":0,
            "_x_boxes":[],"_y_boxes":[],
            "_x_layer_groups":[],"_y_layer_groups":[],
            "_x_axislabel":[],"_y_axislabel":[],
            "_x_rect":xr,"_y_rect":yr,
            "_x_layer_rect":None,"_y_layer_rect":None,
        }

    xc=components(mask,xr)
    yc=components(mask,yr)

    # -------- CAMADA X --------
    xnear,x_layer_rect,xgh,xnearfrac=near_layer_x(mask,xc,xr,x_axis_y)
    (
        xgroups,xspan,xocc,xextent,xratio,xbins
    )=projection_groups(mask,x_layer_rect,"x",xgh)

    # -------- CAMADA Y --------
    ynear,y_layer_rect,ygh,ynearfrac=near_layer_y(mask,yc,yr,y_axis_x)
    (
        ygroups,yspan,yocc,yextent,yratio,ybins
    )=projection_groups(mask,y_layer_rect,"y",ygh)

    # -------- SEGMENTAÇÃO X --------
    xsegcomps=expand_x_for_segmentation(xc,xnear,x_axis_y,xgh)
    xboxes=group_x(xsegcomps,W,H)
    xsegspan=segmentation_span(xboxes,xr,"x")
    xsegok=segmentation_ok(xboxes,xsegspan)

    # -------- SEGMENTAÇÃO Y --------
    ysegcomps=expand_y_for_segmentation(yc,ynear,y_axis_x,ygh)
    yboxes=group_y(ysegcomps,W,H)
    ysegspan=segmentation_span(yboxes,yr,"y")
    ysegok=segmentation_ok(yboxes,ysegspan)

    # -------- EVIDÊNCIAS DE PRESENÇA --------
    xdiscrete=discrete_layer_presence(xgroups,xspan,xnearfrac)
    ydiscrete=discrete_layer_presence(ygroups,yspan,ynearfrac)

    xcongested=congested_layer_presence(
        xocc,xextent,xratio,xnearfrac
    )
    ycongested=congested_layer_presence(
        yocc,yextent,yratio,ynearfrac
    )

    xpresent=combine_presence(
        xsegok,xdiscrete,xcongested,xnearfrac
    )
    ypresent=combine_presence(
        ysegok,ydiscrete,ycongested,ynearfrac
    )

    # -------- CAMADA SECUNDÁRIA --------
    xnear_ids={id(c) for c in xnear}
    ynear_ids={id(c) for c in ynear}

    xaxislabel=probable_axislabel_x(
        xc,xnear_ids,xr,x_axis_y,xgh
    )
    yaxislabel=probable_axislabel_y(
        yc,ynear_ids,yr,y_axis_x,ygh
    )

    return {
        "plot_status":axes.get("plot_status",""),
        "tlx_evaluable":1,
        "tly_evaluable":1,

        "tlx_present_candidate":xpresent,
        "tly_present_candidate":ypresent,
        "tlx_presence_evidence":evidence_label(xsegok,xdiscrete,xcongested),
        "tly_presence_evidence":evidence_label(ysegok,ydiscrete,ycongested),

        "tlx_evidence_segmentation":int(xsegok),
        "tly_evidence_segmentation":int(ysegok),
        "tlx_evidence_discrete_layer":int(xdiscrete),
        "tly_evidence_discrete_layer":int(ydiscrete),
        "tlx_evidence_congested_layer":int(xcongested),
        "tly_evidence_congested_layer":int(ycongested),

        "tlx_near_start_frac":xnearfrac,
        "tly_near_start_frac":ynearfrac,

        "tlx_text_layer_groups":len(xgroups),
        "tly_text_layer_groups":len(ygroups),
        "tlx_text_layer_span":xspan,
        "tly_text_layer_span":yspan,
        "tlx_text_layer_occupied_bins":xocc,
        "tly_text_layer_occupied_bins":yocc,
        "tlx_text_layer_total_bins":xbins,
        "tly_text_layer_total_bins":ybins,
        "tlx_text_layer_occupied_extent":xextent,
        "tly_text_layer_occupied_extent":yextent,
        "tlx_text_layer_occupied_ratio":xratio,
        "tly_text_layer_occupied_ratio":yratio,

        "tlx_segmentation_ok":xsegok,
        "tly_segmentation_ok":ysegok,
        "tlx_n_segmented":len(xboxes),
        "tly_n_segmented":len(yboxes),
        "tlx_segmentation_span":xsegspan,
        "tly_segmentation_span":ysegspan,

        "tlx_axislabel_candidate":int(len(xaxislabel)>0),
        "tly_axislabel_candidate":int(len(yaxislabel)>0),
        "tlx_n_axislabel_candidates":len(xaxislabel),
        "tly_n_axislabel_candidates":len(yaxislabel),

        "tlx_glyph_h_est":xgh,
        "tly_glyph_h_est":ygh,

        "axis_x_coord":x_axis_y,
        "axis_y_coord":y_axis_x,

        "_x_boxes":xboxes,
        "_y_boxes":yboxes,
        "_x_layer_groups":xgroups,
        "_y_layer_groups":ygroups,
        "_x_axislabel":xaxislabel,
        "_y_axislabel":yaxislabel,
        "_x_rect":xr,
        "_y_rect":yr,
        "_x_layer_rect":x_layer_rect,
        "_y_layer_rect":y_layer_rect,
    }


def font_default(size=11):
    try:
        return ImageFont.truetype("DejaVuSans.ttf",size)
    except Exception:
        return ImageFont.load_default()


def draw_box(draw,b,color,width=2):
    if b is None:
        return
    if isinstance(b,dict):
        box=(b["x1"],b["y1"],b["x2"],b["y2"])
    else:
        box=b
    draw.rectangle(box,outline=color,width=width)


def make_overlay(path,res,out_path):
    im=Image.open(path).convert("RGB")
    d=ImageDraw.Draw(im)

    # Bandas amplas
    draw_box(d,res["_x_rect"],(100,180,255),2)
    draw_box(d,res["_y_rect"],(225,185,35),2)

    # Camadas efetivamente usadas para presença
    draw_box(d,res["_x_layer_rect"],(0,105,0),2)
    draw_box(d,res["_y_layer_rect"],(95,0,120),2)

    # Grupos coarse de presença
    for g in res["_x_layer_groups"]:
        draw_box(d,g,(0,105,0),2)
    for g in res["_y_layer_groups"]:
        draw_box(d,g,(95,0,120),2)

    # Segmentação individual
    for b in res["_x_boxes"]:
        draw_box(d,b,(0,200,0),3)
    for b in res["_y_boxes"]:
        draw_box(d,b,(210,0,210),3)

    # Camada secundária
    for b in res["_x_axislabel"]:
        draw_box(d,b,(235,125,20),3)
    for b in res["_y_axislabel"]:
        draw_box(d,b,(0,175,195),3)

    im.save(out_path)


def save_crop(path,res,axis,out_path):
    im=Image.open(path).convert("RGB")

    if axis=="x":
        rect0=res["_x_rect"]
        boxes=res["_x_boxes"]
        groups=res["_x_layer_groups"]
        axislabels=res["_x_axislabel"]
        layer_rect=res["_x_layer_rect"]
        colors=((0,200,0),(0,105,0),(235,125,20))
    else:
        rect0=res["_y_rect"]
        boxes=res["_y_boxes"]
        groups=res["_y_layer_groups"]
        axislabels=res["_y_axislabel"]
        layer_rect=res["_y_layer_rect"]
        colors=((210,0,210),(95,0,120),(0,175,195))

    if rect0 is None:
        Image.new("RGB",(300,80),"white").save(out_path)
        return

    x1,y1,x2,y2=rect0
    crop=im.crop((x1,y1,x2+1,y2+1))
    d=ImageDraw.Draw(crop)

    if layer_rect is not None:
        lr=(
            layer_rect[0]-x1,layer_rect[1]-y1,
            layer_rect[2]-x1,layer_rect[3]-y1
        )
        d.rectangle(lr,outline=colors[1],width=2)

    for g in groups:
        d.rectangle(
            (g["x1"]-x1,g["y1"]-y1,g["x2"]-x1,g["y2"]-y1),
            outline=colors[1],width=2
        )

    for b in boxes:
        d.rectangle(
            (b["x1"]-x1,b["y1"]-y1,b["x2"]-x1,b["y2"]-y1),
            outline=colors[0],width=2
        )

    for b in axislabels:
        d.rectangle(
            (b["x1"]-x1,b["y1"]-y1,b["x2"]-x1,b["y2"]-y1),
            outline=colors[2],width=2
        )

    crop.save(out_path)


def fmt(v,nd=2):
    try:
        if pd.isna(v): return "-"
        return f"{float(v):.{nd}f}"
    except Exception:
        return "-"


def contact_sheet(g,root,outdir,out_path):
    if g.empty:
        return

    rows=math.ceil(len(g)/COLS)
    sheet=Image.new("RGB",(COLS*PANEL_W,rows*PANEL_H),"white")
    d=ImageDraw.Draw(sheet)
    f=font_default(11)

    for idx,row in g.reset_index(drop=True).iterrows():
        col=idx%COLS; rr=idx//COLS
        x0=col*PANEL_W; y0=rr*PANEL_H

        ov=outdir/"overlays"/row["filename"]
        im=Image.open(ov if ov.exists() else root/row["filename"]).convert("RGB")
        im.thumbnail((PANEL_W-20,MAIN_H-8))
        sheet.paste(im,(x0+(PANEL_W-im.width)//2,y0+4))

        cx=Image.open(outdir/"crops_x"/row["filename"]).convert("RGB")
        cx.thumbnail((PANEL_W-24,BAND_H))
        sheet.paste(cx,(x0+12,y0+MAIN_H+2))

        cy=Image.open(outdir/"crops_y"/row["filename"]).convert("RGB")
        cy.thumbnail((PANEL_W-24,BAND_H))
        sheet.paste(cy,(x0+12,y0+MAIN_H+BAND_H+6))

        txt=(
            f"{row['filename']}\n"
            f"X P={row.get('tlx_present_candidate','-')} "
            f"ev={row.get('tlx_presence_evidence','-')} "
            f"grp={row.get('tlx_text_layer_groups','-')} "
            f"occExt={fmt(row.get('tlx_text_layer_occupied_extent'))} | "
            f"S={row.get('tlx_segmentation_ok','-')} "
            f"n={row.get('tlx_n_segmented','-')} "
            f"axisL={row.get('tlx_axislabel_candidate','-')}\n"
            f"Y P={row.get('tly_present_candidate','-')} "
            f"ev={row.get('tly_presence_evidence','-')} "
            f"grp={row.get('tly_text_layer_groups','-')} "
            f"occExt={fmt(row.get('tly_text_layer_occupied_extent'))} | "
            f"S={row.get('tly_segmentation_ok','-')} "
            f"n={row.get('tly_n_segmented','-')} "
            f"axisL={row.get('tly_axislabel_candidate','-')}"
        )

        d.multiline_text(
            (x0+10,y0+MAIN_H+2*BAND_H+12),
            txt,fill="black",font=f,spacing=2
        )

    sheet.save(out_path)



def make_negative_control(path,res,axis,out_path):
    """
    Cria um controle negativo sintético apagando a zona próxima ao eixo
    onde ficam os tick labels, preservando o eixo e, sempre que possível,
    a camada mais distante que pode conter o axis label.

    Estes controles existem SOMENTE para testar especificidade do detector.
    Nunca entram no conjunto experimental nem em estatísticas de conformidade.
    """
    im=Image.open(path).convert("RGB")
    arr=np.asarray(im).copy()
    bg=estimate_background(arr).astype(np.uint8)
    H,W,_=arr.shape

    if axis=="x":
        rect=res.get("_x_rect")
        if rect is None:
            return False
        x1,y1,x2,y2=rect
        depth=max(1,y2-y1+1)
        cut_y2=min(H-1,int(round(y1+NEGATIVE_MASK_NEAR_FRAC*depth)))
        arr[y1:cut_y2+1,x1:x2+1]=bg.reshape(1,1,3)

    else:
        rect=res.get("_y_rect")
        if rect is None:
            return False
        x1,y1,x2,y2=rect
        depth=max(1,x2-x1+1)
        cut_x1=max(0,int(round(x2-NEGATIVE_MASK_NEAR_FRAC*depth)))
        arr[y1:y2+1,cut_x1:x2+1]=bg.reshape(1,1,3)

    Image.fromarray(arr).save(out_path)
    return True


def select_negative_bases(df):
    bases=[]
    for p in PROFILES:
        g=df[
            (df["profile"]==p)
            & (df["tlx_present_candidate"]==1)
            & (df["tly_present_candidate"]==1)
        ].copy()

        if len(g)<NEGATIVE_BASES_PER_PROFILE:
            g=df[df["profile"]==p].copy()

        g=g.sort_values("filename").head(NEGATIVE_BASES_PER_PROFILE)
        bases.extend(g["filename"].tolist())
    return bases


def negative_contact_sheet(negdf,outdir,out_path):
    if negdf.empty:
        return

    cols=4
    pw=420
    ph=330
    ih=255
    rows=math.ceil(len(negdf)/cols)

    sheet=Image.new("RGB",(cols*pw,rows*ph),"white")
    d=ImageDraw.Draw(sheet)
    f=font_default(10)

    for idx,row in negdf.reset_index(drop=True).iterrows():
        col=idx%cols
        rr=idx//cols
        x0=col*pw
        y0=rr*ph

        p=outdir/"negative_controls_overlays"/row["control_filename"]
        im=Image.open(p).convert("RGB")
        im.thumbnail((pw-16,ih-6))
        sheet.paste(im,(x0+(pw-im.width)//2,y0+3))

        txt=(
            f"{row['control_filename']}\\n"
            f"masked={row['masked_axis']} | "
            f"X={row['tlx_present_candidate']} "
            f"Y={row['tly_present_candidate']} | "
            f"PASS={row['negative_control_pass']}"
        )
        d.multiline_text((x0+8,y0+ih+5),txt,fill="black",font=f,spacing=2)

    sheet.save(out_path)


def run_negative_controls(sample_df,results_df,root,outdir):
    controls_dir=outdir/"negative_controls"
    overlays_dir=outdir/"negative_controls_overlays"
    controls_dir.mkdir(parents=True,exist_ok=True)
    overlays_dir.mkdir(parents=True,exist_ok=True)

    bases=select_negative_bases(results_df)
    manifest=[]
    rows=[]

    for fname in bases:
        src=root/fname
        original_res=analyze(src)
        prof=str(results_df.loc[results_df["filename"]==fname,"profile"].iloc[0])

        for axis in ("x","y"):
            cname=f"NEG_{axis.upper()}__{fname}"
            cpath=controls_dir/cname

            ok=make_negative_control(src,original_res,axis,cpath)
            if not ok:
                continue

            cres=analyze(cpath)
            make_overlay(cpath,cres,overlays_dir/cname)

            masked_present=(
                cres["tlx_present_candidate"]
                if axis=="x"
                else cres["tly_present_candidate"]
            )
            other_present=(
                cres["tly_present_candidate"]
                if axis=="x"
                else cres["tlx_present_candidate"]
            )

            pass_flag=int(masked_present==0)

            manifest.append({
                "source_filename":fname,
                "profile":prof,
                "masked_axis":axis.upper(),
                "control_filename":cname,
            })

            rows.append({
                "source_filename":fname,
                "profile":prof,
                "masked_axis":axis.upper(),
                "control_filename":cname,
                "masked_axis_present_after_mask":masked_present,
                "other_axis_present_after_mask":other_present,
                "negative_control_pass":pass_flag,
                "tlx_present_candidate":cres["tlx_present_candidate"],
                "tly_present_candidate":cres["tly_present_candidate"],
                "tlx_presence_evidence":cres.get("tlx_presence_evidence",""),
                "tly_presence_evidence":cres.get("tly_presence_evidence",""),
                "tlx_segmentation_ok":cres.get("tlx_segmentation_ok",0),
                "tly_segmentation_ok":cres.get("tly_segmentation_ok",0),
                "tlx_text_layer_occupied_extent":cres.get("tlx_text_layer_occupied_extent",0),
                "tly_text_layer_occupied_extent":cres.get("tly_text_layer_occupied_extent",0),
                "tlx_near_start_frac":cres.get("tlx_near_start_frac",float("nan")),
                "tly_near_start_frac":cres.get("tly_near_start_frac",float("nan")),
            })

    mdf=pd.DataFrame(manifest)
    ndf=pd.DataFrame(rows)

    mdf.to_csv(
        outdir/"negative_control_manifest_v4.csv",
        index=False,encoding="utf-8-sig"
    )
    ndf.to_csv(
        outdir/"negative_control_results_v4.csv",
        index=False,encoding="utf-8-sig"
    )

    negative_contact_sheet(
        ndf,outdir,outdir/"contact_negative_controls_v4.png"
    )

    return ndf


def main():
    args=parse_args()
    root=args.root.expanduser().resolve()
    outdir=(
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root/OUT_DIRNAME
    )

    for sub in ("overlays","crops_x","crops_y","negative_controls","negative_controls_overlays"):
        (outdir/sub).mkdir(parents=True,exist_ok=True)

    actual_hash=check_axis_hash()
    inv=inventory(root)
    sample,manifest_source=load_manifest(
        root,inv,int(args.units_per_profile),int(args.seed)
    )

    sample.to_csv(
        outdir/"ticklabel_calibration_manifest_v4.csv",
        index=False,encoding="utf-8-sig"
    )

    rows=[]
    errors=[]
    internal_results={}

    print("="*80)
    print("FASE A — CALIBRAÇÃO V4: PRESENÇA DOS TICK LABELS X/Y")
    print("="*80)
    print(f"Detector de eixos V3 SHA-256: {actual_hash}")
    print(f"Imagens: {len(sample)}")
    print(f"Manifesto: {manifest_source}")
    print("OCR: não utilizado")
    print("F: não utilizado na detecção")
    print("Presença V4 = SEG ou camada discreta ou camada congestionada")
    print("Controles negativos sintéticos serão executados após a calibração")
    print()

    for _,r in sample.iterrows():
        path=root/r["filename"]

        try:
            res=analyze(path)

            row={
                "filename":r["filename"],
                "profile":r["profile"],
                "unit_id":r["unit_id"],
                "repeat":r["repeat"],
                **{k:v for k,v in res.items() if not k.startswith("_")},
            }
            rows.append(row)
            internal_results[r["filename"]]=res

            make_overlay(path,res,outdir/"overlays"/r["filename"])
            save_crop(path,res,"x",outdir/"crops_x"/r["filename"])
            save_crop(path,res,"y",outdir/"crops_y"/r["filename"])

        except Exception as exc:
            errors.append({
                "filename":r["filename"],
                "profile":r["profile"],
                "unit_id":r["unit_id"],
                "error_type":type(exc).__name__,
                "error_message":str(exc),
            })

        done=len(rows)+len(errors)
        if done%10==0 or done==len(sample):
            print(f"Processadas: {done}/{len(sample)}")

    df=pd.DataFrame(rows)
    err=pd.DataFrame(errors)

    df.to_csv(
        outdir/"ticklabel_calibration_v4.csv",
        index=False,encoding="utf-8-sig"
    )
    err.to_csv(
        outdir/"ticklabel_calibration_errors_v4.csv",
        index=False,encoding="utf-8-sig"
    )

    for p in PROFILES:
        contact_sheet(
            df[df["profile"]==p],
            root,outdir,
            outdir/f"contact_{p}.png"
        )

    negdf=run_negative_controls(
        sample,df,root,outdir
    )

    lines=[
        "FASE A — CALIBRAÇÃO V4: PRESENÇA DOS TICK LABELS X/Y",
        "="*78,
        f"Imagens processadas: {len(df)}",
        f"Erros: {len(err)}",
        f"Detector de eixos V3 SHA-256: {actual_hash}",
        f"Manifesto: {manifest_source}",
        "",
        "V4:",
        "- presença = SEGMENTAÇÃO OU CAMADA DISCRETA OU CAMADA CONGESTIONADA;",
        "- faixa congestionada usa extensão ocupada, sem exigir múltiplos grupos;",
        "- guarda de proximidade evita promover texto distante como tick label;",
        "- segmentação individual continua diagnóstico separado;",
        "- controles negativos sintéticos testam especificidade;",
        "- sem OCR e sem F.",
        "",
        "POR PERFIL:",
    ]

    for p in PROFILES:
        g=df[df["profile"]==p]
        if g.empty:
            continue

        xe=g["tlx_evaluable"]==1
        ye=g["tly_evaluable"]==1

        xp=pd.to_numeric(
            g.loc[xe,"tlx_present_candidate"],errors="coerce"
        ).fillna(0)
        yp=pd.to_numeric(
            g.loc[ye,"tly_present_candidate"],errors="coerce"
        ).fillna(0)

        xs=pd.to_numeric(
            g.loc[xe,"tlx_segmentation_ok"],errors="coerce"
        ).fillna(0)
        ys=pd.to_numeric(
            g.loc[ye,"tly_segmentation_ok"],errors="coerce"
        ).fillna(0)

        xn=pd.to_numeric(
            g.loc[xe,"tlx_n_segmented"],errors="coerce"
        ).dropna()
        yn=pd.to_numeric(
            g.loc[ye,"tly_n_segmented"],errors="coerce"
        ).dropna()

        xcong=pd.to_numeric(
            g.loc[xe,"tlx_evidence_congested_layer"],errors="coerce"
        ).fillna(0)
        ycong=pd.to_numeric(
            g.loc[ye,"tly_evidence_congested_layer"],errors="coerce"
        ).fillna(0)

        lines.append(
            f"{p}: "
            f"X presente={int(xp.sum())}/{int(xe.sum())}, "
            f"segOK={int(xs.sum())}/{int(xe.sum())}, "
            f"congestionado={int(xcong.sum())}/{int(xe.sum())}, "
            f"mediana nSegX={(xn.median() if len(xn) else float('nan')):.1f}; "
            f"Y presente={int(yp.sum())}/{int(ye.sum())}, "
            f"segOK={int(ys.sum())}/{int(ye.sum())}, "
            f"congestionado={int(ycong.sum())}/{int(ye.sum())}, "
            f"mediana nSegY={(yn.median() if len(yn) else float('nan')):.1f}"
        )

    neg_total=len(negdf)
    neg_pass=int(negdf["negative_control_pass"].sum()) if neg_total else 0

    lines.extend([
        "",
        "CONTROLES NEGATIVOS SINTÉTICOS:",
        f"- controles processados: {neg_total}",
        f"- masked-axis corretamente ausente: {neg_pass}/{neg_total}",
        "- estes controles são apenas calibração e nunca entram nos resultados experimentais;",
        "",
        "AUDITORIA PRINCIPAL:",
        "- primeiro julgar P (presença), independentemente de S (segmentação);",
        "- BC_008_R08 é caso crítico: X deve ser PRESENT via evidência CONGESTED;",
        "- falsos negativos V3 com S=1 devem ser recuperados pela combinação de evidências;",
        "- conferir se axis labels isolados não geram PRESENT nos controles negativos;",
        "- conferir BI/BC inclinados/congestionados;",
        "- NÃO ajustar pela quantidade esperada de labels em F.",
        "",
        "CRITÉRIO PARA HOLDOUT:",
        "- presença visual coerente nos 60 casos;",
        "- controles negativos com especificidade satisfatória;",
        "- ausência de falso negativo sistemático;",
        "- parâmetros de PRESENÇA congelados antes do holdout independente.",
    ])

    summary=outdir/"ticklabel_calibration_summary_v4.txt"
    summary.write_text("\n".join(lines),encoding="utf-8")

    print()
    print("Concluído.")
    print("Resumo:",summary)
    print("CSV:",outdir/"ticklabel_calibration_v4.csv")
    print("Envie resumo, CSV e seis pranchas.")


if __name__=="__main__":
    main()
