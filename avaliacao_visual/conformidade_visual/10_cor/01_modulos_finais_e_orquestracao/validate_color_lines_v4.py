# -*- coding: utf-8 -*-
"""
validate_color_lines_v4.py

VALIDAÇÃO L-Cor V4
==================

Motivação
---------
A V1 funcionou bem para LI, mas falhou sistematicamente em LC porque exigia
que cada cor formasse um componente conexo com grande extensão horizontal.
Em gráficos com duas linhas, cruzamentos, sobreposição parcial e oclusão
podem fragmentar uma das trajetórias em vários componentes.

A V2 muda a unidade de detecção:

    componente conexo  ->  família cromática global + suporte espacial

Para cada imagem:
1. seleciona pixels cromáticos na resolução original;
2. constrói um histograma circular de matiz (HSV);
3. identifica famílias de matiz sem usar F;
4. para cada família, reúne TODOS os pixels compatíveis, mesmo que estejam
   separados em vários componentes;
5. mede a distribuição espacial da família em bins do eixo X;
6. preserva famílias com suporte horizontal suficiente;
7. extrai a cor-núcleo usando pixels de maior saturação;
8. calcula CIELAB/CIEDE2000 somente depois da extração;
9. registra 1 cor para LI e 2 para LC apenas como diagnóstico, não como
   critério de conformidade.

Objetivo específico da V2
-------------------------
- manter vermelho em LI;
- recuperar simultaneamente azul + laranja em LC;
- eliminar cores locais de cruzamentos/antialiasing, como pequenos roxos;
- impedir que preto/cinza seja escolhido como cor-base.

A especificação F NÃO participa da descoberta das cores.

Amostra
-------
Mesma estratégia reprodutível de 40 imagens:
- 10 LI canvas 1200x800
- 10 LI canvas diferente
- 10 LC canvas 1200x800
- 10 LC canvas diferente

Saída
-----
C:/Users/Labvis/Downloads/imagens3120/imagens/_color_lines_validation_v4/

    line_validation_v4.csv
    line_color_validation_v4.csv
    line_validation_contact_sheet_v4.png
    line_validation_summary_v4.txt
    overlays/*.png

Overlay
-------
Para cada família aceita:
- retângulo na própria cor observada = extensão espacial da família;
- texto = cor, suporte X e deltaE00.

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
# CAMINHOS
# ============================================================

ROOT_DIR = Path("C:/Users/Labvis/Downloads/imagens3120/imagens")
OUTPUT_DIR = ROOT_DIR / "_color_lines_validation_v4"
OVERLAY_DIR = OUTPUT_DIR / "overlays"


# ============================================================
# AMOSTRAGEM
# ============================================================

RANDOM_SEED = 20260827
N_PER_PROFILE_CANVAS_CLASS = 10

NAME_RE = re.compile(
    r"^(LI|LC)_(\d{3})_R(\d{2})\.png$",
    re.I
)


# ============================================================
# PALETA F - SOMENTE COMPARAÇÃO POSTERIOR
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
# DETECÇÃO DE FAMÍLIAS CROMÁTICAS
# ============================================================

# Pixels candidatos precisam ser realmente cromáticos.
MIN_SATURATION = 0.22

# Exclui branco/quase branco.
MAX_VALUE = 0.995

# Histograma circular de hue.
N_HUE_BINS = 180
SMOOTH_RADIUS = 2

# Separação entre picos de famílias distintas.
PEAK_MIN_SEPARATION_BINS = 9

# Janela em torno de cada pico de hue.
HUE_WINDOW_BINS = 5

# Número máximo de famílias candidatas antes do filtro espacial.
MAX_HUE_FAMILIES = 8

# Pico precisa ter alguma relevância no histograma.
PEAK_MIN_RELATIVE = 0.003

# Suporte mínimo absoluto.
MIN_FAMILY_PIXELS = 18

# Cor-núcleo: utiliza parte mais saturada da família.
CORE_SAT_QUANTILE = 0.72

# V3: pureza mínima da cor-núcleo.
# A V2 mostrou que as famílias reais de linha apresentaram suporte exato
# elevado, enquanto famílias derivadas de antialiasing/cruzamentos tiveram
# suporte muito baixo. O valor 0.50 é testado aqui empiricamente; ainda não
# é um limiar de conformidade visual, apenas um filtro de qualidade da
# extração cromática.
MIN_CORE_SUPPORT_EXACT = 0.50


# ============================================================
# SUPORTE ESPACIAL
# ============================================================

# Divide o eixo X em bins para medir presença da família ao longo do gráfico.
N_X_BINS = 48

# Um bin conta como ocupado se tiver ao menos este número de pixels da família.
MIN_PIXELS_PER_X_BIN = 2

# Suporte mínimo global de trajetória.
MIN_X_SPAN_FRAC = 0.22
MIN_OCCUPIED_X_BINS = 5
MIN_X_BIN_COVERAGE = 0.10

# Para evitar aceitar apenas legenda ou pequeno trecho local.
# Uma família é forte se satisfizer span + cobertura.
# Para LC, uma linha parcialmente ocluída pode ser menor que a outra,
# portanto NÃO há regra de "72% do maior span" da V1.

# Componentes muito pequenos derivados de cruzamentos tendem a falhar
# nesses limites.


# ============================================================
# V4 - DETECÇÃO DE FAMÍLIAS DERIVADAS DE SOBREPOSIÇÃO
# ============================================================

# A V3 pode aceitar cores de mistura quando duas linhas tracejadas ficam
# quase sobrepostas por grande extensão. A V4 acrescenta uma etapa
# pós-extração que NÃO usa F nem a quantidade esperada de séries.

# Distância espacial usada para verificar se pixels da família candidata
# ficam simultaneamente próximos de duas famílias cromáticas mais básicas.
OVERLAP_DILATION_RADIUS_PX = 5

# Fração mínima dos pixels da candidata que deve ficar perto de AMBAS as
# famílias-base.
OVERLAP_MIN_NEAR_BOTH_FRAC = 0.20

# Evidência complementar por bins em X: em quantos bins ocupados pela
# candidata as duas famílias-base também aparecem geometricamente próximas.
OVERLAP_MIN_XBIN_COOCCURRENCE = 0.25

# Distância vertical máxima entre as medianas das duas famílias-base no
# mesmo bin X para caracterizar quase sobreposição.
OVERLAP_MAX_BASE_Y_GAP_PX = 14

# A cor candidata precisa ser explicável como combinação convexa de
# cor A + cor B + branco (fundo/antialiasing), medida em RGB.
OVERLAP_MAX_MIXTURE_RGB_RESIDUAL = 12.0

# As duas famílias usadas para explicar a mistura devem ter núcleo
# cromático suficientemente estável.
OVERLAP_BASE_MIN_CORE_SUPPORT = 0.70

# Evita usar duas cores praticamente iguais como "bases" de mistura.
OVERLAP_MIN_BASE_RGB_SEPARATION = 45.0

# Cada família-base precisa possuir massa cromática não trivial em relação
# à família candidata.
OVERLAP_MIN_BASE_PIXELS_RELATIVE = 0.35


# ============================================================
# DEDUPLICAÇÃO
# ============================================================

# Famílias próximas em RGB e hue podem ser versões antialiased da mesma linha.
DUPLICATE_RGB_DISTANCE = 28.0
DUPLICATE_HUE_DISTANCE_DEG = 16.0


# ============================================================
# PRANCHA
# ============================================================

PANEL_W = 550
PANEL_H = 400
IMG_W = 505
IMG_H = 305
COLS = 4


# ============================================================
# COR
# ============================================================

def hex_to_rgb(v):
    v = v.lstrip("#")
    return tuple(
        int(v[i:i+2], 16)
        for i in (0, 2, 4)
    )


def rgb_to_hex(rgb):
    vals = [
        max(0, min(255, int(round(float(v)))))
        for v in rgb
    ]
    return "#{:02X}{:02X}{:02X}".format(*vals)


def rgb_to_lab_one(rgb):
    arr = np.array(
        [[[rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]]],
        dtype=np.float64
    )
    return rgb2lab(arr)[0, 0]


F_LAB = {
    name: rgb_to_lab_one(hex_to_rgb(hx))
    for name, hx in F_COLORS.items()
}


def nearest_f(rgb):
    lab = rgb_to_lab_one(rgb)

    distances = {
        name: float(
            deltaE_ciede2000(
                np.array([[lab]], dtype=np.float64),
                np.array([[target]], dtype=np.float64)
            )[0, 0]
        )
        for name, target in F_LAB.items()
    }

    nearest = min(
        distances,
        key=distances.get
    )

    return {
        "lab": lab,
        "nearest_F_name": nearest,
        "nearest_F_hex": F_COLORS[nearest],
        "deltaE00": distances[nearest],
    }


# ============================================================
# HUE CIRCULAR
# ============================================================

def circular_bin_distance(values, center, n=N_HUE_BINS):
    d = np.abs(values - center)
    return np.minimum(d, n - d)


def circular_deg_distance(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def circular_smooth(hist, radius):
    if radius <= 0:
        return hist.astype(float)

    out = np.zeros_like(
        hist,
        dtype=float
    )

    for offset in range(
        -radius,
        radius + 1
    ):
        out += np.roll(
            hist,
            offset
        )

    return out / (
        2 * radius + 1
    )


def select_hue_peaks(hist):
    smooth = circular_smooth(
        hist,
        SMOOTH_RADIUS
    )

    if smooth.max() <= 0:
        return []

    threshold = (
        float(smooth.max())
        * PEAK_MIN_RELATIVE
    )

    order = np.argsort(
        smooth
    )[::-1]

    peaks = []

    for idx in order:
        idx = int(idx)

        if smooth[idx] < threshold:
            break

        if any(
            min(
                abs(idx - p),
                N_HUE_BINS
                - abs(idx - p)
            )
            < PEAK_MIN_SEPARATION_BINS
            for p in peaks
        ):
            continue

        peaks.append(idx)

        if (
            len(peaks)
            >= MAX_HUE_FAMILIES
        ):
            break

    return peaks


# ============================================================
# COR-NÚCLEO
# ============================================================

def representative_core_rgb(
    rgb_pixels,
    saturation_pixels
):
    if len(rgb_pixels) == 0:
        return None, 0, 0.0

    threshold = float(
        np.quantile(
            saturation_pixels,
            CORE_SAT_QUANTILE
        )
    )

    keep = (
        saturation_pixels
        >= threshold
    )

    core = rgb_pixels[keep]

    if len(core) == 0:
        core = rgb_pixels

    packed = (
        (
            core[:, 0].astype(np.uint32)
            << 16
        )
        | (
            core[:, 1].astype(np.uint32)
            << 8
        )
        | core[:, 2].astype(np.uint32)
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

    rgb = (
        (best >> 16) & 255,
        (best >> 8) & 255,
        best & 255
    )

    support_exact = (
        int(counts[best_idx])
        / len(core)
        if len(core)
        else 0.0
    )

    return (
        rgb,
        int(len(core)),
        float(support_exact)
    )


# ============================================================
# SUPORTE X
# ============================================================

def spatial_support(mask):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    h, w = mask.shape

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    x_span_frac = (
        (x_max - x_min + 1)
        / w
    )

    # Bins globais em X.
    edges = np.linspace(
        0,
        w,
        N_X_BINS + 1
    )

    bin_ids = np.clip(
        np.digitize(
            xs,
            edges
        ) - 1,
        0,
        N_X_BINS - 1
    )

    counts = np.bincount(
        bin_ids,
        minlength=N_X_BINS
    )

    occupied = (
        counts
        >= MIN_PIXELS_PER_X_BIN
    )

    n_occupied = int(
        occupied.sum()
    )

    coverage = (
        n_occupied
        / N_X_BINS
    )

    # Maior sequência de bins ocupados.
    longest_run = 0
    current = 0

    for value in occupied:
        if value:
            current += 1
            longest_run = max(
                longest_run,
                current
            )
        else:
            current = 0

    longest_run_frac = (
        longest_run
        / N_X_BINS
    )

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "x_span_frac": float(
            x_span_frac
        ),
        "n_occupied_x_bins": (
            n_occupied
        ),
        "x_bin_coverage": float(
            coverage
        ),
        "longest_x_bin_run": (
            longest_run
        ),
        "longest_x_bin_run_frac": (
            float(longest_run_frac)
        ),
    }



# ============================================================
# V4 - EVIDÊNCIA DE MISTURA / SOBREPOSIÇÃO
# ============================================================

def representative_saturation(rgb):
    arr = np.array(
        [[[
            rgb[0] / 255.0,
            rgb[1] / 255.0,
            rgb[2] / 255.0
        ]]],
        dtype=float
    )
    return float(
        rgb2hsv(arr)[0, 0, 1]
    )


def convex_mixture_rgb_residual(
    target_rgb,
    base_a_rgb,
    base_b_rgb
):
    """
    Menor resíduo RGB entre a cor alvo e uma combinação convexa:

        wA*A + wB*B + wW*branco

    com wA,wB,wW >= 0 e soma = 1.

    Usa grade de 0.01. É uma verificação diagnóstica de mistura
    cromática; não usa a paleta F.
    """
    target = np.array(
        target_rgb,
        dtype=float
    )
    a = np.array(
        base_a_rgb,
        dtype=float
    )
    b = np.array(
        base_b_rgb,
        dtype=float
    )
    white = np.array(
        [255.0, 255.0, 255.0],
        dtype=float
    )

    best_residual = float("inf")
    best_weights = (
        0.0,
        0.0,
        1.0
    )

    # 5151 combinações no máximo; o número de famílias por imagem
    # é pequeno, portanto o custo é baixo.
    for ia in range(101):
        wa = ia / 100.0

        for ib in range(
            101 - ia
        ):
            wb = ib / 100.0
            ww = 1.0 - wa - wb

            pred = (
                wa * a
                + wb * b
                + ww * white
            )

            residual = float(
                np.linalg.norm(
                    target - pred
                )
            )

            if residual < best_residual:
                best_residual = residual
                best_weights = (
                    wa,
                    wb,
                    ww
                )

    return (
        best_residual,
        best_weights
    )


def xbin_overlap_evidence(
    target_mask,
    base_a_mask,
    base_b_mask
):
    """
    Mede, por bins em X, se a família candidata aparece onde as duas
    famílias-base também estão presentes e geometricamente próximas.
    """
    h, w = target_mask.shape

    target_x = np.where(
        target_mask
    )[1]

    if target_x.size == 0:
        return 0.0

    x_min = int(
        target_x.min()
    )
    x_max = int(
        target_x.max()
    )

    if x_max <= x_min:
        return 0.0

    edges = np.linspace(
        x_min,
        x_max + 1,
        N_X_BINS + 1
    )

    occupied = 0
    cooccur = 0

    for i in range(N_X_BINS):
        xa = int(
            math.floor(edges[i])
        )
        xb = int(
            math.ceil(edges[i + 1])
        )

        xa = max(
            0,
            min(w - 1, xa)
        )
        xb = max(
            xa + 1,
            min(w, xb)
        )

        t_slice = target_mask[
            :, xa:xb
        ]

        if int(
            t_slice.sum()
        ) < MIN_PIXELS_PER_X_BIN:
            continue

        occupied += 1

        a_ys = np.where(
            base_a_mask[
                :, xa:xb
            ]
        )[0]

        b_ys = np.where(
            base_b_mask[
                :, xa:xb
            ]
        )[0]

        t_ys = np.where(
            t_slice
        )[0]

        if (
            a_ys.size == 0
            or b_ys.size == 0
            or t_ys.size == 0
        ):
            continue

        ya = float(
            np.median(a_ys)
        )
        yb = float(
            np.median(b_ys)
        )
        yt = float(
            np.median(t_ys)
        )

        base_gap = abs(
            ya - yb
        )

        # A candidata também deve estar na vizinhança vertical
        # definida pelas bases.
        target_near = (
            min(ya, yb)
            - OVERLAP_MAX_BASE_Y_GAP_PX
            <= yt
            <= max(ya, yb)
            + OVERLAP_MAX_BASE_Y_GAP_PX
        )

        if (
            base_gap
            <= OVERLAP_MAX_BASE_Y_GAP_PX
            and target_near
        ):
            cooccur += 1

    if occupied == 0:
        return 0.0

    return float(
        cooccur / occupied
    )


def spatial_overlap_evidence(
    target_mask,
    base_a_mask,
    base_b_mask
):
    """
    Fração dos pixels da família candidata que estão simultaneamente
    próximos das duas famílias-base.
    """
    r = int(
        OVERLAP_DILATION_RADIUS_PX
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * r + 1, 2 * r + 1)
    )

    da = cv2.dilate(
        base_a_mask.astype(
            np.uint8
        ),
        kernel
    ).astype(bool)

    db = cv2.dilate(
        base_b_mask.astype(
            np.uint8
        ),
        kernel
    ).astype(bool)

    n_target = int(
        target_mask.sum()
    )

    if n_target == 0:
        return 0.0

    near_both = (
        target_mask
        & da
        & db
    )

    return float(
        near_both.sum()
        / n_target
    )


def annotate_overlap_derived(
    families
):
    """
    Classifica famílias como:

        BASE
        OVERLAP_DERIVED

    A decisão é independente da paleta F e da quantidade esperada
    de linhas. Uma candidata só é marcada como derivada se houver:

    1. duas outras famílias cromaticamente estáveis;
    2. cor candidata compatível com mistura A+B+branco;
    3. evidência espacial de quase sobreposição.

    O objetivo é tratar casos como linhas azul/laranja tracejadas que
    ficam quase sobrepostas e geram cores intermediárias persistentes.
    """
    for fam in families:
        fam["family_role"] = "BASE"
        fam["overlap_near_both_frac"] = 0.0
        fam["overlap_xbin_cooccurrence"] = 0.0
        fam["overlap_mixture_rgb_residual"] = ""
        fam["overlap_base_a_hex"] = ""
        fam["overlap_base_b_hex"] = ""
        fam["overlap_mix_weight_a"] = ""
        fam["overlap_mix_weight_b"] = ""
        fam["overlap_mix_weight_white"] = ""

    if len(families) < 3:
        return families

    for target_idx, target in enumerate(
        families
    ):
        best = None

        others = [
            i
            for i in range(
                len(families)
            )
            if i != target_idx
        ]

        for p in range(
            len(others)
        ):
            for q in range(
                p + 1,
                len(others)
            ):
                ia = others[p]
                ib = others[q]

                a = families[ia]
                b = families[ib]

                if (
                    a["core_support_exact"]
                    < OVERLAP_BASE_MIN_CORE_SUPPORT
                    or
                    b["core_support_exact"]
                    < OVERLAP_BASE_MIN_CORE_SUPPORT
                ):
                    continue

                # As bases não podem ser quase a mesma cor.
                base_sep = float(
                    np.linalg.norm(
                        np.array(
                            a["observed_rgb"],
                            dtype=float
                        )
                        - np.array(
                            b["observed_rgb"],
                            dtype=float
                        )
                    )
                )

                if (
                    base_sep
                    < OVERLAP_MIN_BASE_RGB_SEPARATION
                ):
                    continue

                # Evita que famílias minúsculas expliquem uma candidata
                # muito mais extensa.
                if (
                    a["family_pixels"]
                    < target["family_pixels"]
                    * OVERLAP_MIN_BASE_PIXELS_RELATIVE
                    or
                    b["family_pixels"]
                    < target["family_pixels"]
                    * OVERLAP_MIN_BASE_PIXELS_RELATIVE
                ):
                    continue

                (
                    mix_residual,
                    mix_weights
                ) = convex_mixture_rgb_residual(
                    target["observed_rgb"],
                    a["observed_rgb"],
                    b["observed_rgb"]
                )

                if (
                    mix_residual
                    > OVERLAP_MAX_MIXTURE_RGB_RESIDUAL
                ):
                    continue

                near_both = (
                    spatial_overlap_evidence(
                        target["_mask"],
                        a["_mask"],
                        b["_mask"]
                    )
                )

                xbin_co = (
                    xbin_overlap_evidence(
                        target["_mask"],
                        a["_mask"],
                        b["_mask"]
                    )
                )

                spatial_ok = (
                    near_both
                    >= OVERLAP_MIN_NEAR_BOTH_FRAC
                    or
                    xbin_co
                    >= OVERLAP_MIN_XBIN_COOCCURRENCE
                )

                if not spatial_ok:
                    continue

                # Escore: prioriza mistura muito bem explicada e
                # forte evidência espacial.
                score = (
                    (1.0 - min(
                        1.0,
                        mix_residual
                        / OVERLAP_MAX_MIXTURE_RGB_RESIDUAL
                    ))
                    + near_both
                    + xbin_co
                )

                candidate = {
                    "score": score,
                    "mix_residual": (
                        mix_residual
                    ),
                    "mix_weights": (
                        mix_weights
                    ),
                    "near_both": (
                        near_both
                    ),
                    "xbin_co": (
                        xbin_co
                    ),
                    "a": a,
                    "b": b,
                }

                if (
                    best is None
                    or candidate["score"]
                    > best["score"]
                ):
                    best = candidate

        if best is not None:
            target[
                "family_role"
            ] = "OVERLAP_DERIVED"

            target[
                "overlap_near_both_frac"
            ] = best[
                "near_both"
            ]

            target[
                "overlap_xbin_cooccurrence"
            ] = best[
                "xbin_co"
            ]

            target[
                "overlap_mixture_rgb_residual"
            ] = best[
                "mix_residual"
            ]

            target[
                "overlap_base_a_hex"
            ] = best[
                "a"
            ]["observed_hex"]

            target[
                "overlap_base_b_hex"
            ] = best[
                "b"
            ]["observed_hex"]

            (
                wa,
                wb,
                ww
            ) = best[
                "mix_weights"
            ]

            target[
                "overlap_mix_weight_a"
            ] = wa

            target[
                "overlap_mix_weight_b"
            ] = wb

            target[
                "overlap_mix_weight_white"
            ] = ww

    return families


# ============================================================
# EXTRAÇÃO DAS FAMÍLIAS
# ============================================================

def extract_line_color_families(img, include_derived=False):
    arr_u8 = np.asarray(
        img.convert("RGB"),
        dtype=np.uint8
    )

    arr = (
        arr_u8.astype(np.float64)
        / 255.0
    )

    hsv = rgb2hsv(arr)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    candidate = (
        (sat >= MIN_SATURATION)
        & (val <= MAX_VALUE)
    )

    if not np.any(candidate):
        return []

    hue_bins = np.floor(
        hue * N_HUE_BINS
    ).astype(int)

    hue_bins = np.clip(
        hue_bins,
        0,
        N_HUE_BINS - 1
    )

    candidate_bins = (
        hue_bins[candidate]
    )

    candidate_sat = (
        sat[candidate]
    )

    # Alta saturação pesa mais, reduzindo antialiasing claro.
    weights = (
        candidate_sat ** 2
    )

    hist = np.bincount(
        candidate_bins,
        weights=weights,
        minlength=N_HUE_BINS
    ).astype(float)

    peaks = select_hue_peaks(
        hist
    )

    families = []

    for peak in peaks:
        dist = circular_bin_distance(
            hue_bins,
            peak
        )

        family_mask = (
            candidate
            & (
                dist
                <= HUE_WINDOW_BINS
            )
        )

        n_pixels = int(
            family_mask.sum()
        )

        if (
            n_pixels
            < MIN_FAMILY_PIXELS
        ):
            continue

        support = spatial_support(
            family_mask
        )

        if support is None:
            continue

        # Regra espacial principal.
        spatial_ok = (
            support["x_span_frac"]
            >= MIN_X_SPAN_FRAC
            and
            support[
                "n_occupied_x_bins"
            ]
            >= MIN_OCCUPIED_X_BINS
            and
            support[
                "x_bin_coverage"
            ]
            >= MIN_X_BIN_COVERAGE
        )

        if not spatial_ok:
            continue

        rgb_pixels = (
            arr_u8[family_mask]
        )

        sat_pixels = (
            sat[family_mask]
        )

        (
            rep_rgb,
            core_pixels,
            support_exact
        ) = representative_core_rgb(
            rgb_pixels,
            sat_pixels
        )

        if rep_rgb is None:
            continue

        # V3: descarta famílias cromáticas difusas, típicas de
        # antialiasing, mistura em cruzamentos ou cores derivadas.
        # Famílias-base devem apresentar uma cor-núcleo dominante.
        if support_exact < MIN_CORE_SUPPORT_EXACT:
            continue

        comparison = nearest_f(
            rep_rgb
        )

        hue_deg = (
            peak
            / N_HUE_BINS
            * 360.0
        )

        families.append({
            "_mask": family_mask.copy(),
            "peak_bin": int(peak),
            "peak_hue_deg": float(
                hue_deg
            ),
            "family_pixels": (
                n_pixels
            ),
            "core_pixels": (
                core_pixels
            ),
            "core_support_exact": (
                support_exact
            ),
            "observed_rgb": (
                rep_rgb
            ),
            "observed_hex": (
                rgb_to_hex(rep_rgb)
            ),
            **support,
            **comparison,
        })

    # --------------------------------------------------------
    # DEDUPLICAÇÃO
    # --------------------------------------------------------
    families.sort(
        key=lambda f: (
            f["x_bin_coverage"],
            f["family_pixels"]
        ),
        reverse=True
    )

    kept = []

    for fam in families:
        duplicate = False

        for old in kept:
            rgb_d = float(
                np.linalg.norm(
                    np.array(
                        fam["observed_rgb"],
                        dtype=float
                    )
                    - np.array(
                        old["observed_rgb"],
                        dtype=float
                    )
                )
            )

            hue_d = (
                circular_deg_distance(
                    fam["peak_hue_deg"],
                    old["peak_hue_deg"]
                )
            )

            if (
                rgb_d
                <= DUPLICATE_RGB_DISTANCE
                and
                hue_d
                <= DUPLICATE_HUE_DISTANCE_DEG
            ):
                duplicate = True
                break

            if (
                fam["observed_hex"]
                == old["observed_hex"]
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(fam)

    # Ordena por hue apenas para apresentação estável.
    kept.sort(
        key=lambda f:
        f["peak_hue_deg"]
    )

    kept = annotate_overlap_derived(
        kept
    )

    if include_derived:
        return kept

    # API usada pelo programa unificado: somente famílias BASE
    # entram como cores observadas.
    return [
        fam
        for fam in kept
        if fam["family_role"] == "BASE"
    ]


# ============================================================
# SELEÇÃO DAS 40 IMAGENS
# ============================================================

def select_validation_images():
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

        profile = (
            m.group(1).upper()
        )

        try:
            with Image.open(path) as img:
                canvas = (
                    "OK"
                    if img.size
                    == (1200, 800)
                    else "NAO_OK"
                )
        except Exception:
            continue

        groups[
            (profile, canvas)
        ].append(path)

    selected = []

    for profile in (
        "LI",
        "LC"
    ):
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

            for path in rng.sample(
                pool,
                k
            ):
                selected.append((
                    profile,
                    canvas,
                    path
                ))

    # Caso-alvo da V4: linhas azul/laranja tracejadas e quase
    # sobrepostas. É incluído explicitamente na CALIBRAÇÃO, não no holdout.
    target_path = (
        ROOT_DIR
        / "LC_018_R05.png"
    )

    if target_path.exists():
        already = {
            p.name
            for _, _, p in selected
        }

        if (
            target_path.name
            not in already
        ):
            try:
                with Image.open(
                    target_path
                ) as img:
                    canvas = (
                        "OK"
                        if img.size
                        == (1200, 800)
                        else "NAO_OK"
                    )

                selected.append((
                    "LC",
                    canvas,
                    target_path
                ))
            except Exception:
                pass

    return selected


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(
    src,
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

    for rank, fam in enumerate(
        families,
        1
    ):
        rgb = tuple(
            fam["observed_rgb"]
        )

        x1 = fam["x_min"]
        x2 = fam["x_max"]
        y1 = fam["y_min"]
        y2 = fam["y_max"]

        # Contorno com a própria cor detectada.
        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline=rgb,
            width=max(
                2,
                img.width // 500
            )
        )

        txt = (
            f"{rank} "
            f"{fam['observed_hex']} "
            f"Xcov={fam['x_bin_coverage']:.2f} "
            f"span={fam['x_span_frac']:.2f} "
            f"dE={fam['deltaE00']:.1f} "
            f"{fam.get('family_role', 'BASE')}"
        )

        draw.text(
            (
                x1,
                max(2, y1 - 13)
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
            f"base={item['n_colors']} "
            f"derived={item.get('n_derived', 0)}"
        )

        draw.text(
            (x0 + 8, y0 + 10),
            title,
            fill=(0, 0, 0),
            font=font
        )

        subtitle = (
            f"Dmax={item['Dmax']:.2f}"
            if item["Dmax"] is not None
            else "Dmax=NA"
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
        "VALIDAÇÃO L-COR V4 "
        "- FAMÍLIA CROMÁTICA + SUPORTE X"
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
    ) in enumerate(
        selected,
        1
    ):
        with Image.open(path) as img:
            all_families = (
                extract_line_color_families(
                    img.convert("RGB"),
                    include_derived=True
                )
            )

        families = [
            f
            for f in all_families
            if f["family_role"] == "BASE"
        ]

        derived_families = [
            f
            for f in all_families
            if f["family_role"]
            == "OVERLAP_DERIVED"
        ]

        dvals = [
            f["deltaE00"]
            for f in families
        ]

        Dmean = (
            float(np.mean(dvals))
            if dvals else None
        )

        Dmax = (
            float(np.max(dvals))
            if dvals else None
        )

        expected_diag = (
            1 if profile == "LI"
            else 2
        )

        count_match = int(
            len(families)
            == expected_diag
        )

        overlay = (
            OVERLAY_DIR
            / f"{path.stem}__v4.png"
        )

        make_overlay(
            path,
            all_families,
            overlay
        )

        image_rows.append({
            "profile": profile,
            "canvas_class": canvas,
            "filename": path.name,

            "expected_color_count_diagnostic": (
                expected_diag
            ),

            "n_line_colors_detected": (
                len(families)
            ),

            "n_overlap_derived_detected": (
                len(derived_families)
            ),

            "count_matches_expected_diagnostic": (
                count_match
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
        })

        for rank, fam in enumerate(
            all_families,
            1
        ):
            color_rows.append({
                "profile": profile,
                "canvas_class": canvas,
                "filename": path.name,
                "color_rank": rank,

                "observed_hex": (
                    fam["observed_hex"]
                ),

                "peak_hue_deg": (
                    round(
                        fam["peak_hue_deg"],
                        4
                    )
                ),

                "family_pixels": (
                    fam["family_pixels"]
                ),

                "core_pixels": (
                    fam["core_pixels"]
                ),

                "core_support_exact": (
                    round(
                        fam[
                            "core_support_exact"
                        ],
                        6
                    )
                ),

                "x_span_frac": (
                    round(
                        fam["x_span_frac"],
                        6
                    )
                ),

                "n_occupied_x_bins": (
                    fam[
                        "n_occupied_x_bins"
                    ]
                ),

                "x_bin_coverage": (
                    round(
                        fam[
                            "x_bin_coverage"
                        ],
                        6
                    )
                ),

                "longest_x_bin_run_frac": (
                    round(
                        fam[
                            "longest_x_bin_run_frac"
                        ],
                        6
                    )
                ),

                "family_role": (
                    fam.get(
                        "family_role",
                        "BASE"
                    )
                ),

                "overlap_near_both_frac": (
                    round(
                        float(
                            fam.get(
                                "overlap_near_both_frac",
                                0.0
                            )
                        ),
                        6
                    )
                ),

                "overlap_xbin_cooccurrence": (
                    round(
                        float(
                            fam.get(
                                "overlap_xbin_cooccurrence",
                                0.0
                            )
                        ),
                        6
                    )
                ),

                "overlap_mixture_rgb_residual": (
                    round(
                        float(
                            fam[
                                "overlap_mixture_rgb_residual"
                            ]
                        ),
                        6
                    )
                    if fam.get(
                        "overlap_mixture_rgb_residual",
                        ""
                    ) != ""
                    else ""
                ),

                "overlap_base_a_hex": (
                    fam.get(
                        "overlap_base_a_hex",
                        ""
                    )
                ),

                "overlap_base_b_hex": (
                    fam.get(
                        "overlap_base_b_hex",
                        ""
                    )
                ),

                "overlap_mix_weight_a": (
                    round(
                        float(
                            fam[
                                "overlap_mix_weight_a"
                            ]
                        ),
                        4
                    )
                    if fam.get(
                        "overlap_mix_weight_a",
                        ""
                    ) != ""
                    else ""
                ),

                "overlap_mix_weight_b": (
                    round(
                        float(
                            fam[
                                "overlap_mix_weight_b"
                            ]
                        ),
                        4
                    )
                    if fam.get(
                        "overlap_mix_weight_b",
                        ""
                    ) != ""
                    else ""
                ),

                "overlap_mix_weight_white": (
                    round(
                        float(
                            fam[
                                "overlap_mix_weight_white"
                            ]
                        ),
                        4
                    )
                    if fam.get(
                        "overlap_mix_weight_white",
                        ""
                    ) != ""
                    else ""
                ),

                "nearest_F_name": (
                    fam[
                        "nearest_F_name"
                    ]
                ),

                "nearest_F_hex": (
                    fam[
                        "nearest_F_hex"
                    ]
                ),

                "deltaE00_to_nearest_F": (
                    round(
                        fam["deltaE00"],
                        6
                    )
                ),
            })

        sheet_items.append({
            "profile": profile,
            "canvas": canvas,
            "filename": path.name,
            "overlay": overlay,
            "n_colors": len(families),
            "n_derived": len(
                derived_families
            ),
            "Dmax": Dmax,
        })

        colors_text = ",".join(
            f["observed_hex"]
            for f in families
        )

        print(
            f"{idx:02d}/{len(selected):02d} | "
            f"{path.name} | "
            f"base={len(families)} "
            f"derived={len(derived_families)} "
            f"[{colors_text}] | "
            f"Dmax="
            f"{Dmax if Dmax is not None else 'NA'}"
        )

    # --------------------------------------------------------
    # CSV IMAGEM
    # --------------------------------------------------------
    image_csv = (
        OUTPUT_DIR
        / "line_validation_v4.csv"
    )

    image_fields = [
        "profile",
        "canvas_class",
        "filename",
        "expected_color_count_diagnostic",
        "n_line_colors_detected",
        "n_overlap_derived_detected",
        "count_matches_expected_diagnostic",
        "Dmean_distinct_colors",
        "Dmax_distinct_colors",
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
    # CSV CORES
    # --------------------------------------------------------
    color_csv = (
        OUTPUT_DIR
        / "line_color_validation_v4.csv"
    )

    color_fields = [
        "profile",
        "canvas_class",
        "filename",
        "color_rank",
        "observed_hex",
        "peak_hue_deg",
        "family_pixels",
        "core_pixels",
        "core_support_exact",
        "x_span_frac",
        "n_occupied_x_bins",
        "x_bin_coverage",
        "longest_x_bin_run_frac",
        "family_role",
        "overlap_near_both_frac",
        "overlap_xbin_cooccurrence",
        "overlap_mixture_rgb_residual",
        "overlap_base_a_hex",
        "overlap_base_b_hex",
        "overlap_mix_weight_a",
        "overlap_mix_weight_b",
        "overlap_mix_weight_white",
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
        / "line_validation_contact_sheet_v4.png"
    )

    make_contact_sheet(
        sheet_items,
        contact
    )

    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------
    matches = sum(
        r[
            "count_matches_expected_diagnostic"
        ]
        for r in image_rows
    )

    li_rows = [
        r for r in image_rows
        if r["profile"] == "LI"
    ]

    lc_rows = [
        r for r in image_rows
        if r["profile"] == "LC"
    ]

    li_matches = sum(
        r[
            "count_matches_expected_diagnostic"
        ]
        for r in li_rows
    )

    lc_matches = sum(
        r[
            "count_matches_expected_diagnostic"
        ]
        for r in lc_rows
    )

    no_color = sum(
        r["n_line_colors_detected"] == 0
        for r in image_rows
    )

    n_derived = sum(
        r["n_overlap_derived_detected"]
        for r in image_rows
    )

    summary = [
        "VALIDAÇÃO L-COR V4",
        "FAMÍLIA CROMÁTICA + SUPORTE X",
        "=" * 72,

        f"Imagens analisadas: "
        f"{len(image_rows)}",

        f"Imagens sem cor detectada: "
        f"{no_color}",

        f"Famílias marcadas OVERLAP_DERIVED: "
        f"{n_derived}",

        f"Contagem compatível "
        f"(diagnóstico): "
        f"{matches}/{len(image_rows)}",

        f"LI compatível: "
        f"{li_matches}/{len(li_rows)}",

        f"LC compatível: "
        f"{lc_matches}/{len(lc_rows)}",

        "",

        "MUDANÇA EM RELAÇÃO À V3:",
        "- mantém família cromática global + suporte espacial em X;",
        f"- mantém core_support_exact >= {MIN_CORE_SUPPORT_EXACT:.2f};",
        "- acrescenta classificação pós-extração BASE / OVERLAP_DERIVED;",
        "- OVERLAP_DERIVED exige mistura cromática A+B+branco compatível;",
        "- exige também evidência espacial de quase sobreposição;",
        "- a quantidade LI=1 / LC=2 NÃO participa dessa classificação;",
        "- F continua fora da etapa de descoberta/classificação da sobreposição;",
        "- LC_018_R05 é incluída explicitamente como caso-alvo de calibração.",

        "",

        "Métrica:",
        "- Dmean sobre cores-base distintas detectadas;",
        "- Dmax sobre cores-base distintas detectadas;",
        "- contagem LI=1 / LC=2 é somente diagnóstico.",

        "",

        "Nenhum limiar de conformidade foi aplicado.",

        "",

        f"Prancha: {contact}",
        f"CSV imagem: {image_csv}",
        f"CSV cores: {color_csv}",
    ]

    summary_path = (
        OUTPUT_DIR
        / "line_validation_summary_v4.txt"
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
