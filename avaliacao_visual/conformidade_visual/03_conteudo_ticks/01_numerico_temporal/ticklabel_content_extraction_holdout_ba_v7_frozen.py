# -*- coding: utf-8 -*-
r"""
HOLDOUT INDEPENDENTE B-A — V7 CONGELADA
========================================

Objetivo
--------
Validar de forma independente o extrator de conteúdo dos tick labels
B-A V7, SEM reajustar parâmetros e SEM comparar ainda com a especificação F.

Protocolo
---------
- importa e executa EXATAMENTE a B-A V7 congelada;
- verifica SHA-256 da V7 antes de processar qualquer imagem;
- verifica também a V4 congelada de presença usada pela V7;
- exclui TODAS as unidades presentes na calibração B-A V7;
- usa 20 novas unidades por perfil:
      5 unidades × 4 condições × 6 perfis = 120 imagens;
- seleciona uma repetição canônica por unidade;
- usa semente fixa, definida no código;
- não há sobreposição de unidade ou imagem com a calibração;
- processa somente arquivos canônicos da RAIZ, sem recursão.

Critério primário do holdout
----------------------------
O holdout valida a EXTRAÇÃO DO CONTEÚDO DOS TICK LABELS VISÍVEIS.

SCALE_* / offset científico:
- continua sendo extraído e salvo como diagnóstico;
- NÃO entra no critério primário de correção dos tick labels;
- NÃO transforma um eixo correto em erro de extração.

IMPORTANTE
----------
Os estados automáticos STRONG/PARTIAL/MULTI etc. NÃO são acurácia.
A acurácia/cobertura independente só será estabelecida após auditoria visual
dos eixos do holdout.

Este programa NÃO recalibra a V7.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import ticklabel_content_extraction_calibration_ba_v7 as ba_v7


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
OUT_DIRNAME = "_ticklabel_content_extraction_holdout_ba_v7_frozen"

EXPECTED_V7_SHA256 = "21423b59ce40eb351693f00b6a197936d936fe53131467f47c1031da864fd688"
EXPECTED_PRESENCE_V4_SHA256 = "d15c325290c7596632f9f4c62d907e7a78d0a794862b690c4c4691ed87ea86b2"

PROFILES = ("BI","BC","LI","LC","SI","SC")
CONDITIONS = ("00","01","10","11")

# Protocolo FIXO do holdout.
UNITS_PER_PROFILE_CONDITION = 5
HOLDOUT_IMAGES_EXPECTED = 120
HOLDOUT_UNITS_EXPECTED = 120
SEED = 20260901


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    p.add_argument("--output-dir",type=Path,default=None)
    p.add_argument("--crosswalk",type=Path,default=None)
    p.add_argument("--calibration-manifest",type=Path,default=None)
    p.add_argument("--tesseract",type=Path,default=None)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_v7():
    path=Path(ba_v7.__file__).resolve()
    actual=sha256(path)

    if actual!=EXPECTED_V7_SHA256:
        raise RuntimeError(
            "A B-A V7 não corresponde à versão congelada.\n"
            f"Arquivo: {path}\n"
            f"SHA esperado: {EXPECTED_V7_SHA256}\n"
            f"SHA encontrado: {actual}\n"
            "Não execute o holdout com uma V7 modificada."
        )

    presence_path=Path(ba_v7.v4.__file__).resolve()
    presence_actual=sha256(presence_path)

    if presence_actual!=EXPECTED_PRESENCE_V4_SHA256:
        raise RuntimeError(
            "A V4 de presença usada pela V7 não corresponde à versão congelada.\n"
            f"Arquivo: {presence_path}\n"
            f"SHA esperado: {EXPECTED_PRESENCE_V4_SHA256}\n"
            f"SHA encontrado: {presence_actual}"
        )

    # Faz também a verificação interna da própria V7.
    ba_v7.check_v4_hash()

    return path,actual,presence_path,presence_actual


def find_calibration_manifest(root: Path, explicit=None):
    if explicit is not None:
        p=explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Manifesto de calibração não encontrado: {p}"
            )
        return p

    p=(
        root/
        "_ticklabel_content_extraction_ba_v7"/
        "ticklabel_ba_v7_manifest.csv"
    )

    if not p.exists():
        raise FileNotFoundError(
            "Manifesto da calibração B-A V7 não encontrado.\n"
            f"Esperado: {p}\n"
            "Use --calibration-manifest se ele estiver em outro local."
        )

    return p.resolve()


def load_calibration_units(cal_path: Path, inv: pd.DataFrame):
    cal=pd.read_csv(cal_path,dtype=str)

    if "filename" not in cal.columns:
        raise RuntimeError(
            "Manifesto da B-A V7 não contém a coluna filename."
        )

    cal_names=cal[["filename"]].drop_duplicates().copy()

    inv2=inv.copy()
    inv2["filename"]=inv2["filename"].astype(str)

    cal2=cal_names.merge(
        inv2[
            ["filename","profile","unit_id","unit_number","repeat"]
        ],
        on="filename",
        how="left",
        validate="one_to_one"
    )

    if cal2["unit_id"].isna().any():
        miss=cal2.loc[
            cal2["unit_id"].isna(),"filename"
        ].tolist()
        raise RuntimeError(
            "Há imagens da calibração que não existem no inventário atual: "
            + ", ".join(miss[:20])
        )

    if len(cal2)!=72:
        raise RuntimeError(
            f"Manifesto da calibração contém {len(cal2)} imagens; esperado=72."
        )

    if cal2["unit_id"].nunique()!=72:
        raise RuntimeError(
            "A calibração B-A V7 deveria conter uma imagem por unidade. "
            f"Unidades únicas encontradas: {cal2['unit_id'].nunique()}."
        )

    return cal2


def select_holdout(
    inv: pd.DataFrame,
    crosswalk_path: Path,
    calibration: pd.DataFrame
):
    if crosswalk_path is None:
        raise RuntimeError(
            "O holdout balanceado requer o crosswalk experimental V3."
        )

    cw=ba_v7.load_unit_crosswalk(crosswalk_path)

    units=(
        inv[["profile","unit_id"]]
        .drop_duplicates()
        .merge(cw,on="unit_id",how="left",validate="one_to_one")
    )

    if units["condition_code"].isna().any():
        n=int(units["condition_code"].isna().sum())
        raise RuntimeError(
            f"{n} unidades canônicas ficaram sem condition_code no crosswalk."
        )

    calibration_units=set(
        calibration["unit_id"].astype(str)
    )

    eligible=units[
        ~units["unit_id"].astype(str).isin(calibration_units)
    ].copy()

    rng=random.Random(SEED)
    chosen_units=[]

    for profile in PROFILES:
        for cond in CONDITIONS:
            g=eligible[
                (eligible["profile"]==profile)
                & (eligible["condition_code"]==cond)
            ].copy()

            unit_ids=sorted(
                g["unit_id"].dropna().astype(str).unique().tolist()
            )

            if len(unit_ids)<UNITS_PER_PROFILE_CONDITION:
                raise RuntimeError(
                    f"Holdout insuficiente para {profile}/{cond}: "
                    f"{len(unit_ids)} unidades elegíveis; "
                    f"necessárias={UNITS_PER_PROFILE_CONDITION}."
                )

            picked=rng.sample(
                unit_ids,
                UNITS_PER_PROFILE_CONDITION
            )

            for unit_id in picked:
                row=g[g["unit_id"]==unit_id].iloc[0].to_dict()
                chosen_units.append(row)

    chosen=pd.DataFrame(chosen_units)

    if len(chosen)!=HOLDOUT_UNITS_EXPECTED:
        raise RuntimeError(
            f"Unidades selecionadas={len(chosen)}; "
            f"esperado={HOLDOUT_UNITS_EXPECTED}."
        )

    if chosen["unit_id"].nunique()!=HOLDOUT_UNITS_EXPECTED:
        raise RuntimeError("Há unidade duplicada na seleção do holdout.")

    rows=[]

    for _,u in chosen.iterrows():
        imgs=inv[
            inv["unit_id"].astype(str)==str(u["unit_id"])
        ].copy()

        if imgs.empty:
            raise RuntimeError(
                f"Nenhuma repetição encontrada para {u['unit_id']}."
            )

        # Todas as repetições canônicas disponíveis têm a mesma chance.
        idx=rng.randrange(len(imgs))
        r=imgs.sort_values("repeat").iloc[idx].to_dict()

        r["condition_code"]=str(u["condition_code"])
        if "condition_label" in chosen.columns:
            r["condition_label"]=u.get("condition_label","")
        if "participant_id" in chosen.columns:
            r["participant_id"]=u.get("participant_id","")

        rows.append(r)

    hold=pd.DataFrame(rows)

    hold=hold.sort_values(
        ["profile","condition_code","unit_id","repeat"]
    ).reset_index(drop=True)

    return hold


def validate_independence(
    hold: pd.DataFrame,
    calibration: pd.DataFrame
):
    cal_units=set(calibration["unit_id"].astype(str))
    hold_units=set(hold["unit_id"].astype(str))

    cal_images=set(calibration["filename"].astype(str))
    hold_images=set(hold["filename"].astype(str))

    overlap_units=sorted(cal_units & hold_units)
    overlap_images=sorted(cal_images & hold_images)

    if overlap_units:
        raise RuntimeError(
            "Sobreposição de unidades entre calibração e holdout: "
            + ", ".join(overlap_units[:20])
        )

    if overlap_images:
        raise RuntimeError(
            "Sobreposição de imagens entre calibração e holdout: "
            + ", ".join(overlap_images[:20])
        )

    if len(hold)!=HOLDOUT_IMAGES_EXPECTED:
        raise RuntimeError(
            f"Holdout contém {len(hold)} imagens; "
            f"esperado={HOLDOUT_IMAGES_EXPECTED}."
        )

    if hold["unit_id"].nunique()!=HOLDOUT_UNITS_EXPECTED:
        raise RuntimeError(
            f"Holdout contém {hold['unit_id'].nunique()} unidades; "
            f"esperado={HOLDOUT_UNITS_EXPECTED}."
        )

    balance=(
        hold.groupby(["profile","condition_code"])
        .size()
        .rename("n")
        .reset_index()
    )

    if not (balance["n"]==UNITS_PER_PROFILE_CONDITION).all():
        raise RuntimeError(
            "O holdout não ficou balanceado em perfil × condição."
        )

    return overlap_units,overlap_images,balance


def process_holdout(
    hold: pd.DataFrame,
    root: Path,
    outdir: Path,
    lang: str
):
    crop_dir=outdir/"crops"
    crop_dir.mkdir(parents=True,exist_ok=True)

    image_rows=[]
    axis_rows=[]
    label_rows=[]
    scale_vote_rows=[]
    numeric_candidate_rows=[]
    errors=[]

    total=len(hold)

    for i,r in hold.iterrows():
        path=root/r["filename"]

        print(
            f"[{i+1:03d}/{total}] {r['filename']} ...",
            end=" ",
            flush=True
        )

        try:
            full_img=Image.open(path).convert("RGB")

            pres=ba_v7.v4.analyze(path)
            axes=ba_v7.v4.axis_v3.analyze(path)

            image_row={
                "filename":r["filename"],
                "profile":r["profile"],
                "unit_id":r["unit_id"],
                "unit_number":r["unit_number"],
                "repeat":r["repeat"],
                "condition_code":r["condition_code"],
            }

            for c in ("condition_label","participant_id"):
                if c in r.index:
                    image_row[c]=r[c]

            for axis in ("X","Y"):
                ar,lrs,svrs,ncrs=ba_v7.axis_extraction(
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

                common={
                    "filename":r["filename"],
                    "profile":r["profile"],
                    "unit_id":r["unit_id"],
                    "condition_code":r["condition_code"],
                }

                for c in ("condition_label","participant_id"):
                    if c in r.index:
                        common[c]=r[c]

                axis_rows.append({
                    **common,
                    **ar,
                })

                for lr in lrs:
                    label_rows.append({
                        **common,
                        **lr,
                    })

                for nc in ncrs:
                    numeric_candidate_rows.append({
                        **common,
                        **nc,
                    })

                for sv in svrs:
                    scale_vote_rows.append({
                        **common,
                        **sv,
                    })

            # Colunas reservadas para adjudicação visual.
            image_row["manual_x_ticklabel_result"]=""
            image_row["manual_y_ticklabel_result"]=""
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
                "condition_code":r["condition_code"],
                "error_type":type(exc).__name__,
                "error_message":str(exc),
            })
            print(
                f"ERRO: {type(exc).__name__}: {exc}",
                flush=True
            )

    return (
        pd.DataFrame(image_rows),
        pd.DataFrame(axis_rows),
        pd.DataFrame(label_rows),
        pd.DataFrame(numeric_candidate_rows),
        pd.DataFrame(scale_vote_rows),
        pd.DataFrame(errors),
    )


def make_audit_all_axes(axes_df: pd.DataFrame):
    if axes_df.empty:
        return pd.DataFrame()

    preferred=[
        "filename","profile","unit_id","condition_code",
        "condition_label","participant_id",
        "axis","kind",
        "presence_candidate","segmentation_ok_v4",
        "n_tick_boxes_after_layer_clip",
        "n_usable_individual","individual_usable_frac",
        "individual_text_candidate",
        "band_ocr_normalized",
        "temporal_years_spatial",
        "numeric_selection_status",
        "numeric_selected_values_spatial",
        "numeric_selection_sources",
        "numeric_selected_candidate_sources",
        "extraction_mode_candidate",
        "extraction_support_candidate",
        # SCALE é mantido apenas como diagnóstico.
        "scale_interpretation_status",
        "scale_text_normalized",
        "scale_factor",
    ]

    cols=[c for c in preferred if c in axes_df.columns]
    audit=axes_df[cols].copy()

    audit["manual_ticklabel_extraction_result"]=""
    audit["manual_visible_ticklabels"]=""
    audit["manual_missing_labels"]=""
    audit["manual_false_labels"]=""
    audit["manual_notes"]=""

    return audit


def make_priority_review(axes_df: pd.DataFrame):
    """
    Lista dirigida para inspeção prioritária.

    SCALE_* não participa da prioridade primária.
    """
    if axes_df.empty:
        return pd.DataFrame()

    support=axes_df["extraction_support_candidate"].astype(str)
    select=axes_df["numeric_selection_status"].astype(str)

    mask=(
        support.isin(["PARTIAL_CANDIDATE","FAIL_CANDIDATE"])
        |
        select.isin([
            "SELECTED_BY_COHERENCE",
            "PARTIAL_UNREADABLE",
            "AMBIGUOUS",
            "INSUFFICIENT",
        ])
    )

    review=axes_df[mask].copy()

    review["manual_ticklabel_extraction_result"]=""
    review["manual_visible_ticklabels"]=""
    review["manual_missing_labels"]=""
    review["manual_false_labels"]=""
    review["manual_notes"]=""

    return review


def make_scale_diagnostic(axes_df: pd.DataFrame):
    if axes_df.empty:
        return pd.DataFrame()

    g=axes_df[axes_df["kind"]=="numeric"].copy()

    cols=[
        c for c in [
            "filename","profile","unit_id","condition_code",
            "axis","kind",
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
        if c in g.columns
    ]

    return g[cols].copy()


def summarize_automatic(axes_df: pd.DataFrame):
    rows=[]

    if axes_df.empty:
        return pd.DataFrame(rows)

    for (profile,axis),g in axes_df.groupby(
        ["profile","axis"]
    ):
        support=g[
            "extraction_support_candidate"
        ].value_counts().to_dict()

        selection=g[
            "numeric_selection_status"
        ].value_counts().to_dict()

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

            "select_coherent_direct":int(
                selection.get("COHERENT_DIRECT",0)
            ),
            "select_by_coherence":int(
                selection.get("SELECTED_BY_COHERENCE",0)
            ),
            "select_partial_unreadable":int(
                selection.get("PARTIAL_UNREADABLE",0)
            ),
            "select_ambiguous":int(
                selection.get("AMBIGUOUS",0)
            ),
            "select_insufficient":int(
                selection.get("INSUFFICIENT",0)
            ),
        })

    return pd.DataFrame(rows)


def write_empty_csv(path: Path, columns):
    pd.DataFrame(columns=list(columns)).to_csv(
        path,index=False,encoding="utf-8-sig"
    )


def main():
    args=parse_args()

    root=args.root.expanduser().resolve()

    outdir=(
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root/OUT_DIRNAME
    )
    outdir.mkdir(parents=True,exist_ok=True)

    v7_path,v7_hash,presence_path,presence_hash=verify_frozen_v7()

    t_version,lang,langs=ba_v7.configure_tesseract(
        args.tesseract
    )

    inv=ba_v7.inventory(root)

    if len(inv)!=3120:
        raise RuntimeError(
            f"Inventário canônico na RAIZ={len(inv)}; esperado=3120."
        )

    if inv["unit_id"].nunique()!=312:
        raise RuntimeError(
            f"Unidades canônicas={inv['unit_id'].nunique()}; esperado=312."
        )

    crosswalk_path=ba_v7.find_crosswalk(
        root,args.crosswalk
    )

    if crosswalk_path is None:
        raise FileNotFoundError(
            "Crosswalk experimental V3 não encontrado."
        )

    cal_path=find_calibration_manifest(
        root,args.calibration_manifest
    )

    calibration=load_calibration_units(
        cal_path,inv
    )

    hold=select_holdout(
        inv,
        crosswalk_path,
        calibration
    )

    overlap_units,overlap_images,balance=validate_independence(
        hold,
        calibration
    )

    hold.to_csv(
        outdir/"ticklabel_ba_v7_holdout_manifest.csv",
        index=False,
        encoding="utf-8-sig"
    )

    balance.to_csv(
        outdir/"ticklabel_ba_v7_holdout_balance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("="*80)
    print("HOLDOUT INDEPENDENTE B-A — V7 CONGELADA")
    print("="*80)
    print(f"B-A V7 SHA-256: {v7_hash}")
    print(f"V4 presença SHA-256: {presence_hash}")
    print(f"Manifesto calibração: {cal_path}")
    print(f"Crosswalk V3: {crosswalk_path}")
    print(f"Semente fixa do holdout: {SEED}")
    print(f"Unidades calibração excluídas: {calibration['unit_id'].nunique()}")
    print(f"Unidades holdout: {hold['unit_id'].nunique()}")
    print(f"Imagens holdout: {len(hold)}")
    print("Balanceamento: 5 unidades por perfil × condição")
    print("Sobreposição de unidades: 0")
    print("Sobreposição de imagens: 0")
    print()
    print(
        "IMPORTANTE: SCALE_* é diagnóstico auxiliar e NÃO entra "
        "no critério primário do holdout."
    )
    print(
        "A saída do holdout NÃO deve ser usada para reajustar a V7."
    )
    print()

    (
        images_df,
        axes_df,
        labels_df,
        numeric_candidates_df,
        scale_votes_df,
        err_df,
    )=process_holdout(
        hold,root,outdir,lang
    )

    images_df.to_csv(
        outdir/"ticklabel_ba_v7_holdout_images.csv",
        index=False,encoding="utf-8-sig"
    )

    axes_df.to_csv(
        outdir/"ticklabel_ba_v7_holdout_axes.csv",
        index=False,encoding="utf-8-sig"
    )

    labels_df.to_csv(
        outdir/"ticklabel_ba_v7_holdout_labels.csv",
        index=False,encoding="utf-8-sig"
    )

    numeric_candidates_df.to_csv(
        outdir/"ticklabel_ba_v7_holdout_numeric_candidates.csv",
        index=False,encoding="utf-8-sig"
    )

    scale_votes_df.to_csv(
        outdir/"ticklabel_ba_v7_holdout_scale_votes.csv",
        index=False,encoding="utf-8-sig"
    )

    if err_df.empty:
        write_empty_csv(
            outdir/"ticklabel_ba_v7_holdout_errors.csv",
            [
                "filename","profile","unit_id","condition_code",
                "error_type","error_message"
            ]
        )
    else:
        err_df.to_csv(
            outdir/"ticklabel_ba_v7_holdout_errors.csv",
            index=False,encoding="utf-8-sig"
        )

    audit_all=make_audit_all_axes(
        axes_df
    )
    audit_all.to_csv(
        outdir/"ticklabel_ba_v7_holdout_audit_all_axes.csv",
        index=False,encoding="utf-8-sig"
    )

    priority=make_priority_review(
        axes_df
    )
    priority.to_csv(
        outdir/"ticklabel_ba_v7_holdout_priority_review.csv",
        index=False,encoding="utf-8-sig"
    )

    scale_diag=make_scale_diagnostic(
        axes_df
    )
    scale_diag.to_csv(
        outdir/"ticklabel_ba_v7_holdout_scale_diagnostic.csv",
        index=False,encoding="utf-8-sig"
    )

    automatic=summarize_automatic(
        axes_df
    )
    automatic.to_csv(
        outdir/"ticklabel_ba_v7_holdout_summary_by_profile_axis.csv",
        index=False,encoding="utf-8-sig"
    )

    # Contatos: todos os 20 casos de cada perfil.
    for profile in PROFILES:
        g=images_df[
            images_df["profile"]==profile
        ]
        if len(g):
            ba_v7.make_contact_sheet(
                g,
                root,
                outdir/f"contact_holdout_BA_V7_{profile}.png"
            )

    # Resumo textual.
    lines=[
        "HOLDOUT INDEPENDENTE B-A — V7 CONGELADA",
        "="*78,
        f"B-A V7 SHA-256: {v7_hash}",
        f"Arquivo B-A V7: {v7_path}",
        f"V4 presença SHA-256: {presence_hash}",
        f"Manifesto da calibração: {cal_path}",
        f"Crosswalk V3: {crosswalk_path}",
        f"Tesseract: {t_version}",
        f"Idioma OCR: {lang}",
        f"Semente fixa: {SEED}",
        "",
        "INDEPENDÊNCIA:",
        f"- imagens canônicas na raiz: {len(inv)}",
        f"- unidades canônicas: {inv['unit_id'].nunique()}",
        f"- imagens de calibração: {len(calibration)}",
        f"- unidades de calibração excluídas: {calibration['unit_id'].nunique()}",
        f"- imagens do holdout: {len(hold)}",
        f"- unidades do holdout: {hold['unit_id'].nunique()}",
        f"- sobreposição de unidades: {len(overlap_units)}",
        f"- sobreposição de imagens: {len(overlap_images)}",
        "- balanceamento: 5 unidades por perfil × condição;",
        "- uma repetição canônica por unidade.",
        "",
        "REGRA DE CONGELAMENTO:",
        "- a V7 é executada sem qualquer ajuste;",
        "- o SHA-256 é verificado antes do processamento;",
        "- o holdout NÃO pode ser usado para recalibrar parâmetros;",
        "- erros observados serão registrados como limitações independentes.",
        "",
        "CRITÉRIO PRIMÁRIO:",
        "- validar se os tick labels VISÍVEIS foram recuperados;",
        "- SCALE_* / offset científico é apenas diagnóstico auxiliar;",
        "- SCALE_* não participa do julgamento primário da extração dos ticks;",
        "- os estados automáticos não equivalem a acurácia.",
        "",
        "AUDITORIA VISUAL:",
        "- auditar os 240 eixos do holdout (120 imagens × X/Y);",
        "- categorias sugeridas:",
        "    EXACT = conteúdo visível recuperado integralmente;",
        "    EQUIVALENT = normalização benigna, conteúdo equivalente;",
        "    PARTIAL = parte dos labels recuperada;",
        "    WRONG = conteúdo incorreto/falso label;",
        "    UNEVALUABLE = imagem/eixo não permite adjudicação segura;",
        "- registrar texto visível, omissões e falsos labels quando necessário.",
        "",
        f"Processadas com sucesso: {len(images_df)}/{len(hold)}",
        f"Erros de execução: {len(err_df)}",
        f"Eixos disponíveis para auditoria: {len(axes_df)}",
        f"Eixos na lista prioritária: {len(priority)}",
        "",
        "RESUMO AUTOMÁTICO POR PERFIL × EIXO:",
    ]

    if not automatic.empty:
        for _,r in automatic.iterrows():
            lines.append(
                f"{r['profile']} {r['axis']} ({r['kind']}): "
                f"n={int(r['n'])}; "
                f"MULTI={int(r['multi_evidence'])}; "
                f"STRONG={int(r['strong_candidate'])}; "
                f"SINGLE={int(r['single_evidence'])}; "
                f"PARTIAL={int(r['partial_candidate'])}; "
                f"FAIL={int(r['fail_candidate'])}; "
                f"SEL_DIRECT={int(r['select_coherent_direct'])}; "
                f"SEL_COH={int(r['select_by_coherence'])}; "
                f"SEL_PART={int(r['select_partial_unreadable'])}; "
                f"SEL_AMB={int(r['select_ambiguous'])}; "
                f"SEL_INSUF={int(r['select_insufficient'])}"
            )

    lines.extend([
        "",
        "PRÓXIMA DECISÃO:",
        "- primeiro realizar a auditoria visual independente;",
        "- NÃO alterar a V7 após observar o holdout;",
        "- se o desempenho independente for satisfatório, fechar a Fase B-A;",
        "- em seguida iniciar B-B: comparação do conteúdo observado com F.",
    ])

    summary_path=(
        outdir/"ticklabel_ba_v7_holdout_summary.txt"
    )
    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print()
    print("Concluído.")
    print("Resumo:",summary_path)
    print(
        "Auditoria completa:",
        outdir/"ticklabel_ba_v7_holdout_audit_all_axes.csv"
    )
    print(
        "Revisão prioritária:",
        outdir/"ticklabel_ba_v7_holdout_priority_review.csv"
    )
    print(
        "Diagnóstico de escala:",
        outdir/"ticklabel_ba_v7_holdout_scale_diagnostic.csv"
    )
    print(
        "Envie o resumo, audit_all_axes, priority_review "
        "e as seis pranchas contact_holdout_BA_V7_*.png."
    )


if __name__=="__main__":
    main()
