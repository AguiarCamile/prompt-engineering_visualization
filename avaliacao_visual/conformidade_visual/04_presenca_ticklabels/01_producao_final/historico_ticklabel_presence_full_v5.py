# -*- coding: utf-8 -*-
r"""
ticklabel_presence_full_v5.py

PROCESSAMENTO INTEGRAL V5 — PRESENÇA DOS TICK LABELS X/Y
========================================================

Detector congelado
------------------
ticklabel_presence_calibration_v4.py
SHA-256 esperado:
d15c325290c7596632f9f4c62d907e7a78d0a794862b690c4c4691ed87ea86b2

Validação independente anterior
-------------------------------
Holdout V4 congelado:
- 120 imagens de 120 unidades novas;
- 0 sobreposição de unidades;
- 0 sobreposição de imagens;
- X presente: 120/120;
- Y presente: 119/120;
- 239/240 instâncias positivas corretamente detectadas;
- um falso negativo conhecido em Y: BI_033_R09;
- parâmetros NÃO reajustados após o holdout.

Objetivo desta etapa
--------------------
Aplicar a V4 congelada às 3.120 imagens canônicas da raiz, sem recursão.

O programa registra separadamente:
- presença automática de tick labels X;
- presença automática de tick labels Y;
- presença simultânea X+Y;
- evidências diagnósticas SEG / DISCRETE / CONGESTED;
- qualidade de segmentação, apenas como diagnóstico;
- metadados experimentais do crosswalk V3.

IMPORTANTE
----------
P=0 NÃO é automaticamente "não conformidade".

Todo P=0 entra em uma lista de auditoria visual para distinguir:
1. ausência visual verdadeira dos tick labels;
2. falso negativo do detector.

Somente depois dessa auditoria deve ser calculada a presença final adjudicada.

Escopo
------
- SEM OCR;
- SEM conteúdo dos labels;
- SEM tipografia;
- SEM avaliação das pequenas marcas físicas dos ticks;
- SEM recalibrar limiares.

Saída padrão
------------
C:\Users\Labvis\Downloads\imagens3120\imagens\
_ticklabel_presence_full_v5
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import ticklabel_presence_calibration_v4 as v4


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
OUT_DIRNAME = "_ticklabel_presence_full_v5"

EXPECTED_V4_SHA256 = "d15c325290c7596632f9f4c62d907e7a78d0a794862b690c4c4691ed87ea86b2"

PROFILES = ("BI","BC","LI","LC","SI","SC")

NAME_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--crosswalk", type=Path, default=None)
    return p.parse_args()


def check_v4_hash():
    path = Path(v4.__file__).resolve()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != EXPECTED_V4_SHA256:
        raise RuntimeError(
            "ticklabel_presence_calibration_v4.py foi alterado.\n"
            "O processamento integral exige exatamente a V4 congelada.\n"
            f"SHA esperado: {EXPECTED_V4_SHA256}\n"
            f"SHA encontrado: {actual}"
        )
    return actual


def inventory(root):
    rows=[]
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        m=NAME_RE.match(p.name)
        if not m:
            continue

        profile=m.group(1).upper()
        unit=int(m.group(2))
        repeat=int(m.group(3))

        rows.append({
            "filename":p.name,
            "path":str(p),
            "profile":profile,
            "unit_number":unit,
            "repeat":repeat,
            "unit_id":f"{profile}_{unit:03d}",
            "technique":profile[0],
            "task":profile[1],
        })

    return pd.DataFrame(rows)


def validate_inventory(inv):
    if len(inv)!=3120:
        raise RuntimeError(
            f"Inventário canônico inválido: {len(inv)} imagens; esperado=3120.\n"
            "O programa processa somente arquivos canônicos na RAIZ e não faz recursão."
        )

    if inv["unit_id"].nunique()!=312:
        raise RuntimeError(
            f"Unidades únicas: {inv['unit_id'].nunique()}; esperado=312."
        )

    counts=inv.groupby("unit_id").size()
    bad=counts[counts!=10]
    if len(bad):
        raise RuntimeError(
            "Há unidades sem exatamente 10 repetições. Exemplos:\n"
            + bad.head(20).to_string()
        )

    prof=inv.groupby("profile").size()
    for p in PROFILES:
        if int(prof.get(p,0))!=520:
            raise RuntimeError(
                f"Perfil {p}: {int(prof.get(p,0))} imagens; esperado=520."
            )


def normalize_crosswalk_columns(cw):
    rename={}
    candidates={
        "participant_id":[
            "participant_id","participant","participant_number",
            "participante","p"
        ],
        "condition_code":[
            "condition_code","condition","prompt_condition_code",
            "condicao_codigo","condition_id"
        ],
        "condition_label":[
            "condition_label","condition_name","prompt_condition",
            "condicao"
        ],
        "examples":[
            "examples","with_examples","has_examples","example_factor"
        ],
        "formal_structure":[
            "formal_structure","with_structure","structure",
            "has_structure","formal_structure_factor"
        ],
    }

    low={str(c).lower():c for c in cw.columns}

    for target,opts in candidates.items():
        if target in cw.columns:
            continue
        for opt in opts:
            if opt.lower() in low:
                src=low[opt.lower()]
                if src not in rename:
                    rename[src]=target
                    break

    return cw.rename(columns=rename)


def find_crosswalk(root,explicit=None):
    if explicit is not None:
        p=explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Crosswalk informado não encontrado: {p}")
        return p

    candidates=[
        root/"_experimental_crosswalk_v3"/"experimental_unit_crosswalk_v3.csv",
        root/"_experimental_crosswalk_v3"/"experimental_image_crosswalk_v3.csv",
        root.parent/"_experimental_crosswalk_v3"/"experimental_unit_crosswalk_v3.csv",
        root.parent/"_experimental_crosswalk_v3"/"experimental_image_crosswalk_v3.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Crosswalk experimental V3 final não encontrado.\n"
        "Esperado preferencialmente em:\n"
        f"{root/'_experimental_crosswalk_v3'/'experimental_unit_crosswalk_v3.csv'}\n"
        "Gere-o com build_experimental_crosswalk_v3.py ou informe --crosswalk CAMINHO."
    )


def attach_crosswalk(df,crosswalk_path):
    cw=pd.read_csv(crosswalk_path,dtype=str)
    cw=normalize_crosswalk_columns(cw)

    if "unit_id" not in cw.columns:
        raise RuntimeError("Crosswalk V3 sem coluna unit_id.")

    if cw["unit_id"].astype(str).nunique()!=312:
        raise RuntimeError(
            f"Crosswalk V3 incompleto: {cw['unit_id'].astype(str).nunique()} unidades; esperado=312."
        )

    required={"participant_id","condition_code","condition_label"}
    missing=required-set(cw.columns)
    if missing:
        raise RuntimeError(
            "Crosswalk V3 sem metadados obrigatórios: "
            + ", ".join(sorted(missing))
        )

    keep=["unit_id"]
    for c in [
        "participant_id","condition_code","condition_label",
        "examples","formal_structure"
    ]:
        if c in cw.columns:
            keep.append(c)

    cwu=cw[keep].drop_duplicates("unit_id").copy()
    out=df.merge(cwu,on="unit_id",how="left")

    if out["participant_id"].isna().any():
        bad=out.loc[out["participant_id"].isna(),"unit_id"].drop_duplicates().tolist()
        raise RuntimeError(
            "Há unidades sem associação no crosswalk V3: "
            + ", ".join(bad[:20])
        )

    if "condition_code" in out.columns:
        out["condition_code"]=(
            out["condition_code"].astype(str)
            .str.replace(r"\.0$","",regex=True)
            .str.zfill(2)
        )

    return out


def status_row(row):
    if int(row.get("tlx_evaluable",0))!=1 or int(row.get("tly_evaluable",0))!=1:
        return "UNEVALUABLE"

    x=int(row.get("tlx_present_candidate",0))==1
    y=int(row.get("tly_present_candidate",0))==1

    if x and y:
        return "BOTH_PRESENT"
    if x:
        return "X_ONLY"
    if y:
        return "Y_ONLY"
    return "NONE"


def font_default(size=10):
    try:
        return ImageFont.truetype("DejaVuSans.ttf",size)
    except Exception:
        return ImageFont.load_default()


def make_review_contact(g,root,outdir,out_path):
    if g.empty:
        return

    cols=4
    pw=470
    ph=350
    ih=260
    rows=math.ceil(len(g)/cols)

    sheet=Image.new("RGB",(cols*pw,rows*ph),"white")
    d=ImageDraw.Draw(sheet)
    f=font_default(10)

    for idx,row in g.reset_index(drop=True).iterrows():
        c=idx%cols
        r=idx//cols
        x0=c*pw
        y0=r*ph

        ov=outdir/"review_overlays"/row["filename"]
        im=Image.open(ov if ov.exists() else root/row["filename"]).convert("RGB")
        im.thumbnail((pw-16,ih-6))
        sheet.paste(im,(x0+(pw-im.width)//2,y0+3))

        txt=(
            f"{row['filename']} | {row['presence_status_auto']}\n"
            f"X P={row.get('tlx_present_candidate','-')} "
            f"ev={row.get('tlx_presence_evidence','-')} "
            f"S={row.get('tlx_segmentation_ok','-')} "
            f"n={row.get('tlx_n_segmented','-')}\n"
            f"Y P={row.get('tly_present_candidate','-')} "
            f"ev={row.get('tly_presence_evidence','-')} "
            f"S={row.get('tly_segmentation_ok','-')} "
            f"n={row.get('tly_n_segmented','-')}"
        )
        d.multiline_text(
            (x0+8,y0+ih+4),
            txt,fill="black",font=f,spacing=2
        )

    sheet.save(out_path)


def add_rates(summary,count_col,total_col="n_images"):
    summary["pct_x_present_auto"]=100*summary["n_x_present_auto"]/summary[total_col]
    summary["pct_y_present_auto"]=100*summary["n_y_present_auto"]/summary[total_col]
    summary["pct_both_present_auto"]=100*summary["n_both_present_auto"]/summary[total_col]
    return summary


def summarize_group(df,cols):
    s=(
        df.groupby(cols,dropna=False)
        .agg(
            n_images=("filename","size"),
            n_x_present_auto=("tlx_present_candidate","sum"),
            n_y_present_auto=("tly_present_candidate","sum"),
            n_both_present_auto=("both_present_auto","sum"),
            n_review=("needs_visual_review","sum"),
            n_x_seg_ok=("tlx_segmentation_ok","sum"),
            n_y_seg_ok=("tly_segmentation_ok","sum"),
        )
        .reset_index()
    )
    return add_rates(s)


def unit_summary(df):
    meta_cols=["unit_id","profile","unit_number","technique","task"]
    for c in [
        "participant_id","condition_code","condition_label",
        "examples","formal_structure"
    ]:
        if c in df.columns:
            meta_cols.append(c)

    agg={
        "filename":"size",
        "tlx_present_candidate":"sum",
        "tly_present_candidate":"sum",
        "both_present_auto":"sum",
        "needs_visual_review":"sum",
        "tlx_segmentation_ok":"sum",
        "tly_segmentation_ok":"sum",
    }

    u=df.groupby(meta_cols,dropna=False).agg(agg).reset_index()
    u=u.rename(columns={
        "filename":"n_images",
        "tlx_present_candidate":"n_x_present_auto",
        "tly_present_candidate":"n_y_present_auto",
        "both_present_auto":"n_both_present_auto",
        "needs_visual_review":"n_review",
        "tlx_segmentation_ok":"n_x_seg_ok",
        "tly_segmentation_ok":"n_y_seg_ok",
    })

    u=add_rates(u)
    u["all_10_x_present_auto"]=(u["n_x_present_auto"]==u["n_images"]).astype(int)
    u["all_10_y_present_auto"]=(u["n_y_present_auto"]==u["n_images"]).astype(int)
    u["all_10_both_present_auto"]=(u["n_both_present_auto"]==u["n_images"]).astype(int)

    return u


def main():
    args=parse_args()
    root=args.root.expanduser().resolve()
    outdir=(
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root/OUT_DIRNAME
    )
    (outdir/"review_overlays").mkdir(parents=True,exist_ok=True)

    actual_hash=check_v4_hash()

    inv=inventory(root)
    validate_inventory(inv)

    crosswalk_path=find_crosswalk(root,args.crosswalk)

    print("="*80)
    print("PROCESSAMENTO INTEGRAL V5 — PRESENÇA DOS TICK LABELS X/Y")
    print("="*80)
    print(f"Detector V4 congelado SHA-256: {actual_hash}")
    print(f"Imagens canônicas na raiz: {len(inv)}")
    print(f"Unidades canônicas: {inv['unit_id'].nunique()}")
    print(f"Crosswalk V3: {crosswalk_path}")
    print("Processamento: somente RAIZ, sem recursão")
    print("P=0 será enviado para auditoria visual; não é automaticamente não conformidade.")
    print()

    rows=[]
    errors=[]
    internal={}

    for _,r in inv.iterrows():
        path=root/r["filename"]

        try:
            res=v4.analyze(path)

            row={
                "filename":r["filename"],
                "path":r["path"],
                "profile":r["profile"],
                "unit_id":r["unit_id"],
                "unit_number":r["unit_number"],
                "repeat":r["repeat"],
                "technique":r["technique"],
                "task":r["task"],
                **{k:v for k,v in res.items() if not k.startswith("_")},
            }

            row["presence_status_auto"]=status_row(row)
            row["both_present_auto"]=int(row["presence_status_auto"]=="BOTH_PRESENT")
            row["needs_visual_review"]=int(row["presence_status_auto"]!="BOTH_PRESENT")

            rows.append(row)
            internal[r["filename"]]=res

            if row["needs_visual_review"]==1:
                v4.make_overlay(
                    path,res,
                    outdir/"review_overlays"/r["filename"]
                )

        except Exception as exc:
            errors.append({
                "filename":r["filename"],
                "profile":r["profile"],
                "unit_id":r["unit_id"],
                "error_type":type(exc).__name__,
                "error_message":str(exc),
            })

        done=len(rows)+len(errors)
        if done%100==0 or done==len(inv):
            print(f"Processadas: {done}/{len(inv)}")

    df=pd.DataFrame(rows)
    err=pd.DataFrame(errors)

    if len(df)!=3120 or len(err)!=0:
        print(
            f"ATENÇÃO: processadas com resultado={len(df)}, erros={len(err)}. "
            "Confira os arquivos de erro antes de interpretar."
        )

    # Anexa desenho experimental somente depois da extração visual.
    df=attach_crosswalk(df,crosswalk_path)

    # Arquivo principal automático.
    df.to_csv(
        outdir/"ticklabel_presence_images_v5.csv",
        index=False,encoding="utf-8-sig"
    )
    err.to_csv(
        outdir/"ticklabel_presence_errors_v5.csv",
        index=False,encoding="utf-8-sig"
    )

    # Casos prioritários.
    review=df[df["needs_visual_review"]==1].copy()
    review["manual_x_present_final"]=""
    review["manual_y_present_final"]=""
    review["manual_classification"]=""
    review["manual_notes"]=""

    review.to_csv(
        outdir/"ticklabel_presence_priority_review_v5.csv",
        index=False,encoding="utf-8-sig"
    )

    # Resumo por unidade.
    units=unit_summary(df)
    units.to_csv(
        outdir/"ticklabel_presence_units_v5.csv",
        index=False,encoding="utf-8-sig"
    )

    # Resumos automáticos.
    by_profile=summarize_group(df,["profile"])
    by_profile.to_csv(
        outdir/"ticklabel_presence_by_profile_v5.csv",
        index=False,encoding="utf-8-sig"
    )

    by_tech=summarize_group(df,["technique"])
    by_tech.to_csv(
        outdir/"ticklabel_presence_by_technique_v5.csv",
        index=False,encoding="utf-8-sig"
    )

    by_task=summarize_group(df,["task"])
    by_task.to_csv(
        outdir/"ticklabel_presence_by_task_v5.csv",
        index=False,encoding="utf-8-sig"
    )

    if "condition_label" in df.columns:
        by_cond=summarize_group(
            df,["condition_code","condition_label"]
        )
        by_cond.to_csv(
            outdir/"ticklabel_presence_by_condition_v5.csv",
            index=False,encoding="utf-8-sig"
        )

        by_prof_cond=summarize_group(
            df,["profile","condition_code","condition_label"]
        )
        by_prof_cond.to_csv(
            outdir/"ticklabel_presence_by_profile_condition_v5.csv",
            index=False,encoding="utf-8-sig"
        )

    # Contact sheets apenas das exceções.
    if len(review):
        for p in PROFILES:
            g=review[review["profile"]==p]
            if len(g):
                make_review_contact(
                    g,root,outdir,
                    outdir/f"contact_review_{p}.png"
                )
        make_review_contact(
            review,root,outdir,
            outdir/"contact_review_all.png"
        )

    # Resumo textual.
    n=len(df)
    nx=int(pd.to_numeric(df["tlx_present_candidate"],errors="coerce").fillna(0).sum())
    ny=int(pd.to_numeric(df["tly_present_candidate"],errors="coerce").fillna(0).sum())
    nb=int(df["both_present_auto"].sum())
    nr=int(df["needs_visual_review"].sum())

    status_counts=df["presence_status_auto"].value_counts().to_dict()

    lines=[
        "PROCESSAMENTO INTEGRAL V5 — PRESENÇA DOS TICK LABELS X/Y",
        "="*78,
        f"Detector V4 congelado SHA-256: {actual_hash}",
        f"Crosswalk V3: {crosswalk_path}",
        f"Imagens processadas: {n}",
        f"Erros: {len(err)}",
        f"Unidades: {df['unit_id'].nunique()}",
        "",
        "RESULTADO AUTOMÁTICO DO DETECTOR:",
        f"X presente automático: {nx}/{n} = {100*nx/n:.4f}%",
        f"Y presente automático: {ny}/{n} = {100*ny/n:.4f}%",
        f"X+Y simultaneamente presentes: {nb}/{n} = {100*nb/n:.4f}%",
        f"Casos enviados à auditoria visual: {nr}",
        "",
        "STATUS AUTOMÁTICO:",
    ]

    for s in ["BOTH_PRESENT","X_ONLY","Y_ONLY","NONE","UNEVALUABLE"]:
        lines.append(f"- {s}: {int(status_counts.get(s,0))}")

    lines.extend([
        "",
        "IMPORTANTE:",
        "- estes são resultados automáticos do detector, não presença final adjudicada;",
        "- todo caso diferente de BOTH_PRESENT deve ser revisado visualmente;",
        "- P=0 pode significar ausência verdadeira OU falso negativo;",
        "- NÃO recalibrar a V4 com base nas exceções;",
        "- conteúdo e tipografia dos tick labels ainda não foram avaliados.",
        "",
        "PRÓXIMA ETAPA:",
        "1. revisar ticklabel_presence_priority_review_v5.csv e as pranchas;",
        "2. preencher manual_x_present_final e manual_y_present_final;",
        "3. produzir a presença final adjudicada;",
        "4. somente então responder às análises experimentais de conformidade;",
        "5. depois avançar para conteúdo e tipografia dos tick labels.",
    ])

    summary=outdir/"ticklabel_presence_summary_v5.txt"
    summary.write_text("\n".join(lines),encoding="utf-8")

    print()
    print("Concluído.")
    print("Resumo:",summary)
    print("CSV principal:",outdir/"ticklabel_presence_images_v5.csv")
    print("Revisão prioritária:",outdir/"ticklabel_presence_priority_review_v5.csv")
    print(f"Casos para revisão visual: {nr}")


if __name__=="__main__":
    main()
