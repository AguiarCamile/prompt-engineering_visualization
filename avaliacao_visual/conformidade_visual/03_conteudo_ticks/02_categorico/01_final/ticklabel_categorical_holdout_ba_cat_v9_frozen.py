# -*- coding: utf-8 -*-
r"""
HOLDOUT INDEPENDENTE B-A-CAT — V9 CONGELADA
============================================

Objetivo
--------
Validar de forma independente o extrator categórico B-A-CAT V9, SEM reajustar
parâmetros e SEM comparar ainda com a especificação F.

Protocolo
---------
- importa e executa EXATAMENTE a B-A-CAT V9 congelada;
- verifica o SHA-256 da V9 antes de processar qualquer imagem;
- verifica as dependências congeladas B-A V7 e V4;
- exclui TODAS as unidades presentes na calibração B-A V7;
- exclui TODAS as unidades usadas no primeiro holdout independente B-A V7;
- usa apenas perfis BI e BC;
- seleciona 5 novas unidades por perfil × condição:
      5 unidades × 4 condições × 2 perfis = 40 imagens;
- seleciona uma repetição canônica por unidade com semente fixa;
- não há sobreposição de unidade ou imagem com calibração ou holdout anterior;
- processa somente arquivos canônicos da RAIZ;
- NÃO carrega a especificação F.

Critérios pré-especificados
---------------------------
Primários:
1. Precisão de CATEGORICAL_COMPLETE para extração integral >= 95%, ideal 100%.
2. Casos manualmente não integrais aceitos como COMPLETE: máximo 1, ideal 0.
3. Orientação categórica errada silenciosamente: 0.
4. Casos não integrais encaminhados para revisão >= 95%, ideal 100%.

Secundários:
- cobertura automática CATEGORICAL_COMPLETE;
- extração integral global;
- resultados por perfil e condição;
- distribuição dos tipos de falha.

IMPORTANTE
----------
- CATEGORICAL_COMPLETE/PARTIAL/AMBIGUOUS são estados automáticos, não acurácia.
- Este holdout NÃO pode ser usado para modificar a V9.
- Falhas observadas devem ser registradas como limitações independentes.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd

import ticklabel_categorical_extraction_calibration_ba_cat_v9 as cat_v9


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
OUT_DIRNAME = "_ticklabel_categorical_holdout_ba_cat_v9_frozen"

EXPECTED_CAT_V9_SHA256 = "65d29b4c1aaad4e054922e37b0cb5496084520fd31b1977bc1d003d9fd2fc5e9"

PROFILES = ("BI", "BC")
CONDITIONS = ("00", "01", "10", "11")

UNITS_PER_PROFILE_CONDITION = 5
HOLDOUT_IMAGES_EXPECTED = 40
HOLDOUT_UNITS_EXPECTED = 40

# Semente definida antes da seleção e inspeção do novo holdout.
SEED = 20260905


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--crosswalk", type=Path, default=None)
    p.add_argument("--calibration-manifest", type=Path, default=None)
    p.add_argument("--previous-holdout-manifest", type=Path, default=None)
    p.add_argument("--tesseract", type=Path, default=None)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_cat_v9():
    path = Path(cat_v9.__file__).resolve()
    actual = sha256(path)

    if actual != EXPECTED_CAT_V9_SHA256:
        raise RuntimeError(
            "A B-A-CAT V9 não corresponde à versão congelada.\n"
            f"Arquivo: {path}\n"
            f"SHA esperado: {EXPECTED_CAT_V9_SHA256}\n"
            f"SHA encontrado: {actual}\n"
            "Não execute o holdout com uma V9 modificada."
        )

    v7_path, v7_hash, presence_path, presence_hash = (
        cat_v9.verify_frozen_dependencies()
    )

    return (
        path, actual,
        v7_path, v7_hash,
        presence_path, presence_hash,
    )


def find_calibration_manifest(root: Path, explicit=None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Manifesto de calibração não encontrado: {p}"
            )
        return p

    p = (
        root
        / "_ticklabel_content_extraction_ba_v7"
        / "ticklabel_ba_v7_manifest.csv"
    )
    if not p.exists():
        raise FileNotFoundError(
            "Manifesto da calibração B-A V7 não encontrado.\n"
            f"Esperado: {p}\n"
            "Use --calibration-manifest se estiver em outro local."
        )
    return p.resolve()


def find_previous_holdout_manifest(root: Path, explicit=None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Manifesto do holdout B-A V7 anterior não encontrado: {p}"
            )
        return p

    p = (
        root
        / "_ticklabel_content_extraction_holdout_ba_v7_frozen"
        / "ticklabel_ba_v7_holdout_manifest.csv"
    )
    if not p.exists():
        raise FileNotFoundError(
            "Manifesto do primeiro holdout B-A V7 não encontrado.\n"
            f"Esperado: {p}\n"
            "Use --previous-holdout-manifest se estiver em outro local."
        )
    return p.resolve()


def load_unit_set_from_manifest(path: Path, inv: pd.DataFrame, label: str):
    df = pd.read_csv(path, dtype=str)

    if "unit_id" in df.columns:
        units = set(df["unit_id"].dropna().astype(str))
        if units:
            return units, df

    if "filename" not in df.columns:
        raise RuntimeError(
            f"{label}: manifesto sem unit_id e sem filename."
        )

    names = df[["filename"]].drop_duplicates().copy()
    inv2 = inv[["filename", "unit_id"]].copy()
    inv2["filename"] = inv2["filename"].astype(str)

    merged = names.merge(
        inv2,
        on="filename",
        how="left",
        validate="one_to_one"
    )

    if merged["unit_id"].isna().any():
        miss = merged.loc[
            merged["unit_id"].isna(), "filename"
        ].tolist()
        raise RuntimeError(
            f"{label}: arquivos ausentes no inventário: "
            + ", ".join(miss[:20])
        )

    return set(merged["unit_id"].astype(str)), merged


def select_holdout(
    inv: pd.DataFrame,
    crosswalk_path: Path,
    excluded_units: set[str],
):
    cw = cat_v9.ba_v7.load_unit_crosswalk(crosswalk_path)

    units = (
        inv[["profile", "unit_id"]]
        .drop_duplicates()
        .merge(cw, on="unit_id", how="left", validate="one_to_one")
    )

    if units["condition_code"].isna().any():
        n = int(units["condition_code"].isna().sum())
        raise RuntimeError(
            f"{n} unidades ficaram sem condition_code no crosswalk."
        )

    eligible = units[
        units["profile"].isin(PROFILES)
        & ~units["unit_id"].astype(str).isin(excluded_units)
    ].copy()

    rng = random.Random(SEED)
    chosen_units = []

    for profile in PROFILES:
        for cond in CONDITIONS:
            g = eligible[
                (eligible["profile"] == profile)
                & (eligible["condition_code"] == cond)
            ].copy()

            unit_ids = sorted(
                g["unit_id"].dropna().astype(str).unique().tolist()
            )

            if len(unit_ids) < UNITS_PER_PROFILE_CONDITION:
                raise RuntimeError(
                    f"Holdout insuficiente para {profile}/{cond}: "
                    f"{len(unit_ids)} elegíveis; "
                    f"necessárias={UNITS_PER_PROFILE_CONDITION}."
                )

            picked = rng.sample(
                unit_ids,
                UNITS_PER_PROFILE_CONDITION
            )

            for unit_id in picked:
                row = g[
                    g["unit_id"].astype(str) == unit_id
                ].iloc[0].to_dict()
                chosen_units.append(row)

    chosen = pd.DataFrame(chosen_units)

    if len(chosen) != HOLDOUT_UNITS_EXPECTED:
        raise RuntimeError(
            f"Unidades selecionadas={len(chosen)}; "
            f"esperado={HOLDOUT_UNITS_EXPECTED}."
        )

    if chosen["unit_id"].nunique() != HOLDOUT_UNITS_EXPECTED:
        raise RuntimeError("Há unidade duplicada no holdout CAT.")

    rows = []

    for _, u in chosen.iterrows():
        imgs = inv[
            inv["unit_id"].astype(str) == str(u["unit_id"])
        ].copy()

        if imgs.empty:
            raise RuntimeError(
                f"Nenhuma repetição encontrada para {u['unit_id']}."
            )

        imgs = imgs.sort_values("repeat").reset_index(drop=True)
        idx = rng.randrange(len(imgs))
        r = imgs.iloc[idx].to_dict()

        r["condition_code"] = str(u["condition_code"])
        for c in ("condition_label", "participant_id"):
            if c in u:
                r[c] = u.get(c, "")

        rows.append(r)

    hold = pd.DataFrame(rows).sort_values(
        ["profile", "condition_code", "unit_id", "repeat"]
    ).reset_index(drop=True)

    return hold


def validate_independence(
    hold: pd.DataFrame,
    calibration_units: set[str],
    previous_holdout_units: set[str],
):
    hold_units = set(hold["unit_id"].astype(str))

    overlap_cal = sorted(hold_units & calibration_units)
    overlap_prev = sorted(hold_units & previous_holdout_units)

    if overlap_cal:
        raise RuntimeError(
            "Sobreposição com calibração: "
            + ", ".join(overlap_cal[:20])
        )

    if overlap_prev:
        raise RuntimeError(
            "Sobreposição com holdout B-A anterior: "
            + ", ".join(overlap_prev[:20])
        )

    if len(hold) != HOLDOUT_IMAGES_EXPECTED:
        raise RuntimeError(
            f"Imagens={len(hold)}; esperado={HOLDOUT_IMAGES_EXPECTED}."
        )

    if hold["unit_id"].nunique() != HOLDOUT_UNITS_EXPECTED:
        raise RuntimeError(
            f"Unidades={hold['unit_id'].nunique()}; "
            f"esperado={HOLDOUT_UNITS_EXPECTED}."
        )

    balance = (
        hold.groupby(["profile", "condition_code"])
        .size()
        .rename("n")
        .reset_index()
    )

    if not (balance["n"] == UNITS_PER_PROFILE_CONDITION).all():
        raise RuntimeError(
            "Holdout CAT não ficou balanceado em perfil × condição."
        )

    return overlap_cal, overlap_prev, balance


def process_holdout(
    hold: pd.DataFrame,
    root: Path,
    outdir: Path,
    lang: str,
):
    image_rows = []
    slot_rows = []
    candidate_rows = []
    errors = []

    total = len(hold)

    for i, r in hold.iterrows():
        path = root / r["filename"]

        print(
            f"[{i+1:02d}/{total}] {r['filename']} ...",
            end=" ",
            flush=True
        )

        try:
            imr, sr, cr = cat_v9.process_image(
                path, r["profile"], lang, outdir
            )

            for c in (
                "unit_id", "unit_number", "repeat",
                "condition_code", "condition_label",
                "participant_id",
            ):
                if c in r.index:
                    imr[c] = r[c]
                    for rr in sr:
                        rr[c] = r[c]
                    for rr in cr:
                        rr[c] = r[c]

            image_rows.append(imr)
            slot_rows.extend(sr)
            candidate_rows.extend(cr)
            print("OK", flush=True)

        except Exception as exc:
            errors.append({
                "filename": r["filename"],
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "condition_code": r["condition_code"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            print(
                f"ERRO: {type(exc).__name__}: {exc}",
                flush=True
            )

    return (
        pd.DataFrame(image_rows),
        pd.DataFrame(slot_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(errors),
    )


def make_audit_all(images_df: pd.DataFrame):
    cols = [
        c for c in [
            "filename", "profile", "unit_id",
            "condition_code", "condition_label",
            "participant_id",
            "plot_bbox_source",
            "bar_orientation_observed",
            "categorical_axis_observed",
            "categorical_slot_strategy",
            "row_groups_detected",
            "row_groups_text",
            "bar_category_groups_detected",
            "continuation_groups_accepted",
            "probable_axis_title_groups_rejected",
            "categorical_axis_status",
            "categorical_n_slots",
            "categorical_n_selected",
            "categorical_n_ambiguous",
            "categorical_n_unreadable",
            "categorical_n_suspicious_edge_slots",
            "categorical_selected_labels_spatial",
            "overlay_path",
        ]
        if c in images_df.columns
    ]

    audit = images_df[cols].copy()

    audit["manual_visible_categories"] = ""
    audit["manual_orientation_result"] = ""
    audit["manual_extraction_result"] = ""
    audit["manual_missing_labels"] = ""
    audit["manual_false_labels"] = ""
    audit["manual_notes"] = ""

    return audit


def make_priority_review(audit: pd.DataFrame):
    if audit.empty:
        return audit.copy()

    mask = ~audit["categorical_axis_status"].eq(
        "CATEGORICAL_COMPLETE"
    )

    if "categorical_n_suspicious_edge_slots" in audit.columns:
        mask = mask | (
            pd.to_numeric(
                audit["categorical_n_suspicious_edge_slots"],
                errors="coerce"
            ).fillna(0) > 0
        )

    return audit[mask].copy()


def write_empty_csv(path: Path, columns):
    pd.DataFrame(columns=list(columns)).to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
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

    (
        cat_path, cat_hash,
        v7_path, v7_hash,
        presence_path, presence_hash,
    ) = verify_frozen_cat_v9()

    t_version, lang, langs = cat_v9.ba_v7.configure_tesseract(
        args.tesseract
    )

    inv = cat_v9.ba_v7.inventory(root)

    if len(inv) != 3120:
        raise RuntimeError(
            f"Inventário canônico na RAIZ={len(inv)}; esperado=3120."
        )

    if inv["unit_id"].nunique() != 312:
        raise RuntimeError(
            f"Unidades canônicas={inv['unit_id'].nunique()}; esperado=312."
        )

    crosswalk_path = cat_v9.ba_v7.find_crosswalk(
        root, args.crosswalk
    )
    if crosswalk_path is None:
        raise FileNotFoundError(
            "Crosswalk experimental V3 não encontrado."
        )

    cal_path = find_calibration_manifest(
        root, args.calibration_manifest
    )
    prev_path = find_previous_holdout_manifest(
        root, args.previous_holdout_manifest
    )

    calibration_units, _ = load_unit_set_from_manifest(
        cal_path, inv, "Calibração B-A V7"
    )
    previous_holdout_units, _ = load_unit_set_from_manifest(
        prev_path, inv, "Holdout B-A V7 anterior"
    )

    excluded_units = calibration_units | previous_holdout_units

    hold = select_holdout(
        inv,
        crosswalk_path,
        excluded_units,
    )

    overlap_cal, overlap_prev, balance = validate_independence(
        hold,
        calibration_units,
        previous_holdout_units,
    )

    manifest_path = outdir / "ba_cat_v9_holdout_manifest.csv"
    hold.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    balance.to_csv(
        outdir / "ba_cat_v9_holdout_balance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 80)
    print("HOLDOUT INDEPENDENTE B-A-CAT — V9 CONGELADA")
    print("=" * 80)
    print(f"CAT V9 SHA-256: {cat_hash}")
    print(f"B-A V7 SHA-256: {v7_hash}")
    print(f"V4 presença SHA-256: {presence_hash}")
    print(f"Manifesto calibração: {cal_path}")
    print(f"Manifesto holdout B-A anterior: {prev_path}")
    print(f"Crosswalk V3: {crosswalk_path}")
    print(f"Semente fixa: {SEED}")
    print(f"Unidades calibração excluídas: {len(calibration_units)}")
    print(
        "Unidades primeiro holdout B-A excluídas: "
        f"{len(previous_holdout_units)}"
    )
    print(f"Unidades totais excluídas: {len(excluded_units)}")
    print(f"Unidades novo holdout: {hold['unit_id'].nunique()}")
    print(f"Imagens novo holdout: {len(hold)}")
    print("Balanceamento: 5 por perfil × condição")
    print("Sobreposição com calibração: 0")
    print("Sobreposição com holdout anterior: 0")
    print()
    print("A V9 está congelada. NÃO reajustar após observar este holdout.")
    print("A especificação F NÃO é carregada.")
    print()

    (
        images_df,
        slots_df,
        candidates_df,
        errors_df,
    ) = process_holdout(
        hold, root, outdir, lang
    )

    images_df.to_csv(
        outdir / "ba_cat_v9_holdout_images.csv",
        index=False,
        encoding="utf-8-sig"
    )
    slots_df.to_csv(
        outdir / "ba_cat_v9_holdout_slots.csv",
        index=False,
        encoding="utf-8-sig"
    )
    candidates_df.to_csv(
        outdir / "ba_cat_v9_holdout_candidates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if errors_df.empty:
        write_empty_csv(
            outdir / "ba_cat_v9_holdout_errors.csv",
            [
                "filename", "profile", "unit_id",
                "condition_code",
                "error_type", "error_message",
            ]
        )
    else:
        errors_df.to_csv(
            outdir / "ba_cat_v9_holdout_errors.csv",
            index=False,
            encoding="utf-8-sig"
        )

    audit = make_audit_all(images_df)
    audit.to_csv(
        outdir / "ba_cat_v9_holdout_audit_all.csv",
        index=False,
        encoding="utf-8-sig"
    )

    priority = make_priority_review(audit)
    priority.to_csv(
        outdir / "ba_cat_v9_holdout_priority_review.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for profile in PROFILES:
        g = images_df[
            images_df["profile"] == profile
        ]
        if len(g):
            cat_v9.make_cat_contact_sheet(
                g,
                root,
                outdir / f"contact_holdout_BA_CAT_V9_{profile}.png"
            )

    status_counts = (
        images_df["categorical_axis_status"]
        .value_counts()
        .to_dict()
        if not images_df.empty
        else {}
    )

    lines = [
        "HOLDOUT INDEPENDENTE B-A-CAT — V9 CONGELADA",
        "=" * 78,
        f"B-A-CAT V9 SHA-256: {cat_hash}",
        f"Arquivo CAT V9: {cat_path}",
        f"B-A V7 SHA-256: {v7_hash}",
        f"V4 presença SHA-256: {presence_hash}",
        f"Tesseract: {t_version}",
        f"Idioma OCR: {lang}",
        f"Semente fixa: {SEED}",
        "",
        "INDEPENDÊNCIA:",
        f"- unidades de calibração excluídas: {len(calibration_units)};",
        (
            "- unidades do primeiro holdout B-A excluídas: "
            f"{len(previous_holdout_units)};"
        ),
        f"- unidades totais excluídas: {len(excluded_units)};",
        f"- unidades do novo holdout: {hold['unit_id'].nunique()};",
        f"- imagens do novo holdout: {len(hold)};",
        f"- sobreposição com calibração: {len(overlap_cal)};",
        f"- sobreposição com holdout anterior: {len(overlap_prev)};",
        "- balanceamento: 5 unidades por perfil × condição;",
        "- uma repetição canônica por unidade.",
        "",
        "REGRA DE CONGELAMENTO:",
        "- a V9 é executada sem qualquer ajuste;",
        "- o SHA-256 é verificado antes do processamento;",
        "- o holdout NÃO pode ser usado para recalibrar parâmetros;",
        "- erros serão registrados como limitações independentes.",
        "",
        "CRITÉRIO PRIMÁRIO PRÉ-ESPECIFICADO:",
        (
            "- precisão de CATEGORICAL_COMPLETE para extração integral "
            ">=95%, ideal 100%;"
        ),
        (
            "- casos manualmente não integrais aceitos como COMPLETE: "
            "máximo 1, ideal 0;"
        ),
        "- orientação categórica errada silenciosamente: 0;",
        (
            "- encaminhamento para revisão entre casos não integrais "
            ">=95%, ideal 100%;"
        ),
        "- cobertura COMPLETE é métrica secundária;",
        "- extração integral global é métrica secundária.",
        "",
        "AUDITORIA MANUAL:",
        "- EXACT/EQUIVALENT = extração integral;",
        "- PARTIAL = conteúdo visível parcialmente recuperado;",
        "- WRONG = falso conteúdo ou estrutura categórica incorreta;",
        "- UNEVALUABLE = adjudicação visual insegura.",
        "",
        f"Processadas com sucesso: {len(images_df)}/{len(hold)}",
        f"Erros de execução: {len(errors_df)}",
        f"Casos em revisão prioritária automática: {len(priority)}",
        "",
        "STATUS AUTOMÁTICO:",
    ]

    for k, n in status_counts.items():
        lines.append(f"- {k}: {int(n)}")

    lines.extend([
        "",
        "PRÓXIMA ETAPA:",
        "- auditar visualmente todas as 40 imagens;",
        "- não modificar a V9 após observar o holdout;",
        "- calcular precisão de COMPLETE, cobertura e recall de revisão;",
        "- se o critério independente for satisfeito, fechar B-A categórico;",
        "- depois integrar com B-B: observado × especificação F.",
    ])

    summary_path = outdir / "ba_cat_v9_holdout_summary.txt"
    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print()
    print("Concluído.")
    print("Resumo:", summary_path)
    print("Manifesto:", manifest_path)
    print(
        "Auditoria:",
        outdir / "ba_cat_v9_holdout_audit_all.csv"
    )
    print(
        "Revisão prioritária:",
        outdir / "ba_cat_v9_holdout_priority_review.csv"
    )
    print(
        "Envie também contact_holdout_BA_CAT_V9_BI.png "
        "e contact_holdout_BA_CAT_V9_BC.png."
    )


if __name__ == "__main__":
    main()
