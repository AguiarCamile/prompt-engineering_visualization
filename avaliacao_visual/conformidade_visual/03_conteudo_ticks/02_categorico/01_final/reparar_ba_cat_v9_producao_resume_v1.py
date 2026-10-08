# -*- coding: utf-8 -*-
r"""
REPARO DA SAÍDA B-A-CAT V9 COM CHECKPOINT
=========================================

Motivo:
O runner com checkpoint gravou os dicionários de cada imagem diretamente
em CSV. Em imagens sem eixo categórico resolvido, o dicionário possui um
conjunto/ordem de campos diferente. Ao fazer append, alguns campos ficaram
deslocados no CSV final.

Este reparo:
1. NÃO altera a B-A-CAT V9 congelada.
2. NÃO carrega F.
3. Identifica apenas linhas estruturalmente corrompidas pelo CSV
   (unit_id ausente ou incompatível com filename).
4. Reprocessa somente essas imagens.
5. Reconstrói o CSV por nomes de colunas usando concat do pandas.
6. Regenera audit, priority review e summary.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

import ticklabel_categorical_extraction_calibration_ba_cat_v9 as cat_v9


ROOT_DEFAULT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES_DEFAULT = ROOT_DEFAULT / "imagens"
OUTDIR_DEFAULT = IMAGES_DEFAULT / "_ticklabel_categorical_production_ba_cat_v9_frozen_resume"

EXPECTED_CAT_V9_SHA256 = "65d29b4c1aaad4e054922e37b0cb5496084520fd31b1977bc1d003d9fd2fc5e9"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--images-root", type=Path, default=IMAGES_DEFAULT)
    p.add_argument("--source-dir", type=Path, default=OUTDIR_DEFAULT)
    p.add_argument("--tesseract", type=Path, default=None)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_from_filename(filename: str) -> str:
    m = re.match(r"^([A-Z]{2}_\d{3})_R\d{2}\.png$", str(filename))
    return m.group(1) if m else ""


def repeat_from_filename(filename: str) -> str:
    m = re.match(r"^[A-Z]{2}_\d{3}_(R\d{2})\.png$", str(filename))
    return m.group(1) if m else ""


def verify_frozen():
    cat_path = Path(cat_v9.__file__).resolve()
    actual = sha256(cat_path)
    if actual != EXPECTED_CAT_V9_SHA256:
        raise RuntimeError(
            "B-A-CAT V9 diferente da versão congelada.\n"
            f"Arquivo: {cat_path}\n"
            f"SHA esperado: {EXPECTED_CAT_V9_SHA256}\n"
            f"SHA encontrado: {actual}"
        )
    (
        v7_path, v7_hash,
        presence_path, presence_hash
    ) = cat_v9.verify_frozen_dependencies()
    return cat_path, actual, v7_path, v7_hash, presence_path, presence_hash


def load_metadata(project_root: Path):
    p = project_root / "prompts_manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {p}")

    df = pd.read_csv(p, dtype=str)
    keep = [
        c for c in [
            "unit_id", "participant_id", "condition", "type", "task",
            "prompt_file", "prompt_sha256"
        ] if c in df.columns
    ]
    return df[keep].drop_duplicates("unit_id").set_index("unit_id")


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    images_root = args.images_root.resolve()
    source_dir = args.source_dir.resolve()

    (
        cat_path, cat_hash,
        v7_path, v7_hash,
        presence_path, presence_hash
    ) = verify_frozen()

    t_version, lang, langs = cat_v9.ba_v7.configure_tesseract(args.tesseract)

    source_images = source_dir / "ba_cat_v9_production_images.csv"
    if not source_images.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {source_images}")

    df = pd.read_csv(source_images, dtype=str)

    if len(df) != 1040:
        raise RuntimeError(f"Linhas no arquivo={len(df)}; esperado=1040.")
    if df["filename"].nunique() != 1040:
        raise RuntimeError(
            f"Filenames únicos={df['filename'].nunique()}; esperado=1040."
        )

    expected_unit = df["filename"].map(unit_from_filename)
    bad_mask = (
        df["unit_id"].isna()
        | df["unit_id"].fillna("").eq("")
        | df["unit_id"].fillna("").ne(expected_unit)
    )

    bad = df[bad_mask].copy()
    good = df[~bad_mask].copy()

    print("=" * 80)
    print("REPARO B-A-CAT V9 — SAÍDA CSV")
    print("=" * 80)
    print(f"CAT V9 SHA-256: {cat_hash}")
    print(f"B-A V7 SHA-256: {v7_hash}")
    print(f"V4 presença SHA-256: {presence_hash}")
    print(f"Tesseract: {t_version}")
    print(f"Idioma OCR: {lang}")
    print(f"Linhas totais: {len(df)}")
    print(f"Linhas estruturalmente íntegras: {len(good)}")
    print(f"Linhas a reprocessar: {len(bad)}")
    print("F NÃO é carregada.")
    print()

    repair_dir = source_dir / "_repair_v1"
    repair_dir.mkdir(parents=True, exist_ok=True)

    # Registra a lista exata que será reprocessada.
    bad[["filename", "profile"]].to_csv(
        repair_dir / "reprocess_list.csv",
        index=False, encoding="utf-8-sig"
    )

    meta = load_metadata(project_root)

    repaired_rows = []
    repair_errors = []

    for i, (_, old) in enumerate(bad.iterrows(), start=1):
        filename = old["filename"]
        profile = old["profile"]
        path = images_root / filename

        print(
            f"[{i:03d}/{len(bad):03d}] {filename} ...",
            end=" ", flush=True
        )

        try:
            imr, sr, cr = cat_v9.process_image(
                path, profile, lang, repair_dir
            )

            unit_id = unit_from_filename(filename)
            repeat = repeat_from_filename(filename)

            imr["unit_id"] = unit_id
            imr["unit_number"] = unit_id.split("_")[1] if unit_id else ""
            imr["repeat"] = repeat

            if unit_id in meta.index:
                mr = meta.loc[unit_id]
                for c in [
                    "participant_id", "condition", "type", "task"
                ]:
                    imr[c] = mr.get(c, "")
            else:
                for c in [
                    "participant_id", "condition", "type", "task"
                ]:
                    imr[c] = ""

            repaired_rows.append(imr)
            print("OK", flush=True)

        except Exception as exc:
            repair_errors.append({
                "filename": filename,
                "profile": profile,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
            print(
                f"ERRO: {type(exc).__name__}: {exc}",
                flush=True
            )

    err_df = pd.DataFrame(repair_errors)
    err_df.to_csv(
        repair_dir / "repair_errors.csv",
        index=False, encoding="utf-8-sig"
    )

    if repair_errors:
        print()
        print("Há erros no reparo. NÃO consolidar ainda.")
        print(repair_dir / "repair_errors.csv")
        return 1

    repaired = pd.DataFrame(repaired_rows)

    # Concat alinha por NOME de coluna, evitando deslocamento.
    combined = pd.concat([good, repaired], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("filename", keep="last")
    combined = combined.sort_values("filename").reset_index(drop=True)

    if len(combined) != 1040 or combined["filename"].nunique() != 1040:
        raise RuntimeError(
            "Consolidação reparada não contém exatamente 1040 imagens únicas."
        )

    # Validação final da chave unit_id.
    expected_unit2 = combined["filename"].map(unit_from_filename)
    bad2 = (
        combined["unit_id"].isna()
        | combined["unit_id"].fillna("").eq("")
        | combined["unit_id"].fillna("").ne(expected_unit2)
    )
    if bad2.any():
        raise RuntimeError(
            f"Ainda restam {int(bad2.sum())} linhas com unit_id inconsistente."
        )

    # Arquivos reparados — não sobrescreve a saída anterior.
    repaired_images_path = repair_dir / "ba_cat_v9_production_images_REPAIRED.csv"
    combined.to_csv(
        repaired_images_path,
        index=False, encoding="utf-8-sig"
    )

    audit = cat_v9.make_audit_all(combined)
    audit.to_csv(
        repair_dir / "ba_cat_v9_production_audit_all_REPAIRED.csv",
        index=False, encoding="utf-8-sig"
    )

    priority = cat_v9.make_priority_review(audit)
    priority.to_csv(
        repair_dir / "ba_cat_v9_production_priority_review_REPAIRED.csv",
        index=False, encoding="utf-8-sig"
    )

    complete = combined[
        combined["categorical_axis_status"].eq("CATEGORICAL_COMPLETE")
    ].copy()
    complete.to_csv(
        repair_dir / "ba_cat_v9_production_complete_auto_candidates_REPAIRED.csv",
        index=False, encoding="utf-8-sig"
    )

    status_counts = (
        combined["categorical_axis_status"]
        .fillna("NO_CATEGORICAL_AXIS_RESOLVED")
        .value_counts(dropna=False)
        .rename_axis("categorical_axis_status")
        .reset_index(name="n")
    )
    status_counts.to_csv(
        repair_dir / "ba_cat_v9_production_summary_status_REPAIRED.csv",
        index=False, encoding="utf-8-sig"
    )

    summary_lines = [
        "B-A-CAT V9 — PRODUÇÃO REPARADA",
        "=" * 72,
        f"CAT V9 SHA-256: {cat_hash}",
        f"B-A V7 SHA-256: {v7_hash}",
        f"V4 presença SHA-256: {presence_hash}",
        f"Imagens totais: {len(combined)}",
        f"Linhas reprocessadas para reparo estrutural: {len(bad)}",
        f"Erros no reparo: {len(repair_errors)}",
        f"Revisão prioritária após reparo: {len(priority)}",
        f"COMPLETE automáticos após reparo: {len(complete)}",
        "",
        "STATUS:",
    ]
    for _, r in status_counts.iterrows():
        summary_lines.append(
            f"- {r['categorical_axis_status']}: {int(r['n'])}"
        )

    (repair_dir / "ba_cat_v9_production_summary_REPAIRED.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    print()
    print("=" * 80)
    print("REPARO CONCLUÍDO")
    print("=" * 80)
    print(f"Imagens consolidadas: {len(combined)}/1040")
    print(f"Linhas reparadas: {len(bad)}")
    print(f"Erros: {len(repair_errors)}")
    print("Pasta:", repair_dir)
    print("Arquivo principal:", repaired_images_path)
    print()
    print("Envie os arquivos *_REPAIRED para a próxima análise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
