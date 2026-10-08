# -*- coding: utf-8 -*-
r"""
B-A-CAT V9 — PRODUÇÃO CONGELADA, COM CHECKPOINT/RESUME
======================================================

Executa a lógica congelada B-A-CAT V9 sobre as 1.040 imagens BI/BC.

Diferença em relação ao runner anterior:
- salva checkpoint após CADA imagem;
- ao reiniciar, lê o checkpoint e pula imagens já concluídas;
- preserva independência de F;
- não altera a lógica B-A-CAT V9.

IMPORTANTE:
A lógica congelada continua sendo:
ticklabel_categorical_extraction_calibration_ba_cat_v9.py
SHA-256 esperado:
65d29b4c1aaad4e054922e37b0cb5496084520fd31b1977bc1d003d9fd2fc5e9

O checkpoint é apenas controle de execução, não recalibração.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import ticklabel_categorical_extraction_calibration_ba_cat_v9 as cat_v9


DEFAULT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120\imagens")
DEFAULT_PROJECT_ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
OUT_DIRNAME = "_ticklabel_categorical_production_ba_cat_v9_frozen_resume"

EXPECTED_CAT_V9_SHA256 = "65d29b4c1aaad4e054922e37b0cb5496084520fd31b1977bc1d003d9fd2fc5e9"
PROFILES = ("BI", "BC")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--tesseract", type=Path, default=None)
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Apaga somente os checkpoints deste runner e reinicia do zero."
    )
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_cat_v9():
    path = Path(cat_v9.__file__).resolve()
    actual = sha256(path)
    if actual != EXPECTED_CAT_V9_SHA256:
        raise RuntimeError(
            "B-A-CAT V9 diferente da versão congelada.\n"
            f"Arquivo: {path}\n"
            f"SHA esperado: {EXPECTED_CAT_V9_SHA256}\n"
            f"SHA encontrado: {actual}"
        )

    (
        v7_path, v7_hash,
        presence_path, presence_hash
    ) = cat_v9.verify_frozen_dependencies()

    return path, actual, v7_path, v7_hash, presence_path, presence_hash


def load_metadata(project_root: Path, explicit=None):
    p = explicit.expanduser().resolve() if explicit else project_root / "prompts_manifest.csv"
    if not p.exists():
        return p, pd.DataFrame()

    df = pd.read_csv(p, dtype=str)
    keep = [
        c for c in [
            "unit_id", "participant_id", "condition", "type", "task",
            "prompt_file", "prompt_sha256"
        ] if c in df.columns
    ]
    if "unit_id" not in keep:
        raise RuntimeError(f"Manifesto sem unit_id: {p}")
    return p, df[keep].drop_duplicates("unit_id")


def append_records_csv(path: Path, records):
    if not records:
        return
    df = pd.DataFrame(records)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(
        path,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig"
    )


def load_completed(checkpoint_path: Path):
    if not checkpoint_path.exists():
        return set()
    df = pd.read_csv(checkpoint_path, dtype=str)
    if "filename" not in df.columns:
        return set()
    return set(df["filename"].dropna().astype(str))


def remove_if_exists(path: Path):
    if path.exists():
        path.unlink()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()

    outdir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / OUT_DIRNAME
    )
    outdir.mkdir(parents=True, exist_ok=True)

    checkpoint_images = outdir / "checkpoint_images.csv"
    checkpoint_slots = outdir / "checkpoint_slots.csv"
    checkpoint_candidates = outdir / "checkpoint_candidates.csv"
    checkpoint_errors = outdir / "checkpoint_errors.csv"
    checkpoint_state = outdir / "checkpoint_state.json"

    if args.fresh:
        for p in [
            checkpoint_images,
            checkpoint_slots,
            checkpoint_candidates,
            checkpoint_errors,
            checkpoint_state,
        ]:
            remove_if_exists(p)
        print("Checkpoints anteriores removidos. Reinício do zero.")

    (
        cat_path, cat_hash,
        v7_path, v7_hash,
        presence_path, presence_hash
    ) = verify_frozen_cat_v9()

    t_version, lang, langs = cat_v9.ba_v7.configure_tesseract(args.tesseract)

    inv = cat_v9.ba_v7.inventory(root)
    if len(inv) != 3120:
        raise RuntimeError(f"Inventário RAIZ={len(inv)}; esperado=3120.")
    if inv["unit_id"].nunique() != 312:
        raise RuntimeError(
            f"Unidades canônicas={inv['unit_id'].nunique()}; esperado=312."
        )

    manifest_path, meta = load_metadata(project_root, args.manifest)
    if not meta.empty:
        inv = inv.merge(meta, on="unit_id", how="left", validate="many_to_one")

    prod = inv[inv["profile"].isin(PROFILES)].copy()
    prod = prod.sort_values(["profile", "unit_id", "repeat"]).reset_index(drop=True)

    if len(prod) != 1040:
        raise RuntimeError(f"Imagens BI/BC={len(prod)}; esperado=1040.")
    if prod["unit_id"].nunique() != 104:
        raise RuntimeError(
            f"Unidades BI/BC={prod['unit_id'].nunique()}; esperado=104."
        )

    manifest_out = outdir / "ba_cat_v9_production_manifest.csv"
    prod.to_csv(manifest_out, index=False, encoding="utf-8-sig")

    completed = load_completed(checkpoint_images)

    print("=" * 80)
    print("B-A-CAT V9 — PRODUÇÃO COM CHECKPOINT/RESUME")
    print("=" * 80)
    print(f"CAT V9 SHA-256: {cat_hash}")
    print(f"B-A V7 SHA-256: {v7_hash}")
    print(f"V4 presença SHA-256: {presence_hash}")
    print(f"Tesseract: {t_version}")
    print(f"Idioma OCR: {lang}")
    print(f"Total BI/BC: {len(prod)}")
    print(f"Já concluídas no checkpoint: {len(completed)}")
    print(f"Restantes: {len(prod) - len(completed)}")
    print("F NÃO é carregada.")
    print()

    total = len(prod)
    success_this_run = 0
    errors_this_run = 0

    for i, r in prod.iterrows():
        filename = r["filename"]

        if filename in completed:
            continue

        path = root / filename

        print(
            f"[{i+1:04d}/{total}] {filename} ...",
            end=" ", flush=True
        )

        try:
            imr, sr, cr = cat_v9.process_image(
                path, r["profile"], lang, outdir
            )

            for c in [
                "unit_id", "unit_number", "repeat",
                "participant_id", "condition", "type", "task"
            ]:
                if c in r.index:
                    imr[c] = r.get(c, "")
                    for rr in sr:
                        rr[c] = r.get(c, "")
                    for rr in cr:
                        rr[c] = r.get(c, "")

            # Ordem importante:
            # slots/candidatos primeiro; a imagem vira "concluída" só no final.
            append_records_csv(checkpoint_slots, sr)
            append_records_csv(checkpoint_candidates, cr)
            append_records_csv(checkpoint_images, [imr])

            completed.add(filename)
            success_this_run += 1

            checkpoint_state.write_text(
                json.dumps(
                    {
                        "last_completed_filename": filename,
                        "completed": len(completed),
                        "total": total,
                        "remaining": total - len(completed),
                        "cat_v9_sha256": cat_hash,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"OK | concluídas={len(completed)}/{total}",
                flush=True
            )

        except KeyboardInterrupt:
            print("\n\nInterrupção solicitada pelo usuário.")
            print("Checkpoint preservado.")
            print(f"Concluídas: {len(completed)}/{total}")
            print("Para continuar, execute o MESMO comando novamente.")
            return

        except Exception as exc:
            err = {
                "filename": filename,
                "profile": r["profile"],
                "unit_id": r["unit_id"],
                "repeat": r.get("repeat", ""),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            append_records_csv(checkpoint_errors, [err])
            errors_this_run += 1
            print(
                f"ERRO: {type(exc).__name__}: {exc}",
                flush=True
            )

    # ------------------------------------------------------------------
    # Finalização a partir dos checkpoints
    # ------------------------------------------------------------------
    images_df = pd.read_csv(checkpoint_images, dtype=str)
    images_df = images_df.drop_duplicates("filename", keep="last")
    images_df = images_df.sort_values("filename").reset_index(drop=True)

    if checkpoint_slots.exists():
        slots_df = pd.read_csv(checkpoint_slots, dtype=str)
        slots_df = slots_df.drop_duplicates(
            ["filename", "slot_index"], keep="last"
        )
    else:
        slots_df = pd.DataFrame()

    if checkpoint_candidates.exists():
        candidates_df = pd.read_csv(checkpoint_candidates, dtype=str)
    else:
        candidates_df = pd.DataFrame()

    if checkpoint_errors.exists():
        errors_df = pd.read_csv(checkpoint_errors, dtype=str)
        # Se uma imagem falhou antes mas depois foi concluída, erro antigo não é final.
        errors_df = errors_df[
            ~errors_df["filename"].isin(set(images_df["filename"]))
        ].copy()
    else:
        errors_df = pd.DataFrame(
            columns=[
                "filename", "profile", "unit_id", "repeat",
                "error_type", "error_message"
            ]
        )

    if len(images_df) != 1040:
        print()
        print("Execução ainda incompleta.")
        print(f"Concluídas: {len(images_df)}/1040")
        print(f"Restantes: {1040 - len(images_df)}")
        print("Execute novamente o mesmo comando para continuar.")
        return

    # Saídas finais.
    images_df.to_csv(
        outdir / "ba_cat_v9_production_images.csv",
        index=False, encoding="utf-8-sig"
    )
    slots_df.to_csv(
        outdir / "ba_cat_v9_production_slots.csv",
        index=False, encoding="utf-8-sig"
    )
    candidates_df.to_csv(
        outdir / "ba_cat_v9_production_candidates.csv",
        index=False, encoding="utf-8-sig"
    )
    errors_df.to_csv(
        outdir / "ba_cat_v9_production_errors.csv",
        index=False, encoding="utf-8-sig"
    )

    audit = cat_v9.make_audit_all(images_df)
    audit.to_csv(
        outdir / "ba_cat_v9_production_audit_all.csv",
        index=False, encoding="utf-8-sig"
    )

    priority = cat_v9.make_priority_review(audit)
    priority.to_csv(
        outdir / "ba_cat_v9_production_priority_review.csv",
        index=False, encoding="utf-8-sig"
    )

    complete = images_df[
        images_df["categorical_axis_status"].eq("CATEGORICAL_COMPLETE")
    ].copy()
    complete.to_csv(
        outdir / "ba_cat_v9_production_complete_auto_candidates.csv",
        index=False, encoding="utf-8-sig"
    )

    status_counts = (
        images_df["categorical_axis_status"]
        .value_counts(dropna=False)
        .rename_axis("categorical_axis_status")
        .reset_index(name="n")
    )
    status_counts.to_csv(
        outdir / "ba_cat_v9_production_summary_status.csv",
        index=False, encoding="utf-8-sig"
    )

    lines = [
        "B-A-CAT V9 — PRODUÇÃO CONGELADA — 1.040 BI/BC",
        "=" * 78,
        f"Arquivo CAT V9: {cat_path}",
        f"CAT V9 SHA-256: {cat_hash}",
        f"B-A V7 SHA-256: {v7_hash}",
        f"V4 presença SHA-256: {presence_hash}",
        f"Tesseract: {t_version}",
        f"Idioma OCR: {lang}",
        "",
        "ESCOPO:",
        f"- imagens BI/BC: {len(prod)}",
        f"- processadas com sucesso: {len(images_df)}",
        f"- erros finais de execução: {len(errors_df)}",
        f"- revisão prioritária automática: {len(priority)}",
        f"- COMPLETE automáticos: {len(complete)}",
        "",
        "INDEPENDÊNCIA:",
        "- F não foi carregada;",
        "- categorias esperadas não foram fornecidas;",
        "- número esperado de categorias não foi fornecido;",
        "- ordem esperada não foi fornecida;",
        "",
        "STATUS AUTOMÁTICOS:",
    ]

    for _, rr in status_counts.iterrows():
        lines.append(
            f"- {rr['categorical_axis_status']}: {int(rr['n'])}"
        )

    summary_path = outdir / "ba_cat_v9_production_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 80)
    print("PRODUÇÃO CONCLUÍDA")
    print("=" * 80)
    print(f"Imagens: {len(images_df)}/1040")
    print(f"Erros finais: {len(errors_df)}")
    print("Resumo:", summary_path)
    print("Auditoria:", outdir / "ba_cat_v9_production_audit_all.csv")
    print("Prioridade:", outdir / "ba_cat_v9_production_priority_review.csv")


if __name__ == "__main__":
    main()
