# -*- coding: utf-8 -*-
r"""
mark_presence_full_v5.py

PROCESSAMENTO FINAL — PRESENÇA DA ESTRUTURA GRÁFICA PRINCIPAL
==============================================================

Aplica EXATAMENTE o detector V3 congelado às 3.120 imagens do experimento.

Princípio
---------
- B -> presença de barras
- L -> presença de linha(s)
- S -> presença de pontos de dispersão

O detector estrutural:
- não consulta as cores normativas de F;
- não usa número esperado de séries/marcas para forçar a decisão;
- usa apenas a técnica indicada pelo prefixo do arquivo para escolher
  o detector correspondente.

Validação prévia
----------------
A V3 foi calibrada e posteriormente testada em holdout independente.
Este programa verifica o SHA-256 do arquivo V3 para impedir que uma versão
alterada seja usada acidentalmente.

SHA-256 esperado de mark_presence_validation_v3.py:
8f33f4daa7e8f1cc8feba58e5ea5f80bc75639334f9e64cf6339408152bc78a2

Pastas padrão
-------------
Imagens:
C:\Users\Labvis\Downloads\imagens3120\imagens

Crosswalk experimental V3:
imagens\_experimental_crosswalk_v3

Saída:
imagens\_mark_presence_full_v5

Arquivos principais
-------------------
mark_presence_images_v5.csv
mark_presence_units_v5.csv
mark_presence_by_profile_v5.csv
mark_presence_by_condition_v5.csv
mark_presence_by_task_v5.csv
mark_presence_by_technique_v5.csv
mark_presence_errors_v5.csv
mark_presence_summary_v5.txt

Opcionalmente:
--save-absent-overlays
salva overlays apenas dos casos classificados como ausência estrutural.

Execução
--------
python mark_presence_full_v5.py
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    r"C:\Users\Labvis\Downloads\imagens3120\imagens"
)
DEFAULT_CROSSWALK_DIRNAME = "_experimental_crosswalk_v3"
DEFAULT_OUTPUT_DIRNAME = "_mark_presence_full_v5"

EXPECTED_V3_SHA256 = "8f33f4daa7e8f1cc8feba58e5ea5f80bc75639334f9e64cf6339408152bc78a2"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

NAME_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.(png|jpg|jpeg)$",
    re.I,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Processamento final da presença da estrutura gráfica principal."
    )

    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    p.add_argument(
        "--crosswalk-dir",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--save-absent-overlays",
        action="store_true",
        help="Salva overlays apenas dos casos classificados como ausência."
    )

    return p.parse_args()


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_frozen_v3():
    module_path = Path(__file__).with_name(
        "mark_presence_validation_v3.py"
    )

    if not module_path.exists():
        raise FileNotFoundError(
            "mark_presence_validation_v3.py não foi encontrado na mesma "
            "pasta do programa."
        )

    actual_hash = sha256_file(module_path)

    if actual_hash != EXPECTED_V3_SHA256:
        raise RuntimeError(
            "O arquivo mark_presence_validation_v3.py não corresponde à "
            "versão congelada validada.\n"
            f"Esperado: {EXPECTED_V3_SHA256}\n"
            f"Encontrado: {actual_hash}"
        )

    try:
        import mark_presence_validation_v3 as v3
    except Exception as exc:
        raise RuntimeError(
            "Falha ao importar mark_presence_validation_v3.py."
        ) from exc

    return v3, actual_hash


def inventory(root: Path):
    rows = []

    for p in sorted(root.iterdir()):
        if not (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ):
            continue

        m = NAME_RE.match(p.name)

        if not m:
            continue

        profile = m.group(1).upper()
        unit_number = int(m.group(2))
        repeat = int(m.group(3))
        unit_id = f"{profile}_{unit_number:03d}"

        rows.append({
            "path": str(p),
            "filename": p.name,
            "profile": profile,
            "technique": profile[0],
            "task": profile[1],
            "unit_id": unit_id,
            "unit_number": unit_number,
            "repeat": repeat,
        })

    return pd.DataFrame(rows)


def load_crosswalk(crosswalk_dir: Path):
    image_path = (
        crosswalk_dir
        / "experimental_image_crosswalk_v3.csv"
    )

    unit_path = (
        crosswalk_dir
        / "experimental_unit_crosswalk_v3.csv"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Crosswalk por imagem não encontrado: {image_path}"
        )

    if not unit_path.exists():
        raise FileNotFoundError(
            f"Crosswalk por unidade não encontrado: {unit_path}"
        )

    image_df = pd.read_csv(
        image_path,
        dtype={"condition_code": str}
    )

    unit_df = pd.read_csv(
        unit_path,
        dtype={"condition_code": str}
    )

    for df in (image_df, unit_df):
        if "condition_code" in df.columns:
            df["condition_code"] = (
                df["condition_code"]
                .astype(str)
                .str.zfill(2)
            )

    return image_df, unit_df


def process_all(
    inv: pd.DataFrame,
    v3,
    save_absent_overlays: bool,
    overlays_dir: Path,
):
    rows = []
    errors = []

    for idx, row in inv.iterrows():
        path = Path(row["path"])

        try:
            result = v3.analyze_one(
                path,
                row["technique"]
            )

            present = bool(
                result["det"]["present"]
            )

            raw_status = (
                "PRESENT"
                if present
                else "ABSENT_OR_UNCERTAIN"
            )

            conformity_status = (
                "MARK_PRESENT"
                if present
                else "MARK_ABSENT"
            )

            out = {
                "filename": row["filename"],
                "profile": row["profile"],
                "technique": row["technique"],
                "task": row["task"],
                "unit_id": row["unit_id"],
                "unit_number": int(row["unit_number"]),
                "repeat": int(row["repeat"]),

                "presence_status_raw": raw_status,
                "mark_present": int(present),
                "mark_absent": int(not present),
                "mark_conformity_status": conformity_status,

                "presence_confidence": float(
                    result["det"]["confidence"]
                ),
                "n_candidates_technique_specific": int(
                    result["n_candidates"]
                ),
                "evidence": result["det"]["evidence"],

                "roi_method": result["roi_method"],
                "roi_x1": int(result["roi"][0]),
                "roi_y1": int(result["roi"][1]),
                "roi_x2": int(result["roi"][2]),
                "roi_y2": int(result["roi"][3]),
            }

            rows.append(out)

            if (
                save_absent_overlays
                and not present
            ):
                overlay = v3.make_overlay(
                    result["image"],
                    row["technique"],
                    result["roi"],
                    result["det"],
                )

                overlay.save(
                    overlays_dir
                    / f"{path.stem}_absent_overlay.png"
                )

        except Exception as exc:
            errors.append({
                "filename": row["filename"],
                "profile": row["profile"],
                "unit_id": row["unit_id"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

        done = len(rows) + len(errors)

        if (
            done % 250 == 0
            or done == len(inv)
        ):
            print(
                f"Processadas: {done}/{len(inv)}"
            )

    return pd.DataFrame(rows), pd.DataFrame(errors)


def attach_image_metadata(
    image_results: pd.DataFrame,
    crosswalk_images: pd.DataFrame,
):
    meta_cols = [
        "filename",
        "participant_id",
        "condition_code",
        "condition_label",
        "with_examples",
        "with_structure",
        "association_method",
        "original_filename",
        "sequence_original",
    ]

    cols = [
        c for c in meta_cols
        if c in crosswalk_images.columns
    ]

    if "filename" not in cols:
        raise ValueError(
            "Crosswalk por imagem não contém filename."
        )

    return image_results.merge(
        crosswalk_images[cols],
        on="filename",
        how="left",
        validate="one_to_one",
    )


def aggregate_units(
    images: pd.DataFrame,
):
    rows = []

    for unit_id, g in images.groupby(
        "unit_id",
        sort=True
    ):
        first = g.iloc[0]

        present = pd.to_numeric(
            g["mark_present"],
            errors="coerce"
        )

        confidence = pd.to_numeric(
            g["presence_confidence"],
            errors="coerce"
        )

        rows.append({
            "unit_id": unit_id,
            "profile": first["profile"],
            "technique": first["technique"],
            "task": first["task"],
            "unit_number": int(first["unit_number"]),
            "n_repeats_found": len(g),

            "n_mark_present": int(present.sum()),
            "n_mark_absent": int(
                (present == 0).sum()
            ),
            "p_mark_present": float(
                present.mean()
            ),
            "p_mark_absent": float(
                (1 - present).mean()
            ),

            "presence_confidence_mean": float(
                confidence.mean()
            ),
            "presence_confidence_median": float(
                confidence.median()
            ),
            "presence_confidence_min": float(
                confidence.min()
            ),
        })

    return pd.DataFrame(rows)


def attach_unit_metadata(
    units: pd.DataFrame,
    crosswalk_units: pd.DataFrame,
):
    meta_cols = [
        "unit_id",
        "participant_id",
        "condition_code",
        "condition_label",
        "with_examples",
        "with_structure",
        "association_method",
        "original_filename",
        "sequence_original",
    ]

    cols = [
        c for c in meta_cols
        if c in crosswalk_units.columns
    ]

    if "unit_id" not in cols:
        raise ValueError(
            "Crosswalk por unidade não contém unit_id."
        )

    return units.merge(
        crosswalk_units[cols],
        on="unit_id",
        how="left",
        validate="one_to_one",
    )


def summarize_group(
    df: pd.DataFrame,
    group_col: str,
):
    if group_col not in df.columns:
        return pd.DataFrame()

    rows = []

    for key, g in df.groupby(
        group_col,
        dropna=False,
        sort=True
    ):
        present = pd.to_numeric(
            g["mark_present"],
            errors="coerce"
        )

        rows.append({
            group_col: key,
            "n_images": len(g),
            "n_mark_present": int(
                present.sum()
            ),
            "n_mark_absent": int(
                (present == 0).sum()
            ),
            "p_mark_present": float(
                present.mean()
            ),
            "p_mark_absent": float(
                (1 - present).mean()
            ),
            "presence_confidence_mean": float(
                pd.to_numeric(
                    g["presence_confidence"],
                    errors="coerce"
                ).mean()
            ),
        })

    return pd.DataFrame(rows)


def build_summary(
    root,
    v3_hash,
    images,
    units,
    errors,
):
    n = len(images)
    n_present = int(
        images["mark_present"].sum()
    )
    n_absent = int(
        images["mark_absent"].sum()
    )

    lines = [
        "PROCESSAMENTO FINAL — PRESENÇA DA ESTRUTURA GRÁFICA PRINCIPAL",
        "=" * 80,
        f"Pasta: {root}",
        f"Detector congelado: mark_presence_validation_v3.py",
        f"SHA-256 do detector: {v3_hash}",
        "",
        f"Imagens processadas: {n}",
        f"Erros: {len(errors)}",
        f"Unidades agregadas: {len(units)}",
        "",
        "RESULTADO GLOBAL:",
        f"MARK_PRESENT: {n_present}",
        f"MARK_ABSENT: {n_absent}",
        (
            f"p_mark_present: {n_present / n:.6f}"
            if n
            else "p_mark_present: NA"
        ),
        "",
        "POR PERFIL:",
    ]

    prof = summarize_group(
        images,
        "profile"
    )

    for _, r in prof.iterrows():
        lines.append(
            f"{r['profile']}: "
            f"{int(r['n_mark_present'])}/{int(r['n_images'])} "
            f"PRESENT "
            f"({float(r['p_mark_present']):.4%})"
        )

    cond = summarize_group(
        images,
        "condition_label"
    )

    if not cond.empty:
        lines.extend([
            "",
            "POR CONDIÇÃO:",
        ])

        for _, r in cond.iterrows():
            lines.append(
                f"{r['condition_label']}: "
                f"{int(r['n_mark_present'])}/{int(r['n_images'])} "
                f"PRESENT "
                f"({float(r['p_mark_present']):.4%})"
            )

    lines.extend([
        "",
        "AGREGAÇÃO POR UNIDADE:",
        "- n_mark_present",
        "- n_mark_absent",
        "- p_mark_present",
        "- p_mark_absent",
        "- confiança média, mediana e mínima",
        "",
        "INTERPRETAÇÃO:",
        "- MARK_PRESENT: a estrutura gráfica principal correspondente à técnica",
        "  solicitada foi detectada;",
        "- MARK_ABSENT: a estrutura gráfica principal correspondente à técnica",
        "  solicitada não foi detectada;",
        "- uma visualização pode conter outra forma de representação e ainda",
        "  ser MARK_ABSENT para a técnica solicitada.",
        "",
        "IMPORTANTE:",
        "- detector V3 aplicado sem alteração de parâmetros;",
        "- somente arquivos na raiz foram processados;",
        "- a decisão estrutural não usa a paleta normativa de F;",
        "- o número esperado de séries/marcas não é usado para forçar a decisão.",
    ])

    return "\n".join(lines)


def main():
    args = parse_args()

    root = args.root.expanduser().resolve()

    crosswalk_dir = (
        args.crosswalk_dir.expanduser().resolve()
        if args.crosswalk_dir is not None
        else root / DEFAULT_CROSSWALK_DIRNAME
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / DEFAULT_OUTPUT_DIRNAME
    )

    overlays_dir = (
        output_dir
        / "absent_overlays"
    )

    if not root.is_dir():
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {root}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.save_absent_overlays:
        overlays_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    v3, actual_hash = load_frozen_v3()

    inv = inventory(root)

    crosswalk_images, crosswalk_units = load_crosswalk(
        crosswalk_dir
    )

    print("=" * 84)
    print("PROCESSAMENTO FINAL — PRESENÇA DA ESTRUTURA GRÁFICA PRINCIPAL")
    print("=" * 84)
    print(f"Imagens canônicas encontradas: {len(inv)}")
    print(f"Detector V3 SHA-256: {actual_hash}")
    print(f"Crosswalk: {crosswalk_dir}")
    print(f"Saída: {output_dir}")
    print()

    if len(inv) != 3120:
        print(
            "AVISO: o inventário não contém exatamente 3.120 imagens."
        )

    image_results, errors = process_all(
        inv,
        v3,
        save_absent_overlays=bool(
            args.save_absent_overlays
        ),
        overlays_dir=overlays_dir,
    )

    image_results = attach_image_metadata(
        image_results,
        crosswalk_images,
    )

    units = aggregate_units(
        image_results
    )

    units = attach_unit_metadata(
        units,
        crosswalk_units,
    )

    # Auditoria do desenho.
    n_missing_condition_images = int(
        image_results["condition_code"]
        .isna()
        .sum()
    )

    n_missing_condition_units = int(
        units["condition_code"]
        .isna()
        .sum()
    )

    if n_missing_condition_images:
        raise RuntimeError(
            f"Há {n_missing_condition_images} imagens sem condição experimental."
        )

    if n_missing_condition_units:
        raise RuntimeError(
            f"Há {n_missing_condition_units} unidades sem condição experimental."
        )

    if not units.empty:
        bad_repeats = units[
            units["n_repeats_found"] != 10
        ]

        if len(bad_repeats):
            print(
                f"AVISO: {len(bad_repeats)} unidades não têm 10 repetições."
            )

    by_profile = summarize_group(
        image_results,
        "profile"
    )

    by_condition = summarize_group(
        image_results,
        "condition_label"
    )

    by_task = summarize_group(
        image_results,
        "task"
    )

    by_technique = summarize_group(
        image_results,
        "technique"
    )

    image_results.to_csv(
        output_dir / "mark_presence_images_v5.csv",
        index=False,
        encoding="utf-8-sig"
    )

    units.to_csv(
        output_dir / "mark_presence_units_v5.csv",
        index=False,
        encoding="utf-8-sig"
    )

    by_profile.to_csv(
        output_dir / "mark_presence_by_profile_v5.csv",
        index=False,
        encoding="utf-8-sig"
    )

    by_condition.to_csv(
        output_dir / "mark_presence_by_condition_v5.csv",
        index=False,
        encoding="utf-8-sig"
    )

    by_task.to_csv(
        output_dir / "mark_presence_by_task_v5.csv",
        index=False,
        encoding="utf-8-sig"
    )

    by_technique.to_csv(
        output_dir / "mark_presence_by_technique_v5.csv",
        index=False,
        encoding="utf-8-sig"
    )

    errors.to_csv(
        output_dir / "mark_presence_errors_v5.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = build_summary(
        root,
        actual_hash,
        image_results,
        units,
        errors,
    )

    summary_path = (
        output_dir
        / "mark_presence_summary_v5.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8"
    )

    print()
    print("=" * 84)
    print("RESULTADO")
    print("=" * 84)
    print(f"Imagens processadas: {len(image_results)}")
    print(f"Erros: {len(errors)}")
    print(f"Unidades: {len(units)}")
    print(
        f"MARK_PRESENT: {int(image_results['mark_present'].sum())}"
    )
    print(
        f"MARK_ABSENT: {int(image_results['mark_absent'].sum())}"
    )
    print()
    print("Abra primeiro:")
    print(summary_path)


if __name__ == "__main__":
    main()
