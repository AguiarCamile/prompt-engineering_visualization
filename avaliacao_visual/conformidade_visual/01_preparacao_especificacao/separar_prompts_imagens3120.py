# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path("C:/Users/Labvis/Downloads/imagens3120")
DEFAULT_INPUT = DEFAULT_ROOT / "prompts_imagens.jsonl"
DEFAULT_OUTPUT = DEFAULT_ROOT / "prompts"

UNIT_RE = re.compile(r"^(BI|BC|LI|LC|SI|SC)_(\d{3})$")
IMAGE_RE = re.compile(
    r"^(BI|BC|LI|LC|SI|SC)_(\d{3})_R(\d{2})\.png$",
    re.IGNORECASE,
)

EXPECTED_TOTAL = 312
EXPECTED_PER_PROFILE = 52
EXPECTED_PROFILES = ("BI", "BC", "LI", "LC", "SI", "SC")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Separa prompts_imagens.jsonl em arquivos TXT por unit_id."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescreve TXT existente apenas se o conteúdo for diferente.",
    )
    parser.add_argument(
        "--allow-nonstandard-count",
        action="store_true",
        help="Permite executar mesmo se o arquivo não tiver 312 unidades/52 por perfil.",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSONL não encontrado: {path}")

    records = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON inválido na linha {line_number}: {exc}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Linha {line_number}: cada linha deve conter um objeto JSON."
                )

            obj["_source_line"] = line_number
            records.append(obj)

    return records


def validate_record(rec: dict):
    line_number = rec["_source_line"]

    unit_id = str(rec.get("unit_id", "")).strip()
    if not UNIT_RE.fullmatch(unit_id):
        raise ValueError(
            f"Linha {line_number}: unit_id inválido: {unit_id!r}"
        )

    prompt_text = rec.get("prompt_text")
    if not isinstance(prompt_text, str):
        raise ValueError(
            f"Linha {line_number} / {unit_id}: prompt_text ausente ou inválido."
        )

    expected_prompt_name = f"{unit_id}.txt"
    prompt_file = rec.get("prompt_file")

    if prompt_file not in (None, ""):
        actual_prompt_name = Path(str(prompt_file)).name
        if actual_prompt_name != expected_prompt_name:
            raise ValueError(
                f"Linha {line_number} / {unit_id}: "
                f"prompt_file={actual_prompt_name!r}, "
                f"esperado={expected_prompt_name!r}."
            )

    images = rec.get("images", [])
    if images is None:
        images = []

    if not isinstance(images, list):
        raise ValueError(
            f"Linha {line_number} / {unit_id}: images deve ser uma lista."
        )

    repeats = set()

    for image in images:
        image_name = Path(str(image)).name
        match = IMAGE_RE.fullmatch(image_name)

        if not match:
            raise ValueError(
                f"Linha {line_number} / {unit_id}: "
                f"nome de imagem inválido: {image_name!r}"
            )

        image_unit_id = f"{match.group(1).upper()}_{match.group(2)}"
        if image_unit_id != unit_id:
            raise ValueError(
                f"Linha {line_number} / {unit_id}: "
                f"{image_name} pertence a {image_unit_id}."
            )

        repeat = int(match.group(3))
        if repeat in repeats:
            raise ValueError(
                f"Linha {line_number} / {unit_id}: "
                f"repetição R{repeat:02d} duplicada."
            )

        repeats.add(repeat)

    return unit_id, prompt_text, images


def main():
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    records = load_jsonl(input_path)

    seen_units = set()
    profile_counts = Counter()
    validated = []

    for rec in records:
        unit_id, prompt_text, images = validate_record(rec)

        if unit_id in seen_units:
            raise ValueError(f"unit_id duplicado: {unit_id}")

        seen_units.add(unit_id)

        profile = unit_id.split("_", 1)[0]
        profile_counts[profile] += 1

        validated.append(
            {
                "unit_id": unit_id,
                "prompt_text": prompt_text,
                "images": images,
                "record": rec,
            }
        )

    if not args.allow_nonstandard_count:
        if len(validated) != EXPECTED_TOTAL:
            raise RuntimeError(
                f"Foram encontradas {len(validated)} unidades; "
                f"esperadas {EXPECTED_TOTAL}."
            )

        wrong_counts = [
            f"{profile}={profile_counts.get(profile, 0)}"
            for profile in EXPECTED_PROFILES
            if profile_counts.get(profile, 0) != EXPECTED_PER_PROFILE
        ]

        if wrong_counts:
            raise RuntimeError(
                "Contagem inesperada por perfil: " + ", ".join(wrong_counts)
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    created = 0
    overwritten = 0
    unchanged = 0

    for item in sorted(validated, key=lambda x: x["unit_id"]):
        unit_id = item["unit_id"]
        prompt_text = item["prompt_text"]
        rec = item["record"]
        images = item["images"]

        output_file = output_dir / f"{unit_id}.txt"

        # O arquivo TXT contém SOMENTE o prompt_text original.
        new_bytes = prompt_text.encode("utf-8")
        new_hash = sha256_bytes(new_bytes)

        action = "CREATED"

        if output_file.exists():
            old_bytes = output_file.read_bytes()
            old_hash = sha256_bytes(old_bytes)

            if old_hash == new_hash:
                action = "UNCHANGED"
                unchanged += 1
            else:
                if not args.overwrite:
                    raise FileExistsError(
                        f"O arquivo já existe com conteúdo diferente: {output_file}\n"
                        "Rode novamente com --overwrite se quiser substituí-lo."
                    )

                output_file.write_bytes(new_bytes)
                action = "OVERWRITTEN"
                overwritten += 1
        else:
            output_file.write_bytes(new_bytes)
            created += 1

        image_names = [Path(str(x)).name for x in images]

        manifest_rows.append(
            {
                "unit_id": unit_id,
                "prompt_file": output_file.name,
                "participant_id": rec.get("participant_id", ""),
                "condition": rec.get("condition", ""),
                "type": rec.get("type", ""),
                "task": rec.get("task", ""),
                "n_images": len(image_names),
                "first_image": image_names[0] if image_names else "",
                "last_image": image_names[-1] if image_names else "",
                "prompt_sha256": new_hash,
                "action": action,
            }
        )

    manifest_path = output_dir / "prompts_manifest.csv"

    manifest_fields = [
        "unit_id",
        "prompt_file",
        "participant_id",
        "condition",
        "type",
        "task",
        "n_images",
        "first_image",
        "last_image",
        "prompt_sha256",
        "action",
    ]

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    generated_units = sorted(
        p.stem
        for p in output_dir.glob("*.txt")
        if UNIT_RE.fullmatch(p.stem)
    )

    missing = sorted(seen_units - set(generated_units))
    unexpected = sorted(set(generated_units) - seen_units)

    summary_path = output_dir / "resumo_prompts.txt"

    summary_lines = [
        "SEPARAÇÃO DOS PROMPTS",
        "=" * 72,
        f"Entrada: {input_path}",
        f"Saída: {output_dir}",
        "",
        f"Registros JSONL: {len(validated)}",
        f"TXT de unidade na pasta: {len(generated_units)}",
        f"Criados nesta execução: {created}",
        f"Sobrescritos nesta execução: {overwritten}",
        f"Já idênticos: {unchanged}",
        "",
        "CONTAGEM POR PERFIL:",
    ]

    for profile in EXPECTED_PROFILES:
        summary_lines.append(
            f"- {profile}: {profile_counts.get(profile, 0)}"
        )

    summary_lines.extend(
        [
            "",
            f"unit_id ausentes: {len(missing)}",
            f"unit_id inesperados: {len(unexpected)}",
            "",
            "REGRA DE VINCULAÇÃO:",
            "BI_001.txt -> BI_001_R01.png ... BI_001_R10.png",
            "BC_001.txt -> BC_001_R01.png ... BC_001_R10.png",
            "...",
            "",
            "IMPORTANTE:",
            "- cada TXT contém somente o prompt_text original;",
            "- metadados ficam no prompts_manifest.csv;",
            "- o conteúdo do prompt não é normalizado nem reescrito.",
        ]
    )

    if missing:
        summary_lines.append("Ausentes: " + ", ".join(missing))

    if unexpected:
        summary_lines.append("Inesperados: " + ", ".join(unexpected))

    summary_path.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    if missing or unexpected:
        raise RuntimeError(
            f"A verificação final encontrou divergências. Veja {summary_path}"
        )

    print("=" * 72)
    print("PROMPTS SEPARADOS COM SUCESSO")
    print("=" * 72)
    print(f"Entrada: {input_path}")
    print(f"Saída: {output_dir}")
    print(f"Total de prompts: {len(validated)}")
    print(f"Criados: {created}")
    print(f"Sobrescritos: {overwritten}")
    print(f"Já idênticos: {unchanged}")
    print()
    for profile in EXPECTED_PROFILES:
        print(f"{profile}: {profile_counts.get(profile, 0)}")
    print()
    print(f"Manifesto: {manifest_path}")
    print(f"Resumo: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        sys.exit(1)
