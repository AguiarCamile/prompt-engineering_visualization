# -*- coding: utf-8 -*-
r"""
EXPORTAR CÓDIGO DO CLASSIFICADOR DE TÉCNICA — V1
=================================================

Cria um único arquivo TXT contendo o código-fonte completo dos principais
scripts relacionados à classificação de técnica gráfica e suas dependências
locais importadas.

Ponto de partida:
- chart_type_multiclass_validation_v1.py
- chart_type_exclusive_holdout_v2.py
- audit_absent_chart_types_v1.py

Também tenta incluir dependências .py locais referenciadas via import/from,
procurando em:
- raiz do projeto
- implementações
- implementações\testes
- subpastas locais do projeto

Ignora:
- .venv
- __pycache__
- site-packages

Saída:
  CODIGO_TECNICA_PARA_REVISAO_V1.txt

O script NÃO executa o classificador e NÃO altera resultados.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
OUT = ROOT / "CODIGO_TECNICA_PARA_REVISAO_V1.txt"

SEEDS = [
    ROOT / "chart_type_multiclass_validation_v1.py",
    ROOT / "chart_type_exclusive_holdout_v2.py",
    ROOT / "audit_absent_chart_types_v1.py",
]

IGNORE_PARTS = {
    ".venv", "__pycache__", "site-packages",
}

def should_ignore(path: Path) -> bool:
    return any(part.lower() in IGNORE_PARTS for part in path.parts)

def all_local_python_files():
    files = []
    for p in ROOT.rglob("*.py"):
        if should_ignore(p):
            continue
        files.append(p)
    return files

def build_module_index(files):
    idx = {}
    for p in files:
        stem = p.stem
        idx.setdefault(stem, []).append(p)

        try:
            rel = p.relative_to(ROOT).with_suffix("")
            dotted = ".".join(rel.parts)
            idx.setdefault(dotted, []).append(p)
        except Exception:
            pass
    return idx

def imported_modules(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except Exception:
        return []

    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module)
    return mods

def resolve_module(mod, index):
    candidates = []
    if mod in index:
        candidates.extend(index[mod])

    last = mod.split(".")[-1]
    if last in index:
        candidates.extend(index[last])

    # Deduplica e prefere arquivos mais próximos da raiz/implementações.
    unique = []
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)

    unique.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return unique

def main():
    files = all_local_python_files()
    index = build_module_index(files)

    missing_seeds = [p for p in SEEDS if not p.exists()]
    if missing_seeds:
        print("ATENÇÃO — arquivos iniciais não encontrados:")
        for p in missing_seeds:
            print(" -", p)

    queue = [p for p in SEEDS if p.exists()]
    included = []
    seen = set()

    while queue:
        p = queue.pop(0)
        if p in seen:
            continue
        seen.add(p)
        included.append(p)

        for mod in imported_modules(p):
            resolved = resolve_module(mod, index)
            # Só inclui módulos locais do projeto; módulos externos não resolvem
            # para um .py sob ROOT.
            for rp in resolved:
                if rp not in seen:
                    queue.append(rp)

    lines = []
    lines.append("=" * 100)
    lines.append("PACOTE DE CÓDIGO — CLASSIFICAÇÃO DE TÉCNICA GRÁFICA")
    lines.append("=" * 100)
    lines.append(f"Raiz: {ROOT}")
    lines.append(f"Arquivos incluídos: {len(included)}")
    lines.append("")
    for i,p in enumerate(included,1):
        try:
            rel = p.relative_to(ROOT)
        except Exception:
            rel = p
        lines.append(f"{i:02d}. {rel}")
    lines.append("")

    for p in included:
        try:
            rel = p.relative_to(ROOT)
        except Exception:
            rel = p

        lines.append("")
        lines.append("=" * 100)
        lines.append(f"ARQUIVO: {rel}")
        lines.append("=" * 100)
        lines.append("")
        try:
            lines.append(p.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            lines.append(f"[ERRO AO LER ARQUIVO: {exc}]")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")

    print("="*80)
    print("EXPORTAÇÃO DO CÓDIGO DA TÉCNICA")
    print("="*80)
    print("Arquivos incluídos:", len(included))
    for p in included:
        print(" -", p)
    print()
    print("Saída:", OUT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
