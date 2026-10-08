# -*- coding: utf-8 -*-
r"""
REVISÃO CEGA DO HOLDOUT — CLASSIFICADOR EXCLUSIVO DE TÉCNICA — 60 CASOS
========================================================================

Objetivo:
validar visualmente a classe ESTRUTURAL observada no holdout independente
usado por chart_type_exclusive_holdout_v2.py.

IMPORTANTE
----------
Durante a revisão, o programa NÃO mostra:
- profile;
- expected_type;
- expected_technique;
- observed_type automático;
- filename.

Assim, a decisão manual é cega tanto para F quanto para a predição automática.

Classes manuais:
BAR
LINE
SCATTER
OTHER
EMPTY_NO_DATA
UNCERTAIN

Observação:
- O classificador automático só possui BAR, LINE, SCATTER e OTHER.
- EMPTY_NO_DATA será tratado posteriormente como pertencente à classe
  automática OTHER para avaliar o classificador de 4 classes.
- Se você reconhecer uma estrutura específica não contemplada, por exemplo
  STEM, use OTHER e registre "STEM" em manual_notes.

Entrada:
C:\Users\Labvis\Downloads\imagens3120\imagens\
  _chart_type_exclusive_holdout_v2\chart_type_exclusive_holdout_v2.csv

Imagens:
C:\Users\Labvis\Downloads\imagens3120\imagens\*.png

Saídas:
C:\Users\Labvis\Downloads\imagens3120\
  REVISAO_MANUAL_HOLDOUT_TECNICA_60_PROGRESS.csv
  REVISAO_MANUAL_HOLDOUT_TECNICA_60_FINAL.csv
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES = ROOT / "imagens"
QUEUE = (
    IMAGES
    / "_chart_type_exclusive_holdout_v2"
    / "chart_type_exclusive_holdout_v2.csv"
)

PROGRESS = ROOT / "REVISAO_MANUAL_HOLDOUT_TECNICA_60_PROGRESS.csv"
FINAL = ROOT / "REVISAO_MANUAL_HOLDOUT_TECNICA_60_FINAL.csv"

CLASS_VALUES = [
    "",
    "BAR",
    "LINE",
    "SCATTER",
    "OTHER",
    "EMPTY_NO_DATA",
    "UNCERTAIN",
]

MANUAL_FIELDS = [
    "manual_visible_type",
    "manual_notes",
]

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def normalize_row(r):
    for f in MANUAL_FIELDS:
        r.setdefault(f, "")
    return r

def is_reviewed(r):
    return str(r.get("manual_visible_type", "")).strip() != ""

def image_path(r):
    return IMAGES / Path(r["filename"]).name

class Reviewer(tk.Tk):
    def __init__(self, rows):
        super().__init__()
        self.title("Revisão cega — holdout técnica exclusiva — 60 casos")
        self.geometry("1400x900")
        self.minsize(1100, 760)

        self.rows = rows
        self.index = self.first_unreviewed()
        self.photo = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.load_current()

    def first_unreviewed(self):
        for i, r in enumerate(self.rows):
            if not is_reviewed(r):
                return i
        return 0

    def build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self.lbl_progress = ttk.Label(
            top, text="", font=("Segoe UI", 11, "bold")
        )
        self.lbl_progress.pack(side="left")

        self.lbl_blind = ttk.Label(
            top,
            text="Revisão cega: classe esperada e classe automática ocultas",
            font=("Segoe UI", 10),
        )
        self.lbl_blind.pack(side="right")

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(main)
        right = ttk.Frame(main, width=430)
        main.add(left, weight=3)
        main.add(right, weight=2)

        imgf = ttk.LabelFrame(left, text="Imagem")
        imgf.pack(fill="both", expand=True)

        self.img_label = ttk.Label(imgf, anchor="center")
        self.img_label.pack(fill="both", expand=True, padx=6, pady=6)

        manual = ttk.LabelFrame(
            right, text="Classificação visual manual", padding=10
        )
        manual.pack(fill="both", expand=True)

        ttk.Label(
            manual,
            text="Qual é a técnica/estrutura principal visível?",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        self.var_class = tk.StringVar()
        self.cmb = ttk.Combobox(
            manual,
            textvariable=self.var_class,
            values=CLASS_VALUES,
            state="readonly",
            width=24,
        )
        self.cmb.pack(anchor="w", pady=(8, 16))

        ttk.Label(manual, text="Notas:").pack(anchor="w")
        self.txt_notes = tk.Text(manual, height=7, wrap="word")
        self.txt_notes.pack(fill="x", pady=(2, 12))

        rules = (
            "BAR: barras são a marca principal.\n"
            "LINE: trajetória/linha é a marca principal.\n"
            "SCATTER: pontos dispersos são a marca principal.\n"
            "OTHER: outra estrutura visual principal.\n"
            "EMPTY_NO_DATA: sem estrutura de dados principal visível.\n"
            "UNCERTAIN: não é possível decidir com segurança.\n\n"
            "Se reconhecer STEM, selecione OTHER e registre STEM nas notas."
        )
        ttk.Label(
            manual,
            text=rules,
            foreground="#555555",
            justify="left",
        ).pack(anchor="w")

        nav = ttk.Frame(self, padding=8)
        nav.pack(fill="x")

        ttk.Button(nav, text="◀ Anterior", command=self.previous).pack(side="left")
        ttk.Button(nav, text="Salvar", command=self.save_current).pack(
            side="left", padx=6
        )
        ttk.Button(nav, text="Salvar e próxima ▶", command=self.save_next).pack(
            side="left"
        )
        ttk.Button(
            nav,
            text="Próxima não revisada",
            command=self.next_unreviewed
        ).pack(side="left", padx=12)
        ttk.Button(nav, text="Exportar FINAL", command=self.export_final).pack(
            side="right"
        )

        self.bind("<Control-s>", lambda e: self.save_current())
        self.bind("<Control-Right>", lambda e: self.save_next())
        self.bind("<Control-Left>", lambda e: self.previous())

    def load_image(self, path: Path):
        if not path.exists():
            self.img_label.configure(
                text=f"Imagem não encontrada:\n{path}",
                image=""
            )
            self.photo = None
            return

        img = Image.open(path).convert("RGB")
        max_w, max_h = 900, 700
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        size = (
            max(1, int(img.width * scale)),
            max(1, int(img.height * scale)),
        )
        if size != img.size:
            img = img.resize(size, Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.photo, text="")

    def load_current(self):
        r = self.rows[self.index]
        reviewed = sum(is_reviewed(x) for x in self.rows)

        self.lbl_progress.config(
            text=f"Caso {self.index+1}/{len(self.rows)} | "
                 f"revisados: {reviewed}/{len(self.rows)}"
        )

        self.var_class.set(r.get("manual_visible_type", ""))
        self.txt_notes.delete("1.0", "end")
        self.txt_notes.insert("1.0", r.get("manual_notes", ""))

        self.load_image(image_path(r))

    def persist(self):
        r = self.rows[self.index]
        r["manual_visible_type"] = self.var_class.get().strip()
        r["manual_notes"] = self.txt_notes.get("1.0", "end").strip()

    def validate(self):
        if not self.var_class.get().strip():
            messagebox.showwarning(
                "Campo obrigatório",
                "Selecione a classe visual manual."
            )
            return False
        return True

    def save_current(self, silent=False):
        if not self.validate():
            return False
        self.persist()
        write_csv(PROGRESS, self.rows)
        if not silent:
            messagebox.showinfo(
                "Salvo",
                f"Progresso salvo:\n{PROGRESS}"
            )
        return True

    def save_next(self):
        if not self.save_current(True):
            return
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.load_current()
        else:
            self.export_final()

    def previous(self):
        self.persist()
        write_csv(PROGRESS, self.rows)
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def next_unreviewed(self):
        self.persist()
        write_csv(PROGRESS, self.rows)

        for off in range(1, len(self.rows) + 1):
            j = (self.index + off) % len(self.rows)
            if not is_reviewed(self.rows[j]):
                self.index = j
                self.load_current()
                return

        messagebox.showinfo(
            "Revisão",
            "Todos os 60 casos já foram revisados."
        )

    def export_final(self):
        self.persist()
        write_csv(PROGRESS, self.rows)

        missing = [
            i + 1 for i, r in enumerate(self.rows)
            if not is_reviewed(r)
        ]
        if missing:
            messagebox.showwarning(
                "Revisão incompleta",
                f"Ainda faltam {len(missing)} casos."
            )
            return

        write_csv(FINAL, self.rows)
        messagebox.showinfo(
            "Concluído",
            f"Arquivo FINAL:\n{FINAL}"
        )

    def on_close(self):
        self.persist()
        write_csv(PROGRESS, self.rows)
        self.destroy()

def main():
    if not QUEUE.exists():
        print("ERRO: holdout não encontrado:")
        print(QUEUE)
        return 1

    rows = [normalize_row(r) for r in read_csv(QUEUE)]

    if len(rows) != 60:
        print(
            f"ATENÇÃO: esperadas 60 imagens no holdout, "
            f"encontradas {len(rows)}."
        )

    if PROGRESS.exists():
        old_rows = read_csv(PROGRESS)
        old = {r["filename"]: r for r in old_rows}
        for r in rows:
            p = old.get(r["filename"])
            if p:
                for f in MANUAL_FIELDS:
                    r[f] = p.get(f, r.get(f, ""))

    print("=" * 76)
    print("REVISÃO CEGA — HOLDOUT DO CLASSIFICADOR EXCLUSIVO DE TÉCNICA")
    print("=" * 76)
    print("Casos:", len(rows))
    print("Holdout:", QUEUE)
    print("Progresso:", PROGRESS)
    print()
    print("A interface NÃO mostra expected_type nem observed_type automático.")

    app = Reviewer(rows)
    app.mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
