# -*- coding: utf-8 -*-
r"""
REVISOR MANUAL B-A CATEGÓRICO — PRODUÇÃO (280 IMAGENS)
======================================================

Objetivo:
Revisar visualmente, sem consultar F, as 280 imagens categóricas selecionadas
para B-B.

Entrada esperada:
C:\Users\Labvis\Downloads\imagens3120\
  FILA_REVISAO_BA_CAT_PRODUCAO_280.csv
  revisao_ba_cat_producao_280\...

Saídas:
C:\Users\Labvis\Downloads\imagens3120\
  REVISAO_MANUAL_BA_CAT_PRODUCAO_280_PROGRESS.csv
  REVISAO_MANUAL_BA_CAT_PRODUCAO_280_FINAL.csv

O progresso é salvo após CADA imagem.
Se o programa for fechado, basta executar novamente para continuar.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
QUEUE = ROOT / "FILA_REVISAO_BA_CAT_PRODUCAO_280.csv"
IMAGE_ROOT = ROOT / "revisao_ba_cat_producao_280"

PROGRESS = ROOT / "REVISAO_MANUAL_BA_CAT_PRODUCAO_280_PROGRESS.csv"
FINAL = ROOT / "REVISAO_MANUAL_BA_CAT_PRODUCAO_280_FINAL.csv"

ORIENTATION_VALUES = ["", "CORRECT", "WRONG", "UNEVALUABLE"]
EXTRACTION_VALUES = ["", "EXACT", "EQUIVALENT", "PARTIAL", "WRONG", "UNEVALUABLE"]

MANUAL_FIELDS = [
    "manual_visible_categories",
    "manual_orientation_result",
    "manual_extraction_result",
    "manual_missing_labels",
    "manual_false_labels",
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
    return (
        str(r.get("manual_extraction_result", "")).strip() != ""
        or str(r.get("manual_orientation_result", "")).strip() != ""
        or str(r.get("manual_visible_categories", "")).strip() != ""
    )

def expected_image_path(row):
    # Prefer separated folder created by preparar_revisao_ba_cat_producao_280.py
    p = IMAGE_ROOT / row["profile"] / row["categorical_axis_status"] / row["filename"]
    if p.exists():
        return p

    # Fallback to original 3,120 root
    p2 = ROOT / "imagens" / row["filename"]
    if p2.exists():
        return p2

    return p

class Reviewer(tk.Tk):
    def __init__(self, rows):
        super().__init__()

        self.title("Revisão manual B-A categórica — produção")
        self.geometry("1500x980")
        self.minsize(1200, 800)

        self.rows = rows
        self.index = self.first_unreviewed_index()
        self.photo = None

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.build_ui()
        self.load_current()

    def first_unreviewed_index(self):
        for i, r in enumerate(self.rows):
            if not is_reviewed(r):
                return i
        return 0

    def build_ui(self):
        # Top info
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self.lbl_progress = ttk.Label(top, text="", font=("Segoe UI", 11, "bold"))
        self.lbl_progress.pack(side="left")

        self.lbl_info = ttk.Label(top, text="", font=("Segoe UI", 10))
        self.lbl_info.pack(side="right")

        # Main panes
        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(main)
        right = ttk.Frame(main, width=460)
        main.add(left, weight=3)
        main.add(right, weight=2)

        # Image area
        img_frame = ttk.LabelFrame(left, text="Imagem")
        img_frame.pack(fill="both", expand=True)

        self.img_label = ttk.Label(img_frame, anchor="center")
        self.img_label.pack(fill="both", expand=True, padx=6, pady=6)

        img_buttons = ttk.Frame(left)
        img_buttons.pack(fill="x", pady=(4, 0))

        ttk.Button(
            img_buttons,
            text="Abrir imagem no visualizador do Windows",
            command=self.open_external
        ).pack(side="left")

        # Automatic extraction
        auto = ttk.LabelFrame(right, text="B-A automática (sem F)", padding=8)
        auto.pack(fill="x", pady=(0, 8))

        self.lbl_auto_status = ttk.Label(auto, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_auto_status.pack(anchor="w")

        ttk.Label(auto, text="Categorias extraídas automaticamente:").pack(anchor="w", pady=(6, 2))
        self.txt_auto = tk.Text(auto, height=5, wrap="word")
        self.txt_auto.pack(fill="x")
        self.txt_auto.configure(state="disabled")

        # Manual entry
        manual = ttk.LabelFrame(right, text="Adjudicação manual B-A", padding=8)
        manual.pack(fill="both", expand=True)

        ttk.Label(
            manual,
            text="Categorias visíveis, na ordem espacial (separar com |):"
        ).pack(anchor="w")
        self.txt_visible = tk.Text(manual, height=5, wrap="word")
        self.txt_visible.pack(fill="x", pady=(2, 8))

        row1 = ttk.Frame(manual)
        row1.pack(fill="x", pady=3)
        ttk.Label(row1, text="Orientação/eixo categórico:", width=26).pack(side="left")
        self.var_orientation = tk.StringVar()
        self.cmb_orientation = ttk.Combobox(
            row1, textvariable=self.var_orientation,
            values=ORIENTATION_VALUES, state="readonly", width=22
        )
        self.cmb_orientation.pack(side="left")

        row2 = ttk.Frame(manual)
        row2.pack(fill="x", pady=3)
        ttk.Label(row2, text="Resultado da extração:", width=26).pack(side="left")
        self.var_extraction = tk.StringVar()
        self.cmb_extraction = ttk.Combobox(
            row2, textvariable=self.var_extraction,
            values=EXTRACTION_VALUES, state="readonly", width=22
        )
        self.cmb_extraction.pack(side="left")

        ttk.Label(manual, text="Rótulos visíveis ausentes da extração automática:").pack(anchor="w", pady=(8, 2))
        self.txt_missing = tk.Text(manual, height=3, wrap="word")
        self.txt_missing.pack(fill="x")

        ttk.Label(manual, text="Falsos rótulos incluídos pela extração automática:").pack(anchor="w", pady=(8, 2))
        self.txt_false = tk.Text(manual, height=3, wrap="word")
        self.txt_false.pack(fill="x")

        ttk.Label(manual, text="Notas:").pack(anchor="w", pady=(8, 2))
        self.txt_notes = tk.Text(manual, height=4, wrap="word")
        self.txt_notes.pack(fill="x")

        rules = (
            "EXACT = leitura integral literal\n"
            "EQUIVALENT = apenas normalização benigna\n"
            "PARTIAL = parte correta, parte faltante\n"
            "WRONG = conteúdo/estrutura incorreta\n"
            "UNEVALUABLE = não é possível decidir\n\n"
            "Regra central: registrar somente o que está visível. NÃO consultar F."
        )
        ttk.Label(manual, text=rules, foreground="#555555").pack(anchor="w", pady=(8, 0))

        # Bottom navigation
        nav = ttk.Frame(self, padding=8)
        nav.pack(fill="x")

        ttk.Button(nav, text="◀ Anterior", command=self.previous).pack(side="left")
        ttk.Button(nav, text="Salvar", command=self.save_current).pack(side="left", padx=6)
        ttk.Button(nav, text="Salvar e próxima ▶", command=self.save_and_next).pack(side="left")

        ttk.Button(nav, text="Ir para próxima não revisada", command=self.jump_next_unreviewed).pack(side="left", padx=12)

        ttk.Button(nav, text="Exportar FINAL", command=self.export_final).pack(side="right")

        # Shortcuts
        self.bind("<Control-s>", lambda e: self.save_current())
        self.bind("<Control-Right>", lambda e: self.save_and_next())
        self.bind("<Control-Left>", lambda e: self.previous())

    def load_current(self):
        if not self.rows:
            return

        row = self.rows[self.index]
        reviewed = sum(is_reviewed(r) for r in self.rows)

        self.lbl_progress.config(
            text=f"Imagem {self.index + 1}/{len(self.rows)} | revisadas: {reviewed}/{len(self.rows)}"
        )
        self.lbl_info.config(
            text=f"{row['filename']} | {row['profile']} | {row['unit_id']} | {row['categorical_axis_status']}"
        )

        # Auto text
        auto = str(row.get("categorical_selected_labels_spatial", "") or "")
        if not auto.strip():
            auto = str(row.get("categorical_axis_observed", "") or "")

        self.txt_auto.configure(state="normal")
        self.txt_auto.delete("1.0", "end")
        self.txt_auto.insert("1.0", auto)
        self.txt_auto.configure(state="disabled")

        # Manual fields
        self.set_text(self.txt_visible, row.get("manual_visible_categories", ""))
        self.var_orientation.set(row.get("manual_orientation_result", ""))
        self.var_extraction.set(row.get("manual_extraction_result", ""))
        self.set_text(self.txt_missing, row.get("manual_missing_labels", ""))
        self.set_text(self.txt_false, row.get("manual_false_labels", ""))
        self.set_text(self.txt_notes, row.get("manual_notes", ""))

        self.load_image(expected_image_path(row))

    def set_text(self, widget, value):
        widget.delete("1.0", "end")
        widget.insert("1.0", str(value or ""))

    def get_text(self, widget):
        return widget.get("1.0", "end").strip()

    def load_image(self, path: Path):
        if not path.exists():
            self.img_label.configure(text=f"Imagem não encontrada:\n{path}", image="")
            self.photo = None
            return

        img = Image.open(path).convert("RGB")

        # Preserve readability while fitting screen.
        max_w = 930
        max_h = 720
        scale = min(max_w / img.width, max_h / img.height, 1.0)
        new_size = (
            max(1, int(img.width * scale)),
            max(1, int(img.height * scale))
        )
        if new_size != img.size:
            img = img.resize(new_size, Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.photo, text="")

    def open_external(self):
        row = self.rows[self.index]
        path = expected_image_path(row)
        if path.exists():
            os.startfile(path)
        else:
            messagebox.showerror("Erro", f"Imagem não encontrada:\n{path}")

    def persist_row_from_form(self):
        row = self.rows[self.index]
        row["manual_visible_categories"] = self.get_text(self.txt_visible)
        row["manual_orientation_result"] = self.var_orientation.get().strip()
        row["manual_extraction_result"] = self.var_extraction.get().strip()
        row["manual_missing_labels"] = self.get_text(self.txt_missing)
        row["manual_false_labels"] = self.get_text(self.txt_false)
        row["manual_notes"] = self.get_text(self.txt_notes)

    def validate_current(self):
        visible = self.get_text(self.txt_visible)
        orientation = self.var_orientation.get().strip()
        extraction = self.var_extraction.get().strip()

        if not extraction:
            messagebox.showwarning(
                "Campo obrigatório",
                "Selecione o resultado da extração."
            )
            return False

        if not orientation:
            messagebox.showwarning(
                "Campo obrigatório",
                "Selecione o resultado da orientação/eixo categórico."
            )
            return False

        if extraction != "UNEVALUABLE" and not visible:
            messagebox.showwarning(
                "Campo obrigatório",
                "Transcreva as categorias visíveis ou marque UNEVALUABLE."
            )
            return False

        return True

    def save_current(self, silent=False):
        if not self.validate_current():
            return False

        self.persist_row_from_form()
        write_csv(PROGRESS, self.rows)

        if not silent:
            messagebox.showinfo(
                "Salvo",
                f"Progresso salvo:\n{PROGRESS}"
            )
        return True

    def save_and_next(self):
        if not self.save_current(silent=True):
            return
        if self.index < len(self.rows) - 1:
            self.index += 1
            self.load_current()
        else:
            self.export_final()

    def previous(self):
        self.persist_row_from_form()
        write_csv(PROGRESS, self.rows)

        if self.index > 0:
            self.index -= 1
            self.load_current()

    def jump_next_unreviewed(self):
        self.persist_row_from_form()
        write_csv(PROGRESS, self.rows)

        for offset in range(1, len(self.rows) + 1):
            j = (self.index + offset) % len(self.rows)
            if not is_reviewed(self.rows[j]):
                self.index = j
                self.load_current()
                return

        messagebox.showinfo("Revisão", "Todas as imagens já possuem alguma revisão.")

    def export_final(self):
        self.persist_row_from_form()
        write_csv(PROGRESS, self.rows)

        missing = [
            r["filename"]
            for r in self.rows
            if not (
                str(r.get("manual_extraction_result", "")).strip()
                and str(r.get("manual_orientation_result", "")).strip()
            )
        ]

        if missing:
            messagebox.showwarning(
                "Revisão incompleta",
                f"Ainda faltam {len(missing)} imagens.\n"
                "O progresso foi salvo, mas o arquivo FINAL não foi criado."
            )
            return

        write_csv(FINAL, self.rows)
        messagebox.showinfo(
            "Concluído",
            f"Revisão completa.\n\nArquivo FINAL:\n{FINAL}"
        )

    def on_close(self):
        # Save current form without forcing validation.
        self.persist_row_from_form()
        write_csv(PROGRESS, self.rows)
        self.destroy()

def main():
    if not QUEUE.exists():
        print("ERRO: fila não encontrada:", QUEUE)
        return 1

    base = [normalize_row(r) for r in read_csv(QUEUE)]

    if PROGRESS.exists():
        progress_rows = read_csv(PROGRESS)
        progress_by_file = {r["filename"]: r for r in progress_rows}
        for r in base:
            p = progress_by_file.get(r["filename"])
            if p:
                for f in MANUAL_FIELDS:
                    r[f] = p.get(f, r.get(f, ""))
        print(f"Progresso anterior carregado: {PROGRESS}")

    print("=" * 72)
    print("REVISOR MANUAL B-A CATEGÓRICO — PRODUÇÃO")
    print("=" * 72)
    print("Imagens:", len(base))
    print("Já revisadas:", sum(is_reviewed(r) for r in base))
    print("Fila:", QUEUE)
    print("Progresso:", PROGRESS)
    print()

    app = Reviewer(base)
    app.mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
