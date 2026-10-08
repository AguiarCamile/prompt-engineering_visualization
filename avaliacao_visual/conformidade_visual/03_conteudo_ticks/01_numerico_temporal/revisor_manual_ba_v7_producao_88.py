# -*- coding: utf-8 -*-
r"""
REVISOR MANUAL B-A V7 — PRODUÇÃO (88 EIXOS)
===========================================

Objetivo:
Revisar visualmente, sem consultar F, os 88 eixos da produção V7 que ficaram
no gate manual.

Entrada esperada:
C:\Users\Labvis\Downloads\imagens3120\
  FILA_REVISAO_BA_V7_PRODUCAO_88.csv

Imagens:
C:\Users\Labvis\Downloads\imagens3120\imagens\*.png

Saídas:
C:\Users\Labvis\Downloads\imagens3120\
  REVISAO_MANUAL_BA_V7_PRODUCAO_88_PROGRESS.csv
  REVISAO_MANUAL_BA_V7_PRODUCAO_88_FINAL.csv

O progresso é salvo após cada eixo. O mesmo PNG pode aparecer duas vezes
quando X e Y precisam de revisão separada.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
QUEUE = ROOT / "FILA_REVISAO_BA_V7_PRODUCAO_88.csv"
IMAGES = ROOT / "imagens"

PROGRESS = ROOT / "REVISAO_MANUAL_BA_V7_PRODUCAO_88_PROGRESS.csv"
FINAL = ROOT / "REVISAO_MANUAL_BA_V7_PRODUCAO_88_FINAL.csv"

RESULT_VALUES = ["", "EXACT", "EQUIVALENT", "PARTIAL", "WRONG", "UNEVALUABLE"]

MANUAL_FIELDS = [
    "manual_ticklabel_extraction_result",
    "manual_visible_ticklabels",
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
    return str(r.get("manual_ticklabel_extraction_result", "")).strip() != ""

def image_path(row):
    return IMAGES / row["filename"]

def auto_observed_text(row):
    kind = row.get("kind", "")
    lines = []

    if kind == "temporal":
        years = row.get("temporal_years_spatial", "")
        if years:
            lines.append("temporal_years_spatial: " + years)

    if kind == "numeric":
        eff = row.get("numeric_effective_values_spatial", "")
        sel = row.get("numeric_selected_values_spatial", "")
        raw = row.get("numeric_lane_ocr_raw_tokens", "")

        if eff:
            lines.append("numeric_effective_values_spatial: " + eff)
        if sel:
            lines.append("numeric_selected_values_spatial: " + sel)
        if raw:
            lines.append("numeric_lane_ocr_raw_tokens: " + raw)

    ind = row.get("individual_text_candidate", "")
    band = row.get("band_ocr_normalized", "")
    if ind:
        lines.append("individual_text_candidate: " + ind)
    if band:
        lines.append("band_ocr_normalized: " + band)

    if not lines:
        lines.append("(sem conteúdo automático utilizável)")

    return "\n".join(lines)

class Reviewer(tk.Tk):
    def __init__(self, rows):
        super().__init__()
        self.title("Revisão manual B-A V7 — produção")
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
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")

        self.lbl_progress = ttk.Label(top, text="", font=("Segoe UI", 11, "bold"))
        self.lbl_progress.pack(side="left")

        self.lbl_info = ttk.Label(top, text="", font=("Segoe UI", 10))
        self.lbl_info.pack(side="right")

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(main)
        right = ttk.Frame(main, width=480)
        main.add(left, weight=3)
        main.add(right, weight=2)

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

        auto = ttk.LabelFrame(right, text="B-A automática V7 (sem F)", padding=8)
        auto.pack(fill="x", pady=(0,8))

        self.lbl_auto_status = ttk.Label(auto, text="", font=("Segoe UI", 10, "bold"))
        self.lbl_auto_status.pack(anchor="w")

        self.txt_auto = tk.Text(auto, height=9, wrap="word")
        self.txt_auto.pack(fill="x", pady=(6,0))
        self.txt_auto.configure(state="disabled")

        manual = ttk.LabelFrame(right, text="Adjudicação manual B-A", padding=8)
        manual.pack(fill="both", expand=True)

        ttk.Label(
            manual,
            text="Tick labels VISÍVEIS no eixo indicado, na ordem espacial (separar com |):"
        ).pack(anchor="w")
        self.txt_visible = tk.Text(manual, height=5, wrap="word")
        self.txt_visible.pack(fill="x", pady=(2,8))

        row = ttk.Frame(manual)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Resultado da extração:", width=25).pack(side="left")
        self.var_result = tk.StringVar()
        self.cmb_result = ttk.Combobox(
            row, textvariable=self.var_result,
            values=RESULT_VALUES, state="readonly", width=22
        )
        self.cmb_result.pack(side="left")

        ttk.Label(manual, text="Rótulos visíveis ausentes da extração automática:").pack(anchor="w", pady=(8,2))
        self.txt_missing = tk.Text(manual, height=3, wrap="word")
        self.txt_missing.pack(fill="x")

        ttk.Label(manual, text="Falsos rótulos incluídos pela extração automática:").pack(anchor="w", pady=(8,2))
        self.txt_false = tk.Text(manual, height=3, wrap="word")
        self.txt_false.pack(fill="x")

        ttk.Label(manual, text="Notas:").pack(anchor="w", pady=(8,2))
        self.txt_notes = tk.Text(manual, height=4, wrap="word")
        self.txt_notes.pack(fill="x")

        rules = (
            "EXACT = todos os tick labels visíveis recuperados integralmente\n"
            "EQUIVALENT = somente normalização benigna\n"
            "PARTIAL = parte dos ticks foi recuperada corretamente\n"
            "WRONG = conteúdo incorreto, falso tick ou troca de valor\n"
            "UNEVALUABLE = não é possível adjudicar com segurança\n\n"
            "Regra central: olhar apenas para a imagem e para B-A. NÃO consultar F."
        )
        ttk.Label(manual, text=rules, foreground="#555555").pack(anchor="w", pady=(8,0))

        nav = ttk.Frame(self, padding=8)
        nav.pack(fill="x")

        ttk.Button(nav, text="◀ Anterior", command=self.previous).pack(side="left")
        ttk.Button(nav, text="Salvar", command=self.save_current).pack(side="left", padx=6)
        ttk.Button(nav, text="Salvar e próxima ▶", command=self.save_and_next).pack(side="left")
        ttk.Button(nav, text="Próxima não revisada", command=self.jump_next_unreviewed).pack(side="left", padx=12)
        ttk.Button(nav, text="Exportar FINAL", command=self.export_final).pack(side="right")

        self.bind("<Control-s>", lambda e: self.save_current())
        self.bind("<Control-Right>", lambda e: self.save_and_next())
        self.bind("<Control-Left>", lambda e: self.previous())

    def set_text(self, widget, value):
        widget.delete("1.0", "end")
        widget.insert("1.0", str(value or ""))

    def get_text(self, widget):
        return widget.get("1.0", "end").strip()

    def load_current(self):
        row = self.rows[self.index]
        reviewed = sum(is_reviewed(r) for r in self.rows)

        self.lbl_progress.config(
            text=f"Eixo {self.index+1}/{len(self.rows)} | revisados: {reviewed}/{len(self.rows)}"
        )
        self.lbl_info.config(
            text=(
                f"{row['filename']} | {row['profile']} | "
                f"EIXO {row['axis']} | {row['kind']} | "
                f"{row.get('extraction_support_candidate','')}"
            )
        )

        self.lbl_auto_status.config(
            text=(
                f"axis={row['axis']}  kind={row['kind']}  "
                f"support={row.get('extraction_support_candidate','')}  "
                f"numeric_status={row.get('numeric_selection_status','')}"
            )
        )

        self.txt_auto.configure(state="normal")
        self.txt_auto.delete("1.0", "end")
        self.txt_auto.insert("1.0", auto_observed_text(row))
        self.txt_auto.configure(state="disabled")

        self.var_result.set(row.get("manual_ticklabel_extraction_result",""))
        self.set_text(self.txt_visible, row.get("manual_visible_ticklabels",""))
        self.set_text(self.txt_missing, row.get("manual_missing_labels",""))
        self.set_text(self.txt_false, row.get("manual_false_labels",""))
        self.set_text(self.txt_notes, row.get("manual_notes",""))

        self.load_image(image_path(row))

    def load_image(self, path: Path):
        if not path.exists():
            self.img_label.configure(text=f"Imagem não encontrada:\n{path}", image="")
            self.photo = None
            return

        img = Image.open(path).convert("RGB")
        max_w, max_h = 930, 720
        scale = min(max_w/img.width, max_h/img.height, 1.0)
        new_size = (max(1,int(img.width*scale)), max(1,int(img.height*scale)))
        if new_size != img.size:
            img = img.resize(new_size, Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.photo, text="")

    def open_external(self):
        path = image_path(self.rows[self.index])
        if path.exists():
            os.startfile(path)
        else:
            messagebox.showerror("Erro", f"Imagem não encontrada:\n{path}")

    def persist_form(self):
        row = self.rows[self.index]
        row["manual_ticklabel_extraction_result"] = self.var_result.get().strip()
        row["manual_visible_ticklabels"] = self.get_text(self.txt_visible)
        row["manual_missing_labels"] = self.get_text(self.txt_missing)
        row["manual_false_labels"] = self.get_text(self.txt_false)
        row["manual_notes"] = self.get_text(self.txt_notes)

    def validate_current(self):
        result = self.var_result.get().strip()
        visible = self.get_text(self.txt_visible)

        if not result:
            messagebox.showwarning("Campo obrigatório","Selecione o resultado da extração.")
            return False

        if result != "UNEVALUABLE" and not visible:
            messagebox.showwarning(
                "Campo obrigatório",
                "Transcreva os tick labels visíveis ou marque UNEVALUABLE."
            )
            return False

        return True

    def save_current(self, silent=False):
        if not self.validate_current():
            return False
        self.persist_form()
        write_csv(PROGRESS, self.rows)
        if not silent:
            messagebox.showinfo("Salvo", f"Progresso salvo:\n{PROGRESS}")
        return True

    def save_and_next(self):
        if not self.save_current(silent=True):
            return
        if self.index < len(self.rows)-1:
            self.index += 1
            self.load_current()
        else:
            self.export_final()

    def previous(self):
        self.persist_form()
        write_csv(PROGRESS, self.rows)
        if self.index > 0:
            self.index -= 1
            self.load_current()

    def jump_next_unreviewed(self):
        self.persist_form()
        write_csv(PROGRESS, self.rows)
        for offset in range(1, len(self.rows)+1):
            j = (self.index+offset) % len(self.rows)
            if not is_reviewed(self.rows[j]):
                self.index = j
                self.load_current()
                return
        messagebox.showinfo("Revisão","Todos os 88 eixos já possuem revisão.")

    def export_final(self):
        self.persist_form()
        write_csv(PROGRESS, self.rows)

        missing = [
            r["filename"] + ":" + r["axis"]
            for r in self.rows
            if not str(r.get("manual_ticklabel_extraction_result","")).strip()
        ]

        if missing:
            messagebox.showwarning(
                "Revisão incompleta",
                f"Ainda faltam {len(missing)} eixos.\n"
                "O progresso foi salvo, mas o arquivo FINAL não foi criado."
            )
            return

        write_csv(FINAL, self.rows)
        messagebox.showinfo(
            "Concluído",
            f"Revisão completa.\n\nArquivo FINAL:\n{FINAL}"
        )

    def on_close(self):
        self.persist_form()
        write_csv(PROGRESS, self.rows)
        self.destroy()

def main():
    if not QUEUE.exists():
        print("ERRO: fila não encontrada:", QUEUE)
        return 1

    base = [normalize_row(r) for r in read_csv(QUEUE)]

    if PROGRESS.exists():
        progress_rows = read_csv(PROGRESS)
        lookup = {(r["filename"],r["axis"]): r for r in progress_rows}
        for r in base:
            p = lookup.get((r["filename"],r["axis"]))
            if p:
                for f in MANUAL_FIELDS:
                    r[f] = p.get(f, r.get(f,""))
        print("Progresso anterior carregado:", PROGRESS)

    print("=" * 72)
    print("REVISOR MANUAL B-A V7 — PRODUÇÃO")
    print("=" * 72)
    print("Eixos:", len(base))
    print("Imagens únicas:", len({r['filename'] for r in base}))
    print("Já revisados:", sum(is_reviewed(r) for r in base))
    print("Fila:", QUEUE)
    print("Progresso:", PROGRESS)
    print()

    app = Reviewer(base)
    app.mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
