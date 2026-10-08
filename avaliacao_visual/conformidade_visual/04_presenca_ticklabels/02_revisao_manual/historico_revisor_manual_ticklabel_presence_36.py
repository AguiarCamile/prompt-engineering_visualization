# -*- coding: utf-8 -*-
"""
REVISÃO MANUAL — PRESENÇA DOS TICK LABELS — 36 CASOS
=====================================================

Entrada:
C:\Users\Labvis\Downloads\imagens3120\imagens\
  _ticklabel_presence_full_v5\ticklabel_presence_priority_review_v5.csv

Imagens:
C:\Users\Labvis\Downloads\imagens3120\imagens\*.png

Saídas:
C:\Users\Labvis\Downloads\imagens3120\
  REVISAO_MANUAL_TICKLABEL_PRESENCE_36_PROGRESS.csv
  REVISAO_MANUAL_TICKLABEL_PRESENCE_36_FINAL.csv

A pergunta é apenas visual:
- existem tick labels visíveis no eixo X?
- existem tick labels visíveis no eixo Y?

Não consultar F.
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
    IMAGES / "_ticklabel_presence_full_v5"
    / "ticklabel_presence_priority_review_v5.csv"
)

PROGRESS = ROOT / "REVISAO_MANUAL_TICKLABEL_PRESENCE_36_PROGRESS.csv"
FINAL = ROOT / "REVISAO_MANUAL_TICKLABEL_PRESENCE_36_FINAL.csv"

YN_VALUES = ["", "YES", "NO", "UNEVALUABLE"]
CLASS_VALUES = [
    "",
    "BOTH_PRESENT",
    "X_ONLY",
    "Y_ONLY",
    "NONE_PRESENT",
    "UNEVALUABLE",
]

MANUAL_FIELDS = [
    "manual_x_present_final",
    "manual_y_present_final",
    "manual_classification",
    "manual_notes",
]

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def normalize(r):
    for f in MANUAL_FIELDS:
        r.setdefault(f, "")
    return r

def is_reviewed(r):
    return (
        str(r.get("manual_x_present_final","")).strip() != ""
        and str(r.get("manual_y_present_final","")).strip() != ""
        and str(r.get("manual_classification","")).strip() != ""
    )

def image_path(r):
    return IMAGES / Path(r["filename"]).name

class Reviewer(tk.Tk):
    def __init__(self, rows):
        super().__init__()
        self.title("Revisão manual — presença dos tick labels — 36 casos")
        self.geometry("1480x940")
        self.minsize(1180, 780)
        self.rows = rows
        self.index = self.first_unreviewed()
        self.photo = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build()
        self.load()

    def first_unreviewed(self):
        for i,r in enumerate(self.rows):
            if not is_reviewed(r):
                return i
        return 0

    def build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        self.lbl_progress = ttk.Label(top, font=("Segoe UI",11,"bold"))
        self.lbl_progress.pack(side="left")
        self.lbl_info = ttk.Label(top, font=("Segoe UI",10))
        self.lbl_info.pack(side="right")

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(main)
        right = ttk.Frame(main, width=480)
        main.add(left, weight=3)
        main.add(right, weight=2)

        imgf = ttk.LabelFrame(left, text="Imagem")
        imgf.pack(fill="both", expand=True)
        self.img_label = ttk.Label(imgf, anchor="center")
        self.img_label.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Button(
            left,
            text="Abrir imagem no visualizador do Windows",
            command=self.open_external
        ).pack(anchor="w", pady=4)

        auto = ttk.LabelFrame(
            right,
            text="Resultado automático V5 (apenas apoio)",
            padding=8
        )
        auto.pack(fill="x", pady=(0,8))
        self.txt_auto = tk.Text(auto, height=10, wrap="word")
        self.txt_auto.pack(fill="x")
        self.txt_auto.configure(state="disabled")

        manual = ttk.LabelFrame(
            right,
            text="Adjudicação manual visual — sem F",
            padding=8
        )
        manual.pack(fill="both", expand=True)

        row = ttk.Frame(manual)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Tick labels visíveis no eixo X:", width=30).pack(side="left")
        self.var_x = tk.StringVar()
        ttk.Combobox(
            row, textvariable=self.var_x,
            values=YN_VALUES, state="readonly", width=18
        ).pack(side="left")

        row = ttk.Frame(manual)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Tick labels visíveis no eixo Y:", width=30).pack(side="left")
        self.var_y = tk.StringVar()
        ttk.Combobox(
            row, textvariable=self.var_y,
            values=YN_VALUES, state="readonly", width=18
        ).pack(side="left")

        row = ttk.Frame(manual)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Classificação final:", width=30).pack(side="left")
        self.var_class = tk.StringVar()
        ttk.Combobox(
            row, textvariable=self.var_class,
            values=CLASS_VALUES, state="readonly", width=22
        ).pack(side="left")

        ttk.Label(manual, text="Notas:").pack(anchor="w", pady=(12,2))
        self.txt_notes = tk.Text(manual, height=6, wrap="word")
        self.txt_notes.pack(fill="x")

        rules = (
            "BOTH_PRESENT: X=YES e Y=YES\n"
            "X_ONLY: X=YES e Y=NO\n"
            "Y_ONLY: X=NO e Y=YES\n"
            "NONE_PRESENT: X=NO e Y=NO\n"
            "UNEVALUABLE: não é possível decidir visualmente.\n\n"
            "Avalie somente se há rótulos de ticks visíveis. "
            "Não avaliar o conteúdo dos valores e NÃO consultar F."
        )
        ttk.Label(manual, text=rules, foreground="#555555").pack(anchor="w", pady=(12,0))

        nav = ttk.Frame(self, padding=8)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀ Anterior", command=self.previous).pack(side="left")
        ttk.Button(nav, text="Salvar", command=self.save_current).pack(side="left", padx=6)
        ttk.Button(nav, text="Salvar e próxima ▶", command=self.save_next).pack(side="left")
        ttk.Button(nav, text="Próxima não revisada", command=self.next_unreviewed).pack(side="left", padx=12)
        ttk.Button(nav, text="Exportar FINAL", command=self.export_final).pack(side="right")

    def auto_text(self, r):
        fields = [
            ("presence_status_auto", r.get("presence_status_auto","")),
            ("x_present_auto", r.get("x_present_auto","")),
            ("y_present_auto", r.get("y_present_auto","")),
            ("both_present_auto", r.get("both_present_auto","")),
            ("x_segmentation_ok", r.get("x_segmentation_ok","")),
            ("y_segmentation_ok", r.get("y_segmentation_ok","")),
            ("x_ocr_nonempty", r.get("x_ocr_nonempty","")),
            ("y_ocr_nonempty", r.get("y_ocr_nonempty","")),
            ("visual_review_reason", r.get("visual_review_reason","")),
        ]
        return "\n".join(f"{k}: {v}" for k,v in fields)

    def load_image(self, path):
        if not path.exists():
            self.img_label.configure(
                text=f"Imagem não encontrada:\n{path}", image=""
            )
            self.photo = None
            return
        img = Image.open(path).convert("RGB")
        scale = min(900/img.width, 700/img.height, 1.0)
        size = (
            max(1,int(img.width*scale)),
            max(1,int(img.height*scale))
        )
        if size != img.size:
            img = img.resize(size, Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.photo, text="")

    def load(self):
        r = self.rows[self.index]
        reviewed = sum(is_reviewed(x) for x in self.rows)
        self.lbl_progress.config(
            text=f"Caso {self.index+1}/{len(self.rows)} | revisados: {reviewed}/{len(self.rows)}"
        )
        self.lbl_info.config(
            text=f'{r.get("filename","")} | {r.get("profile","")} | '
                 f'{r.get("technique","")} | {r.get("task","")}'
        )

        self.txt_auto.configure(state="normal")
        self.txt_auto.delete("1.0","end")
        self.txt_auto.insert("1.0", self.auto_text(r))
        self.txt_auto.configure(state="disabled")

        self.var_x.set(r.get("manual_x_present_final",""))
        self.var_y.set(r.get("manual_y_present_final",""))
        self.var_class.set(r.get("manual_classification",""))
        self.txt_notes.delete("1.0","end")
        self.txt_notes.insert("1.0", r.get("manual_notes",""))

        self.load_image(image_path(r))

    def persist(self):
        r = self.rows[self.index]
        r["manual_x_present_final"] = self.var_x.get().strip()
        r["manual_y_present_final"] = self.var_y.get().strip()
        r["manual_classification"] = self.var_class.get().strip()
        r["manual_notes"] = self.txt_notes.get("1.0","end").strip()

    def expected_class(self, x, y):
        if x == "UNEVALUABLE" or y == "UNEVALUABLE":
            return "UNEVALUABLE"
        if x == "YES" and y == "YES":
            return "BOTH_PRESENT"
        if x == "YES" and y == "NO":
            return "X_ONLY"
        if x == "NO" and y == "YES":
            return "Y_ONLY"
        if x == "NO" and y == "NO":
            return "NONE_PRESENT"
        return ""

    def validate(self):
        x = self.var_x.get().strip()
        y = self.var_y.get().strip()
        c = self.var_class.get().strip()

        if not x or not y or not c:
            messagebox.showwarning(
                "Campos obrigatórios",
                "Preencha X, Y e a classificação final."
            )
            return False

        expected = self.expected_class(x,y)
        if expected and c != expected:
            ok = messagebox.askyesno(
                "Classificação inconsistente",
                f"Com X={x} e Y={y}, a classificação esperada seria "
                f"{expected}.\n\nDeseja salvar mesmo assim?"
            )
            if not ok:
                return False
        return True

    def save_current(self, silent=False):
        if not self.validate():
            return False
        self.persist()
        write_csv(PROGRESS, self.rows)
        if not silent:
            messagebox.showinfo("Salvo", f"Progresso salvo:\n{PROGRESS}")
        return True

    def save_next(self):
        if not self.save_current(True):
            return
        if self.index < len(self.rows)-1:
            self.index += 1
            self.load()
        else:
            self.export_final()

    def previous(self):
        self.persist()
        write_csv(PROGRESS, self.rows)
        if self.index > 0:
            self.index -= 1
            self.load()

    def next_unreviewed(self):
        self.persist()
        write_csv(PROGRESS, self.rows)
        for off in range(1,len(self.rows)+1):
            j = (self.index+off) % len(self.rows)
            if not is_reviewed(self.rows[j]):
                self.index = j
                self.load()
                return
        messagebox.showinfo("Revisão","Todos os 36 casos já foram revisados.")

    def open_external(self):
        p = image_path(self.rows[self.index])
        if p.exists():
            os.startfile(p)

    def export_final(self):
        self.persist()
        write_csv(PROGRESS, self.rows)
        missing = [r["filename"] for r in self.rows if not is_reviewed(r)]
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
        print("ERRO: fila de 36 casos não encontrada:")
        print(QUEUE)
        return 1

    rows = [normalize(r) for r in read_csv(QUEUE)]

    if len(rows) != 36:
        print(f"ATENÇÃO: esperados 36 casos, encontrados {len(rows)}.")

    if PROGRESS.exists():
        old = {r["filename"]: r for r in read_csv(PROGRESS)}
        for r in rows:
            p = old.get(r["filename"])
            if p:
                for f in MANUAL_FIELDS:
                    r[f] = p.get(f, r.get(f,""))

    print("="*72)
    print("REVISÃO MANUAL — PRESENÇA DOS TICK LABELS")
    print("="*72)
    print("Casos:", len(rows))
    print("Fila:", QUEUE)
    print("Progresso:", PROGRESS)

    app = Reviewer(rows)
    app.mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
