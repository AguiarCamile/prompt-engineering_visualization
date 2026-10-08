# -*- coding: utf-8 -*-
r"""
REVISÃO CEGA — 55 DISCREPÂNCIAS DE TÉCNICA NA PRODUÇÃO
=======================================================

Objetivo
--------
Revisar manualmente, SEM consultar F e SEM mostrar a classificação automática,
as imagens em que a classe observada pelo classificador exclusivo V2 difere
da técnica associada ao perfil experimental.

Esta fila é apenas uma fila de AUDITORIA B-A. A decisão normativa B-B será
feita somente depois da transcrição/classificação visual manual.

Entrada
-------
C:\Users\Labvis\Downloads\imagens3120\
  CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_FINAL.csv

Imagens
-------
C:\Users\Labvis\Downloads\imagens3120\imagens\*.png

Saídas
------
C:\Users\Labvis\Downloads\imagens3120\
  REVISAO_MANUAL_TECNICA_PRODUCAO_55_PROGRESS.csv
  REVISAO_MANUAL_TECNICA_PRODUCAO_55_FINAL.csv

Blindagem da interface
----------------------
A interface NÃO mostra:
- filename
- profile
- unit_id
- técnica esperada
- observed_type automático
- descritores automáticos

Classes manuais
---------------
BAR
LINE
SCATTER
OTHER
EMPTY_NO_DATA
UNCERTAIN

Se reconhecer claramente uma estrutura STEM, use OTHER e escreva STEM em notas.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES = ROOT / "imagens"

SOURCE = ROOT / "CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_FINAL.csv"
PROGRESS = ROOT / "REVISAO_MANUAL_TECNICA_PRODUCAO_55_PROGRESS.csv"
FINAL = ROOT / "REVISAO_MANUAL_TECNICA_PRODUCAO_55_FINAL.csv"

EXPECTED_BY_PROFILE = {
    "BI": "BAR",
    "BC": "BAR",
    "LI": "LINE",
    "LC": "LINE",
    "SI": "SCATTER",
    "SC": "SCATTER",
}

# Conflitos de especificação conhecidos para a dimensão técnica.
# Eles não entram como "não conformes" normativos. Na produção atual,
# nenhum deles pertence à fila de discrepâncias, mas a exclusão fica explícita.
SPEC_CONFLICT_UNITS = {"SC_024", "SC_037"}

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

def build_queue(rows):
    queue = []
    for r in rows:
        profile = str(r.get("profile", "")).strip().upper()
        unit_id = str(r.get("unit_id", "")).strip()
        observed = str(r.get("observed_type", "")).strip().upper()

        expected = EXPECTED_BY_PROFILE.get(profile)
        if expected is None:
            continue
        if unit_id in SPEC_CONFLICT_UNITS:
            continue

        if observed != expected:
            rr = dict(r)
            rr["candidate_expected_profile_type"] = expected
            rr["manual_visible_type"] = ""
            rr["manual_notes"] = ""
            queue.append(rr)

    queue.sort(key=lambda r: r["filename"])
    return queue

def is_reviewed(r):
    return str(r.get("manual_visible_type", "")).strip() != ""

def image_path(r):
    return IMAGES / Path(r["filename"]).name

class Reviewer(tk.Tk):
    def __init__(self, rows):
        super().__init__()
        self.title("Revisão cega — técnica — 55 discrepâncias")
        self.geometry("1380x900")
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
            top, font=("Segoe UI", 11, "bold")
        )
        self.lbl_progress.pack(side="left")

        ttk.Label(
            top,
            text="Cego para F e para a classificação automática",
            font=("Segoe UI", 10),
        ).pack(side="right")

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(main)
        right = ttk.Frame(main, width=430)
        main.add(left, weight=3)
        main.add(right, weight=2)

        image_frame = ttk.LabelFrame(left, text="Imagem")
        image_frame.pack(fill="both", expand=True)

        self.img_label = ttk.Label(image_frame, anchor="center")
        self.img_label.pack(fill="both", expand=True, padx=6, pady=6)

        manual = ttk.LabelFrame(
            right,
            text="Classificação visual manual",
            padding=10,
        )
        manual.pack(fill="both", expand=True)

        ttk.Label(
            manual,
            text="Qual é a técnica/estrutura principal visível?",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        self.var_class = tk.StringVar()
        ttk.Combobox(
            manual,
            textvariable=self.var_class,
            values=CLASS_VALUES,
            state="readonly",
            width=24,
        ).pack(anchor="w", pady=(8, 16))

        ttk.Label(manual, text="Notas:").pack(anchor="w")
        self.txt_notes = tk.Text(manual, height=8, wrap="word")
        self.txt_notes.pack(fill="x", pady=(2, 12))

        rules = (
            "BAR: barras são a marca principal.\n"
            "LINE: linha/trajetória é a marca principal.\n"
            "SCATTER: pontos dispersos são a marca principal.\n"
            "OTHER: outra estrutura principal.\n"
            "EMPTY_NO_DATA: não há estrutura de dados principal visível.\n"
            "UNCERTAIN: não é possível decidir visualmente.\n\n"
            "Se identificar STEM, marque OTHER e escreva STEM nas notas.\n\n"
            "Não consulte F, prompt, profile ou resultados automáticos."
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
            command=self.next_unreviewed,
        ).pack(side="left", padx=12)
        ttk.Button(
            nav,
            text="Exportar FINAL",
            command=self.export_final,
        ).pack(side="right")

    def load_image(self, path: Path):
        if not path.exists():
            self.photo = None
            self.img_label.configure(
                text=f"Imagem não encontrada:\n{path}",
                image=""
            )
            return

        img = Image.open(path).convert("RGB")
        scale = min(900 / img.width, 700 / img.height, 1.0)
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
            "Todos os casos já foram revisados."
        )

    def export_final(self):
        self.persist()
        write_csv(PROGRESS, self.rows)

        missing = [
            r["filename"]
            for r in self.rows
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
    if not SOURCE.exists():
        print("ERRO: arquivo de produção não encontrado:")
        print(SOURCE)
        return 1

    rows = read_csv(SOURCE)
    queue = build_queue(rows)

    if len(queue) != 55:
        print(
            f"ATENÇÃO: a produção atual gera {len(queue)} discrepâncias; "
            "eram esperadas 55."
        )

    if PROGRESS.exists():
        old = {
            r["filename"]: r
            for r in read_csv(PROGRESS)
        }
        for r in queue:
            p = old.get(r["filename"])
            if p:
                for field in MANUAL_FIELDS:
                    r[field] = p.get(field, "")

    print("=" * 76)
    print("REVISÃO CEGA — 55 DISCREPÂNCIAS DE TÉCNICA")
    print("=" * 76)
    print("Casos:", len(queue))
    print("Progresso:", PROGRESS)
    print("FINAL:", FINAL)
    print()
    print("A interface não mostra F, profile, filename ou observed_type automático.")

    app = Reviewer(queue)
    app.mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
