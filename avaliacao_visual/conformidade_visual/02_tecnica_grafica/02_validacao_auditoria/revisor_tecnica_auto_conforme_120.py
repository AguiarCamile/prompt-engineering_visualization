# -*- coding: utf-8 -*-
r"""
AUDITORIA CEGA DE CONTROLE — 120 AUTO-CONFORMES DE TÉCNICA
===========================================================

Seleciona 20 imagens por perfil entre os casos em que observed_type automático
coincide com a técnica do perfil e que não pertençam aos conflitos de
especificação SC_024/SC_037.

A interface NÃO mostra:
- filename
- profile
- expected_type
- observed_type automático
- descritores automáticos

Classes:
BAR, LINE, SCATTER, OTHER, EMPTY_NO_DATA, UNCERTAIN

Saídas:
AMOSTRA_TECNICA_AUTO_CONFORME_120.csv
REVISAO_MANUAL_TECNICA_AUTO_CONFORME_120_PROGRESS.csv
REVISAO_MANUAL_TECNICA_AUTO_CONFORME_120_FINAL.csv
"""

from __future__ import annotations
import csv, random
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
IMAGES = ROOT / "imagens"
SOURCE = ROOT / "CHART_TYPE_EXCLUSIVE_PRODUCTION_V2_FROZEN_FINAL.csv"
SAMPLE = ROOT / "AMOSTRA_TECNICA_AUTO_CONFORME_120.csv"
PROGRESS = ROOT / "REVISAO_MANUAL_TECNICA_AUTO_CONFORME_120_PROGRESS.csv"
FINAL = ROOT / "REVISAO_MANUAL_TECNICA_AUTO_CONFORME_120_FINAL.csv"

SEED = 20260913
N_PER_PROFILE = 20
PROFILES = ["BI","BC","LI","LC","SI","SC"]
EXPECTED = {
    "BI":"BAR","BC":"BAR",
    "LI":"LINE","LC":"LINE",
    "SI":"SCATTER","SC":"SCATTER",
}
SPEC_CONFLICT_UNITS = {"SC_024","SC_037"}

VALUES = ["","BAR","LINE","SCATTER","OTHER","EMPTY_NO_DATA","UNCERTAIN"]

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def build_sample(rows):
    if SAMPLE.exists():
        s = read_csv(SAMPLE)
        if len(s) == 120:
            return s

    rng = random.Random(SEED)
    selected = []

    for profile in PROFILES:
        expected = EXPECTED[profile]
        candidates = [
            dict(r) for r in rows
            if r["profile"] == profile
            and r["unit_id"] not in SPEC_CONFLICT_UNITS
            and str(r["observed_type"]).upper() == expected
        ]
        rng.shuffle(candidates)
        chosen = candidates[:N_PER_PROFILE]
        if len(chosen) != N_PER_PROFILE:
            raise RuntimeError(
                f"{profile}: somente {len(chosen)} casos elegíveis."
            )
        selected.extend(chosen)

    # Randomiza ordem global para reduzir aprendizagem por perfil.
    rng.shuffle(selected)

    for i, r in enumerate(selected, 1):
        r["blind_case_id"] = f"CASE_{i:03d}"
        r["manual_visible_type"] = ""
        r["manual_notes"] = ""

    write_csv(SAMPLE, selected)
    return selected

def reviewed(r):
    return str(r.get("manual_visible_type","")).strip() != ""

class App(tk.Tk):
    def __init__(self, rows):
        super().__init__()
        self.title("Auditoria cega — 120 auto-conformes de técnica")
        self.geometry("1380x900")
        self.rows = rows
        self.i = next((i for i,r in enumerate(rows) if not reviewed(r)), 0)
        self.photo = None
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.build()
        self.load()

    def build(self):
        top = ttk.Frame(self, padding=8); top.pack(fill="x")
        self.lbl = ttk.Label(top, font=("Segoe UI",11,"bold")); self.lbl.pack(side="left")
        ttk.Label(
            top,
            text="Cego para F, perfil e classificação automática"
        ).pack(side="right")

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(main); right = ttk.Frame(main, width=420)
        main.add(left, weight=3); main.add(right, weight=2)

        fr = ttk.LabelFrame(left, text="Imagem")
        fr.pack(fill="both", expand=True)
        self.img = ttk.Label(fr, anchor="center")
        self.img.pack(fill="both", expand=True, padx=6, pady=6)

        mf = ttk.LabelFrame(right, text="Classificação visual", padding=10)
        mf.pack(fill="both", expand=True)
        ttk.Label(
            mf,
            text="Qual é a técnica/estrutura principal visível?",
            font=("Segoe UI",10,"bold")
        ).pack(anchor="w")

        self.var = tk.StringVar()
        ttk.Combobox(
            mf, textvariable=self.var, values=VALUES,
            state="readonly", width=24
        ).pack(anchor="w", pady=(8,16))

        ttk.Label(mf, text="Notas:").pack(anchor="w")
        self.notes = tk.Text(mf, height=8, wrap="word")
        self.notes.pack(fill="x")

        ttk.Label(
            mf,
            text=(
                "Se reconhecer STEM, use OTHER e registre STEM nas notas.\n"
                "Não consulte F, nomes de arquivo ou resultados automáticos."
            ),
            foreground="#555555",
            justify="left"
        ).pack(anchor="w", pady=(14,0))

        nav = ttk.Frame(self, padding=8); nav.pack(fill="x")
        ttk.Button(nav, text="◀ Anterior", command=self.prev).pack(side="left")
        ttk.Button(nav, text="Salvar", command=self.save).pack(side="left", padx=6)
        ttk.Button(nav, text="Salvar e próxima ▶", command=self.next).pack(side="left")
        ttk.Button(nav, text="Exportar FINAL", command=self.export).pack(side="right")

    def persist(self):
        self.rows[self.i]["manual_visible_type"] = self.var.get().strip()
        self.rows[self.i]["manual_notes"] = self.notes.get("1.0","end").strip()

    def load(self):
        r = self.rows[self.i]
        n = sum(reviewed(x) for x in self.rows)
        self.lbl.config(text=f"Caso {self.i+1}/{len(self.rows)} | revisados {n}/{len(self.rows)}")
        self.var.set(r.get("manual_visible_type",""))
        self.notes.delete("1.0","end")
        self.notes.insert("1.0", r.get("manual_notes",""))

        p = IMAGES / r["filename"]
        im = Image.open(p).convert("RGB")
        scale = min(900/im.width, 700/im.height, 1.0)
        size = (int(im.width*scale), int(im.height*scale))
        if size != im.size:
            im = im.resize(size, Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(im)
        self.img.configure(image=self.photo)

    def valid(self):
        if not self.var.get().strip():
            messagebox.showwarning("Obrigatório","Selecione uma classe.")
            return False
        return True

    def save(self, quiet=False):
        if not self.valid(): return False
        self.persist(); write_csv(PROGRESS,self.rows)
        if not quiet:
            messagebox.showinfo("Salvo", str(PROGRESS))
        return True

    def next(self):
        if not self.save(True): return
        if self.i < len(self.rows)-1:
            self.i += 1; self.load()
        else:
            self.export()

    def prev(self):
        self.persist(); write_csv(PROGRESS,self.rows)
        if self.i > 0:
            self.i -= 1; self.load()

    def export(self):
        self.persist(); write_csv(PROGRESS,self.rows)
        missing = [r for r in self.rows if not reviewed(r)]
        if missing:
            messagebox.showwarning("Incompleto", f"Faltam {len(missing)} casos.")
            return
        write_csv(FINAL,self.rows)
        messagebox.showinfo("Concluído", str(FINAL))

    def close(self):
        self.persist(); write_csv(PROGRESS,self.rows); self.destroy()

def main():
    if not SOURCE.exists():
        print("ERRO:", SOURCE)
        return 1
    rows = build_sample(read_csv(SOURCE))

    if PROGRESS.exists():
        old = {r["blind_case_id"]:r for r in read_csv(PROGRESS)}
        for r in rows:
            o = old.get(r["blind_case_id"])
            if o:
                r["manual_visible_type"] = o.get("manual_visible_type","")
                r["manual_notes"] = o.get("manual_notes","")

    print("Amostra cega:", len(rows))
    print("Seed:", SEED)
    print("20 por perfil.")
    App(rows).mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
