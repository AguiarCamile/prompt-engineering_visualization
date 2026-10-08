# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(r"C:\Users\Labvis\Downloads\imagens3120")
QUEUE = ROOT / "FILA_AUDITORIA_BA_V7_AUTO_NONCONFORME_X22.csv"
IMAGES = ROOT / "imagens"

PROGRESS = ROOT / "AUDITORIA_BA_V7_AUTO_NONCONFORME_X22_PROGRESS.csv"
FINAL = ROOT / "AUDITORIA_BA_V7_AUTO_NONCONFORME_X22_FINAL.csv"

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
    lines = []
    lines.append("automatic_observed_display: " + str(row.get("automatic_observed_display","")))
    kind = row.get("kind","")
    if kind == "temporal" and row.get("temporal_years_spatial",""):
        lines.append("temporal_years_spatial: " + row["temporal_years_spatial"])
    if kind == "numeric":
        if row.get("numeric_effective_values_spatial",""):
            lines.append("numeric_effective_values_spatial: " + row["numeric_effective_values_spatial"])
        if row.get("numeric_selected_values_spatial",""):
            lines.append("numeric_selected_values_spatial: " + row["numeric_selected_values_spatial"])
        if row.get("numeric_lane_ocr_raw_tokens",""):
            lines.append("numeric_lane_ocr_raw_tokens: " + row["numeric_lane_ocr_raw_tokens"])
    if row.get("individual_text_candidate",""):
        lines.append("individual_text_candidate: " + row["individual_text_candidate"])
    if row.get("band_ocr_normalized",""):
        lines.append("band_ocr_normalized: " + row["band_ocr_normalized"])
    return "\n".join(lines)

class Reviewer(tk.Tk):
    def __init__(self, rows):
        super().__init__()
        self.title("Auditoria B-A V7 — não conformes automáticos — eixo X")
        self.geometry("1500x980")
        self.minsize(1200,800)
        self.rows = rows
        self.index = self.first_unreviewed_index()
        self.photo = None
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.load_current()

    def first_unreviewed_index(self):
        for i,r in enumerate(self.rows):
            if not is_reviewed(r):
                return i
        return 0

    def build_ui(self):
        top=ttk.Frame(self,padding=8); top.pack(fill="x")
        self.lbl_progress=ttk.Label(top,text="",font=("Segoe UI",11,"bold")); self.lbl_progress.pack(side="left")
        self.lbl_info=ttk.Label(top,text="",font=("Segoe UI",10)); self.lbl_info.pack(side="right")

        main=ttk.Panedwindow(self,orient="horizontal"); main.pack(fill="both",expand=True,padx=8,pady=4)
        left=ttk.Frame(main); right=ttk.Frame(main,width=480)
        main.add(left,weight=3); main.add(right,weight=2)

        imgf=ttk.LabelFrame(left,text="Imagem"); imgf.pack(fill="both",expand=True)
        self.img_label=ttk.Label(imgf,anchor="center"); self.img_label.pack(fill="both",expand=True,padx=6,pady=6)
        ttk.Button(left,text="Abrir imagem no visualizador do Windows",command=self.open_external).pack(anchor="w",pady=4)

        auto=ttk.LabelFrame(right,text="B-A automática — caso que resultou NONCONFORMING",padding=8); auto.pack(fill="x",pady=(0,8))
        self.lbl_auto=ttk.Label(auto,text="",font=("Segoe UI",10,"bold")); self.lbl_auto.pack(anchor="w")
        self.txt_auto=tk.Text(auto,height=10,wrap="word"); self.txt_auto.pack(fill="x",pady=(6,0)); self.txt_auto.configure(state="disabled")

        manual=ttk.LabelFrame(right,text="Auditoria manual B-A (sem F)",padding=8); manual.pack(fill="both",expand=True)
        ttk.Label(manual,text="Tick labels VISÍVEIS no eixo indicado, na ordem espacial (separar com |):").pack(anchor="w")
        self.txt_visible=tk.Text(manual,height=5,wrap="word"); self.txt_visible.pack(fill="x",pady=(2,8))

        row=ttk.Frame(manual); row.pack(fill="x",pady=3)
        ttk.Label(row,text="Resultado da extração:",width=25).pack(side="left")
        self.var_result=tk.StringVar()
        self.cmb=ttk.Combobox(row,textvariable=self.var_result,values=RESULT_VALUES,state="readonly",width=22); self.cmb.pack(side="left")

        ttk.Label(manual,text="Rótulos visíveis ausentes da extração automática:").pack(anchor="w",pady=(8,2))
        self.txt_missing=tk.Text(manual,height=3,wrap="word"); self.txt_missing.pack(fill="x")
        ttk.Label(manual,text="Falsos rótulos incluídos pela extração automática:").pack(anchor="w",pady=(8,2))
        self.txt_false=tk.Text(manual,height=3,wrap="word"); self.txt_false.pack(fill="x")
        ttk.Label(manual,text="Notas:").pack(anchor="w",pady=(8,2))
        self.txt_notes=tk.Text(manual,height=4,wrap="word"); self.txt_notes.pack(fill="x")

        ttk.Label(
            manual,
            text=(
                "EXACT = B-A automática recuperou todos os ticks visíveis.\n"
                "EQUIVALENT = apenas diferença benigna de representação.\n"
                "PARTIAL = parte dos ticks foi recuperada.\n"
                "WRONG = conteúdo automático incorreto/falso.\n"
                "UNEVALUABLE = não é possível decidir.\n\n"
                "IMPORTANTE: NÃO consultar F. Estamos auditando apenas B-A."
            ),
            foreground="#555555"
        ).pack(anchor="w",pady=(8,0))

        nav=ttk.Frame(self,padding=8); nav.pack(fill="x")
        ttk.Button(nav,text="◀ Anterior",command=self.previous).pack(side="left")
        ttk.Button(nav,text="Salvar",command=self.save_current).pack(side="left",padx=6)
        ttk.Button(nav,text="Salvar e próxima ▶",command=self.save_and_next).pack(side="left")
        ttk.Button(nav,text="Próxima não revisada",command=self.jump).pack(side="left",padx=12)
        ttk.Button(nav,text="Exportar FINAL",command=self.export_final).pack(side="right")

    def set_text(self,w,v):
        w.delete("1.0","end"); w.insert("1.0",str(v or ""))

    def get_text(self,w):
        return w.get("1.0","end").strip()

    def load_current(self):
        r=self.rows[self.index]
        reviewed=sum(is_reviewed(x) for x in self.rows)
        self.lbl_progress.config(text=f"Caso {self.index+1}/{len(self.rows)} | revisados: {reviewed}/{len(self.rows)}")
        self.lbl_info.config(text=f"{r['filename']} | {r['profile']} | EIXO {r['axis']} | {r['kind']}")
        self.lbl_auto.config(text=f"support={r.get('extraction_support_candidate','')} | numeric_status={r.get('numeric_selection_status','')}")
        self.txt_auto.configure(state="normal"); self.txt_auto.delete("1.0","end"); self.txt_auto.insert("1.0",auto_observed_text(r)); self.txt_auto.configure(state="disabled")
        self.var_result.set(r.get("manual_ticklabel_extraction_result",""))
        self.set_text(self.txt_visible,r.get("manual_visible_ticklabels",""))
        self.set_text(self.txt_missing,r.get("manual_missing_labels",""))
        self.set_text(self.txt_false,r.get("manual_false_labels",""))
        self.set_text(self.txt_notes,r.get("manual_notes",""))
        self.load_image(image_path(r))

    def load_image(self,path):
        if not path.exists():
            self.img_label.configure(text=f"Imagem não encontrada:\n{path}",image=""); self.photo=None; return
        img=Image.open(path).convert("RGB")
        scale=min(930/img.width,720/img.height,1.0)
        ns=(max(1,int(img.width*scale)),max(1,int(img.height*scale)))
        if ns!=img.size: img=img.resize(ns,Image.LANCZOS)
        self.photo=ImageTk.PhotoImage(img); self.img_label.configure(image=self.photo,text="")

    def open_external(self):
        p=image_path(self.rows[self.index])
        if p.exists(): os.startfile(p)

    def persist(self):
        r=self.rows[self.index]
        r["manual_ticklabel_extraction_result"]=self.var_result.get().strip()
        r["manual_visible_ticklabels"]=self.get_text(self.txt_visible)
        r["manual_missing_labels"]=self.get_text(self.txt_missing)
        r["manual_false_labels"]=self.get_text(self.txt_false)
        r["manual_notes"]=self.get_text(self.txt_notes)

    def validate(self):
        if not self.var_result.get().strip():
            messagebox.showwarning("Campo obrigatório","Selecione o resultado da extração."); return False
        if self.var_result.get().strip()!="UNEVALUABLE" and not self.get_text(self.txt_visible):
            messagebox.showwarning("Campo obrigatório","Transcreva os ticks visíveis ou marque UNEVALUABLE."); return False
        return True

    def save_current(self,silent=False):
        if not self.validate(): return False
        self.persist(); write_csv(PROGRESS,self.rows)
        if not silent: messagebox.showinfo("Salvo",f"Progresso salvo:\n{PROGRESS}")
        return True

    def save_and_next(self):
        if not self.save_current(True): return
        if self.index<len(self.rows)-1:
            self.index+=1; self.load_current()
        else: self.export_final()

    def previous(self):
        self.persist(); write_csv(PROGRESS,self.rows)
        if self.index>0: self.index-=1; self.load_current()

    def jump(self):
        self.persist(); write_csv(PROGRESS,self.rows)
        for off in range(1,len(self.rows)+1):
            j=(self.index+off)%len(self.rows)
            if not is_reviewed(self.rows[j]):
                self.index=j; self.load_current(); return
        messagebox.showinfo("Revisão","Todos os casos já foram revisados.")

    def export_final(self):
        self.persist(); write_csv(PROGRESS,self.rows)
        missing=[r["filename"]+":"+r["axis"] for r in self.rows if not r.get("manual_ticklabel_extraction_result","").strip()]
        if missing:
            messagebox.showwarning("Revisão incompleta",f"Ainda faltam {len(missing)} casos."); return
        write_csv(FINAL,self.rows)
        messagebox.showinfo("Concluído",f"Arquivo FINAL:\n{FINAL}")

    def on_close(self):
        self.persist(); write_csv(PROGRESS,self.rows); self.destroy()

def main():
    if not QUEUE.exists():
        print("ERRO: fila não encontrada:",QUEUE); return 1
    rows=[normalize_row(r) for r in read_csv(QUEUE)]
    if PROGRESS.exists():
        old={(r["filename"],r["axis"]):r for r in read_csv(PROGRESS)}
        for r in rows:
            p=old.get((r["filename"],r["axis"]))
            if p:
                for f in MANUAL_FIELDS: r[f]=p.get(f,r.get(f,""))
    print("="*72)
    print("AUDITORIA B-A V7 — NÃO CONFORMES AUTOMÁTICOS — EIXO X")
    print("="*72)
    print("Casos:",len(rows))
    print("Fila:",QUEUE)
    app=Reviewer(rows); app.mainloop()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
