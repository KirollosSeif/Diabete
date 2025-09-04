# streamlit_prod/main_streamlit_prod.py
import sys
import re
import unicodedata
import csv
import os
import json
import subprocess
from uuid import uuid4
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ========== PATH & IMPORT BASE ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_best_model, preprocess_for_inference  # noqa: E402

# === Chat RAG import (dopo sys.path, con fallback da file) ===
_CHAT_IMPORT_ERROR = ""
answer_with_rag = None
try:
    # prova come package (se la cartella è visibile come modulo)
    from streamlit_prod.chatbot import answer_with_rag  # type: ignore
except Exception as _e1:
    try:
        # fallback: importa direttamente dal file chatbot.py
        import importlib.util, types
        CHATBOT_PATH = Path(__file__).resolve().parent / "chatbot.py"
        if CHATBOT_PATH.exists():
            spec = importlib.util.spec_from_file_location("chatbot_module", CHATBOT_PATH)
            mod = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            answer_with_rag = getattr(mod, "answer_with_rag", None)
            if answer_with_rag is None:
                _CHAT_IMPORT_ERROR = "funzione answer_with_rag non trovata in chatbot.py"
        else:
            _CHAT_IMPORT_ERROR = f"chatbot.py non trovato in {CHATBOT_PATH}"
    except Exception as _e2:
        _CHAT_IMPORT_ERROR = f"{_e1} | {_e2}"

# ========== PAGE CONFIG & THEME ==========
st.set_page_config(page_title="Valutazione Rischio Diabete", page_icon="🩺", layout="wide")
st.markdown("""
<style>
html, body { background: #f6f7fb; }
div.block-container{
  padding: 1.8rem 1rem 1.4rem !important;  /* spazio sopra leggero */
  max-width: 1480px !important;
}
section[data-testid="stSidebar"]{display:none}
header [data-testid="baseButton-headerNoPadding"]{visibility:hidden}

/* ------- WOW HOME ------- */
.hero-wow{
  position: relative; border-radius: 28px; padding: clamp(1.4rem, 2.2vw, 2.2rem);
  overflow: hidden; border: 1px solid rgba(16,24,40,.08);
  background:
    radial-gradient(1200px 600px at 8% 10%, rgba(56,189,248,.18), transparent 60%),
    radial-gradient(900px 500px at 92% 30%, rgba(168,85,247,.18), transparent 60%),
    linear-gradient(180deg, #ffffff, #f9fbff);
  box-shadow: 0 18px 46px rgba(16,24,40,.10); margin-bottom: 1.1rem;
}
.hero-wow:before{ content: ""; position:absolute; inset:-20%;
  background: conic-gradient(from 180deg at 50% 50%, rgba(99,102,241,.1), rgba(56,189,248,.08), rgba(236,72,153,.08), rgba(99,102,241,.1));
  filter: blur(40px); opacity:.6; }
.hero-inner{ position: relative; z-index: 2; }
.gradient-title{
  font-weight: 900; letter-spacing: .2px; line-height: 1.08;
  font-size: clamp(1.8rem, 3.4vw, 3.2rem);
  background: linear-gradient(90deg, #0ea5e9, #6366f1, #f43f5e, #0ea5e9);
  -webkit-background-clip: text; background-clip: text; color: transparent;
  background-size: 300% 100%; animation: shimmer 8s linear infinite; margin: 0 0 .4rem;
}
@keyframes shimmer { 0% { background-position: 0% 50% } 100% { background-position: 300% 50% } }
.subtitle{ margin: 0; opacity: .9; font-size: clamp(.98rem, 1.4vw, 1.05rem); }

/* Feature cards */
.feature-grid{ display:grid; grid-template-columns:1fr; gap:.9rem; }
@media (min-width: 880px){ .feature-grid{ grid-template-columns: 1fr 1fr 1fr; } }
.card{
  border: 1px solid rgba(16,24,40,.08); border-radius: 20px; background: #fff;
  box-shadow: 0 8px 22px rgba(16,24,40,.06); padding: 1.05rem 1.2rem;
  transition: transform .14s ease, box-shadow .14s ease;
}
.card:hover{ transform: translateY(-1px); box-shadow: 0 14px 28px rgba(16,24,40,.08); }
.card .icon{ font-size: 1.4rem; }

.badge{
  display:inline-block; padding:.34rem .7rem; border-radius:999px;
  border:1px solid rgba(16,24,40,.12); background:#fff; font-size:.8rem;
  color:#111; font-weight:700;
}

/* Pulsanti */
.stButton>button[kind="primary"]{
  border-radius: 14px; font-weight: 800; padding: .82rem 1.1rem;
  box-shadow: 0 14px 30px rgba(16,24,40,.14);
}
.stButton>button{ border-radius: 12px; font-weight: 650 }

/* Barra probabilità */
.prob-wrap{ width:100%; height: 10px; border-radius: 999px; border: 1px solid rgba(16,24,40,.12); background: #f2f4f7; overflow:hidden; }
.prob-fill{ height:100%; background: linear-gradient(90deg,#60a5fa,#a78bfa); }

/* Risultato card colorata */
.result{ border-left-width: 6px; border-left-style: solid; padding-left: 1rem; }

/* Griglia contatti */
.contact-grid{ display:grid; grid-template-columns:1fr; gap:1rem; }
@media (min-width: 780px){ .contact-grid{ grid-template-columns: 1fr 1fr; } }

/* Row azioni centrata */
.action-center{ display:flex; justify-content:center; gap:1rem; margin:.6rem 0 0; }

/* ====== Floating Chat (usa l’ULTIMO Expander della pagina) ====== */
div[data-testid="stExpander"]:last-of-type > details{
  position: fixed !important;
  right: 18px; bottom: 18px;
  width: 360px; max-width: 92vw;
  z-index: 9999;
  border-radius: 14px;
  box-shadow: 0 16px 38px rgba(16,24,40,.18);
  overflow: hidden;
  border: 1px solid rgba(16,24,40,.12);
}
div[data-testid="stExpander"]:last-of-type > details:not([open]) { width: 58px; }

/* Header dell'expander = bottone della chat */
div[data-testid="stExpander"]:last-of-type .streamlit-expanderHeader{
  background: #ffffff;
  color: #111 !important;
  font-weight: 800;
  padding:.6rem .8rem;
  display:flex; align-items:center; gap:.5rem;
  border-bottom: 1px solid rgba(16,24,40,.06);
}
div[data-testid="stExpander"]:last-of-type .streamlit-expanderHeader p{ margin:0; }
div[data-testid="stExpander"]:last-of-type .streamlit-expanderHeader:before{
  content: "💬"; display:inline-block; font-size: 1.05rem;
}

/* Corpo chat */
div[data-testid="stExpander"]:last-of-type [data-testid="stMarkdownContainer"]{
  max-height: 42vh; overflow: auto; padding-right: .25rem;
}
div[data-testid="stExpander"]:last-of-type .stTextInput>div>div>input{ font-size: .95rem; }
div[data-testid="stExpander"]:last-of-type .stButton>button{
  width: 100%; border-radius: 10px; font-weight: 750;
}

/* Dark mode */
@media (prefers-color-scheme: dark){
  html, body { background: #0f1117; }
  .hero-wow{ border-color: rgba(255,255,255,.08); background: linear-gradient(180deg,#0f1117,#12131a); color:#e9e9ea }
  .card{ background:#12131a; color:#e9e9ea; border-color: rgba(255,255,255,.08); }
  .prob-wrap{ border-color: rgba(255,255,255,.1); background: #141722; }
}
</style>
""", unsafe_allow_html=True)

# ========== FILES & CONFIG ==========
CONTACTS_CSV = PROJECT_ROOT / "data" / "ospedali_milano_comuni_mapping.csv"
LOG_CSV      = PROJECT_ROOT / "data" / "prod_interactions_v1.csv"
MILANO_LAT, MILANO_LON = 45.4642, 9.1900

# Auto-commit/push via git CLI
GIT_AUTOCOMMIT_ENABLED = True
GIT_BRANCH = "main"

# ========== UTILS NORMALIZZAZIONE ==========
def _slug(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def _norm_text(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "comune": "comune", "citta": "comune", "comuni": "comune",
        "ospedale_di_riferimento_linea_daria_macro_area": "ospedale",
        "ospedale_di_riferimento_macro_area": "ospedale",
        "ospedale_di_riferimento": "ospedale",
        "ospedale": "ospedale",
        "indirizzo_ospedale": "indirizzo", "indirizzo": "indirizzo",
        "telefono": "telefono",
        "prenotazioni_cup": "prenotazioni", "prenotazioni": "prenotazioni", "cup": "prenotazioni",
        "note": "note",
        "lat": "lat", "latitude": "lat", "latitudine": "lat",
        "lon": "lon", "lng": "lon", "longitude": "lon", "longitudine": "lon",
    }
    new_cols = {c: mapping.get(_slug(c), _slug(c)) for c in df.columns}
    df = df.rename(columns=new_cols)
    for c in ["comune","ospedale","indirizzo","telefono","prenotazioni","note"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for req in ["comune","ospedale","indirizzo","telefono"]:
        if req not in df.columns:
            df[req] = pd.NA
    return df

@st.cache_data(show_spinner=False)
def load_contacts():
    if not CONTACTS_CSV.exists():
        return pd.DataFrame(), []
    try:
        df = pd.read_csv(CONTACTS_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(CONTACTS_CSV, encoding="latin-1", dtype=str, keep_default_na=False)
    df = _canonicalize_columns(df)
    comuni = (
        pd.Series(df["comune"].astype(str).str.strip())
          .replace({"": np.nan}).dropna().drop_duplicates()
          .sort_values(key=lambda s: s.str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.casefold())
          .tolist()
    )
    return df, comuni

contacts_df, comuni_options = load_contacts()

# ========== SESSION ==========
st.session_state.setdefault("view", "home")
st.session_state.setdefault("last_pred", None)
st.session_state.setdefault("last_prob", None)
st.session_state.setdefault("last_form", None)
st.session_state.setdefault("selected_comune", None)
st.session_state.setdefault("sid", str(uuid4()))
st.session_state.setdefault("chat_history", [])  # [(role, content), ...]
st.session_state.setdefault("chat_input", "")

def go(view: str):
    st.session_state.view = view

# ========== LOG CSV ==========
LOG_COLUMNS = [
    "timestamp","session_id","event_type",
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
    "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth",
    "DiffWalk","Sex","Age","Education","Income",
    "predicted_class","probability",
    "comune","ospedale","telefono","indirizzo","prenotazioni","note"
]

def _bootstrap_log():
    try:
        LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_CSV.exists() or LOG_CSV.stat().st_size == 0:
            with LOG_CSV.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(LOG_COLUMNS)
    except Exception as e:
        print("[log bootstrap]", e)

_bootstrap_log()

# --- Git auto-commit/push (best-effort, silenzioso) ---
def _run_git(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=25
        )
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out
    except Exception as e:
        return 1, f"{e}"

def git_auto_commit_push(file_path: Path, message: str | None = None):
    if not GIT_AUTOCOMMIT_ENABLED:
        return
    try:
        rel = str(file_path.relative_to(PROJECT_ROOT))
    except Exception:
        rel = str(file_path)

    if message is None:
        message = f"Auto-update {rel} — {datetime.now().isoformat(timespec='seconds')}"

    _run_git(["git", "pull", "--rebase", "--autostash", "origin", GIT_BRANCH])  # << qui
    _run_git(["git", "add", "-f", rel])                                         # << forza add
    code, out = _run_git(["git", "commit", "-m", message])
    if code != 0 and ("nothing to commit" in out.lower() or "no changes added to commit" in out.lower()):
        return
    _run_git(["git", "push", "origin", GIT_BRANCH])
    
def append_log_row(row: dict):
    try:
        LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not LOG_CSV.exists()) or (LOG_CSV.stat().st_size == 0)
        safe = {k: row.get(k, "") for k in LOG_COLUMNS}
        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if write_header: w.writeheader()
            w.writerow(safe)
    except Exception as e:
        print("[append_log_row]", e)
        return
    git_auto_commit_push(LOG_CSV)

def build_form_dict(**vals) -> dict:
    bmi = vals["peso"] / max((vals["altezza_cm"]/100.0)**2, 1e-6)
    return {
        "HighBP":int(vals["highbp"]), "HighChol":int(vals["highchol"]), "CholCheck":int(vals["cholcheck"]),
        "BMI":round(float(bmi),1), "Smoker":int(vals["smoker"]), "Stroke":int(vals["stroke"]),
        "HeartDiseaseorAttack":int(vals["heartdisease"]), "PhysActivity":int(vals["physactivity"]),
        "Fruits":int(vals["fruits"]), "Veggies":int(vals["veggies"]), "HvyAlcoholConsump":int(vals["hvyalcoh"]),
        "AnyHealthcare":int(vals["anyhealthcare"]), "NoDocbcCost":int(vals["nomedicalcare"]),
        "GenHlth":int(vals["genhlth"]), "MentHlth":int(vals["menthlth"]), "PhysHlth":int(vals["physhlth"]),
        "DiffWalk":int(vals["diffwalk"]), "Sex":int(vals["gender"]), "Age":int(vals["age"]),
        "Education":int(vals["education"]), "Income":int(vals["income"]),
    }

def log_prediction(form_d: dict, cls: int, prob: float):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": st.session_state.sid,
        "event_type": "prediction",
        **form_d,
        "predicted_class": int(cls),
        "probability": round(float(prob), 4),
        "comune": "", "ospedale": "", "telefono": "", "indirizzo": "", "prenotazioni": "", "note": ""
    }
    append_log_row(row)

def log_contact_view(comune: str, contact_row: dict):
    form_d = st.session_state.last_form or {}
    cls = st.session_state.last_pred
    prob = st.session_state.last_prob
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": st.session_state.sid,
        "event_type": "contact_view",
        **{k: form_d.get(k, "") for k in [
            "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity",
            "Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth",
            "DiffWalk","Sex","Age","Education","Income"
        ]},
        "predicted_class": (int(cls) if cls is not None else ""),
        "probability": (round(float(prob), 4) if prob is not None else ""),
        "comune": comune,
        "ospedale": str(contact_row.get("ospedale","")).strip(),
        "telefono": str(contact_row.get("telefono","")).strip(),
        "indirizzo": str(contact_row.get("indirizzo","")).strip(),
        "prenotazioni": str(contact_row.get("prenotazioni","")).strip(),
        "note": str(contact_row.get("note","")).strip(),
    }
    append_log_row(row)

# ========== MODEL: PREDIZIONE CON PROB ==========
def predict_with_proba(model, model_type: str, X: pd.DataFrame):
    """
    Ritorna (label_classe, probabilità_della_label) con mapping corretto usando model.classes_.
    """
    if model_type == "sklearn":
        classes = getattr(model, "classes_", None)

        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)[0]
            cls_idx = int(np.argmax(p))
            cls_label = int(classes[cls_idx]) if classes is not None else cls_idx
            return cls_label, float(p[cls_idx])

        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            # binario: s è 1D → usa sigmoide e soglia 0.5
            if s.ndim == 1:
                p1 = 1/(1+np.exp(-s[0]))
                cls_label = 1 if p1 >= .5 else 0
                return int(cls_label), float(p1 if cls_label == 1 else 1-p1)
            # multi-classe: softmax
            p = np.exp(s[0]-np.max(s[0])); p /= p.sum()
            cls_idx = int(np.argmax(p))
            cls_label = int(classes[cls_idx]) if classes is not None else cls_idx
            return cls_label, float(p[cls_idx])

        # fallback: niente proba -> usa predict e prob dummy
        pred = model.predict(X)[0]
        return int(pred), 0.50

    # Keras / altri: si assume output in [0..K-1]
    p = model.predict(X, verbose=0)[0]
    cls_idx = int(np.argmax(p))
    return cls_idx, float(p[cls_idx])

# Allineamento feature al training (ordine, tipi, NaN)
def _align_features_for_inference(X: pd.DataFrame, model, meta: dict | None) -> pd.DataFrame:
    # 1) ordine colonne dal training
    feat_order = None
    if meta and isinstance(meta, dict):
        feat_order = meta.get("feature_order") or meta.get("feature_names_out")
    if feat_order is None:
        feat_order = list(getattr(model, "feature_names_in_", [])) or list(X.columns)

    X = X.reindex(columns=feat_order, fill_value=0)

    # 2) tipi numerici
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 3) imputazione minima: usa valori salvati nel meta, altrimenti 0
    if X.isna().any().any():
        impute = (meta or {}).get("impute_values", {}) if isinstance(meta, dict) else {}
        for c in X.columns:
            if X[c].isna().any():
                if c in impute:
                    X[c] = X[c].fillna(impute[c])
        X = X.fillna(0)

    return X

# ========== HOME ==========
def render_home():
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="hero-wow">
            <div class="hero-inner">
              <div class="gradient-title">Valutazione del Rischio Diabete</div>
              <p class="subtitle">Predizione <b>0/1/2</b> con probabilità. Contatti rapidi agli ospedali del tuo comune.</p>
            </div>
           </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    st.markdown(
        """<div class="card"><div class="icon">⚡</div>
           <div style="font-weight:800; font-size:1.05rem; margin-top:.2rem">Valutazione immediata</div>
           <div style="opacity:.9">Inserisci poche informazioni, calcoliamo il <b>BMI</b> e stimiamo il rischio con <b>probabilità</b>.</div>
           </div>""", unsafe_allow_html=True)
    st.markdown(
        """<div class="card"><div class="icon">🧠</div>
           <div style="font-weight:800; font-size:1.05rem; margin-top:.2rem">Modelli ottimizzati</div>
           <div style="opacity:.9">Selezione automatica tra pipeline <b>Sklearn</b> e rete <b>Keras</b> con preprocess dedicato.</div>
           </div>""", unsafe_allow_html=True)
    st.markdown(
        """<div class="card"><div class="icon">🏥</div>
           <div style="font-weight:800; font-size:1.05rem; margin-top:.2rem">Contatti ospedalieri</div>
           <div style="opacity:.9">Cerca il tuo <b>comune</b> e ottieni <b>indirizzo</b>, <b>telefono</b> e <b>Prenotazioni/CUP</b>.</div>
           </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    left, center, right = st.columns([1, 1.2, 1], gap="large")
    with center:
        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.button("📝 Apri il form", type="primary", use_container_width=True, on_click=lambda: go("form"))
        with c2:
            st.button("📍 Vai ai contatti", use_container_width=True, on_click=lambda: go("contacts"))

    st.write("---")
    st.markdown(
        '<div style="text-align:center; opacity:.85">'
        'Creato da <b>Dicorato Martina</b> e <b>Kirollos Seif</b> — v1.0'
        '</div>',
        unsafe_allow_html=True
    )

    render_floating_chat()  # chat in basso a destra

# ========== CONTATTI ==========
def show_contacts_ui():
    st.markdown("### 📍 Prenota vicino a te", help="Cerca il comune e visualizza l'ospedale di riferimento e i contatti.")
    if contacts_df.empty or not comuni_options:
        st.warning("Contatti non trovati. Verifica il file `data/ospedali_milano_comuni_mapping.csv`.")
        return

    q = st.text_input("🔎 Cerca comune", placeholder="Scrivi almeno 2 lettere…")
    if q and len(q.strip()) >= 2:
        qn = _norm_text(q)
        options = [c for c in comuni_options if qn in _norm_text(c)]
    else:
        options = comuni_options

    st.caption(f"Risultati: {len(options)} comuni")

    if options:
        show_list = options[:60]
        rows = (len(show_list) + 5) // 6
        for r in range(rows):
            cols = st.columns(6, gap="small")
            for i, col in enumerate(cols):
                idx = r*6 + i
                if idx >= len(show_list): break
                label = show_list[idx]
                with col:
                    if st.button(label, key=f"pill-{label}", use_container_width=True):
                        st.session_state.selected_comune = label

    if st.session_state.selected_comune:
        sel = st.session_state.selected_comune
        st.markdown(f"#### 📌 Contatti per **{sel}**")
        mask = contacts_df["comune"].astype(str).map(_norm_text).eq(_norm_text(sel))
        sub = contacts_df.loc[mask].copy()
        if sub.empty:
            st.info("Nessun contatto trovato per il comune selezionato.")
            return

        st.markdown('<div class="contact-grid">', unsafe_allow_html=True)
        for i, (_, row) in enumerate(sub.iterrows()):
            osp = (row.get("ospedale") or "Ospedale di riferimento").strip()
            tel = str(row.get("telefono", "")).strip()
            ind = str(row.get("indirizzo", "")).strip()
            cup = str(row.get("prenotazioni", "")).strip()
            note = str(row.get("note", "")).strip()
            tel_html = f'<a href="tel:{tel}">{tel}</a>' if tel else ''
            st.markdown(
                f"""
                <div class="card">
                  <div style="font-weight:750; font-size:1.02rem; margin-bottom:.25rem">{osp}</div>
                  <div style="margin:.2rem 0">{'📞 <b>Telefono:</b> ' + tel_html if tel_html else ''}</div>
                  <div style="margin:.2rem 0">{'📍 <b>Indirizzo:</b> ' + ind if ind else ''}</div>
                  <div style="margin:.2rem 0">{'🗓️ <b>Prenotazioni/CUP:</b> ' + cup if cup else ''}</div>
                  <div style="margin:.2rem 0; opacity:.9">{'📝 <b>Note:</b> ' + note if note else ''}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if i == 0:
                log_contact_view(sel, row.to_dict())

        st.markdown('</div>', unsafe_allow_html=True)

        csv_bytes = sub.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Scarica contatti (CSV)",
            csv_bytes,
            file_name=f"contatti_{_norm_text(sel)}.csv",
            use_container_width=True,
        )

def render_contacts():
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    top = st.columns([1,3])[0]
    if top.button("⬅️ Torna alla Home"):
        go("home")
        st.stop()

    st.markdown("## 🗂️ Contatti ospedalieri per comune")

    if st.button("🔄 Ricarica contatti"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        global contacts_df, comuni_options
        contacts_df, comuni_options = load_contacts()
        st.success("Contatti ricaricati.")

    show_contacts_ui()

    st.markdown("### 🗺️ Mappa (se disponibile)")
    lat_col = next((c for c in contacts_df.columns if c.lower() in ["lat","latitude","latitudine"]), None)
    lon_col = next((c for c in contacts_df.columns if c.lower() in ["lon","lng","longitude","longitudine"]), None)

    def _zoom_for_bbox(df_sel: pd.DataFrame, lat_col: str, lon_col: str) -> float:
        if df_sel.empty:
            return 11.0
        lat_span = float(df_sel[lat_col].max() - df_sel[lat_col].min())
        lon_span = float(df_sel[lon_col].max() - df_sel[lon_col].min())
        span = max(lat_span, lon_span)
        if span < 0.005: return 14.5
        if span < 0.01:  return 14.0
        if span < 0.02:  return 13.5
        if span < 0.05:  return 13.0
        if span < 0.1:   return 12.0
        return 11.0

    if lat_col and lon_col:
        try:
            dfm = contacts_df.copy()
            dfm[lat_col] = pd.to_numeric(dfm[lat_col], errors="coerce")
            dfm[lon_col] = pd.to_numeric(dfm[lon_col], errors="coerce")
            dfm = dfm.dropna(subset=[lat_col, lon_col])
            if not dfm.empty:
                layer_all = pdk.Layer(
                    "ScatterplotLayer", data=dfm,
                    get_position=[lon_col, lat_col],
                    get_radius=600, get_fill_color=[200, 0, 0, 200],
                    pickable=True, radius_min_pixels=4, radius_max_pixels=20,
                )
                layers = [layer_all]

                if st.session_state.selected_comune:
                    sel = _norm_text(st.session_state.selected_comune)
                    df_sel = dfm[dfm["comune"].astype(str).map(_norm_text).eq(sel)]
                    if not df_sel.empty:
                        layer_sel = pdk.Layer(
                            "ScatterplotLayer", data=df_sel,
                            get_position=[lon_col, lat_col],
                            get_radius=900, get_fill_color=[255, 120, 0, 230],
                            pickable=True, radius_min_pixels=6, radius_max_pixels=24,
                        )
                        layers.append(layer_sel)
                        center_lat = float(df_sel[lat_col].mean())
                        center_lon = float(df_sel[lon_col].mean())
                        zoom = _zoom_for_bbox(df_sel, lat_col, lon_col)
                    else:
                        center_lat, center_lon, zoom = MILANO_LAT, MILANO_LON, 11.0
                else:
                    center_lat, center_lon, zoom = MILANO_LAT, MILANO_LON, 11.0

                view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0, bearing=0)
                tooltip = {"html": "<b>{comune}</b><br/>{ospedale}<br/>{indirizzo}<br/>{telefono}",
                           "style": {"backgroundColor": "rgba(0,0,0,0.85)", "color": "white"}}
                st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip), use_container_width=True)
            else:
                st.info("Coordinate presenti ma non valide.")
        except Exception as e:
            st.warning(f"Mappa non disponibile: {e}")
    else:
        st.info("Per la mappa, aggiungi colonne **Latitudine/Longitudine** (o lat/lon) nel CSV.")

    render_floating_chat()  # chat in basso a destra

# ========== FORM ==========
def render_form():
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    top = st.columns([1,3])[0]
    if top.button("⬅️ Torna alla Home"):
        go("home")
        st.stop()

    st.markdown("### 📋 Valutazione personale")
    st.caption("Compila i campi. Il BMI è calcolato automaticamente.")

    s=lambda t: st.selectbox(t,[0,1]); sl=lambda t,a,b,d: st.slider(t,a,b,d)
    c1,c2=st.columns(2)
    with c1:
        gender = st.selectbox("Sesso (0=femmina, 1=maschio)", [0, 1])
        age = st.slider("Età", 18, 90, 40)
        highbp = st.selectbox("Pressione alta? (0=no, 1=si)", [0, 1])
        highchol = st.selectbox("Colesterolo alto? (0=no, 1=si)", [0, 1])
        cholcheck = st.selectbox("Hai controllato il colesterolo negli ultimi 5 anni? (0=no, 1=si)", [0, 1])
        smoker = st.selectbox("Fumi? (0=no, 1=si)", [0, 1])
        stroke = st.selectbox("Ictus in passato? (0=no, 1=si)", [0, 1])
        heartdisease = st.selectbox("Malattie cardiache? (0=no, 1=si)", [0, 1])
        physactivity = st.selectbox("Attività fisica regolare? (0=no, 1=si)", [0, 1])
        fruits = st.selectbox("Frutta regolare? (0=no, 1=si)", [0, 1])
    with c2:
        veggies = st.selectbox("Verdura regolare? (0=no, 1=si)", [0, 1])
        hvyalcoh = st.selectbox("Consumo elevato di alcol? (0=no, 1=si)", [0, 1])
        anyhealthcare = st.selectbox("Accesso a servizi sanitari? (0=no, 1=si)", [0, 1])
        nomedicalcare = st.selectbox("Eviti cure per costi? (0=no, 1=si)", [0, 1])
        genhlth = st.slider("Salute generale quantificata (1 ottima – 5 pessima)", 1, 5, 3)
        menthlth = st.slider("Giorni con problemi mentali negli ultimi 30 gg", 0, 30, 2)
        physhlth = st.slider("Giorni con problemi fisici negli ultimi 30 gg", 0, 30, 2)
        diffwalk = st.selectbox("Difficoltà a camminare? negli ultimi 30 gg", [0, 1])
        education = st.slider("Livello di Istruzione (1 Elementare - 2 Scuole Medie - 3 Diploma Superiore - 4 Formazione post Diploma – 5 Laurea Triennale - 6 Laurea Magistrale)", 1, 6, 4)
        income = st.slider("Livello Reddito (1–8)", 1, 8, 4)
    b1,b2=st.columns(2)
    with b1: peso=st.number_input("Peso (kg)",30.0,250.0,70.0,0.5)
    with b2: h=st.number_input("Altezza (cm)",100.0,220.0,170.0,0.5)
    bmi = peso/((max(h,.0001)/100)**2); st.caption(f"👉 BMI: **{bmi:.2f}**")

    if st.button("🔎 Calcola risultato", type="primary", use_container_width=True):
        form_d = build_form_dict(
            gender=gender, age=age, highbp=highbp, highchol=highchol, cholcheck=cholcheck,
            smoker=smoker, stroke=stroke, heartdisease=heartdisease, physactivity=physactivity,
            fruits=fruits, veggies=veggies, hvyalcoh=hvyalcoh, anyhealthcare=anyhealthcare,
            nomedicalcare=nomedicalcare, genhlth=genhlth, menthlth=menthlth, physhlth=physhlth,
            diffwalk=diffwalk, education=education, income=income, peso=peso, altezza_cm=h
        )
        st.session_state.last_form = form_d

        rec = pd.DataFrame([form_d])

        try:
            model, model_type, meta = load_best_model()
            X = preprocess_for_inference(rec, meta)
            X = _align_features_for_inference(X, model, meta)

            # Guard-rails: evita input degenerati
            if X.isna().any().any():
                st.error(f"NaN nell'input del modello: {X.isna().sum().to_dict()}")
                return
            if X.shape[1] == 0:
                st.error("Nessuna feature dopo il preprocess: controlla preprocess_for_inference/meta.")
                return

            cls, prob = predict_with_proba(model, model_type, X)

        except Exception as e:
            st.error(f"Errore durante la predizione: {e}")
            return

        st.session_state.last_pred = cls
        st.session_state.last_prob = prob

        try:
            log_prediction(form_d, cls, prob)
        except Exception:
            pass

        if cls == 2:
            color, label = "#ef4444", "Rischio alto (2)"
        elif cls == 1:
            color, label = "#f59e0b", "Rischio medio (1)"
        else:
            color, label = "#10b981", "Basso rischio (0)"

        prob_pct = int(prob * 100)
        st.markdown(
            f"""
            <div class="card result" style="border-left-color:{color}">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
                <div style="font-weight:800;font-size:1.05rem">Risultato: {label}</div>
                <span class="badge">Probabilità: {prob_pct}%</span>
              </div>
              <div style="margin-top:.6rem" class="prob-wrap">
                <div class="prob-fill" style="width:{prob_pct}%"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.last_pred is not None:
        cls = int(st.session_state.last_pred)
        if cls == 0:
            st.info("✅ **Basso rischio** — se vuoi puoi comunque **prenotare una visita di controllo**.")
        else:
            st.warning("⚠️ **Rischio medio/alto** — è **fortemente consigliato** prenotare un **controllo medico**.")
        st.button("🔎 Cerca contatti ospedale", use_container_width=True, on_click=lambda: go("contacts"))

    render_floating_chat()  # chat in basso a destra

# ========== FLOATING CHAT ==========
_SITE_FAQ = (
    "Come usare il sito:\n"
    "1) Vai su '📝 Apri il form' e compila i campi.\n"
    "2) Premi '🔎 Calcola risultato' per ottenere classe e probabilità.\n"
    "3) Se vuoi prenotare, vai su '📍 Vai ai contatti' e cerca il tuo comune per telefono/indirizzo/CUP.\n"
    "⚠️ Non è un dispositivo medico: per dubbi, rivolgersi al medico."
)

def render_floating_chat():
    """Expander flottante sempre visibile in basso a destra (pallino chiuso, finestra quando aperto)."""
    label = " "  # l'icona 💬 viene inserita via CSS nell'header
    with st.expander(label, expanded=False):
        if _CHAT_IMPORT_ERROR:
            st.warning(f"Chat non disponibile: {_CHAT_IMPORT_ERROR}")
        elif answer_with_rag is None:
            st.warning("Chat non disponibile (modulo chatbot mancante).")
        else:
            if not st.session_state.chat_history:
                # Primo messaggio dell'assistente in NERO per leggibilità
                st.session_state.chat_history = [("assistant",
                    "Ciao! Posso aiutarti a capire a cosa serve il sito, come usarlo (form/contatti) "
                    "e rispondere a domande sul diabete in generale. Scrivimi qui sotto. 🙂")]

            for role, content in st.session_state.chat_history:
                bubble_bg = "#eef2ff" if role == "assistant" else "#e0f2fe"
                align = "flex-start" if role == "assistant" else "flex-end"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:{align}; margin:.2rem 0">
                      <div style="max-width:90%; background:{bubble_bg}; padding:.6rem .75rem; border-radius:12px; box-shadow:0 1px 3px rgba(16,24,40,.12)">
                        <div style="opacity:.7; font-size:.8rem; margin-bottom:.15rem; color:#111">{'Assistente' if role=='assistant' else 'Tu'}</div>
                        <div style="white-space:pre-wrap; color:#111">{content}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.form("chat_form", clear_on_submit=True):
                q = st.text_input("Scrivi qui la tua domanda…", value=st.session_state.get("chat_input",""))
                send = st.form_submit_button("Invia")
            if send and q.strip():
                user_q = q.strip()
                st.session_state.chat_history.append(("user", user_q))
                try:
                    ans = answer_with_rag(user_q, site_faq=_SITE_FAQ)
                except Exception as e:
                    ans = f"⚠️ Errore: {e}"
                st.session_state.chat_history.append(("assistant", ans))
                st.session_state.chat_input = ""

                # forza il re-render per mostrare SUBITO la nuova risposta
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()

# ========== ROUTING ==========
if st.session_state.view == "home":
    render_home()
elif st.session_state.view == "form":
    render_form()
elif st.session_state.view == "contacts":
    render_contacts()
else:
    go("home")
    render_home()
