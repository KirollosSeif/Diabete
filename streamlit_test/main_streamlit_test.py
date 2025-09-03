# streamlit_test/main_streamlit_test.py
import sys
from pathlib import Path
from datetime import datetime, timezone
import os
import subprocess

import pandas as pd
import streamlit as st

# === PATH PROGETTO / IMPORT ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # root del progetto
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_best_model, predict_with_model, preprocess_for_inference  # noqa: E402

# === CONFIG & STILE ===
st.set_page_config(page_title="Rischio Diabete — Demo", page_icon="🩺", layout="wide")
st.markdown("""
<style>
/* Nascondi sidebar e bottone header */
section[data-testid="stSidebar"]{display:none}
header [data-testid="baseButton-headerNoPadding"]{visibility:hidden}

/* Un filo più in basso rispetto a prima */
div.block-container{padding-top:2.2rem;padding-bottom:1.6rem}

/* Hero box */
.hero{
  background: radial-gradient(1200px 600px at 8% 10%, rgba(0,120,255,.08), transparent 60%),
              radial-gradient(1000px 500px at 90% 30%, rgba(255,60,140,.08), transparent 60%);
  border:1px solid rgba(0,0,0,.06); border-radius:24px; padding:2rem;
  box-shadow:0 10px 30px rgba(0,0,0,.06);
}
/* Centra solo il titolo dell’hero (e volendo anche il testo) */
.hero h1{ text-align:center; margin:0 0 .25rem 0; }
.hero p{ text-align:center; margin:0; } /* opzionale: centra il sottotitolo */

/* Card generiche */
.card{
  border:1px solid rgba(0,0,0,.06); border-radius:18px; padding:1rem; background:#fff;
  box-shadow:0 6px 16px rgba(0,0,0,.05);
}

/* Bottoni base */
.btn{
  display:inline-block; padding:.7rem 1rem; border-radius:14px; font-weight:600;
  border:1px solid rgba(0,0,0,.1); background:#fff;
}

/* Barra superiore (se usata) */
.topbar{display:flex; gap:.5rem; margin-bottom:.5rem}
</style>



""", unsafe_allow_html=True)

DATA_DIR = PROJECT_ROOT / "data"
METRICS_DIR = DATA_DIR / "metrics"
FEEDBACK = DATA_DIR / "feedback_test.csv"   # <<< unico cambio di percorso

def _auto_git_sync(file_paths, commit_msg="Auto: update feedback_test.csv"):
    """
    Add/commit/pull --rebase --autostash/push dei file passati.
    Se git fallisce, NON blocca il salvataggio locale.
    """
    try:
        repo_root = PROJECT_ROOT
        rels = [str(Path(p).resolve().relative_to(repo_root)) for p in file_paths]

        # 1) assicurati di essere allineata col remoto (autostash evita l'errore che vedi)
        subprocess.run(
            ["git", "pull", "--rebase", "--autostash", "origin", "main"],
            cwd=repo_root, check=False, capture_output=True
        )
        # 2) aggiungi (forzando in caso di .gitignore su data/)
        subprocess.run(
            ["git", "add", "-f"] + rels,
            cwd=repo_root, check=True, capture_output=True
        )
        # 3) commit (se non c'è nulla da committare, non esplodiamo)
        cp = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=repo_root, check=False, capture_output=True, text=True
        )
        # 4) push (se c’è un commit)
        if cp.returncode == 0:
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=repo_root, check=False, capture_output=True
            )
        st.toast("📤 Sincronizzato su GitHub (test).", icon="✅")
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or b"").decode(errors="ignore")
        st.toast("⚠️ Sync Git fallita (dati salvati solo in locale).", icon="⚠️")
        st.caption(f"Dettagli Git: {msg.strip()[:400]}")



# === ROUTER STATO ===
st.session_state.setdefault("view", "home")
def go(view: str): st.session_state.view = view

# === HELPER: COSTRUISCI RECORD DAL FORM ===
def _build_record(**vals) -> pd.DataFrame:
    bmi = vals["peso"] / max((vals["altezza_cm"]/100.0)**2, 1e-6)
    return pd.DataFrame([{
        "HighBP":int(vals["highbp"]), "HighChol":int(vals["highchol"]), "CholCheck":int(vals["cholcheck"]),
        "BMI":round(float(bmi),1), "Smoker":int(vals["smoker"]), "Stroke":int(vals["stroke"]),
        "HeartDiseaseorAttack":int(vals["heartdisease"]), "PhysActivity":int(vals["physactivity"]),
        "Fruits":int(vals["fruits"]), "Veggies":int(vals["veggies"]), "HvyAlcoholConsump":int(vals["hvyalcoh"]),
        "AnyHealthcare":int(vals["anyhealthcare"]), "NoDocbcCost":int(vals["nomedicalcare"]),
        "GenHlth":int(vals["genhlth"]), "MentHlth":int(vals["menthlth"]), "PhysHlth":int(vals["physhlth"]),
        "DiffWalk":int(vals["diffwalk"]), "Sex":int(vals["gender"]), "Age":int(vals["age"]),
        "Education":int(vals["education"]), "Income":int(vals["income"]),
    }])

# === HOME ===
def render_home():
    st.markdown(
        """
        <div class="hero">
          <h1>🩺 Valutazione del Rischio Diabete (demo)</h1>
          <p class="small">
            Inserisci poche informazioni, ottieni la <b>classe 0/1/2</b> e salva un feedback per il miglioramento.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🎯 Obiettivo\n- Predizione rapida\n- Feedback per retrain\n- Report automatici")
    with c2:
        st.markdown("### 📦 Modello")
        try:
            _, model_type, meta = load_best_model()
            sk = meta.get("sklearn_score"); ke = meta.get("keras_score")
            st.markdown(f"In uso: **{model_type}**")
            st.caption(f"Sklearn CV: {sk:.4f}" if sk is not None else "Sklearn CV: n/d")
            st.caption(f"Keras Val: {ke:.4f}" if ke is not None else "Keras Val: n/d")
        except Exception as e:
            st.warning(f"Modello non caricato: {e}")
    with c3:
        st.markdown("### 🔒 Privacy\nDati di test salvati in `data/feedback_test.csv`.")
    st.write("")
    a, b, _ = st.columns([1,1,1])
    left, center, right = st.columns([1, 1.2, 1])  # colonna centrale leggermente più larga
    with center:
        b1, b2 = st.columns(2)
        with b1:
            go_form = st.button("📝 Apri Form", type="primary", use_container_width=True, key="home_btn_form")
        with b2:
            go_monitor = st.button("📈 Apri Monitoraggio", use_container_width=True, key="home_btn_monitor")

    if go_form:
        go("form")
    if go_monitor:
        go("monitor")
    # --- Mini grafico attività (ultimi 30 giorni) ---
    try:
        if FEEDBACK.exists():
            dfh = pd.read_csv(FEEDBACK)
            if not dfh.empty and "timestamp" in dfh.columns:
                t = pd.to_datetime(dfh["timestamp"], utc=True, errors="coerce").dropna()
                if not t.empty:
                    # conteggio per giorno (UTC) e reindex sugli ultimi 30 giorni
                    daily = (
                        t.dt.floor("D")
                         .value_counts()
                         .rename_axis("day")
                         .sort_index()
                         .rename("invii")
                    )
                    idx = pd.date_range(
                        end=pd.Timestamp.now(tz="UTC").normalize(),
                        periods=30, freq="D", tz="UTC"
                    )
                    daily = daily.reindex(idx, fill_value=0)

                    st.markdown("#### 📈 Attività recente (ultimi 30 giorni)")
                    st.line_chart(daily)
            else:
                st.caption("Nessun dato attività ancora — compila il form per vedere il grafico.")
        else:
            st.caption("Il file di feedback non esiste ancora.")
    except Exception as e:
        st.caption(f"Grafico attività non disponibile: {e}")






# === FORM (con feedback) ===
def render_form():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if st.button("⬅️ Home"): go("home"); st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## 📝 Form di autovalutazione")
    try:
        model, model_type, meta = load_best_model()
        st.caption(f"Selezione automatica → usando: **{model_type}**")
    except Exception as e:
        st.error(f"Errore nel caricamento del modello: {e}"); return

    col1, col2 = st.columns(2)
    with col1:
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
    with col2:
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

    c1, c2 = st.columns(2)
    with c1: peso = st.number_input("Peso (kg)", 30.0, 250.0, 70.0, 0.5)
    with c2: altezza_cm = st.number_input("Altezza (cm)", 100.0, 220.0, 170.0, 0.5)
    bmi = peso / max((altezza_cm/100.0)**2, 1e-6)
    st.caption(f"👉 BMI: **{bmi:.2f}**")

    if st.button("🧪 Calcola predizione", type="primary"):
        rec = _build_record(
            gender=gender, age=age, highbp=highbp, highchol=highchol, cholcheck=cholcheck,
            smoker=smoker, stroke=stroke, heartdisease=heartdisease, physactivity=physactivity,
            fruits=fruits, veggies=veggies, hvyalcoh=hvyalcoh, anyhealthcare=anyhealthcare,
            nomedicalcare=nomedicalcare, genhlth=genhlth, menthlth=menthlth, physhlth=physhlth,
            diffwalk=diffwalk, education=education, income=income, peso=peso, altezza_cm=altezza_cm
        )
        X = preprocess_for_inference(rec, meta)
        try:
            pred = predict_with_model(model, model_type, X)
            pred_class = int(pred[0])
        except Exception as e:
            st.error(f"Errore durante la predizione: {e}"); return

        st.session_state["pending_record"] = rec
        st.session_state["pending_model_type"] = model_type

        # nome artefatto (per report)
        artifacts_dir = PROJECT_ROOT / "data" / "grid_search_results"
        if model_type == "sklearn":
            cand = list(artifacts_dir.glob("*_optimized_model.pkl"))
            model_artifact = cand[0].name if cand else "unknown.pkl"
        else:
            model_artifact = "best_keras_model.keras" if (artifacts_dir / "best_keras_model.keras").exists() else "best_keras_model.h5"
        st.session_state["pending_model_artifact"] = model_artifact
        st.session_state["pending_pred_class"] = pred_class

        st.success(f"Predizione: **{pred_class}**  (0=No, 1=Pre, 2=Diabete)")
        st.info("Conferma il feedback e poi salva.")

    if st.session_state.get("pending_record") is not None:
        pred_class = st.session_state.get("pending_pred_class", 0)
        fb = st.radio("Questo risultato è corretto?", ["Sì", "No"], horizontal=True, index=0)
        label = pred_class if fb == "Sì" else st.selectbox("Valore corretto:", [0, 1, 2], index=pred_class)

        if st.button("💾 Salva con feedback"):
            FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
            out = st.session_state["pending_record"].copy()
            out["Predicted"] = pred_class
            out["Diabetes_012"] = int(label)
            out["timestamp"] = datetime.now(timezone.utc).isoformat()
            out["model_type"] = st.session_state["pending_model_type"]
            out["model_artifact"] = st.session_state["pending_model_artifact"]

            if FEEDBACK.exists():
                df_old = pd.read_csv(FEEDBACK)
                df_new = pd.concat([df_old, out], ignore_index=True)
            else:
                df_new = out
            df_new.to_csv(FEEDBACK, index=False)

            _auto_git_sync([FEEDBACK])

            for k in ["pending_record","pending_model_type","pending_model_artifact","pending_pred_class"]:
                st.session_state[k] = None
            st.success("✅ Salvato in data/feedback_test.csv.")  # testo aggiornato

# === REPORT INLINE (semplice e affidabile) ===
def _build_reports_inline():
    """Genera weekly/by_model/confusion partendo *sempre* da FEEDBACK."""
    if not FEEDBACK.exists():
        return False, f"Non trovo {FEEDBACK}. Fai almeno un invio dal Form."
    df = pd.read_csv(FEEDBACK)
    if df.empty:
        return False, f"{FEEDBACK.name} è vuoto."

    df["is_correct"] = (df["Predicted"] == df["Diabetes_012"]).astype(int)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Weekly (settimana = inizio lunedì)
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        weekly = (
            df.assign(week_start=df["timestamp"].dt.to_period("W-MON").apply(lambda p: p.start_time))
              .groupby("week_start")
              .agg(tests=("is_correct","size"), accuracy=("is_correct","mean"))
              .reset_index()
        )
        weekly["accuracy"] = weekly["accuracy"].round(4)
        weekly.to_csv(METRICS_DIR / "weekly_report.csv", index=False)

    # by model
    if "model_artifact" in df.columns:
        by_model = (
            df.groupby("model_artifact")
              .agg(tests=("is_correct","size"), accuracy=("is_correct","mean"))
              .reset_index()
        )
        by_model["accuracy"] = by_model["accuracy"].round(4)
        by_model.to_csv(METRICS_DIR / "by_model_report.csv", index=False)

    # confusion
    cm = pd.crosstab(df["Diabetes_012"], df["Predicted"], rownames=["True"], colnames=["Pred"])
    cm.to_csv(METRICS_DIR / "confusion_matrix_overall.csv")

    return True, "Report rigenerati dal CSV."

# === MONITORAGGIO ===
def render_monitor():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    if st.button("⬅️ Home"): go("home"); st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## 📈 Monitoraggio modello")
    weekly_path = METRICS_DIR / "weekly_report.csv"
    by_model_path = METRICS_DIR / "by_model_report.csv"
    cm_path = METRICS_DIR / "confusion_matrix_overall.csv"

    # se i file non esistono ancora, creali adesso
    if not (weekly_path.exists() or by_model_path.exists() or cm_path.exists()):
        ok, msg = _build_reports_inline()
        (st.success if ok else st.warning)(msg)

    # bottone aggiorna (rigenera sempre dai feedback)
    if st.button("🔄 Aggiorna report"):
        ok, msg = _build_reports_inline()
        (st.success if ok else st.warning)(msg)

    # UI
    if weekly_path.exists():
        weekly = pd.read_csv(weekly_path, parse_dates=["week_start"])
        st.subheader("Andamento settimanale")
        if not weekly.empty:
            st.line_chart(weekly.set_index("week_start")[["tests","accuracy"]])
            last = weekly.sort_values("week_start").tail(1)
            c1, c2 = st.columns(2)
            c1.metric("Test ultima settimana", int(last["tests"].iloc[0]))
            c2.metric("Accuracy ultima settimana", f"{last['accuracy'].iloc[0]*100:.1f}%")
        else:
            st.info("Nessun dato settimanale ancora.")

    if by_model_path.exists():
        st.subheader("Prestazioni per modello")
        st.dataframe(pd.read_csv(by_model_path), use_container_width=True)

    if cm_path.exists():
        st.subheader("Confusion matrix (complessiva)")
        st.dataframe(pd.read_csv(cm_path, index_col=0), use_container_width=True)

# === ROUTING ===
v = st.session_state.view
if v == "home":    render_home()
elif v == "form":  render_form()
elif v == "monitor": render_monitor()
else:              go("home")
