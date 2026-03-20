"""
interface.py — F1 Podium Predictor · Interface Streamlit professionnelle
Grille, calendrier et circuits viennent exclusivement de l'API OpenF1.
Le pipeline de données se lance automatiquement en fond au démarrage.
"""
import os
import time
import pickle
import threading
import pandas as pd
import torch
import streamlit as st

import pipeline
from dashboard import gui as f1_viz
from data_fetcher import (
    get_current_grid, get_circuit_list, get_next_race,
    get_acronym, preload_acronyms,
)
from HybridPodium import HybridNet, train_hybrid_model
from LapTimePredictor import ExpertPodiumPredictor

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS — Thème F1 : carbone noir + rouge E10600
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --red:    #E10600;
    --dark:   #0d0d0d;
    --panel:  #1a1a1a;
    --border: #2a2a2a;
    --muted:  #555;
    --text:   #e0e0e0;
    --font-t: 'Barlow Condensed', sans-serif;
    --font-m: 'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--dark) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"]  { background: transparent !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: var(--font-t) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--red) !important;
    border-bottom: 2px solid var(--red) !important;
}

/* Boutons */
.stButton > button {
    background: var(--red) !important;
    color: white !important;
    border: none !important;
    font-family: var(--font-t) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-m) !important;
    border-radius: 2px !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div > div { background: var(--red) !important; }

/* Métriques */
[data-testid="metric-container"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label {
    font-family: var(--font-t) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-m) !important;
    color: var(--text) !important;
}

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stAlert"] { background: var(--panel) !important; border-radius: 2px !important; }

/* Composants custom */
.page-title   { font-family:var(--font-t);font-weight:900;font-size:2.8rem;letter-spacing:-0.01em;color:white;line-height:1; }
.page-title span { color:var(--red); }
.f1-label     { font-family:var(--font-t);font-size:0.68rem;font-weight:600;letter-spacing:0.2em;text-transform:uppercase;color:var(--muted); }
.f1-tag       { display:inline-block;background:var(--red);color:white;font-family:var(--font-t);font-weight:700;font-size:0.68rem;letter-spacing:0.15em;text-transform:uppercase;padding:2px 8px;border-radius:2px; }

/* Podium */
.podium-card  { background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:1.5rem 1rem;text-align:center;position:relative;overflow:hidden; }
.podium-card::before { content:'';position:absolute;top:0;left:0;right:0;height:3px; }
.p1::before { background:#FFD700; }
.p2::before { background:#C0C0C0; }
.p3::before { background:#CD7F32; }
.pod-pos  { font-family:var(--font-t);font-size:0.72rem;font-weight:700;letter-spacing:0.2em;color:var(--muted);text-transform:uppercase; }
.pod-name { font-family:var(--font-t);font-weight:900;font-size:3rem;letter-spacing:-0.02em;color:white;line-height:1.1; }
.pod-team { font-family:var(--font-m);font-size:0.68rem;color:var(--muted);margin-top:0.3rem; }
.pod-scr  { font-family:var(--font-m);font-size:0.75rem;color:var(--red);margin-top:0.5rem;font-weight:600; }

/* Classement */
.rank-table   { background:var(--panel);border:1px solid var(--border);border-radius:4px;overflow:hidden;margin-top:1.2rem; }
.rank-row     { display:flex;align-items:center;gap:1rem;padding:0.55rem 1rem;border-bottom:1px solid var(--border); }
.rank-row:last-child { border-bottom:none; }
.rk-pos       { width:28px;font-family:var(--font-m);font-size:0.75rem;color:var(--muted);flex-shrink:0; }
.rk-drv       { width:48px;font-family:var(--font-m);font-weight:600;font-size:0.9rem;color:white;flex-shrink:0; }
.rk-team      { flex-grow:1;font-family:var(--font-m);font-size:0.7rem;color:var(--muted); }
.rk-scr       { font-family:var(--font-m);font-size:0.75rem;color:var(--red);font-weight:600;flex-shrink:0; }

/* Loader */
.loader-box   { background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:2.5rem 2rem;text-align:center; }
.loader-title { font-family:var(--font-t);font-weight:700;font-size:1rem;letter-spacing:0.2em;text-transform:uppercase;color:var(--text);margin:0.8rem 0 0.3rem; }
.loader-step  { font-family:var(--font-m);font-size:0.75rem;color:var(--muted); }

/* Status dots */
.status-grid  { font-family:var(--font-m);font-size:0.7rem;color:var(--muted);text-align:right;line-height:2.2; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
MODEL_LSTM   = "expert_model.pth"
MODEL_HYBRID = "hybrid_model.pth"
MAP_LSTM     = "circuit_map.pkl"
MAP_HYBRID   = "hybrid_mappings.pkl"
RESULTS_FILE = "f1_race_results.pkl"
GLOBAL_MAX_LAP_TIME = 150.0
SEQUENCE_LENGTH = 5

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "pipeline_launched": False,
    "api_loaded":        False,
    "circuits":          [],
    "grid":              [],
    "next_race":         None,
    "prediction":        None,
    "pred_circuit":      None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────
# PIPELINE — DÉMARRAGE AUTO
# ─────────────────────────────────────────────────────────────
if not st.session_state.pipeline_launched:
    st.session_state.pipeline_launched = True
    pipeline.start_in_background()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _models_ok() -> bool:
    return all(os.path.exists(f) for f in [MODEL_LSTM, MODEL_HYBRID, MAP_LSTM, MAP_HYBRID])

def _data_ok() -> bool:
    return os.path.exists(RESULTS_FILE)

def _dot(ok: bool) -> str:
    return "🟢" if ok else "🔴"

@st.cache_data(ttl=3600, show_spinner=False)
def _load_api():
    circuits  = get_circuit_list()
    grid      = get_current_grid()
    next_race = get_next_race()
    if grid:
        preload_acronyms(grid)
    return circuits, grid, next_race

@st.cache_resource(show_spinner=False)
def _load_models():
    try:
        dev  = torch.device("cpu")
        ckpt = torch.load(MODEL_LSTM, map_location=dev)
        lstm = ExpertPodiumPredictor(ckpt["num_circuits"])
        lstm.load_state_dict(ckpt["state_dict"])
        lstm.eval()
        with open(MAP_LSTM, "rb") as f:
            lm = pickle.load(f)
        with open(MAP_HYBRID, "rb") as f:
            hm = pickle.load(f)
        hyb = HybridNet(len(hm["d"]), len(hm["t"]), len(hm["c"]))
        hyb.load_state_dict(torch.load(MODEL_HYBRID, map_location=dev))
        hyb.eval()
        return lstm, hyb, lm, hm
    except Exception as e:
        return None, None, None, str(e)

def _predict(circuit, grid, lstm, hybrid, lm, hm):
    cl = lm.get(circuit, 0)
    ch = hm["c"].get(circuit, 0)
    base = torch.tensor([[[90.0 / GLOBAL_MAX_LAP_TIME]] * SEQUENCE_LENGTH], dtype=torch.float32).unsqueeze(-1)
    lap  = torch.tensor([[0.5]], dtype=torch.float32)
    top_teams = {"Red Bull Racing", "McLaren", "Ferrari"}

    times, ref = [], {}
    with torch.no_grad():
        for num, team in grid:
            try:
                p = lstm(base, torch.tensor([num], dtype=torch.long), torch.tensor([cl], dtype=torch.long), lap).item()
            except Exception:
                p = None
            if p is not None:
                times.append({"d": num, "t": team, "p": p})
                if team not in ref or p < ref[team]:
                    ref[team] = p

    known = {x["d"] for x in times}
    for num, team in grid:
        if num not in known:
            times.append({"d": num, "t": team, "p": ref.get(team, 0.65)})

    best = min(x["p"] for x in times if x["p"] > 0)
    lb = []
    with torch.no_grad():
        for x in times:
            deficit = x["p"] / best
            score = hybrid(
                torch.tensor([hm["d"].get(x["d"], 0)]),
                torch.tensor([hm["t"].get(x["t"], 0)]),
                torch.tensor([ch]),
                torch.tensor([[deficit]]),
            ).item()
            if x["t"] in top_teams:
                score += 0.05
            lb.append({"driver": x["d"], "acr": get_acronym(x["d"]), "team": x["t"], "score": score})

    lb.sort(key=lambda x: x["score"], reverse=True)
    return lb

def _podium_card(pos_label, css, acr, team, score):
    st.markdown(f"""
    <div class="podium-card {css}">
        <div class="pod-pos">{pos_label}</div>
        <div class="pod-name">{acr}</div>
        <div class="pod-team">{team}</div>
        <div class="pod-scr">score {score:.3f}</div>
    </div>""", unsafe_allow_html=True)

def _rank_table(lb):
    rows = "".join(f"""
    <div class="rank-row">
        <span class="rk-pos">P{i}</span>
        <span class="rk-drv">{r['acr']}</span>
        <span class="rk-team">{r['team']}</span>
        <span class="rk-scr">{r['score']:.3f}</span>
    </div>""" for i, r in enumerate(lb[3:], 4))
    st.markdown(f'<div class="rank-table">{rows}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# EN-TÊTE
# ─────────────────────────────────────────────────────────────
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem">
        <div class="f1-label">Deep Learning · OpenF1 API · PyTorch</div>
        <div class="page-title">F1 <span>PODIUM</span> PREDICTOR</div>
    </div>""", unsafe_allow_html=True)

with h_right:
    is_running = pipeline.state["running"]
    st.markdown(f"""
    <div class="status-grid" style="padding-top:1.8rem">
        {_dot(not is_running and _data_ok())} DONNÉES<br>
        {_dot(_models_ok())} MODÈLES<br>
        {'🔴 SYNC' if is_running else '⬛ IDLE'} PIPELINE
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_pred, tab_data, tab_stats, tab_dash = st.tabs([
    "🔮  PRÉDICTION", "📡  DONNÉES", "📊  ANALYSE", "🏁  DASHBOARD"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════
with tab_pred:

    # Pipeline encore en cours → loader + polling
    if pipeline.state["running"]:
        st.markdown(f"""
        <div class="loader-box">
            <div class="f1-tag">LIVE UPDATE</div>
            <div class="loader-title">SYNCHRONISATION EN COURS</div>
            <div class="loader-step">{pipeline.state['step']}</div>
        </div>""", unsafe_allow_html=True)
        st.progress(pipeline.state["progress"])
        time.sleep(1.5)
        st.rerun()

    elif pipeline.state["error"]:
        st.error(f"❌ Pipeline : {pipeline.state['error']}")

    else:
        # Chargement API (une seule fois, mis en cache 1h)
        if not st.session_state.api_loaded:
            with st.spinner("Récupération du calendrier et de la grille (API OpenF1)..."):
                c, g, nr = _load_api()
                st.session_state.circuits  = c
                st.session_state.grid      = g
                st.session_state.next_race = nr
                st.session_state.api_loaded = True

        circuits  = st.session_state.circuits
        grid      = st.session_state.grid
        next_race = st.session_state.next_race

        # ── Sélecteur de circuit ──────────────────────────────
        col_sel, col_btn = st.columns([2, 1])
        with col_sel:
            default = 0
            if next_race and circuits:
                loc = next_race.get("location", "")
                if loc in circuits:
                    default = circuits.index(loc)

            selected = st.selectbox(
                "Circuit",
                circuits if circuits else ["—"],
                index=default,
                label_visibility="collapsed",
            )

            if next_race:
                date = next_race.get("date_start", "")[:10]
                loc  = next_race.get("location", "?")
                st.markdown(
                    f'<span class="f1-label">Prochaine course : </span>'
                    f'<span class="f1-tag">{loc} · {date}</span>',
                    unsafe_allow_html=True,
                )

        with col_btn:
            simulate = st.button("⚡ SIMULER", use_container_width=True,
                                 disabled=(not circuits or not grid))

        st.markdown("<div style='margin-top:1.5rem'>", unsafe_allow_html=True)

        # ── Lancement simulation ──────────────────────────────
        if simulate and selected and selected != "—":
            if not _models_ok():
                st.error("Modèles introuvables. Lance l'entraînement dans l'onglet Données.")
            elif not grid:
                st.error("Grille vide. Problème de connexion API.")
            else:
                with st.status(f"Simulation — {selected}", expanded=True) as status:
                    st.write(f"Spécialisation HybridNet pour {selected}...")
                    ok = train_hybrid_model(selected)
                    if not ok:
                        status.update(label="Données insuffisantes ❌", state="error")
                    else:
                        st.write("Calcul LSTM + HybridNet...")
                        st.cache_resource.clear()
                        lstm, hyb, lm, hm = _load_models()
                        if lstm is None:
                            status.update(label="Erreur modèle ❌", state="error")
                            st.error(hm)
                        else:
                            result = _predict(selected, grid, lstm, hyb, lm, hm)
                            st.session_state.prediction = result
                            st.session_state.pred_circuit = selected
                            status.update(label="Simulation terminée ✅", state="complete")

        # ── Affichage résultat ────────────────────────────────
        if st.session_state.prediction:
            lb  = st.session_state.prediction
            ckt = st.session_state.pred_circuit

            st.markdown(f"""
            <div style="margin:1.5rem 0 1rem">
                <div class="f1-label">Résultat de simulation</div>
                <div style="font-family:var(--font-t);font-weight:900;font-size:1.8rem;
                            letter-spacing:0.06em;text-transform:uppercase">{ckt}</div>
            </div>""", unsafe_allow_html=True)

            # Podium : P2 | P1 | P3
            c2, c1, c3 = st.columns(3)
            with c1: _podium_card("P1 · VAINQUEUR", "p1", lb[0]["acr"], lb[0]["team"], lb[0]["score"])
            with c2: _podium_card("P2", "p2", lb[1]["acr"], lb[1]["team"], lb[1]["score"])
            with c3: _podium_card("P3", "p3", lb[2]["acr"], lb[2]["team"], lb[2]["score"])

            # P4 → fin
            st.markdown("<div class='f1-label' style='margin-top:1.5rem'>Classement complet</div>", unsafe_allow_html=True)
            _rank_table(lb)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — DONNÉES
# ══════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("<div class='f1-label' style='margin-bottom:1rem'>État du système</div>", unsafe_allow_html=True)

    # ── Métriques ─────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        val = "—"
        if _data_ok():
            df_r = pd.read_pickle(RESULTS_FILE)
            val  = str(df_r["session_key"].nunique())
        st.metric("Courses en base", val)
    with m2:
        val = "—"
        if os.path.exists("f1_master_database.pkl"):
            df_m = pd.read_pickle("f1_master_database.pkl")
            val  = f"{len(df_m):,}"
        st.metric("Tours enregistrés", val)
    with m3:
        val = "—"
        if os.path.exists("f1_hybrid_dataset.pkl"):
            df_h = pd.read_pickle("f1_hybrid_dataset.pkl")
            val  = f"{len(df_h):,}"
        st.metric("Entrées hybrides", val)
    with m4:
        st.metric("Modèles", "✅ Prêts" if _models_ok() else "❌ Manquants")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Pipeline ──────────────────────────────────────────────
    st.markdown("<div class='f1-label' style='margin-bottom:0.8rem'>Pipeline de données</div>", unsafe_allow_html=True)

    col_pipe, col_refresh = st.columns([3, 1])
    with col_pipe:
        if pipeline.state["running"]:
            st.progress(pipeline.state["progress"])
            st.markdown(f"<div class='loader-step'>{pipeline.state['step']}</div>", unsafe_allow_html=True)
        elif pipeline.state["error"]:
            st.error(pipeline.state["error"])
        elif pipeline.state["done"]:
            st.success(pipeline.state["step"])
        else:
            st.info("Pipeline en attente.")

    with col_refresh:
        st.markdown("<div style='padding-top:0.3rem'>", unsafe_allow_html=True)
        if st.button("🔄 FORCER LA MAJ", use_container_width=True,
                     disabled=pipeline.state["running"]):
            _load_api.clear()
            st.session_state.api_loaded = False
            pipeline.start_in_background()
            time.sleep(0.3)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Entraînement ──────────────────────────────────────────
    st.markdown("<div class='f1-label' style='margin-bottom:1rem'>Entraînement des modèles</div>", unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("""
        <div style="background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:1.2rem;margin-bottom:0.8rem">
            <div class="f1-label">LSTM — Rythme de course</div>
            <p style="font-family:var(--font-m);font-size:0.72rem;color:var(--muted);margin:0.5rem 0 0">
            Entraînement global sur toutes les courses.<br>À relancer après une mise à jour majeure.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("🧠 ENTRAÎNER LSTM", use_container_width=True):
            if not os.path.exists("f1_master_database.pkl"):
                st.error("Lance d'abord une mise à jour des données.")
            else:
                with st.spinner("Entraînement LSTM... (peut prendre quelques minutes)"):
                    import subprocess, sys
                    r = subprocess.run([sys.executable, "LapTimePredictor.py"], capture_output=True)
                    if r.returncode == 0:
                        st.success("Modèle LSTM entraîné ✅")
                        st.cache_resource.clear()
                    else:
                        st.error(f"Erreur : {r.stderr.decode()[:300]}")

    with col_t2:
        circuits_t = st.session_state.circuits or get_circuit_list()
        circuit_train = st.selectbox(
            "Circuit cible", circuits_t if circuits_t else ["—"], key="train_c"
        )
        if st.button("🎯 ENTRAÎNER HYBRIDNET", use_container_width=True):
            if not os.path.exists("f1_hybrid_dataset.pkl"):
                st.error("Lance d'abord une mise à jour des données.")
            else:
                with st.status(f"Spécialisation — {circuit_train}...") as s:
                    ok = train_hybrid_model(circuit_train)
                    s.update(
                        label=f"HybridNet prêt pour {circuit_train} ✅" if ok else "Données insuffisantes ❌",
                        state="complete" if ok else "error"
                    )
                    if ok:
                        st.cache_resource.clear()


# ══════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE
# ══════════════════════════════════════════════════════════════
with tab_stats:
    st.markdown("<div class='f1-label' style='margin-bottom:1rem'>Analyse des données historiques</div>", unsafe_allow_html=True)

    if not _data_ok():
        st.info("Lance une mise à jour des données pour voir les statistiques.")
    else:
        import plotly.graph_objects as go

        df_r = pd.read_pickle(RESULTS_FILE)
        df_r["position"] = pd.to_numeric(df_r["position"], errors="coerce")
        df_r = df_r.dropna(subset=["position"])
        df_r = df_r[df_r["position"] <= 20]

        CHART_STYLE = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono", color="#555", size=11),
            margin=dict(l=0, r=30, t=40, b=0),
            height=300,
        )

        col_a, col_b = st.columns(2)

        with col_a:
            wins = df_r[df_r["position"] == 1].groupby("team_name").size().sort_values()
            fig  = go.Figure(go.Bar(
                x=wins.values, y=wins.index, orientation="h",
                marker_color="#E10600", text=wins.values, textposition="outside",
                textfont=dict(color="#aaa"),
            ))
            fig.update_layout(
                title=dict(text="VICTOIRES PAR ÉCURIE", font=dict(family="Barlow Condensed", size=13, color="#555"), x=0),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
                **CHART_STYLE
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            pods = df_r[df_r["position"] <= 3].groupby("team_name").size().sort_values()
            fig2 = go.Figure(go.Bar(
                x=pods.values, y=pods.index, orientation="h",
                marker_color="#2a2a2a", text=pods.values, textposition="outside",
                textfont=dict(color="#aaa"),
                marker_line=dict(width=1, color="#555"),
            ))
            fig2.update_layout(
                title=dict(text="PODIUMS PAR ÉCURIE", font=dict(family="Barlow Condensed", size=13, color="#555"), x=0),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
                **CHART_STYLE
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='f1-label' style='margin-bottom:0.5rem'>Top 10 Pilotes — Position moyenne</div>", unsafe_allow_html=True)

        avg = (
            df_r.groupby("driver_number")["position"].mean()
            .sort_values().head(10).reset_index()
        )
        avg["acr"] = avg["driver_number"].apply(lambda x: get_acronym(int(x)))
        avg["position"] = avg["position"].round(2)

        colors = ["#E10600", "#C0C0C0", "#CD7F32"] + ["#2a2a2a"] * 7
        fig3 = go.Figure(go.Bar(
            x=avg["acr"], y=avg["position"],
            marker_color=colors,
            text=avg["position"], textposition="outside",
            textfont=dict(color="#aaa"),
        ))
        fig3.update_layout(
            yaxis=dict(autorange="reversed", showgrid=False, title="Pos. moy.",
                       titlefont=dict(color="#555")),
            xaxis=dict(showgrid=False),
            **CHART_STYLE
        )
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — DASHBOARD (session viewer complet)
# ══════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("<div class='f1-label' style='margin-bottom:1rem'>Session Explorer — Temps au tour · Stratégie · Pit stops</div>", unsafe_allow_html=True)
    try:
        f1_viz.render()
    except SystemExit:
        pass  # st.stop() dans gui.py lève SystemExit — on l'absorbe proprement
    except Exception as e:
        st.error(f"Erreur dashboard : {e}")


# ─────────────────────────────────────────────────────────────
# POLLING GLOBAL — Si pipeline tourne, rerun toutes les 2s
# ─────────────────────────────────────────────────────────────
if pipeline.state["running"]:
    time.sleep(2)
    st.rerun()
