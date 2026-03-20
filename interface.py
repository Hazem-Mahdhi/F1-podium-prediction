import streamlit as st
import torch
import pickle
import pandas as pd
import os
import time

# On importe les fonctions principales de tes scripts de données
from build_master_db import update_master_database
from get_race_results import build_results_db
from fusion_data import create_hybrid_dataset

# Import de tes modules
from LapTimePredictor import ExpertPodiumPredictor 
from HybridPodium import HybridNet, train_hybrid_model
from pilotes import get_driver_acronym

from dashboard import gui as f1_viz

# --- CONFIGURATION ---
st.set_page_config(page_title="F1 Prédiction Podium", page_icon="🏎️", layout="wide")

MODEL_LSTM = "expert_model.pth"
MODEL_HYBRID = "hybrid_model.pth"
MAP_LSTM = "circuit_map.pkl"
MAP_HYBRID = "hybrid_mappings.pkl"
RESULTS_FILE = "f1_race_results.pkl"

GLOBAL_MAX_LAP_TIME = 150.0
SEQUENCE_LENGTH = 5

# --- CALENDRIER & GRILLE ---
GRID_2025 = [
    (1, "Red Bull Racing"), (30, "Red Bull Racing"),
    (16, "Ferrari"), (44, "Ferrari"),
    (63, "Mercedes"), (12, "Mercedes"),
    (4, "McLaren"), (81, "McLaren"),
    (14, "Aston Martin"), (18, "Aston Martin"),
    (10, "Alpine"), (61, "Alpine"),
    (23, "Williams"), (55, "Williams"),
    (31, "Haas F1 Team"), (87, "Haas F1 Team"),
    (22, "RB"), (3, "RB"),
    (27, "Sauber"), (5, "Sauber"),
]

# On garde juste ça pour que les noms soient jolis (pas EDG pour Antonelli)
# Mais cela n'impacte plus le calcul du score.
ROOKIE_NAMES = {
    12: "ANT", 30: "LAW", 61: "DOO", 87: "BEA", 5: "BOR", 6: "HAD"
}

# --- FONCTIONS ---
def get_last_race_info():
    if os.path.exists(RESULTS_FILE):
        try:
            df = pd.read_pickle(RESULTS_FILE)
            if not df.empty:
                df = df.sort_values(['year', 'session_key'], ascending=False)
                last = df.iloc[0]
                return f"{last['circuit']} {last['year']}"
        except: pass
    return "Aucune donnée"

@st.cache_resource
def load_models():
    try:
        # On force le chargement sur CPU pour éviter les erreurs CUDA si tu n'as pas de GPU nvidia
        device = torch.device('cpu')
        
        # LSTM
        ckpt_lstm = torch.load(MODEL_LSTM, map_location=device)
        lstm_model = ExpertPodiumPredictor(ckpt_lstm['num_circuits'])
        lstm_model.load_state_dict(ckpt_lstm['state_dict'])
        lstm_model.eval()
        with open(MAP_LSTM, 'rb') as f: lstm_circuits = pickle.load(f)
        
        # Hybride
        with open(MAP_HYBRID, 'rb') as f: maps_h = pickle.load(f)
        hybrid_model = HybridNet(len(maps_h['d']), len(maps_h['t']), len(maps_h['c']))
        hybrid_model.load_state_dict(torch.load(MODEL_HYBRID, map_location=device))
        hybrid_model.eval()
        
        return lstm_model, hybrid_model, lstm_circuits, maps_h
    except Exception as e:
        # En cas d'erreur, on la retourne pour l'afficher
        return None, None, None, str(e)

def run_prediction(circuit_name, lstm_model, hybrid_model, lstm_circuits, maps_h):
    c_id_lstm = lstm_circuits.get(circuit_name, 0)
    c_id_hybrid = maps_h['c'].get(circuit_name, 0)
    
    # Phase 1: Temps (LSTM)
    base_input = torch.tensor([[[90.0/GLOBAL_MAX_LAP_TIME]] * SEQUENCE_LENGTH], dtype=torch.float32)
    lap_progress = torch.tensor([[0.5]], dtype=torch.float32)

    predicted_times = []
    team_pace_ref = {} # Pour stocker le temps de référence de l'écurie

    # 1. On calcule d'abord pour les pilotes connus du modèle
    for d_num, team in GRID_2025:
        try:
            # Si le pilote est connu dans le mapping LSTM
            d_tensor = torch.tensor([d_num], dtype=torch.long)
            c_tensor = torch.tensor([c_id_lstm], dtype=torch.long)
            pred_norm = lstm_model(base_input, d_tensor, c_tensor, lap_progress).item()
            
            # On enregistre le temps
            predicted_times.append({'driver': d_num, 'team': team, 'raw_pace': pred_norm})
            
            # On définit ce temps comme référence pour l'équipe (le meilleur des deux)
            if team not in team_pace_ref or pred_norm < team_pace_ref[team]:
                team_pace_ref[team] = pred_norm
        except:
            pass # Si erreur (pilote inconnu), on passe au tour suivant

    # 2. On gère les inconnus (Rookies) en leur donnant le temps de l'équipe
    # Aucune pénalité : Temps Rookie = Temps Coéquipier
    known_drivers = [p['driver'] for p in predicted_times]
    
    for d_num, team in GRID_2025:
        if d_num not in known_drivers:
            if team in team_pace_ref:
                # ÉGALITÉ PARFAITE avec le coéquipier
                pred_norm = team_pace_ref[team] 
            else:
                pred_norm = 0.65 # Fallback si toute l'équipe est inconnue
            
            predicted_times.append({'driver': d_num, 'team': team, 'raw_pace': pred_norm})

    # Normalisation
    valid_paces = [p['raw_pace'] for p in predicted_times if p['raw_pace'] > 0]
    best_pace = min(valid_paces) if valid_paces else 0.5
    
    # Phase 2: Classement (Hybride)
    leaderboard = []
    for p in predicted_times:
        pace_deficit = p['raw_pace'] / best_pace
        
        d_id = maps_h['d'].get(p['driver'], 0)
        t_id = maps_h['t'].get(p['team'], 0)
        
        d_in = torch.tensor([d_id])
        t_in = torch.tensor([t_id])
        c_in = torch.tensor([c_id_hybrid])
        pace_in = torch.tensor([[pace_deficit]])
        
        score = hybrid_model(d_in, t_in, c_in, pace_in).item()
        
        # Boost Top Teams
        if p['team'] in ["Red Bull Racing", "McLaren", "Ferrari"]:
            score += 0.05
            
        # Nom Affichage
        if p['driver'] in ROOKIE_NAMES:
            acr = ROOKIE_NAMES[p['driver']]
        else:
            acr = get_driver_acronym(p['driver'])
            if acr == "???": acr = f"#{p['driver']}"
            
        leaderboard.append({
            'Pilote': acr, 
            'Écurie': p['team'], 
            'Score IA': score,
            'Pace': pace_deficit
        })
        
    df_res = pd.DataFrame(leaderboard)
    df_res = df_res.sort_values('Score IA', ascending=False).reset_index(drop=True)
    df_res.index += 1
    return df_res

# --- INTERFACE ---
st.title("🏎️ F1 Podium prediction")

tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "⚙️ Entraînement", "📊 Visualisation"])

with tab1:
    last_race_txt = get_last_race_info()
    st.info(f"📅 **Dernière course en base :** {last_race_txt}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        circuit_list = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami", "Emilia-Romagna", "Monaco", "Canada", "Spain", "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore", "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi", "Sakhir", "Jeddah", "Melbourne", "Suzuka", "Shanghai", "Imola", "Montréal", "Barcelona", "Spielberg", "Silverstone", "Budapest", "Spa-Francorchamps", "Zandvoort", "Monza", "Baku", "Marina Bay", "Austin", "Mexico City", "São Paulo", "Lusail", "Yas Island"]
        circuit_list.sort()
        idx_def = circuit_list.index("Lusail") if "Lusail" in circuit_list else 0
        selected_circuit = st.selectbox("Choisir le Circuit", circuit_list, index=idx_def)
        
        if st.button("🚀 Lancer la Simulation", type="primary"):
            # On récupère 4 valeurs. La dernière peut être une erreur.
            lstm, hybrid, l_maps, h_maps = load_models()
            
            # Si le premier modèle est None, c'est qu'il y a eu un crash
            if lstm is None:
                error_msg = h_maps # Dans ce cas, la 4ème variable contient le message d'erreur
                st.error(f"Erreur de chargement des modèles : {error_msg}")
                st.warning("Essaie de relancer l'entraînement dans l'onglet 'Atelier'.")
            else:
                with st.spinner(f"Simulation de la course à {selected_circuit}..."):
                    time.sleep(0.5)
                    results = run_prediction(selected_circuit, lstm, hybrid, l_maps, h_maps)
                    st.session_state['results'] = results
                    st.success("Terminé !")

    with col2:
        if 'results' in st.session_state:
            df = st.session_state['results']
            st.subheader(f"🏆 Podium : {selected_circuit}")
            c2, c1, c3 = st.columns([1, 1, 1])
            with c1: st.markdown(f"<div style='text-align: center; background: #FFD700; padding: 10px; border-radius: 10px;'><h1>🥇<br>{df.iloc[0]['Pilote']}</h1><small>{df.iloc[0]['Écurie']}</small></div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div style='text-align: center; background: #C0C0C0; padding: 10px; border-radius: 10px;'><h2>🥈<br>{df.iloc[1]['Pilote']}</h2><small>{df.iloc[1]['Écurie']}</small></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div style='text-align: center; background: #CD7F32; padding: 10px; border-radius: 10px;'><h2>🥉<br>{df.iloc[2]['Pilote']}</h2><small>{df.iloc[2]['Écurie']}</small></div>", unsafe_allow_html=True)
            st.divider()
            st.dataframe(df.style.background_gradient(subset=['Score IA'], cmap="Greens"))

# === ONGLET 2 : ATELIER & DATA ===
with tab2:
    st.header("⚙️ Atelier Technique")
    
    col_data, col_train = st.columns(2)
    
    # --- COLONNE GAUCHE : DATA ---
    with col_data:
        st.subheader("📡 Mise à jour Données")
        st.info("Récupère les derniers résultats (2024/2025) depuis l'API OpenF1.")
        
        if st.button("🔄 Lancer la Mise à jour API", help="Attention, cela peut prendre quelques minutes."):
            status = st.status("Connexion à l'API...", expanded=True)
            
            try:
                # Étape 1 : Tours
                status.write("1. Téléchargement des Tours (Laps)...")
                # On redirige les prints vers le terminal pour ne pas polluer, 
                # mais on affiche l'avancement ici
                update_master_database()
                st.toast("Tours mis à jour !", icon="✅")
                
                # Étape 2 : Résultats
                status.write("2. Téléchargement des Classements...")
                build_results_db()
                st.toast("Classements mis à jour !", icon="✅")
                
                # Étape 3 : Fusion
                status.write("3. Fusion et Calcul des Rythmes...")
                create_hybrid_dataset()
                st.toast("Fusion terminée !", icon="✅")
                
                status.update(label="Base de données à jour !", state="complete", expanded=False)
                st.success("Toutes les données sont synchronisées.")
                time.sleep(2)
                st.rerun() # Rafraîchit la page pour mettre à jour la date en haut
                
            except Exception as e:
                status.update(label="Erreur", state="error")
                st.error(f"Une erreur est survenue : {e}")

    # --- COLONNE DROITE : IA ---
    with col_train:
        st.subheader("🧠 Entraînement IA")
        st.info("Spécialise le modèle pour un circuit précis.")
        
        train_c = st.selectbox("Circuit Cible", circuit_list, key="train_c")
        
        if st.button("💪 Entraîner le Modèle", type="primary"):
            # Vérification que les données existent
            if not os.path.exists("f1_hybrid_dataset.pkl"):
                st.error("Données manquantes ! Lance la mise à jour à gauche d'abord.")
            else:
                with st.status(f"Optimisation pour {train_c}...", expanded=True) as status:
                    status.write("Filtrage des données historiques...")
                    success = train_hybrid_model(train_c)
                    
                    if success:
                        status.write("Convergence des réseaux de neurones...")
                        time.sleep(1)
                        status.update(label="Entraînement réussi !", state="complete", expanded=False)
                        st.success(f"Le modèle est maintenant expert pour {train_c}.")
                        st.cache_resource.clear()
                    else:
                        status.update(label="Échec", state="error")
                        st.error("Pas assez de données pour ce circuit.")

with tab3:
    st.header("📊 Visualisation des Données")
    f1_viz.render()