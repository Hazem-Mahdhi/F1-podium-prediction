import torch
import pickle
import numpy as np
import pandas as pd
import os
from LapTimePredictor import ExpertPodiumPredictor 
from HybridPodium import HybridNet, train_hybrid_model # <-- On importe la fonction d'entraînement
from pilotes import get_driver_acronym

# Fichiers
MODEL_LSTM = "expert_model.pth"
MODEL_HYBRID = "hybrid_model.pth"
MAP_LSTM = "circuit_map.pkl"
MAP_HYBRID = "hybrid_mappings.pkl"
RESULTS_FILE = "f1_race_results.pkl"

GLOBAL_MAX_LAP_TIME = 150.0
SEQUENCE_LENGTH = 5

# --- CALENDRIER 2025 ---
CALENDAR_2025 = [
    "Australia", "China", "Japan", "Bahrain", "Saudi Arabia", 
    "Miami", "Emilia-Romagna", "Monaco", "Spain", "Canada", 
    "Austria", "Great Britain", "Belgium", "Hungary", "Netherlands", 
    "Italy", "Azerbaijan", "Singapore", "United States", "Mexico", 
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
]

# --- GRILLE 2025 ---
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

def print_last_race_and_upcoming():
    if os.path.exists(RESULTS_FILE):
        df = pd.read_pickle(RESULTS_FILE)
        if not df.empty:
            df = df.sort_values(['year', 'session_key'], ascending=[False, False])
            last = df.iloc[0]
            print(f"\n📅 DERNIÈRE DONNÉE : {last['circuit']} {last['year']}")
            
            # Suggestion prochaine course
            try:
                # Mapping simple pour les noms communs
                map_c = {"Lusail": "Qatar", "Yas Island": "Abu Dhabi", "Sakhir": "Bahrain"}
                last_name = map_c.get(last['circuit'], last['circuit'])
                
                for i, c in enumerate(CALENDAR_2025):
                    if c in last_name or last_name in c:
                        if i + 1 < len(CALENDAR_2025):
                            print(f"👉 PROCHAINE COURSE : {CALENDAR_2025[i+1]}")
                        break
            except: pass
    print("-" * 50)

def predict():
    print_last_race_and_upcoming()
    
    # 1. Choix du circuit AVANT de charger le modèle Hybride
    # (Car on va le ré-entraîner maintenant)
    circuit_name = input("\nCircuit pour 2025 (ex: Lusail) : ")
    
    # --- ENTRAÎNEMENT AUTOMATIQUE ---
    print("\n🧠 Mise à jour du cerveau stratégique...")
    success = train_hybrid_model(circuit_name)
    if not success:
        print("❌ Échec de l'entraînement. Vérifie le nom du circuit.")
        return

    # 2. Chargement des Modèles (maintenant qu'ils sont à jour)
    print("\n📂 Chargement des modèles...")
    try:
        # LSTM (Physicien - Lui n'a pas besoin de bouger)
        ckpt_lstm = torch.load(MODEL_LSTM)
        lstm_model = ExpertPodiumPredictor(ckpt_lstm['num_circuits'])
        lstm_model.load_state_dict(ckpt_lstm['state_dict'])
        lstm_model.eval()
        with open(MAP_LSTM, 'rb') as f: lstm_circuits = pickle.load(f)
        
        # Hybride (Stratège - Vient d'être mis à jour)
        with open(MAP_HYBRID, 'rb') as f: maps_h = pickle.load(f)
        hybrid_model = HybridNet(len(maps_h['d']), len(maps_h['t']), len(maps_h['c']))
        hybrid_model.load_state_dict(torch.load(MODEL_HYBRID))
        hybrid_model.eval()
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        return

    # Gestion ID Circuit
    c_id_lstm = lstm_circuits.get(circuit_name, 0)
    c_id_hybrid = maps_h['c'].get(circuit_name, 0)

    print(f"🏁 Simulation Pipeline 2025 à {circuit_name}...")
    
    # --- PHASE 1 : ESTIMATION TEMPS (LSTM) ---
    predicted_times = []
    base_input = torch.tensor([[[90.0/GLOBAL_MAX_LAP_TIME]] * SEQUENCE_LENGTH], dtype=torch.float32)
    lap_progress = torch.tensor([[0.5]], dtype=torch.float32)

    with torch.no_grad():
        for d_num, team in GRID_2025:
            try:
                d_tensor = torch.tensor([d_num], dtype=torch.long)
                c_tensor = torch.tensor([c_id_lstm], dtype=torch.long)
                pred_norm = lstm_model(base_input, d_tensor, c_tensor, lap_progress).item()
            except:
                pred_norm = 0.65 
            predicted_times.append({'driver': d_num, 'team': team, 'raw_pace': pred_norm})

    valid_paces = [p['raw_pace'] for p in predicted_times if p['raw_pace'] > 0]
    best_pace = min(valid_paces) if valid_paces else 0.5
    
    # --- PHASE 2 : CLASSEMENT (HYBRIDE) ---
    leaderboard = []
    with torch.no_grad():
        for p in predicted_times:
            pace_deficit = p['raw_pace'] / best_pace
            
            d_id = maps_h['d'].get(p['driver'], 0)
            t_id = maps_h['t'].get(p['team'], 0)
            c_id = c_id_hybrid
            
            d_in = torch.tensor([d_id])
            t_in = torch.tensor([t_id])
            c_in = torch.tensor([c_id])
            pace_in = torch.tensor([[pace_deficit]])
            
            score = hybrid_model(d_in, t_in, c_in, pace_in).item()
            
            if p['team'] in ["Red Bull Racing", "McLaren", "Ferrari"]:
                score += 0.05
            
            leaderboard.append({'driver': p['driver'], 'team': p['team'], 'score': score})

    leaderboard.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n🏆 PODIUM PRÉDIT ({circuit_name} 2025)")
    print("="*50)
    emojis = ["🥇", "🥈", "🥉"]
    for i, res in enumerate(leaderboard[:12]): 
        acr = get_driver_acronym(res['driver'])
        if acr == "???": acr = f"#{res['driver']}"
        prefix = emojis[i] if i < 3 else f"P{i+1} "
        print(f"{prefix} {acr:<5} ({res['team']}) \t[Score: {res['score']:.3f}]")

if __name__ == "__main__":
    predict()