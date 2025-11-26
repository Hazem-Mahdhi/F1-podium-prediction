import torch
import numpy as np
import pandas as pd
import requests
import pickle # Pour charger le dico des circuits
from sessions import get_session_list
from pilotes import get_driver_acronym 
# Assure-toi d'importer la nouvelle classe ExpertPodiumPredictor
from LapTimePredictor import ExpertPodiumPredictor 

# --- CONFIGURATION ---
BASE_URL = "https://api.openf1.org/v1"
MODEL_FILE = "expert_model.pth"
CIRCUIT_MAP_FILE = "circuit_map.pkl"
GLOBAL_MAX_LAP_TIME = 150.0
SEQUENCE_LENGTH = 5

def get_drivers_in_session(session_key):
    try:
        url = f"{BASE_URL}/drivers"
        r = requests.get(url, params={'session_key': session_key}, timeout=10)
        data = r.json()
        return list(set([d['driver_number'] for d in data]))
    except: return []

def predict_race_podium():
    print("🧠 Chargement du cerveau Expert...")
    
    # 1. Chargement du Mapping Circuit
    try:
        with open(CIRCUIT_MAP_FILE, 'rb') as f:
            circuit_to_idx = pickle.load(f)
    except FileNotFoundError:
        print("❌ Dictionnaire circuits introuvable. Relance l'entraînement.")
        return

    # 2. Chargement du Modèle
    try:
        checkpoint = torch.load(MODEL_FILE, map_location=torch.device("cpu"))
        # On récupère le nombre de circuits qu'il a appris
        num_circuits_learned = checkpoint['num_circuits']
        
        model = ExpertPodiumPredictor(num_circuits_learned)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    except FileNotFoundError:
        print(f"❌ Fichier {MODEL_FILE} introuvable.")
        return

    # 3. Interface Utilisateur
    print("\n--- 🏁 PRÉDICTEUR EXPERT 🏁 ---")
    year_input = input("Année (ex: 2023) : ")
    country_input = input("Pays/Circuit (ex: Monaco) : ") 
    
    # Recherche de session
    sessions = get_session_list(year=year_input, country_name=country_input, session_name="Race")
    if not sessions:
        print("❌ Session introuvable.")
        return

    target_session = sessions[0]
    session_key = target_session['session_key']
    real_circuit_name = target_session['location']
    print(f"✅ Session : {real_circuit_name} (Key: {session_key})")

    # --- POINT 1 : VÉRIFICATION DU CIRCUIT ID ---
    # Le nom dans l'API (ex: "Monaco") doit correspondre à ce qu'on a appris
    if real_circuit_name not in circuit_to_idx:
        print(f"⚠️ ATTENTION : Le modèle n'a jamais vu le circuit '{real_circuit_name}' !")
        print("Il va utiliser un profil 'Inconnu' (ID 0). La précision sera faible.")
        circuit_id = 0
    else:
        circuit_id = circuit_to_idx[real_circuit_name]
        print(f"🗺️  Circuit identifié par le modèle : ID {circuit_id}")

    # 4. Participants
    drivers = get_drivers_in_session(session_key)
    if not drivers: return

    # 5. Simulation
    # On simule un tour "lancé" à mi-course (Tour 35/70)
    simulated_lap_progress = 0.5 
    
    # On calibre la "base" sur une valeur moyenne générique (90s)
    # Le modèle va la corriger grâce à l'embedding du circuit (Monaco -> +Lent, Monza -> +Vite)
    base_lap_time = 90.0 / GLOBAL_MAX_LAP_TIME 
    input_seq = torch.tensor([[[base_lap_time]] * SEQUENCE_LENGTH], dtype=torch.float32)
    
    leaderboard = []
    print("🔮 Simulation de course (Prise en compte Pilote + Circuit + Carburant)...")
    
    with torch.no_grad():
        for driver_num in drivers:
            # Tenseurs d'entrée
            d_input = torch.tensor([driver_num], dtype=torch.long)
            c_input = torch.tensor([circuit_id], dtype=torch.long)
            l_input = torch.tensor([[simulated_lap_progress]], dtype=torch.float32)
            
            # Prédiction
            pred_norm = model(input_seq, d_input, c_input, l_input).item()
            pred_time = pred_norm * GLOBAL_MAX_LAP_TIME
            
            leaderboard.append({'num': driver_num, 'time': pred_time})

    # Classement
    leaderboard.sort(key=lambda x: x['time'])
    
    print("\n" + "="*35)
    print(f"🏆 PODIUM PRÉDIT : {real_circuit_name} {year_input}")
    print("="*35)
    
    positions = ["🥇", "🥈", "🥉"]
    for i in range(3):
        if i < len(leaderboard):
            d = leaderboard[i]
            acr = get_driver_acronym(d['num'], session_key)
            print(f"{positions[i]} {acr} \t: {d['time']:.3f}s")
            
    print("="*35)
    
    # Top 10
    for i in range(3, min(10, len(leaderboard))):
        d = leaderboard[i]
        acr = get_driver_acronym(d['num'], session_key)
        print(f"P{i+1} {acr} \t: {d['time']:.3f}s")

if __name__ == "__main__":
    predict_race_podium()