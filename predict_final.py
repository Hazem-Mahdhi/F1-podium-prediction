import torch
import pandas as pd
import pickle
from DirectPodium import RankNet
from pilotes import get_driver_acronym  # On importe ta fonction

MODEL_FILE = "direct_podium_model.pth"
MAPPING_FILE = "mappings.pkl"

# --- GRILLE 2025 ---
GRID_2025 = [
    (1, "Red Bull Racing"),      # Verstappen
    (11, "Red Bull Racing"),     # Perez
    (16, "Ferrari"),             # Leclerc
    (44, "Ferrari"),             # Hamilton
    (63, "Mercedes"),            # Russell
    (4, "McLaren"),              # Norris
    (81, "McLaren"),             # Piastri
    (14, "Aston Martin"),        # Alonso
    (18, "Aston Martin"),        # Stroll
    (10, "Alpine"),              # Gasly
    (31, "Alpine"),              # Ocon (ou Doohan en 2025)
    (27, "Haas F1 Team"),        # Hulkenberg (Sauber en 2025?)
    (23, "Williams"),            # Albon
    (22, "RB"),                  # Tsunoda
    # Ajoute les autres si besoin
]

def predict():
    # 1. Chargement
    try:
        with open(MAPPING_FILE, 'rb') as f:
            maps = pickle.load(f)
        model = RankNet(len(maps['driver']), len(maps['team']), len(maps['circuit']))
        model.load_state_dict(torch.load(MODEL_FILE))
        model.eval()
    except:
        print("❌ Modèle ou Mapping manquant. Relance l'entraînement DirectPodium.py")
        return

    driver_map = maps['driver']
    team_map = maps['team']
    circuit_map = maps['circuit']
    
    print("\n--- 🔮 PRÉDICTION PODIUM 2025 ---")
    circuit_name = input("Circuit (ex: Monaco, Monza, Lusail) : ")
    
    if circuit_name not in circuit_map:
        print(f"❌ Circuit '{circuit_name}' inconnu. Essaie : {list(circuit_map.keys())[:3]}...")
        return
        
    circuit_id = circuit_map[circuit_name]
    
    # 2. Simulation
    results = []
    print(f"🏁 Simulation de course à {circuit_name}...")
    
    with torch.no_grad():
        for driver_num, team_name in GRID_2025:
            if driver_num not in driver_map or team_name not in team_map:
                continue
                
            d_id = torch.tensor([driver_map[driver_num]])
            t_id = torch.tensor([team_map[team_name]])
            c_id = torch.tensor([circuit_id])
            
            score = model(d_id, t_id, c_id).item()
            
            # Appel API pour l'acronyme (ex: 1 -> VER)
            acro = get_driver_acronym(driver_num)
            
            results.append({
                'acronym': acro,
                'team': team_name,
                'score': score
            })
            
    # 3. Affichage Propre
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*30)
    print(f"🏆 PODIUM PRÉVU : {circuit_name}")
    print("="*30)
    
    emojis = ["🥇", "🥈", "🥉"]
    
    for i, res in enumerate(results):
        rank = emojis[i] if i < 3 else f"P{i+1} "
        print(f"{rank} : {res['acronym']} \t({res['team']})")

if __name__ == "__main__":
    predict()