import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
import os

DATA_FILE = "f1_race_results.pkl"
MODEL_FILE = "direct_podium_model.pth"
MAPPING_FILE = "mappings.pkl"

# ==========================================
# 1. PRÉPARATION DES DONNÉES (MAPPING)
# ==========================================
def prepare_data():
    df = pd.read_pickle(DATA_FILE)
    
    # --- CORRECTION ANTI-BUG ---
    # 1. On force la position en numérique (les "NC" deviennent NaN)
    df['position'] = pd.to_numeric(df['position'], errors='coerce')
    # 2. On supprime les lignes sans position valide
    df = df.dropna(subset=['position'])
    # 3. On supprime les positions > 20 (cas rares)
    df = df[df['position'] <= 20]
    
    print(f"✅ Données propres : {len(df)} lignes")
    
    # Création des dictionnaires (Mappings)
    drivers = df['driver_number'].unique()
    teams = df['team_name'].unique()
    circuits = df['circuit'].unique()
    
    driver_map = {d: i for i, d in enumerate(drivers)}
    team_map = {t: i for i, t in enumerate(teams)}
    circuit_map = {c: i for i, c in enumerate(circuits)}
    
    with open(MAPPING_FILE, 'wb') as f:
        pickle.dump({'driver': driver_map, 'team': team_map, 'circuit': circuit_map}, f)
        
    X_driver = [driver_map[d] for d in df['driver_number']]
    X_team = [team_map[t] for t in df['team_name']]
    X_circuit = [circuit_map[c] for c in df['circuit']]
    
    # CIBLE : Score normalisé (1.0 = P1, 0.0 = P20)
    Y = 1.0 - ((df['position'].values - 1) / 19.0)
    
    return (torch.tensor(X_driver, dtype=torch.long),
            torch.tensor(X_team, dtype=torch.long),
            torch.tensor(X_circuit, dtype=torch.long),
            torch.tensor(Y, dtype=torch.float32).unsqueeze(-1),
            len(drivers), len(teams), len(circuits))

# ==========================================
# 2. LE MODÈLE NEURAL
# ==========================================
class RankNet(nn.Module):
    def __init__(self, n_drivers, n_teams, n_circuits):
        super(RankNet, self).__init__()
        self.emb_driver = nn.Embedding(n_drivers, 8)
        self.emb_team = nn.Embedding(n_teams, 12)
        self.emb_circuit = nn.Embedding(n_circuits, 8)
        
        self.fc1 = nn.Linear(28, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, d, t, c):
        x = torch.cat((self.emb_driver(d), self.emb_team(t), self.emb_circuit(c)), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 3. ENTRAÎNEMENT
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print("❌ Lance d'abord get_race_results.py !")
        exit()
        
    d, t, c, y, nd, nt, nc = prepare_data()
    
    dataset = TensorDataset(d, t, c, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = RankNet(nd, nt, nc)
    # Baisse du Learning Rate pour éviter les NaNs
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.MSELoss()
    
    print(f"🚀 Entraînement sur {len(d)} résultats...")
    
    for epoch in range(50):
        total_loss = 0
        for bd, bt, bc, by in loader:
            optimizer.zero_grad()
            pred = model(bd, bt, bc)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} : Loss {total_loss/len(loader):.4f}")
            
    torch.save(model.state_dict(), MODEL_FILE)
    print("💾 Modèle de classement sauvegardé.")