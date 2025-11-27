import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
import os

DATA_FILE = "f1_hybrid_dataset.pkl"
MODEL_FILE = "hybrid_model.pth"
MAPPING_FILE = "hybrid_mappings.pkl"

# ==========================================
# 1. PRÉPARATION INTELLIGENTE
# ==========================================
def prepare_data(target_circuit):
    print(f"   🔄 Filtrage des données pour : {target_circuit}...")
    
    if not os.path.exists(DATA_FILE):
        print("❌ Fichier f1_hybrid_dataset.pkl manquant.")
        return None

    df = pd.read_pickle(DATA_FILE)
    df = df[df['position'] <= 20] 

    # --- STRATÉGIE DE FILTRAGE ---
    # 1. Forme du moment (2025)
    mask_current_form = (df['year'] == 2025)
    
    # 2. Spécialistes du circuit (Historique 2023-2024 sur CE circuit)
    mask_track_history = (df['circuit'] == target_circuit) & (df['year'] < 2025)
    
    # 3. Filet de sécurité (Fin 2024 si début 2025 vide)
    mask_safety_net = (df['year'] == 2024) & (df['session_key'] > df[df['year']==2024]['session_key'].max() - 7)

    if len(df[mask_current_form]) > 0:
        final_df = df[mask_current_form | mask_track_history].copy()
    else:
        final_df = df[mask_safety_net | mask_track_history].copy()

    if len(final_df) < 10:
        print("   ⚠️ Peu de données trouvées. Le modèle risque d'être imprécis.")

    # Mappings sur tout l'historique pour ne rien perdre
    drivers = df['driver_number'].unique()
    teams = df['team_name'].unique()
    circuits = df['circuit'].unique()
    
    d_map = {d: i for i, d in enumerate(drivers)}
    t_map = {t: i for i, t in enumerate(teams)}
    c_map = {c: i for i, c in enumerate(circuits)}
    
    with open(MAPPING_FILE, 'wb') as f:
        pickle.dump({'d': d_map, 't': t_map, 'c': c_map}, f)
        
    # Tenseurs sur les données FILTRÉES
    X_d = [d_map[d] for d in final_df['driver_number']]
    X_t = [t_map[t] for t in final_df['team_name']]
    X_c = [c_map[c] for c in final_df['circuit']]
    X_pace = final_df['pace_deficit'].values 
    Y = 1.0 - ((final_df['position'].values - 1) / 19.0)
    
    return (torch.tensor(X_d, dtype=torch.long),
            torch.tensor(X_t, dtype=torch.long),
            torch.tensor(X_c, dtype=torch.long),
            torch.tensor(X_pace, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(Y, dtype=torch.float32).unsqueeze(-1),
            len(drivers), len(teams), len(circuits))

# ==========================================
# 2. MODÈLE
# ==========================================
class HybridNet(nn.Module):
    def __init__(self, n_d, n_t, n_c):
        super(HybridNet, self).__init__()
        self.emb_d = nn.Embedding(n_d, 8)
        self.emb_t = nn.Embedding(n_t, 16) 
        self.emb_c = nn.Embedding(n_c, 8)
        self.fc1 = nn.Linear(33, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) 
        
    def forward(self, d, t, c, pace):
        x = torch.cat((self.emb_d(d), self.emb_t(t), self.emb_c(c), pace), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 3. FONCTION D'ENTRAÎNEMENT (Appelable)
# ==========================================
def train_hybrid_model(target_circuit):
    print(f"\n⚙️ DÉMARRAGE ENTRAÎNEMENT SPÉCIFIQUE : {target_circuit}")
    
    data = prepare_data(target_circuit)
    if not data: return False
    
    d, t, c, p, y, nd, nt, nc = data
    dataset = TensorDataset(d, t, c, p, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = HybridNet(nd, nt, nc)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    # Entraînement rapide (le dataset filtré est petit)
    for epoch in range(80): 
        for batch in loader:
            optimizer.zero_grad()
            pred = model(*batch[:-1])
            loss = criterion(pred, batch[-1])
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"✅ Modèle optimisé pour {target_circuit} prêt !")
    return True

if __name__ == "__main__":
    # Test manuel si on lance ce fichier directement
    c = input("Circuit test : ")
    train_hybrid_model(c)