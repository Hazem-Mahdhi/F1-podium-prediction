import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import pickle # Pour sauvegarder le dictionnaire des circuits

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 5
GLOBAL_MAX_LAP_TIME = 150.0
HISTORY_FILE = "f1_master_database.pkl"
TENSORS_FILE = "f1_master_tensors_v2.pt" # Nouveau nom pour éviter les conflits
CIRCUIT_MAP_FILE = "circuit_map.pkl"     # Pour se souvenir quel ID = quel Circuit

def create_dataset_complex(master_df, sequence_length=5):
    print(f"🔨 Construction du dataset EXPERT sur {len(master_df)} tours...")
    
    # 1. Création du Dictionnaire de Circuits (Mapping)
    # On récupère tous les noms uniques de circuits
    unique_circuits = master_df['circuit'].unique()
    circuit_to_idx = {name: i for i, name in enumerate(unique_circuits)}
    
    # On sauvegarde ce mapping pour la prédiction plus tard
    with open(CIRCUIT_MAP_FILE, 'wb') as f:
        pickle.dump(circuit_to_idx, f)
    print(f"🌍 {len(unique_circuits)} circuits identifiés et indexés.")

    clean_sessions = []
    
    # 2. Nettoyage (Méthode du Quantile vue précédemment)
    grouped = master_df.groupby(['session_key', 'driver_number'])
    for (session_key, driver_num), group_df in grouped:
        if group_df.empty: continue
        # On garde les 40% meilleurs tours (Rythme de course pur)
        threshold = group_df['lap_duration'].quantile(0.40)
        clean_df = group_df[group_df['lap_duration'] <= threshold].copy()
        if len(clean_df) > sequence_length:
            clean_sessions.append(clean_df)

    if not clean_sessions: return None, None
    master_df_clean = pd.concat(clean_sessions)

    # 3. Création des Séquences
    X_time, X_driver, X_circuit, X_lap_num, Y = [], [], [], [], []
    
    grouped_clean = master_df_clean.groupby(['session_key', 'driver_number'])
    total = len(grouped_clean)
    count = 0

    for (session_key, driver_num), group_df in grouped_clean:
        count += 1
        if count % 200 == 0: print(f"  > Traitement {count}/{total}...")
        
        group_df = group_df.sort_values('lap_number').reset_index(drop=True)
        
        # Récupération de l'ID du circuit pour cette session
        circuit_name = group_df['circuit'].iloc[0]
        circuit_id = circuit_to_idx[circuit_name]
        
        lap_durations_norm = group_df['lap_duration'].values / GLOBAL_MAX_LAP_TIME
        lap_numbers = group_df['lap_number'].values / 70.0 # Normalisation approx (tour 70 = 1.0)

        for i in range(sequence_length, len(group_df)):
            # Inputs
            seq_time = lap_durations_norm[i-sequence_length : i]
            current_lap_progression = lap_numbers[i] # Point 3 : Progression dans la course
            
            X_time.append(seq_time)
            X_driver.append(driver_num)
            X_circuit.append(circuit_id) # Point 1 : ID du Circuit
            X_lap_num.append(current_lap_progression)
            
            # Cible
            Y.append(lap_durations_norm[i])

    # Conversion Tenseurs
    t_X_time = torch.tensor(np.array(X_time), dtype=torch.float32).unsqueeze(-1)
    t_X_driver = torch.tensor(np.array(X_driver), dtype=torch.long)
    t_X_circuit = torch.tensor(np.array(X_circuit), dtype=torch.long)
    t_X_lap = torch.tensor(np.array(X_lap_num), dtype=torch.float32).unsqueeze(-1)
    t_Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1)

    print(f"✅ Dataset final : {len(t_Y)} échantillons.")
    # On retourne aussi la taille du dictionnaire circuits pour dimensionner le modèle
    return t_X_time, t_X_driver, t_X_circuit, t_X_lap, t_Y, len(unique_circuits)

# ==========================================
# LE MODÈLE EXPERT (Multi-Embedding)
# ==========================================
class ExpertPodiumPredictor(nn.Module):
    def __init__(self, num_circuits):
        super(ExpertPodiumPredictor, self).__init__()
        
        # 1. Analyse Temporelle (LSTM)
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        
        # 2. Profil Pilote (Embedding)
        self.driver_emb = nn.Embedding(num_embeddings=100, embedding_dim=8)
        
        # 3. Profil Circuit (Embedding) - POINT 1
        # Le modèle va apprendre un vecteur pour "Monaco", un pour "Monza", etc.
        self.circuit_emb = nn.Embedding(num_embeddings=num_circuits + 1, embedding_dim=8)
        
        # 4. Fusion
        # 64 (Temps) + 8 (Pilote) + 8 (Circuit) + 1 (Numéro de tour - POINT 3)
        fusion_size = 64 + 8 + 8 + 1 
        
        self.fc1 = nn.Linear(fusion_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x_time, x_driver, x_circuit, x_lap):
        # LSTM
        lstm_out, _ = self.lstm(x_time)
        feat_time = lstm_out[:, -1, :] 
        
        # Embeddings
        feat_driver = self.driver_emb(x_driver)
        feat_circuit = self.circuit_emb(x_circuit)
        
        # Concaténation de TOUTES les infos
        combined = torch.cat((feat_time, feat_driver, feat_circuit, x_lap), dim=1)
        
        x = self.relu(self.fc1(combined))
        return self.fc2(x)

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Chargement / Création Données
    if os.path.exists(TENSORS_FILE):
        print(f"⚡ Chargement Cache {TENSORS_FILE}...")
        saved = torch.load(TENSORS_FILE)
        X_t, X_d, X_c, X_l, Y = saved['data']
        num_circuits = saved['num_circuits']
    else:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_pickle(HISTORY_FILE)
            # Filtre tours aberrants absolus (>200s)
            df = df[df['lap_duration'] < 200]
            
            res = create_dataset_complex(df, SEQUENCE_LENGTH)
            if res[0] is not None:
                X_t, X_d, X_c, X_l, Y, num_circuits = res
                torch.save({'data': (X_t, X_d, X_c, X_l, Y), 'num_circuits': num_circuits}, TENSORS_FILE)
            else:
                raise ValueError("Dataset vide.")
        else:
            raise ValueError("Lance le crawler d'abord !")

    # 2. Training Setup
    idx = np.arange(len(X_t))
    np.random.shuffle(idx)
    split = int(0.8 * len(X_t))
    train_i, val_i = idx[:split], idx[split:]

    train_ds = TensorDataset(X_t[train_i], X_d[train_i], X_c[train_i], X_l[train_i], Y[train_i])
    val_ds = TensorDataset(X_t[val_i], X_d[val_i], X_c[val_i], X_l[val_i], Y[val_i])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = ExpertPodiumPredictor(num_circuits)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n🚀 Démarrage Entraînement Expert (Circuits: {num_circuits})")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch in train_loader:
            xt, xd, xc, xl, y = batch
            optimizer.zero_grad()
            pred = model(xt, xd, xc, xl)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.5f}")

    # Sauvegarde Modèle + Méta-données (Important pour la taille des embeddings)
    torch.save({
        'state_dict': model.state_dict(),
        'num_circuits': num_circuits
    }, "expert_model.pth")
    print("\n💾 Modèle Expert sauvegardé !")