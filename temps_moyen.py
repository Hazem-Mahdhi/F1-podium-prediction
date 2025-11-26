import pandas as pd
import numpy as np
import torch

import requests
import pandas as pd
import numpy as np
import torch

BASE_URL = "https://api.openf1.org/v1"
TARGET_SESSION_KEY = 9161 # Clé de session réelle (ex: Course d'Imola 2024)
DRIVER_NUMBER = 1        # Numéro de pilote réel (Max Verstappen)

def fetch_laps_data(session_key, driver_number):
    """
    Récupère les données de tours d'un pilote pour une session donnée.
    """
    laps_url = f"{BASE_URL}/laps"
    params = {
        'session_key': session_key,
        'driver_number': driver_number
    }
    
    # ... (code de requête omis pour la concision)
    # response = requests.get(laps_url, params=params)
    # df = pd.DataFrame(response.json())
    
    # Remplaçons par une simulation de la réponse pour l'instant pour continuer sur le formatage
    data = {
        'lap_number': range(1, 16),
        'lap_duration': [95.1, 94.8, 94.5, 94.2, 94.1, 
                        94.0, 93.9, 94.3, 94.5, 94.8, 
                        95.0, 95.2, 95.5, 95.8, 96.0],
        'is_pit_in_lap': [False] * 15 # Ajout d'une feature utile
    }
    df = pd.DataFrame(data)
    # --- Fin de la simulation ---
    
    # Le nettoyage des données est crucial : on retire les tours non pertinents pour le rythme (pit-stops, Safety Car)
    df_cleaned = df[df['lap_duration'].notna()].copy() 
    return df_cleaned.sort_values(by='lap_number').reset_index(drop=True)

laps_df_clean = fetch_laps_data(TARGET_SESSION_KEY, DRIVER_NUMBER)

def create_sequence(df, sequence_length):
    """
    Transforme un DataFrame de tours consécutifs en séquences (X) et cibles (Y).

    :param df: DataFrame contenant les données de tours d'UN SEUL pilote.
    :param sequence_length: Longueur T de la séquence (nombre de tours consécutifs).
    :return: Listes des séquences (X) et des cibles (Y).
    """

    X, Y = [],
    features = df['lap_duration'].values

    for i in range(len(features) - sequence_length):
        seq_x = features[i:i + sequence_length]
        seq_y = features[i + sequence_length]
        X.append(seq_x)
        Y.append(seq_y)

    # On convertit les listes NumPy en tenseurs PyTorch
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)
    
    # Les tenseurs X pour un LSTM doivent avoir la forme [samples, sequence_length, features]
    # Comme nous n'avons qu'une seule feature (F=1), nous devons l'ajouter
    X = X.unsqueeze(-1)
    
    return X, Y


SEQUENCE_LENGTH = 5 
X_tensor, Y_tensor = create_sequence(laps_df_clean, SEQUENCE_LENGTH)

print(f"Forme du Tenseur d'Entrée X (LSTM) : {X_tensor.shape}")
print(f"Forme du Tenseur de Cible Y (Régression) : {Y_tensor.shape}")
print(f"Exemple de Séquence X (5 tours) : {X_tensor[0].squeeze().tolist()}")
print(f"Exemple de Cible Y (Tour 6) : {Y_tensor[0].item()}")
