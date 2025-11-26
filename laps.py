import requests
import pandas as pd

BASE_URL = "https://api.openf1.org/v1"

def fetch_laps_data(session_key, driver_number=None):
    """
    Récupère les données de tours et les renvoie sous forme de DataFrame.
    """
    laps_url = f"{BASE_URL}/laps"
    params = {'session_key': session_key}
    
    if driver_number:
        params['driver_number'] = driver_number
    
    try:
        response = requests.get(laps_url, params=params)
        response.raise_for_status() 
        data = response.json()
        
        # Convertir la liste de dictionnaires JSON en DataFrame pandas
        df = pd.DataFrame(data)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return pd.DataFrame()

# --- Exemple d'utilisation (en supposant que vous ayez obtenu la clé) ---

# Utilisez la clé obtenue à l'étape 2 (exemple fictif si l'étape 2 n'a pas été exécutée)
TARGET_SESSION_KEY = 9523 

# Récupérer les tours pour tous les pilotes de cette session
laps_df = fetch_laps_data(TARGET_SESSION_KEY)

if not laps_df.empty:
    print(f"\n✅ Données de tours récupérées pour la session {TARGET_SESSION_KEY}.")
    print(f"Nombre de lignes : {len(laps_df)}")
    print("\nCinq premières lignes du DataFrame :")
    print(laps_df[['driver_number', 'lap_number', 'lap_duration', 'is_pit_out_lap', 'stint_number']].head())
    
    # Exemple d'analyse : Trouver le meilleur temps au tour
    fastest_lap = laps_df['lap_duration'].min()
    print(f"\nMeilleur temps au tour de la session : {fastest_lap:.3f} secondes.")