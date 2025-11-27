import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import os
import time
from sessions import get_session_list

# --- CONFIGURATION ---
BASE_URL = "https://api.openf1.org/v1"
MASTER_DB_FILE = "f1_master_database.pkl"
YEARS_TO_UPDATE = [2024, 2025] # On cible la fin 2024 et le début 2025

# Session robuste
def create_robust_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

http = create_robust_session()

def get_drivers_in_session(session_key):
    try:
        response = http.get(f"{BASE_URL}/drivers", params={'session_key': session_key}, timeout=10)
        return [d['driver_number'] for d in response.json()]
    except: return []

def get_laps(session_key, driver_number):
    try:
        response = http.get(f"{BASE_URL}/laps", params={'session_key': session_key, 'driver_number': driver_number}, timeout=15)
        return pd.DataFrame(response.json())
    except: return pd.DataFrame()

def update_master_database():
    # 1. Chargement de l'existant
    if os.path.exists(MASTER_DB_FILE):
        print(f"📂 Chargement de la base existante...")
        master_df = pd.read_pickle(MASTER_DB_FILE)
        existing_keys = master_df['session_key'].unique().tolist()
        print(f"   -> {len(existing_keys)} sessions déjà en stock (Dernière: {master_df['session_key'].max()})")
    else:
        master_df = pd.DataFrame()
        existing_keys = []

    print(f"🚀 Recherche de nouvelles données pour : {YEARS_TO_UPDATE}")
    
    for year in YEARS_TO_UPDATE:
        print(f"\n📅 Vérification SAISON {year}...")
        try:
            sessions = get_session_list(year=year, session_name="Race")
        except:
            print(f"   Pas de session trouvée ou erreur API pour {year}.")
            continue
            
        for session in sessions:
            s_key = session['session_key']
            s_name = session['location']
            
            # --- LE FILTRE : On ne télécharge que ce qu'on n'a pas ---
            if s_key in existing_keys:
                print(f"  ✅ {s_name} ({year}) déjà téléchargé. On passe.")
                continue
                
            print(f"  📥 TÉLÉCHARGEMENT : {s_name} ({year})...")
            
            drivers = get_drivers_in_session(s_key)
            if not drivers:
                print("     ⚠️ Course pas encore commencée ou pas de données.")
                continue
            
            session_laps = []
            for d_num in drivers:
                df_laps = get_laps(s_key, d_num)
                if not df_laps.empty:
                    if 'is_pit_out_lap' in df_laps.columns:
                        df_laps = df_laps[df_laps['is_pit_out_lap'] == False]
                    df_laps = df_laps.dropna(subset=['lap_duration'])
                    df_laps['year'] = year
                    df_laps['circuit'] = s_name
                    df_laps['session_key'] = s_key
                    df_laps['driver_number'] = d_num
                    session_laps.append(df_laps)
                time.sleep(0.1) 

            # Sauvegarde immédiate
            if session_laps:
                new_race_df = pd.concat(session_laps, ignore_index=True)
                
                # Optimisation types
                new_race_df['driver_number'] = new_race_df['driver_number'].astype('int16')
                new_race_df['session_key'] = new_race_df['session_key'].astype('int32')
                
                if master_df.empty:
                    master_df = new_race_df
                else:
                    master_df = pd.concat([master_df, new_race_df], ignore_index=True)
                
                master_df.to_pickle(MASTER_DB_FILE)
                print(f"     💾 {s_name} Ajouté ! (Total DB: {len(master_df)} tours)")
                existing_keys.append(s_key) # On met à jour la liste locale
            else:
                print("     ❌ Pas de tours valides récupérés.")

    print("\n🎉 Base de données Tours à jour !")

if __name__ == "__main__":
    update_master_database()