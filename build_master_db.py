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
YEARS_TO_CRAWL = [2023, 2024, 2025] # On se concentre sur l'historique connu. 2025 viendra quand les courses auront lieu.

# --- CRÉATION D'UNE SESSION ROBUSTE (Anti-Crash) ---
def create_robust_session():
    session = requests.Session()
    # Si ça échoue (500, 502, 504), on réessaie 3 fois avec un délai exponentiel
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

# On utilise cette session globale
http = create_robust_session()

def get_drivers_in_session(session_key):
    """Récupère la liste des numéros de pilotes"""
    url = f"{BASE_URL}/drivers"
    try:
        response = http.get(url, params={'session_key': session_key}, timeout=10)
        drivers = response.json()
        return [d['driver_number'] for d in drivers]
    except Exception as e:
        print(f"    ⚠️ Erreur Drivers: {e}")
        return []

def get_laps(session_key, driver_number):
    """Récupère les tours"""
    url = f"{BASE_URL}/laps"
    try:
        response = http.get(url, params={'session_key': session_key, 'driver_number': driver_number}, timeout=15)
        return pd.DataFrame(response.json())
    except Exception:
        return pd.DataFrame()

# ==========================================
# CRAWLER INTELLIGENT
# ==========================================

def update_master_database():
    # 1. Chargement de l'existant (pour ne pas tout refaire)
    if os.path.exists(MASTER_DB_FILE):
        print(f"📂 Chargement de la base existante...")
        master_df = pd.read_pickle(MASTER_DB_FILE)
        existing_sessions = master_df['session_key'].unique()
        print(f"   -> {len(master_df)} tours déjà en banque ({len(existing_sessions)} sessions).")
    else:
        master_df = pd.DataFrame()
        existing_sessions = []

    print(f"🚀 Démarrage du Crawler pour : {YEARS_TO_CRAWL}")
    
    new_data_buffer = [] # Tampon pour les nouvelles données
    
    for year in YEARS_TO_CRAWL:
        print(f"\n📅 SAISON {year}...")
        
        try:
            sessions = get_session_list(year=year, session_name="Race")
        except Exception as e:
            print(f"❌ Erreur critique récupération sessions {year}: {e}")
            continue
            
        for session in sessions:
            s_key = session['session_key']
            s_name = session['location']
            
            # Si on a déjà cette course, on passe !
            if s_key in existing_sessions:
                print(f"  ⏭️  {s_name} ({year}) déjà téléchargé.")
                continue
                
            print(f"  📥 Téléchargement : {s_name} ({year}) [Key: {s_key}]")
            
            drivers = get_drivers_in_session(s_key)
            if not drivers:
                print("     ⚠️ 0 pilotes trouvés (Session future ou bug API).")
                continue
            
            print(f"     -> Récupération de {len(drivers)} pilotes...")
            
            session_laps = []
            for d_num in drivers:
                df_laps = get_laps(s_key, d_num)
                
                if not df_laps.empty:
                    # Nettoyage
                    if 'is_pit_out_lap' in df_laps.columns:
                        df_laps = df_laps[df_laps['is_pit_out_lap'] == False]
                    df_laps = df_laps.dropna(subset=['lap_duration'])
                    
                    # Enrichissement
                    df_laps['year'] = year
                    df_laps['circuit'] = s_name
                    df_laps['session_key'] = s_key
                    df_laps['driver_number'] = d_num
                    
                    session_laps.append(df_laps)
                
                # PAUSE OBLIGATOIRE (Anti-Ban)
                time.sleep(0.2) 

            # Sauvegarde intermédiaire après CHAQUE course
            if session_laps:
                race_df = pd.concat(session_laps, ignore_index=True)
                
                # Optimisation Types
                race_df['driver_number'] = race_df['driver_number'].astype('int16')
                race_df['session_key'] = race_df['session_key'].astype('int32')
                
                # On ajoute au buffer et on sauvegarde tout de suite le gros fichier
                if master_df.empty:
                    master_df = race_df
                else:
                    master_df = pd.concat([master_df, race_df], ignore_index=True)
                
                master_df.to_pickle(MASTER_DB_FILE)
                print(f"     ✅ {s_name} sauvegardé ! (Total DB: {len(master_df)} tours)")
            else:
                print("     ❌ Aucune donnée de tour valide.")

    print("\n🎉 CRAWLING TERMINÉ !")

if __name__ == "__main__":
    update_master_database()