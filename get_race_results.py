import requests
import pandas as pd
import time
import os
from sessions import get_session_list

BASE_URL = "https://api.openf1.org/v1"
RESULTS_FILE = "f1_race_results.pkl"
YEARS_TO_UPDATE = [2024, 2025]

def get_session_results(session_key):
    try:
        # Résultats
        r = requests.get(f"{BASE_URL}/session_result", params={'session_key': session_key}, timeout=10)
        results = r.json()
        # Drivers pour les teams
        r_d = requests.get(f"{BASE_URL}/drivers", params={'session_key': session_key}, timeout=10)
        drivers_data = r_d.json()
        driver_teams = {d['driver_number']: d['team_name'] for d in drivers_data}
        
        data = []
        for res in results:
            d_num = res['driver_number']
            data.append({
                'driver_number': d_num,
                'team_name': driver_teams.get(d_num, "Unknown"),
                'position': res['position'],
                'grid_position': res.get('grid_position', res['position']),
                'points': res.get('points', 0),
                'status': 'DNF' if res.get('dnf') else 'Finished'
            })
        return pd.DataFrame(data)
    except: return pd.DataFrame()

def build_results_db():
    if os.path.exists(RESULTS_FILE):
        print("📂 Chargement résultats existants...")
        existing_df = pd.read_pickle(RESULTS_FILE)
        existing_keys = existing_df['session_key'].unique().tolist()
        print(f"   -> {len(existing_keys)} courses connues.")
    else:
        existing_df = pd.DataFrame()
        existing_keys = []

    new_data_found = False
    
    for year in YEARS_TO_UPDATE:
        print(f"\n📅 Vérification Résultats {year}...")
        try:
            sessions = get_session_list(year=year, session_name="Race")
        except: continue
            
        for s in sessions:
            s_key = s['session_key']
            s_name = s['location']
            
            if s_key in existing_keys:
                print(f"✅ {s_name} déjà fait.")
                continue
            
            print(f"📥 Récupération classement : {s_name}...")
            df = get_session_results(s_key)
            
            if not df.empty:
                df['session_key'] = s_key
                df['year'] = year
                df['circuit'] = s_name
                df['country'] = s['country_name']
                
                if existing_df.empty:
                    existing_df = df
                else:
                    existing_df = pd.concat([existing_df, df], ignore_index=True)
                
                # On force le type int pour la clé
                existing_df['session_key'] = existing_df['session_key'].astype('int32')
                existing_df.to_pickle(RESULTS_FILE)
                new_data_found = True
                existing_keys.append(s_key)
            
            time.sleep(0.5)
            
    if new_data_found:
        print(f"\n🎉 Base Résultats mise à jour ! Total : {len(existing_df)} lignes.")
    else:
        print("\n✨ Rien de nouveau sous le soleil.")

if __name__ == "__main__":
    build_results_db()