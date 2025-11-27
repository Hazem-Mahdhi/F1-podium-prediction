import pandas as pd
import numpy as np
import os

# Fichiers sources (Assure-toi qu'ils existent via tes crawlers précédents)
RESULTS_FILE = "f1_race_results.pkl"      # Contient les positions
TIMES_FILE = "f1_master_database.pkl"     # Contient les temps au tour
OUTPUT_FILE = "f1_hybrid_dataset.pkl"

def create_hybrid_dataset():
    print("🔄 Chargement des bases de données...")
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(TIMES_FILE):
        print("❌ Fichiers manquants. Lance 'build_master_db.py' et 'get_race_results.py' d'abord.")
        return

    df_res = pd.read_pickle(RESULTS_FILE)
    df_times = pd.read_pickle(TIMES_FILE)

    # 1. Calcul du "Rythme de Course" (Race Pace) pour chaque pilote/course
    print("⏱️  Calcul du rythme moyen par pilote (Médiane)...")
    
    # On filtre les tours aberrants (Safety Car, etc) pour avoir le vrai rythme
    # On considère qu'un tour > 110% du meilleur tour global est "pollué"
    df_times = df_times[df_times['lap_duration'] < 150] # Filtre grossier
    
    # Calcul de la médiane par session/pilote (La médiane ignore les arrêts aux stands)
    pace_df = df_times.groupby(['session_key', 'driver_number'])['lap_duration'].median().reset_index()
    pace_df.rename(columns={'lap_duration': 'race_pace'}, inplace=True)

    # 2. Fusion avec les Résultats
    print("🔗 Fusion des Positions et des Temps...")
    merged_df = pd.merge(df_res, pace_df, on=['session_key', 'driver_number'], how='inner')

    # 3. Normalisation Intelligente (Le "Pace Deficit")
    # Au lieu du temps brut (80s), on calcule le % de retard sur le meilleur de la course.
    # 1.00 = Le plus rapide. 1.01 = 1% plus lent.
    
    # Trouver le meilleur rythme de chaque session
    session_best = merged_df.groupby('session_key')['race_pace'].min().reset_index()
    session_best.rename(columns={'race_pace': 'best_session_pace'}, inplace=True)
    
    merged_df = pd.merge(merged_df, session_best, on='session_key')
    merged_df['pace_deficit'] = merged_df['race_pace'] / merged_df['best_session_pace']

    # Sauvegarde
    merged_df.to_pickle(OUTPUT_FILE)
    print(f"✅ Dataset Hybride créé : {len(merged_df)} performances enregistrées.")
    print(merged_df[['circuit', 'driver_number', 'team_name', 'position', 'pace_deficit']].head())

if __name__ == "__main__":
    create_hybrid_dataset()