"""
F1 Podium Prediction — Point d'entrée principal
Détecte automatiquement la prochaine course et prédit le podium.
"""

import os
import sys
import torch
import pickle
from datetime import datetime, timezone
import requests

# Pipeline de données
from build_master_db import update_master_database
from get_race_results import build_results_db
from fusion_data import create_hybrid_dataset

# Modèles
from LapTimePredictor import ExpertPodiumPredictor
from HybridPodium import HybridNet, train_hybrid_model
from pilotes import get_driver_acronym
from sessions import get_session_list

# --- CONFIG ---
BASE_URL = "https://api.openf1.org/v1"
MODEL_LSTM   = "expert_model.pth"
MODEL_HYBRID = "hybrid_model.pth"
MAP_LSTM     = "circuit_map.pkl"
MAP_HYBRID   = "hybrid_mappings.pkl"

GLOBAL_MAX_LAP_TIME = 150.0
SEQUENCE_LENGTH = 5

GRID_2025 = [
    (1,  "Red Bull Racing"), (30, "Red Bull Racing"),
    (16, "Ferrari"),         (44, "Ferrari"),
    (63, "Mercedes"),        (12, "Mercedes"),
    (4,  "McLaren"),         (81, "McLaren"),
    (14, "Aston Martin"),    (18, "Aston Martin"),
    (10, "Alpine"),          (61, "Alpine"),
    (23, "Williams"),        (55, "Williams"),
    (31, "Haas F1 Team"),    (87, "Haas F1 Team"),
    (22, "RB"),              (3,  "RB"),
    (27, "Sauber"),          (5,  "Sauber"),
]

ROOKIE_NAMES = {12: "ANT", 30: "LAW", 61: "DOO", 87: "BEA", 5: "BOR"}

# ─────────────────────────────────────────────
# 1. DÉTECTION AUTOMATIQUE DE LA PROCHAINE COURSE
# ─────────────────────────────────────────────
def get_next_race():
    """
    Interroge l'API OpenF1 pour trouver la prochaine course à venir.
    Retourne un dict {circuit, session_key} ou None.
    """
    current_year = datetime.now().year
    now_utc = datetime.now(timezone.utc)

    for year in [current_year, current_year + 1]:
        try:
            sessions = get_session_list(year=year, session_name="Race")
        except Exception:
            continue

        for s in sessions:
            date_str = s.get("date_start", "")
            if not date_str:
                continue
            try:
                # L'API renvoie des dates ISO 8601 (ex: "2025-03-16T15:00:00")
                race_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if race_dt.tzinfo is None:
                    race_dt = race_dt.replace(tzinfo=timezone.utc)
                if race_dt > now_utc:
                    return {
                        "circuit": s["location"],
                        "country": s.get("country_name", ""),
                        "session_key": s["session_key"],
                        "date": race_dt.strftime("%d %B %Y"),
                    }
            except Exception:
                continue

    return None


# ─────────────────────────────────────────────
# 2. MISE À JOUR DYNAMIQUE DES ANNÉES
# ─────────────────────────────────────────────
def patch_years():
    """
    Met à jour dynamiquement YEARS_TO_UPDATE dans les modules de data
    pour toujours cibler l'année précédente + l'année courante.
    """
    import build_master_db
    import get_race_results
    year = datetime.now().year
    build_master_db.YEARS_TO_UPDATE = [year - 1, year]
    get_race_results.YEARS_TO_UPDATE = [year - 1, year]


# ─────────────────────────────────────────────
# 3. PRÉDICTION
# ─────────────────────────────────────────────
def predict(circuit_name: str):
    """Lance la prédiction complète pour un circuit donné."""

    if not os.path.exists(MODEL_LSTM):
        print("❌ Modèle LSTM introuvable. Relance l'entraînement (LapTimePredictor.py).")
        return

    # Chargement modèles
    device = torch.device("cpu")
    ckpt   = torch.load(MODEL_LSTM, map_location=device)
    lstm   = ExpertPodiumPredictor(ckpt["num_circuits"])
    lstm.load_state_dict(ckpt["state_dict"])
    lstm.eval()

    with open(MAP_LSTM, "rb") as f:
        lstm_circuits = pickle.load(f)

    with open(MAP_HYBRID, "rb") as f:
        maps_h = pickle.load(f)

    hybrid = HybridNet(len(maps_h["d"]), len(maps_h["t"]), len(maps_h["c"]))
    hybrid.load_state_dict(torch.load(MODEL_HYBRID, map_location=device))
    hybrid.eval()

    c_id_lstm   = lstm_circuits.get(circuit_name, 0)
    c_id_hybrid = maps_h["c"].get(circuit_name, 0)

    base_input   = torch.tensor([[[90.0 / GLOBAL_MAX_LAP_TIME] * SEQUENCE_LENGTH]], dtype=torch.float32).unsqueeze(-1)
    lap_progress = torch.tensor([[0.5]], dtype=torch.float32)

    # Phase 1 : Rythme LSTM
    predicted_times = []
    team_pace_ref   = {}

    with torch.no_grad():
        for d_num, team in GRID_2025:
            try:
                pred = lstm(
                    base_input,
                    torch.tensor([d_num], dtype=torch.long),
                    torch.tensor([c_id_lstm], dtype=torch.long),
                    lap_progress,
                ).item()
            except Exception:
                pred = None

            if pred is not None:
                predicted_times.append({"driver": d_num, "team": team, "raw_pace": pred})
                if team not in team_pace_ref or pred < team_pace_ref[team]:
                    team_pace_ref[team] = pred

    # Rookies → rythme coéquipier
    known = {p["driver"] for p in predicted_times}
    for d_num, team in GRID_2025:
        if d_num not in known:
            fallback = team_pace_ref.get(team, 0.65)
            predicted_times.append({"driver": d_num, "team": team, "raw_pace": fallback})

    best_pace = min(p["raw_pace"] for p in predicted_times if p["raw_pace"] > 0)

    # Phase 2 : Score Hybride
    leaderboard = []
    with torch.no_grad():
        for p in predicted_times:
            pace_deficit = p["raw_pace"] / best_pace
            score = hybrid(
                torch.tensor([maps_h["d"].get(p["driver"], 0)]),
                torch.tensor([maps_h["t"].get(p["team"],   0)]),
                torch.tensor([c_id_hybrid]),
                torch.tensor([[pace_deficit]]),
            ).item()

            if p["team"] in ("Red Bull Racing", "McLaren", "Ferrari"):
                score += 0.05

            acr = ROOKIE_NAMES.get(p["driver"]) or get_driver_acronym(p["driver"])
            if acr == "???":
                acr = f"#{p['driver']}"

            leaderboard.append({"Pilote": acr, "Écurie": p["team"], "Score": score})

    leaderboard.sort(key=lambda x: x["Score"], reverse=True)
    return leaderboard


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   🏎️  F1 PODIUM PREDICTION — AUTO UPDATE")
    print("=" * 55)

    # 0. Patch années dynamiques
    patch_years()

    # 1. Mise à jour des données
    print("\n📡 Étape 1/4 — Mise à jour des données API OpenF1...")
    update_master_database()
    build_results_db()

    print("\n🔗 Étape 2/4 — Fusion et calcul des rythmes...")
    create_hybrid_dataset()

    # 2. Détection prochaine course
    print("\n🗓️  Étape 3/4 — Détection de la prochaine course...")
    next_race = get_next_race()

    if not next_race:
        print("⚠️  Aucune course à venir trouvée dans l'API.")
        circuit = input("Entrez manuellement le nom du circuit : ").strip()
    else:
        circuit = next_race["circuit"]
        print(f"✅ Prochaine course détectée : {circuit} ({next_race['date']})")

    # 3. Entraînement spécialisé pour ce circuit
    print(f"\n🧠 Étape 4/4 — Spécialisation du modèle pour {circuit}...")
    success = train_hybrid_model(circuit)

    if not success:
        print("❌ Données insuffisantes pour ce circuit. Essaie un autre nom.")
        sys.exit(1)

    # 4. Prédiction
    print(f"\n{'=' * 55}")
    print(f"  🏆 PODIUM PRÉDIT — {circuit.upper()}")
    print(f"{'=' * 55}")

    results = predict(circuit)

    if not results:
        print("❌ La prédiction a échoué.")
        sys.exit(1)

    emojis = ["🥇", "🥈", "🥉"]
    for i, row in enumerate(results[:10]):
        prefix = emojis[i] if i < 3 else f"P{i+1:>2}"
        print(f"  {prefix}  {row['Pilote']:<6} ({row['Écurie']:<20})  score: {row['Score']:.3f}")

    print("=" * 55)


if __name__ == "__main__":
    main()
