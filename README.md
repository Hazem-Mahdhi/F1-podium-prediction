# 🏎️ F1 Podium Prediction

> Système de prédiction du podium de Formule 1 basé sur des données réelles et deux réseaux de neurones entraînés sur l'historique des courses.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![OpenF1](https://img.shields.io/badge/API-OpenF1-E10600)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Présentation

Ce projet prédit le podium de la prochaine course de F1 à partir de données réelles récupérées via l'API OpenF1. Il combine un modèle **LSTM** pour estimer le rythme de course de chaque pilote et un **réseau de neurones hybride** pour produire un classement final.

Le système se met **automatiquement à jour** à chaque lancement : il récupère les dernières courses, détecte la prochaine étape du calendrier et génère une prédiction fraîche.

---

## Architecture ML

Le pipeline de prédiction repose sur deux modèles complémentaires :

```
Données Brutes (OpenF1 API)
        │
        ▼
┌───────────────────┐
│  Pipeline Data    │  get_race_results.py + build_master_db.py + fusion_data.py
│  (résultats +     │  → f1_race_results.pkl
│   temps au tour)  │  → f1_master_database.pkl
└───────────────────┘  → f1_hybrid_dataset.pkl
        │
        ▼
┌───────────────────┐
│  Modèle 1 : LSTM  │  LapTimePredictor.py
│  "Le Physicien"   │  Prédit le rythme de course (temps au tour)
│                   │  de chaque pilote sur le circuit cible
└───────────────────┘
        │  pace_deficit par pilote
        ▼
┌───────────────────┐
│  Modèle 2 :       │  HybridPodium.py
│  HybridNet        │  Classement final (position normalisée)
│  "Le Stratège"    │  → Embeddings pilote + équipe + circuit
│                   │  → Intègre le rythme prédit en entrée
└───────────────────┘
        │
        ▼
   Podium prédit
```

### Détail des modèles

| Modèle | Architecture | Données d'entrée | Sortie |
|--------|-------------|-----------------|--------|
| **LSTM Expert** | LSTM(64) + Embeddings pilote/circuit | Séquences de 5 tours + progression course | Temps au tour normalisé |
| **HybridNet** | 3 Embeddings + 3 couches FC + Dropout | Pilote, écurie, circuit, pace_deficit | Score de classement [0–1] |

---

## Démarrage rapide

### Prérequis

- Python 3.10+
- Git

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/Hazem-Mahdhi/F1-podium-prediction.git
cd F1-podium-prediction

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Lancement

#### Interface Web (Streamlit)
```bash
streamlit run interface.py
```
Ouvre automatiquement `http://localhost:8501` dans ton navigateur.

#### Script CLI (auto-update + prédiction)
```bash
python main.py
```
Ce script :
1. Récupère automatiquement les dernières courses depuis l'API OpenF1
2. Détecte la prochaine course du calendrier
3. Spécialise le modèle pour ce circuit
4. Affiche le podium prédit