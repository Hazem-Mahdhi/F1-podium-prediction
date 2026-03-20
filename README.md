# 🏎️ F1 Podium Prediction

> Système de prédiction du podium de Formule 1 basé sur des données réelles et deux réseaux de neurones entraînés sur l'historique des courses.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![OpenF1](https://img.shields.io/badge/API-OpenF1-E10600)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Présentation

Ce projet prédit le podium de la prochaine course de F1 à partir de données réelles récupérées via l'API OpenF1. Il combine un modèle **LSTM** pour estimer le rythme de course de chaque pilote et un **réseau de neurones hybride** pour produire un classement final.

Le système se met **automatiquement à jour** à chaque lancement : il récupère les dernières courses, détecte la prochaine étape du calendrier et génère une prédiction fraîche.

---

## 🧠 Architecture ML

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
   🏆 Podium prédit
```

### Détail des modèles

| Modèle | Architecture | Données d'entrée | Sortie |
|--------|-------------|-----------------|--------|
| **LSTM Expert** | LSTM(64) + Embeddings pilote/circuit | Séquences de 5 tours + progression course | Temps au tour normalisé |
| **HybridNet** | 3 Embeddings + 3 couches FC + Dropout | Pilote, écurie, circuit, pace_deficit | Score de classement [0–1] |

---

## 🚀 Démarrage rapide

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

### ▶️ Lancement

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

#### Entraînement complet des modèles (première fois)
```bash
# Étape 1 : Collecte des données
python get_race_results.py   # Classements de courses
python build_master_db.py    # Temps au tour

# Étape 2 : Fusion
python fusion_data.py

# Étape 3 : Entraînement des modèles
python LapTimePredictor.py   # LSTM
python HybridPodium.py       # HybridNet (spécifique à un circuit)
```

---

## 📁 Structure du projet

```
F1-podium-prediction/
│
├── main.py                  # ⭐ Point d'entrée principal (auto-update + prédiction)
├── interface.py             # Interface Streamlit
│
├── 📡 Data Pipeline
│   ├── sessions.py          # Wrapper API OpenF1 — sessions
│   ├── get_race_results.py  # Collecte des classements de courses
│   ├── build_master_db.py   # Collecte des temps au tour
│   └── fusion_data.py       # Fusion et calcul du pace_deficit
│
├── 🧠 Modèles ML
│   ├── LapTimePredictor.py  # LSTM — prédiction de rythme
│   ├── HybridPodium.py      # HybridNet — classement final
│   └── pilotes.py           # Helpers pilotes (acronymes, numéros)
│
├── 🔮 Prédiction
│   └── predict_hybrid.py    # CLI de prédiction (pipeline hybride)
│
└── requirements.txt
```

---

## 📊 Données

Les données sont récupérées en temps réel depuis [**OpenF1 API**](https://openf1.org) (API publique et gratuite) :

| Endpoint | Utilisation |
|----------|------------|
| `/sessions` | Calendrier des courses |
| `/session_result` | Classements officiels |
| `/laps` | Temps au tour par pilote |
| `/drivers` | Profils pilotes (numéro, équipe, acronyme) |

Les données sont persistées localement en `.pkl` (format pickle/pandas) pour éviter de re-télécharger à chaque lancement. Seules les nouvelles courses sont récupérées.

---

## ⚙️ Stack Technique

| Domaine | Technologie |
|---------|------------|
| Langage | Python 3.10+ |
| Deep Learning | PyTorch 2.x (LSTM, Embedding, FC layers) |
| Data | pandas, NumPy |
| Interface | Streamlit |
| API | OpenF1 (REST, JSON) |
| Persistance | pickle / pandas `.pkl` |

---

## 🌐 Hébergement (gratuit)

L'interface Streamlit peut être déployée gratuitement sur **[Streamlit Community Cloud](https://streamlit.io/cloud)** :

1. Fork ce repo sur ton compte GitHub
2. Connecte-toi sur [share.streamlit.io](https://share.streamlit.io)
3. Sélectionne le repo → fichier `interface.py`
4. Deploy 🚀

> ⚠️ Note : les fichiers `.pkl` (modèles entraînés) ne persistent pas entre les déploiements cloud. Voir la section suivante.

---

## 📝 Notes

- Le projet cible la saison **2024–2025** par défaut. Les années sont mises à jour automatiquement via `main.py`.
- Les modèles `.pth` et les bases de données `.pkl` ne sont pas versionnés (`.gitignore`). Il faut lancer le pipeline d'entraînement une première fois en local.
- Pour les nouveaux pilotes (rookies 2025), le rythme est estimé à partir de leur coéquipier.

---

## 👤 Auteur

**Hazem Mahdhi**  
Projet réalisé dans le cadre d'un apprentissage personnel du Machine Learning appliqué à la data sportive.

[![GitHub](https://img.shields.io/badge/GitHub-Hazem--Mahdhi-181717?logo=github)](https://github.com/Hazem-Mahdhi)
