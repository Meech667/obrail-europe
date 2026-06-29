# ObRail Europe — MSPR Bloc E6.2 / E6.4

**EPSI 2025-2026 — Développeur en Intelligence Artificielle et Data Science**

Projet réalisé pour ObRail Europe, observatoire indépendant spécialisé dans le ferroviaire et la mobilité durable.

---

## Objectif

Développer des modèles d'IA répondant à deux enjeux métiers :

- **Régression CO2** : estimer les émissions futures d'une liaison ferroviaire selon des scénarios de développement du réseau
- **Classification** : identifier automatiquement les liaisons candidates à remplacer un vol aérien

---

## Structure du projet

```
obrail-europe/
├── 01_regression_co2.ipynb                       # Enjeu régression — 3 modèles, GridSearchCV, CV
├── 02_classification_substitution_avion.ipynb    # Enjeu classification — 3 modèles, GridSearchCV, CV
├── api.py                                        # API REST commune (FastAPI) — 2 routes
├── predict.py                                    # Script prédiction classification standalone
├── predict_co2.py                                # Script prédiction régression CO2 standalone
├── requirements.txt                              # Dépendances Python
├── models/                                       # Modèles sauvegardés (.joblib)
│   ├── classification_substitution_avion.joblib
│   ├── regression_co2.joblib
│   ├── encoders.joblib
│   └── scaler.joblib
├── figures/
│   ├── CLF/                                      # Visualisations classification (11 figures)
│   └── REG/                                      # Visualisations régression (8 figures)
└── eda/                                          # Cadrage EDA et modélisation complémentaire
    ├── 00_cadrage_eda.ipynb
    ├── 01_modelisation_resultats.ipynb
    └── figures/                                  # Visualisations EDA (8 figures)
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Données
Le dataset `eu_trips_v2.csv` est disponible ici :
https://drive.google.com/file/d/1vt7VH9y0hb1HF0VJOSGWkntRVcbhafPr/view?usp=sharing
Télécharger et placer à la racine du projet avant d'exécuter les notebooks.

---

## Lancement des notebooks

Exécuter dans l'ordre depuis la racine du projet :

```bash
# Enjeu 1 — Régression CO2
jupyter notebook 01_regression_co2.ipynb

# Enjeu 2 — Classification substitution avion/train
jupyter notebook 02_classification_substitution_avion.ipynb
```

Ou via VSCode : ouvrir le fichier `.ipynb` → **Kernel** → **Restart & Run All**

---

## Lancement de l'API

```bash
uvicorn api:app --reload
```

Documentation interactive disponible sur : **http://localhost:8000/docs**

---

## Routes API

| Route | Méthode | Description |
|---|---|---|
| GET / | GET | Message d'accueil et liste des routes |
| GET /health | GET | Statut des deux modèles chargés |
| POST /predict/substitution | POST | Classification — liaison substituable à l'avion (0 ou 1) |
| POST /predict/co2 | POST | Régression — estimation CO2 futur selon scénario |

### Exemple — POST /predict/substitution

```json
{
  "distance_km": 800,
  "duration_minutes": 195,
  "n_stops": 3,
  "co2_estime": 450000,
  "consommation_totale": 16000,
  "type_train": "electric",
  "country": "FR"
}
```

### Exemple — POST /predict/co2

```json
{
  "distance_km": 400,
  "duration_minutes": 120,
  "n_stops": 2,
  "consommation_energy": 10.0,
  "gco2_per_kwh": 21.7,
  "consommation_totale": 4000,
  "type_train": "diesel",
  "scenario": "diesel_50_electrique"
}
```

Scénarios disponibles : `reference`, `diesel_50_electrique`, `conso_moins_15`, `distance_moins_10`

---

## Modèles et performances

| Enjeu | Modèles testés | Modèle retenu | Métrique principale |
|---|---|---|---|
| Régression CO2 | Régression Linéaire, Random Forest, XGBoost | XGBoost | R² = 1.000, MAE = 0.939 kgCO2 |
| Classification substitution | Logistic Regression, Random Forest, XGBoost | XGBoost | F1-macro = 0.997, AUC = 1.000 |

Cross-validation 5 folds réalisée sur les deux modèles — écart-type < 0.001 confirmant la stabilité.

---

## Scripts de prédiction standalone

```bash
# Test classification
python predict.py

# Test régression CO2
python predict_co2.py
```

---

## CI/CD — Pipeline GitHub Actions

[![CI/CD ObRail Europe](https://github.com/Meech667/obrail-europe/actions/workflows/ci.yml/badge.svg)](https://github.com/Meech667/obrail-europe/actions/workflows/ci.yml)

Le pipeline s'exécute automatiquement sur chaque `push` et `pull request` vers la branche `main`.

### Ce que teste le pipeline

| Test | Route | Description |
|---|---|---|
| `test_health` | GET /health | Vérifie que l'API démarre et que les 2 modèles sont chargés |
| `test_predict_substitution_substituable` | POST /predict/substitution | Paris→Marseille classifiée substituable (résultat = 1) |
| `test_predict_substitution_non_substituable` | POST /predict/substitution | Liaison 5800 km classifiée non substituable (résultat = 0) |
| `test_predict_co2_reference` | POST /predict/co2 | Scénario référence retourne un float positif |
| `test_predict_co2_electrification` | POST /predict/co2 | Scénario électrification réduit les émissions vs référence |

> Si les fichiers `.joblib` sont absents, les tests sont automatiquement ignorés (`skip`) sans faire échouer le pipeline.

### Lancer les tests localement

```bash
# Installer les dépendances (inclut pytest, httpx, pytest-cov)
pip install -r requirements.txt

# Lancer tous les tests
pytest tests/

# Avec rapport de couverture de code
pytest tests/ --cov=api --cov-report=term-missing -v
```

---

## Monitoring

Système de journalisation et de surveillance conforme RGPD — aucune donnée personnelle dans les logs.

| Composant | Fichier | Description |
|---|---|---|
| Logs API | `logs/api.log` | Journal de toutes les requêtes (features numériques uniquement) |
| Monitoring | `monitoring.py` | Métriques, taux d'erreur, durée, alertes automatiques |
| Drift detection | `detect_drift.py` | Détection dérive des données vs statistiques d'entraînement |
| Dashboard | `dashboard.py` | Tableau de bord Streamlit temps réel |

```bash
# Lancer le monitoring (métriques + détection d'incidents)
python monitoring/monitoring.py

# Détecter le drift des données (feedback loop MLOps)
python monitoring/detect_drift.py

# Tableau de bord visuel
streamlit run monitoring/dashboard.py   # → http://localhost:8501
```

**Outils prévus en production :** Prometheus · Grafana · Loki · Evidently AI

Voir [monitoring/MONITORING.md](monitoring/MONITORING.md) pour la documentation complète.

---

## Équipe

Projet réalisé en groupe dans le cadre de la MSPR Bloc E6.2 / E6.4 — EPSI 2025-2026.

**Membres de l'équipe :** Messa · Adam · Akram · Aymen

Lien du repo : https://github.com/Meech667/obrail-europe
