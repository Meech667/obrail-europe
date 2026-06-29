# Monitoring ObRail Europe — Documentation

## Architecture du monitoring

```
                    ┌──────────────────────────┐
                    │        api.py             │
                    │  (FastAPI — uvicorn)      │
                    │  logging → logs/api.log   │
                    └────────────┬─────────────┘
                                 │
                          logs/api.log
                                 │
           ┌─────────────────────┼──────────────────────┐
           │                     │                      │
  monitoring/             monitoring/             monitoring/
  monitoring.py           detect_drift.py         dashboard.py
  (métriques +            (détection              (Streamlit —
   incidents)              data drift)             visuel temps réel)
           │                     │                      │
  logs/monitoring_        logs/drift_           http://localhost:8501
  report.txt              report.txt
```

---

## Journalisation

### Format des logs

```
YYYY-MM-DD HH:MM:SS,mmm - LEVEL - MESSAGE key1=val1 key2=val2 ...
```

**Exemples de lignes :**

```
# Démarrage réussi de l'API
2024-01-15 10:00:01,001 - INFO - Modèle classification_substitution_avion chargé avec succès
2024-01-15 10:00:01,045 - INFO - Modèle regression_co2 chargé avec succès

# Requête de classification (succès)
2024-01-15 10:30:45,123 - INFO - REQUETE route=/predict/substitution distance_km=800.0 duration_minutes=195.0 n_stops=3 co2_estime=450000.0 consommation_totale=16000.0 resultat=1 proba=0.9876 duree_ms=45.2

# Requête de régression CO2 (succès)
2024-01-15 10:31:02,456 - INFO - REQUETE route=/predict/co2 distance_km=400.0 duration_minutes=120.0 n_stops=2 consommation_energy=10.0 gco2_per_kwh=21.7 consommation_totale=4000.0 scenario=reference co2_estime_kg=87.5234 duree_ms=32.1

# Erreur (modèle indisponible)
2024-01-15 10:32:10,789 - ERROR - ERREUR route=/predict/substitution status=503 duree_ms=0.3
```

### Emplacement des fichiers

| Fichier | Description |
|---|---|
| `logs/api.log` | Journal de toutes les requêtes API (succès + erreurs) |
| `logs/monitoring_report.txt` | Rapport généré par `monitoring/monitoring.py` |
| `logs/drift_report.txt` | Rapport généré par `monitoring/detect_drift.py` |
| `logs/.gitkeep` | Fichier vide pour versionner le dossier `logs/` sur GitHub |

> Les fichiers `.log` et `*_report.txt` sont dans `.gitignore` (données de production,
> potentiellement volumineuses). Seul `.gitkeep` est versionné.

### Politique RGPD

**Aucune donnée personnelle** n'est enregistrée dans les logs :

- Pas d'adresse IP ni identifiant réseau
- Pas de nom, prénom, email ou toute autre PII (Personally Identifiable Information)
- Pas d'horodatage lié à un utilisateur spécifique
- Seules les **features numériques des liaisons ferroviaires** sont loggées :
  distance (km), durée (min), nombre d'arrêts, émissions CO2 estimées

Cette politique est conforme au **Règlement Général sur la Protection des Données (RGPD)**,
article 5 — principe de minimisation des données.

---

## Comment lancer le monitoring

```bash
# Depuis la racine du projet
python monitoring/monitoring.py

# Résultat :
#   - Métriques affichées dans le terminal
#   - Alertes en rouge si incidents détectés
#   - Rapport sauvegardé dans logs/monitoring_report.txt
```

**Incidents détectés automatiquement :**
- Taux d'erreur > 5 %
- Durée moyenne de traitement > 2 secondes
- Proportion de substitutions > 90 % ou < 10 % (anomalie de distribution)

---

## Comment détecter le drift des données

```bash
# Depuis la racine du projet
python monitoring/detect_drift.py

# Résultat :
#   - Z-score par feature affiché dans le terminal
#   - Alerte en rouge si z-score > 2 sur une feature
#   - Rapport sauvegardé dans logs/drift_report.txt
```

**Features surveillées :**

| Feature | Moyenne référence | Écart-type référence |
|---|---|---|
| `distance_km` | 99.6 km | 142.4 km |
| `duration_minutes` | 95.4 min | 82.3 min |
| `n_stops` | 14.2 | 9.5 |
| `co2_estime` | 251 165 gCO2 | 539 481 gCO2 |

---

## Comment voir le dashboard

```bash
# Depuis la racine du projet
streamlit run monitoring/dashboard.py

# Accessible sur : http://localhost:8501
```

**Contenu du dashboard :**
- Métriques clés (total requêtes, erreurs, taux d'erreur, durée moyenne)
- Graphique en barres : requêtes par route
- Camembert : distribution substituable / non substituable
- Timeline : requêtes par minute dans le temps
- Tableau des 10 dernières erreurs
- Alertes automatiques en rouge si incident détecté
- Bouton "Actualiser les données"

---

## Feedback loop MLOps

```
Requêtes API
     │
     ▼
logs/api.log  ──────────────────────────────────────►  monitoring/detect_drift.py
                                                              │
                                              ┌───────────────┴────────────────┐
                                              │                                │
                                         Drift détecté ?                  Pas de drift
                                              │                                │
                                              ▼                           Continuer la
                                    1. Alerter l'équipe MLOps             surveillance
                                    2. Lancer réentraînement
                                    3. Valider via CI/CD (pytest)
                                    4. Redéployer le modèle
                                              │
                                              ▼
                                     Nouveau modèle en prod
                                     → cycle recommence
```

Le drift est considéré significatif quand la **moyenne des features récentes**
s'écarte de plus de **2 écarts-types** des statistiques du jeu d'entraînement
(test de z-score unilatéral).

---

## Ce qui serait ajouté en production

| Outil | Rôle |
|---|---|
| **Prometheus** | Collecte les métriques exposées par l'API (requêtes/s, latence P50/P95/P99, taux erreur) via `prometheus-fastapi-instrumentator` |
| **Grafana** | Visualise les métriques Prometheus sous forme de tableaux de bord interactifs avec alertes configurables (email, Slack, PagerDuty) |
| **Loki** | Centralise et indexe les logs de toutes les instances de l'API pour recherche et corrélation avec les métriques Grafana |
| **Evidently AI** | Automatise la détection de drift ML avec rapports HTML et webhooks pour déclencher le réentraînement sans intervention humaine |

### Pourquoi ces outils ?

- **Prometheus** : standard industriel pour le monitoring d'APIs REST, s'intègre nativement
  avec FastAPI en quelques lignes. Expose des métriques temps réel via `/metrics`.
- **Grafana** : interface de visualisation puissante, compatible Prometheus, avec alerting
  avancé (seuils configurables, silences, escalades).
- **Loki** : conçu spécifiquement pour les logs, s'intègre avec Grafana pour corréler
  logs et métriques au même endroit sans indexer le contenu des logs (économique).
- **Evidently AI** : dédié au monitoring ML, génère des rapports de drift automatiques
  en HTML, s'intègre dans Airflow ou MLflow pour déclencher le réentraînement.
