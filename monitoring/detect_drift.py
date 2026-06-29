"""
detect_drift.py — Détection de dérive des données (data drift) ObRail Europe
Usage : python monitoring/detect_drift.py  (depuis la racine du projet)
        python detect_drift.py             (depuis monitoring/)

Compare les distributions des features entrantes (issues de ../logs/api.log) avec
les statistiques de référence du jeu d'entraînement.
Déclenche une alerte si une feature dépasse 2 écarts-types.

--------------------------------------------------------------------------
OUTILS PRODUCTION :
En production, ce script serait remplacé par :
  - Evidently AI : génère des rapports HTML automatisés qui comparent
    les distributions courantes vs. celles de référence. Peut déclencher
    des webhooks pour alerter l'équipe MLOps dès qu'un seuil est dépassé.
  - WhyLogs : profilele les données entrantes en temps réel et génère
    des "profils" statistiques comparables à un profil de référence.
    S'intègre nativement dans les pipelines MLflow et Airflow.

FEEDBACK LOOP MLOps :
  Si drift détecté sur une feature :
    1. Alerter l'équipe MLOps (Slack, email, PagerDuty)
    2. Déclencher le réentraînement du modèle sur les données récentes
    3. Valider le nouveau modèle via le pipeline CI/CD (tests pytest)
    4. Redéployer automatiquement si les métriques s'améliorent
  Ce cycle évite la dégradation silencieuse du modèle en production.
--------------------------------------------------------------------------
"""

import re
import os
from datetime import datetime

# Racine du projet — fonctionne quelle que soit la commande d'appel
_RACINE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE     = os.path.join(_RACINE, "logs", "api.log")
RAPPORT_FILE = os.path.join(_RACINE, "logs", "drift_report.txt")

# Codes couleur ANSI
ROUGE = "\033[91m"
VERT  = "\033[92m"
JAUNE = "\033[93m"
RESET = "\033[0m"
GRAS  = "\033[1m"

# Statistiques de référence issues du jeu d'entraînement
STATS_REFERENCE = {
    "distance_km":      {"moyenne": 99.6,     "ecart_type": 142.4},
    "duration_minutes": {"moyenne": 95.4,     "ecart_type": 82.3},
    "n_stops":          {"moyenne": 14.2,     "ecart_type": 9.5},
    "co2_estime":       {"moyenne": 251165.0, "ecart_type": 539481.0},
}

# Alerte si la moyenne récente s'écarte de plus de 2 écarts-types de la référence
SEUIL_ZSCORE = 2.0

# Nombre minimum de requêtes pour calculer un z-score fiable
MIN_REQUETES = 5


# ---------------------------------------------------------------------------
# Extraction des features depuis les logs
# ---------------------------------------------------------------------------

def extraire_features(lignes):
    """
    Parse les lignes INFO REQUETE de /predict/substitution
    et extrait les valeurs numériques des features surveillées.
    Retourne un dict {feature: [valeurs]}.
    """
    features = {k: [] for k in STATS_REFERENCE}

    for ligne in lignes:
        if " - INFO - REQUETE " not in ligne:
            continue
        route_m = re.search(r'route=(\S+)', ligne)
        if not route_m or route_m.group(1) not in ("/predict/substitution", "/predict"):
            continue

        distance_m = re.search(r'distance_km=([\d.]+)', ligne)
        duree_m    = re.search(r'duration_minutes=([\d.]+)', ligne)
        stops_m    = re.search(r'n_stops=(\d+)', ligne)
        co2_m      = re.search(r'co2_estime=([\d.]+)', ligne)

        if distance_m:
            features["distance_km"].append(float(distance_m.group(1)))
        if duree_m:
            features["duration_minutes"].append(float(duree_m.group(1)))
        if stops_m:
            features["n_stops"].append(float(stops_m.group(1)))
        if co2_m:
            features["co2_estime"].append(float(co2_m.group(1)))

    return features


# ---------------------------------------------------------------------------
# Calcul du drift par feature
# ---------------------------------------------------------------------------

def analyser_drift(features):
    """
    Pour chaque feature, calcule le z-score entre la moyenne récente
    et la moyenne de référence (normalisée par l'écart-type de référence).
    Retourne (resultats, alertes).
    """
    resultats = {}
    alertes   = []

    for feature, valeurs in features.items():
        ref = STATS_REFERENCE[feature]

        if len(valeurs) < MIN_REQUETES:
            resultats[feature] = {
                "n":      len(valeurs),
                "statut": f"Insuffisant ({len(valeurs)} < {MIN_REQUETES} requis)",
                "alerte": False,
            }
            continue

        moy_recente = sum(valeurs) / len(valeurs)
        zscore = abs(moy_recente - ref["moyenne"]) / ref["ecart_type"] if ref["ecart_type"] > 0 else 0.0
        alerte = zscore > SEUIL_ZSCORE

        resultats[feature] = {
            "n":           len(valeurs),
            "moy_recente": moy_recente,
            "moy_ref":     ref["moyenne"],
            "std_ref":     ref["ecart_type"],
            "zscore":      zscore,
            "alerte":      alerte,
        }

        if alerte:
            alertes.append(
                f"{feature} : moyenne récente={moy_recente:.1f} "
                f"vs référence={ref['moyenne']:.1f} "
                f"(z-score={zscore:.2f} > seuil {SEUIL_ZSCORE})"
            )

    return resultats, alertes


# ---------------------------------------------------------------------------
# Génération du rapport
# ---------------------------------------------------------------------------

def generer_rapport(resultats, alertes, n_requetes):
    """Génère le rapport de drift et l'écrit dans RAPPORT_FILE."""
    lignes = [
        "=" * 60,
        "RAPPORT DÉTECTION DRIFT — ObRail Europe",
        f"Généré le      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Requêtes analysées : {n_requetes}",
        "=" * 60,
        "",
        "--- ANALYSE PAR FEATURE ---",
    ]

    for feature, r in resultats.items():
        ref = STATS_REFERENCE[feature]
        lignes.append(f"\n  {feature}")
        lignes.append(f"    Référence (entraînement) : moy={ref['moyenne']:.1f}, σ={ref['ecart_type']:.1f}")

        if "moy_recente" in r:
            statut = "DRIFT DÉTECTÉ" if r["alerte"] else "normal"
            lignes.append(f"    Données récentes : n={r['n']}, moy={r['moy_recente']:.1f}")
            lignes.append(f"    Z-score          : {r['zscore']:.3f}  →  {statut}")
        else:
            lignes.append(f"    Statut           : {r['statut']}")

    lignes += ["", "--- ALERTES DRIFT ---"]
    if alertes:
        for alerte in alertes:
            lignes.append(f"  [DRIFT] {alerte}")
        lignes += [
            "",
            "  Action recommandée :",
            "    1. Alerter l'équipe MLOps",
            "    2. Déclencher le réentraînement du modèle sur les données récentes",
            "    3. Valider via le pipeline CI/CD puis redéployer",
        ]
    else:
        lignes.append("  Aucun drift détecté — distributions stables.")

    lignes += [
        "",
        "--- OUTILS PRODUCTION ---",
        "  Ce script serait remplacé en production par :",
        "  - Evidently AI : rapports HTML automatisés avec seuils configurables",
        "  - WhyLogs      : profiling temps réel des distributions entrantes",
        "  Ces outils s'intègrent dans la pipeline MLOps pour déclencher",
        "  automatiquement le réentraînement sans intervention humaine.",
    ]

    rapport = "\n".join(lignes) + "\n"

    os.makedirs(os.path.dirname(RAPPORT_FILE), exist_ok=True)
    with open(RAPPORT_FILE, "w", encoding="utf-8") as f:
        f.write(rapport)

    return rapport


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{GRAS}Détection de drift — ObRail Europe...{RESET}\n")

    if not os.path.exists(LOG_FILE):
        print(f"{JAUNE}Aucune donnée : {LOG_FILE} absent ou vide.{RESET}")
    else:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lignes = f.readlines()

        features = extraire_features(lignes)
        n_requetes = len(features["distance_km"])
        resultats, alertes = analyser_drift(features)
        rapport = generer_rapport(resultats, alertes, n_requetes)

        print(rapport)

        if alertes:
            print(f"\n{ROUGE}{GRAS}{'=' * 60}{RESET}")
            print(f"{ROUGE}{GRAS}  DRIFT DÉTECTÉ — RÉENTRAÎNEMENT RECOMMANDÉ :{RESET}")
            print(f"{ROUGE}{GRAS}{'=' * 60}{RESET}")
            for alerte in alertes:
                print(f"{ROUGE}  [DRIFT] {alerte}{RESET}")
            print(f"{ROUGE}{GRAS}{'=' * 60}{RESET}")
            print(f"\n{ROUGE}Action : déclencher le réentraînement du modèle.{RESET}")
        else:
            print(f"\n{VERT}Distributions stables — aucun drift détecté.{RESET}")

        print(f"\nRapport sauvegardé : {RAPPORT_FILE}")
