"""
monitoring.py — Analyse des logs et détection automatique d'incidents ObRail Europe
Usage : python monitoring/monitoring.py  (depuis la racine du projet)
        python monitoring.py             (depuis monitoring/)

Lit ../logs/api.log, calcule les métriques, détecte les incidents et génère
un rapport dans ../logs/monitoring_report.txt.
Affiche les alertes en rouge dans le terminal si un incident est détecté.
"""

import re
import os
from collections import defaultdict
from datetime import datetime

# Racine du projet — fonctionne quelle que soit la commande d'appel
_RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE     = os.path.join(_RACINE, "logs", "api.log")
RAPPORT_FILE = os.path.join(_RACINE, "logs", "monitoring_report.txt")

# Codes couleur ANSI pour le terminal
ROUGE = "\033[91m"
VERT  = "\033[92m"
JAUNE = "\033[93m"
RESET = "\033[0m"
GRAS  = "\033[1m"

# Seuils de détection d'incidents
SEUIL_TAUX_ERREUR = 0.05   # 5 % d'erreurs → alerte
SEUIL_DUREE_MS    = 2000   # 2 secondes de latence moyenne → alerte
SEUIL_SUB_MAX     = 0.90   # > 90 % de substitutions → anomalie
SEUIL_SUB_MIN     = 0.10   # < 10 % de substitutions → anomalie


# ---------------------------------------------------------------------------
# Lecture et parsing
# ---------------------------------------------------------------------------

def lire_logs():
    """Lit le fichier de logs et retourne les lignes brutes."""
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return f.readlines()


def parser_logs(lignes):
    """
    Parse les lignes de log et retourne deux listes :
      - requetes : lignes INFO REQUETE (succès)
      - erreurs  : lignes ERROR ERREUR (incidents)
    """
    requetes = []
    erreurs  = []

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue

        ts_m = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', ligne)
        timestamp = ts_m.group(1) if ts_m else "inconnu"

        if " - INFO - REQUETE " in ligne:
            route_m    = re.search(r'route=(\S+)', ligne)
            duree_m    = re.search(r'duree_ms=([\d.]+)', ligne)
            resultat_m = re.search(r'resultat=(\d)', ligne)
            co2_m      = re.search(r'co2_estime_kg=([\d.]+)', ligne)

            if route_m and duree_m:
                requetes.append({
                    "timestamp":     timestamp,
                    "route":         route_m.group(1),
                    "duree_ms":      float(duree_m.group(1)),
                    "resultat":      int(resultat_m.group(1)) if resultat_m else None,
                    "co2_estime_kg": float(co2_m.group(1)) if co2_m else None,
                })

        elif " - ERROR - ERREUR " in ligne:
            route_m  = re.search(r'route=(\S+)', ligne)
            status_m = re.search(r'status=(\d+)', ligne)
            erreurs.append({
                "timestamp": timestamp,
                "route":     route_m.group(1) if route_m else "inconnu",
                "status":    int(status_m.group(1)) if status_m else 0,
                "ligne":     ligne,
            })

    return requetes, erreurs


# ---------------------------------------------------------------------------
# Calcul des métriques
# ---------------------------------------------------------------------------

def calculer_metriques(requetes, erreurs):
    """Agrège les données parsées en métriques exploitables."""
    total_req = len(requetes) + len(erreurs)
    total_err = len(erreurs)
    taux_erreur = total_err / total_req if total_req > 0 else 0.0

    # Nombre de requêtes par route (succès + erreurs)
    requetes_par_route = defaultdict(int)
    for r in requetes:
        requetes_par_route[r["route"]] += 1
    for e in erreurs:
        requetes_par_route[e["route"]] += 1

    # Durée moyenne par route (sur les requêtes réussies)
    durees_par_route = defaultdict(list)
    for r in requetes:
        durees_par_route[r["route"]].append(r["duree_ms"])
    duree_moy_par_route = {
        route: sum(d) / len(d)
        for route, d in durees_par_route.items()
    }

    # Distribution des prédictions de substitution
    resultats_sub = [
        r["resultat"]
        for r in requetes
        if r["route"] in ("/predict/substitution", "/predict") and r["resultat"] is not None
    ]
    n_substituable = sum(resultats_sub)
    taux_substituable = n_substituable / len(resultats_sub) if resultats_sub else None

    # CO2 moyen prédit
    co2_values = [
        r["co2_estime_kg"]
        for r in requetes
        if r["route"] == "/predict/co2" and r["co2_estime_kg"] is not None
    ]
    co2_moyen = sum(co2_values) / len(co2_values) if co2_values else None

    return {
        "total_req":           total_req,
        "total_err":           total_err,
        "taux_erreur":         taux_erreur,
        "requetes_par_route":  dict(requetes_par_route),
        "duree_moy_par_route": duree_moy_par_route,
        "resultats_sub":       resultats_sub,
        "n_substituable":      n_substituable,
        "taux_substituable":   taux_substituable,
        "co2_moyen":           co2_moyen,
    }


# ---------------------------------------------------------------------------
# Détection automatique d'incidents
# ---------------------------------------------------------------------------

def detecter_incidents(metriques):
    """
    Détecte automatiquement les incidents selon les seuils définis.
    Retourne la liste des alertes (vide si système nominal).
    """
    alertes = []

    if metriques["taux_erreur"] > SEUIL_TAUX_ERREUR:
        alertes.append(
            f"TAUX D'ERREUR ÉLEVÉ : {metriques['taux_erreur']:.1%} "
            f"> seuil {SEUIL_TAUX_ERREUR:.0%}"
        )

    for route, duree in metriques["duree_moy_par_route"].items():
        if duree > SEUIL_DUREE_MS:
            alertes.append(
                f"LATENCE EXCESSIVE sur {route} : {duree:.0f} ms "
                f"> seuil {SEUIL_DUREE_MS} ms"
            )

    taux_sub = metriques["taux_substituable"]
    if taux_sub is not None:
        if taux_sub > SEUIL_SUB_MAX:
            alertes.append(
                f"ANOMALIE DISTRIBUTION : {taux_sub:.1%} de substitutions "
                f"> seuil max {SEUIL_SUB_MAX:.0%} — possible drift du modèle"
            )
        elif taux_sub < SEUIL_SUB_MIN:
            alertes.append(
                f"ANOMALIE DISTRIBUTION : {taux_sub:.1%} de substitutions "
                f"< seuil min {SEUIL_SUB_MIN:.0%} — possible drift du modèle"
            )

    return alertes


# ---------------------------------------------------------------------------
# Génération du rapport
# ---------------------------------------------------------------------------

def generer_rapport(metriques, alertes, erreurs):
    """Génère le rapport texte et l'écrit dans RAPPORT_FILE."""
    lignes = [
        "=" * 60,
        "RAPPORT DE MONITORING — ObRail Europe",
        f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "--- MÉTRIQUES GÉNÉRALES ---",
        f"  Total requêtes traitées : {metriques['total_req']}",
        f"  Total erreurs            : {metriques['total_err']}",
        f"  Taux d'erreur            : {metriques['taux_erreur']:.1%}",
        "",
        "--- REQUÊTES PAR ROUTE ---",
    ]

    for route, count in sorted(metriques["requetes_par_route"].items()):
        lignes.append(f"  {route:<40} {count} requête(s)")

    lignes += ["", "--- DURÉE MOYENNE PAR ROUTE (ms) ---"]
    for route, duree in sorted(metriques["duree_moy_par_route"].items()):
        lignes.append(f"  {route:<40} {duree:.1f} ms")

    lignes += ["", "--- PRÉDICTIONS /predict/substitution ---"]
    if metriques["resultats_sub"]:
        n_total = len(metriques["resultats_sub"])
        n_sub   = metriques["n_substituable"]
        lignes += [
            f"  Prédictions totales    : {n_total}",
            f"  Substituable   (= 1)   : {n_sub}",
            f"  Non substituable (= 0) : {n_total - n_sub}",
            f"  Taux substituable      : {metriques['taux_substituable']:.1%}",
        ]
    else:
        lignes.append("  Aucune prédiction de substitution enregistrée.")

    lignes += ["", "--- PRÉDICTIONS /predict/co2 ---"]
    if metriques["co2_moyen"] is not None:
        lignes.append(f"  CO2 moyen prédit : {metriques['co2_moyen']:.2f} kgCO2")
    else:
        lignes.append("  Aucune prédiction CO2 enregistrée.")

    lignes += ["", "--- ALERTES INCIDENTS ---"]
    if alertes:
        for alerte in alertes:
            lignes.append(f"  [ALERTE] {alerte}")
    else:
        lignes.append("  Aucune alerte — système nominal.")

    if erreurs:
        lignes += ["", "--- DERNIÈRES ERREURS (10 max) ---"]
        for e in erreurs[-10:]:
            lignes.append(f"  [{e['timestamp']}] route={e['route']} status={e['status']}")

    rapport = "\n".join(lignes) + "\n"

    os.makedirs(os.path.dirname(RAPPORT_FILE), exist_ok=True)
    with open(RAPPORT_FILE, "w", encoding="utf-8") as f:
        f.write(rapport)

    return rapport


# ---------------------------------------------------------------------------
# Affichage terminal
# ---------------------------------------------------------------------------

def afficher_rapport(rapport, alertes):
    """Affiche le rapport et les alertes colorées dans le terminal."""
    print(rapport)

    if alertes:
        print(f"\n{ROUGE}{GRAS}{'=' * 60}{RESET}")
        print(f"{ROUGE}{GRAS}  INCIDENTS DÉTECTÉS AUTOMATIQUEMENT :{RESET}")
        print(f"{ROUGE}{GRAS}{'=' * 60}{RESET}")
        for alerte in alertes:
            print(f"{ROUGE}  [ALERTE] {alerte}{RESET}")
        print(f"{ROUGE}{GRAS}{'=' * 60}{RESET}")
    else:
        print(f"\n{VERT}  Système nominal — aucun incident détecté.{RESET}")

    print(f"\nRapport sauvegardé : {RAPPORT_FILE}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{GRAS}Analyse des logs ObRail Europe...{RESET}\n")

    lignes = lire_logs()
    if not lignes:
        print(f"{JAUNE}Aucune donnée : {LOG_FILE} absent ou vide.{RESET}")
    else:
        requetes, erreurs = parser_logs(lignes)
        metriques = calculer_metriques(requetes, erreurs)
        alertes = detecter_incidents(metriques)
        rapport = generer_rapport(metriques, alertes, erreurs)
        afficher_rapport(rapport, alertes)
