"""
dashboard.py — Tableau de bord monitoring ObRail Europe
Lancement : streamlit run monitoring/dashboard.py  (depuis la racine du projet)
            streamlit run dashboard.py              (depuis monitoring/)
Accessible sur : http://localhost:8501

Lit ../logs/api.log en temps réel et affiche métriques, graphiques et alertes.
Fonctionne même si le fichier de logs est absent ou vide.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from collections import defaultdict
from datetime import datetime

# Racine du projet — fonctionne quelle que soit la commande d'appel
_RACINE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(_RACINE, "logs", "api.log")

# Seuils d'alerte (identiques à monitoring.py)
SEUIL_TAUX_ERREUR = 0.05
SEUIL_DUREE_MS    = 2000
SEUIL_SUB_MAX     = 0.90
SEUIL_SUB_MIN     = 0.10

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ObRail Europe — Dashboard Monitoring",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Lecture et parsing des logs (mis en cache 30 secondes)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def charger_logs():
    """Lit et parse ../logs/api.log. Retourne (requetes, erreurs)."""
    if not os.path.exists(LOG_FILE):
        return [], []

    requetes = []
    erreurs  = []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue

            ts_m = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', ligne)
            timestamp = ts_m.group(1) if ts_m else None

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
                    "message":   ligne,
                })

    return requetes, erreurs


# ---------------------------------------------------------------------------
# En-tête du dashboard
# ---------------------------------------------------------------------------

st.title("ObRail Europe — Dashboard Monitoring")
st.caption(
    f"Source : `{LOG_FILE}` | "
    f"Cache rafraîchi toutes les 30 s | "
    f"Dernière consultation : {datetime.now().strftime('%H:%M:%S')}"
)

# Bouton de rafraîchissement manuel
if st.button("Actualiser les données"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

requetes, erreurs = charger_logs()

total_req   = len(requetes) + len(erreurs)
total_err   = len(erreurs)
taux_erreur = total_err / total_req if total_req > 0 else 0.0
duree_moy   = (
    sum(r["duree_ms"] for r in requetes) / len(requetes)
    if requetes else 0.0
)

resultats_sub = [
    r["resultat"]
    for r in requetes
    if r["route"] in ("/predict/substitution", "/predict") and r["resultat"] is not None
]
taux_sub = sum(resultats_sub) / len(resultats_sub) if resultats_sub else None

# ---------------------------------------------------------------------------
# Détection automatique des alertes
# ---------------------------------------------------------------------------

alertes = []
if taux_erreur > SEUIL_TAUX_ERREUR:
    alertes.append(f"Taux d'erreur élevé : {taux_erreur:.1%} > seuil {SEUIL_TAUX_ERREUR:.0%}")
if duree_moy > SEUIL_DUREE_MS:
    alertes.append(f"Latence excessive : {duree_moy:.0f} ms > seuil {SEUIL_DUREE_MS} ms")
if taux_sub is not None:
    if taux_sub > SEUIL_SUB_MAX:
        alertes.append(
            f"Anomalie distribution : {taux_sub:.1%} de substitutions "
            f"> seuil max {SEUIL_SUB_MAX:.0%} — drift possible"
        )
    elif taux_sub < SEUIL_SUB_MIN:
        alertes.append(
            f"Anomalie distribution : {taux_sub:.1%} de substitutions "
            f"< seuil min {SEUIL_SUB_MIN:.0%} — drift possible"
        )

# Affichage des alertes en rouge en haut du dashboard
if alertes:
    st.subheader("Alertes")
    for alerte in alertes:
        st.error(f"[INCIDENT] {alerte}")
    st.divider()

# ---------------------------------------------------------------------------
# Section 1 : Métriques clés
# ---------------------------------------------------------------------------

st.subheader("Métriques clés")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total requêtes",  total_req)
col2.metric("Erreurs",         total_err)
col3.metric("Taux d'erreur",   f"{taux_erreur:.1%}")
col4.metric("Durée moy. (ms)", f"{duree_moy:.1f}")

st.divider()

# ---------------------------------------------------------------------------
# Section 2 : Graphiques — requêtes par route et distribution substitution
# ---------------------------------------------------------------------------

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("Requêtes par route")
    if total_req > 0:
        comptage = defaultdict(int)
        for r in requetes:
            comptage[r["route"]] += 1
        for e in erreurs:
            comptage[e["route"]] += 1
        df_routes = pd.DataFrame.from_dict(
            comptage, orient="index", columns=["Requêtes"]
        ).sort_values("Requêtes", ascending=False)
        st.bar_chart(df_routes)
    else:
        st.info("Aucune requête enregistrée.")

with col_g2:
    st.subheader("Distribution substitution avion/train")
    if resultats_sub:
        n_sub = sum(resultats_sub)
        n_non = len(resultats_sub) - n_sub
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [n_sub, n_non],
            labels=[f"Substituable ({n_sub})", f"Non substituable ({n_non})"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Classification substitution")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Aucune prédiction de substitution disponible.")

st.divider()

# ---------------------------------------------------------------------------
# Section 3 : Timeline des requêtes dans le temps
# ---------------------------------------------------------------------------

st.subheader("Timeline des requêtes (par minute)")
if requetes:
    df_time = pd.DataFrame(requetes)
    df_time["timestamp"] = pd.to_datetime(df_time["timestamp"], errors="coerce")
    df_time = df_time.dropna(subset=["timestamp"])
    if not df_time.empty:
        df_time["minute"] = df_time["timestamp"].dt.floor("min")
        timeline = df_time.groupby("minute").size().reset_index(name="requêtes")
        timeline = timeline.set_index("minute")
        st.line_chart(timeline)
    else:
        st.info("Impossible de parser les timestamps.")
else:
    st.info("Aucune donnée temporelle disponible.")

st.divider()

# ---------------------------------------------------------------------------
# Section 4 : 10 dernières erreurs
# ---------------------------------------------------------------------------

st.subheader("10 dernières erreurs")
if erreurs:
    df_erreurs = pd.DataFrame(erreurs[-10:][::-1])[
        ["timestamp", "route", "status", "message"]
    ]
    df_erreurs.columns = ["Timestamp", "Route", "Code HTTP", "Message"]
    st.dataframe(df_erreurs, use_container_width=True)
else:
    st.success("Aucune erreur enregistrée — système nominal.")

st.divider()

# ---------------------------------------------------------------------------
# Section 5 : Logs bruts — api.log
# ---------------------------------------------------------------------------

st.subheader("Logs en temps réel — api.log")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        toutes_lignes = f.readlines()

    if toutes_lignes:
        vingt_dernieres = "".join(toutes_lignes[-20:])
        st.code(vingt_dernieres, language="log")

        if st.button("Voir tous les logs"):
            st.code("".join(toutes_lignes), language="log")
    else:
        st.info("Le fichier api.log est vide.")
else:
    st.warning(f"Fichier introuvable : `{LOG_FILE}`")

st.divider()
st.caption("ObRail Europe — MSPR TPRE622 EPSI 2025-2026 | Monitoring MLOps")
