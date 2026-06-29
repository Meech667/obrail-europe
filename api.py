"""
api.py — API commune ObRail Europe
Démarrage : uvicorn api:app --reload
Documentation : http://localhost:8000/docs

Routes exposées :
    GET  /               → message d'accueil
    GET  /health         → statut de l'API et des modèles chargés
    POST /predict                  → classification substitution (compat. v1)
    POST /predict/substitution     → classification substitution avion/train
    POST /predict/co2              → régression émissions CO2 futures

Monitoring en production (à brancher sur Prometheus / Evidently) :
    - Taux de requêtes par minute par route
    - Distribution des prédictions dans le temps
      (proportion de 1 sur /predict/substitution, moyenne kgCO2 sur /predict/co2)
    - Data drift : comparer les distributions des features en entrée
      vs. les distributions du jeu d'entraînement à l'aide d'Evidently ou WhyLogs
    - Latence P50 / P95 / P99 par route
    - Taux d'erreurs HTTP 422 (payload invalide) et 500 (erreur modèle)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import time

# ---------------------------------------------------------------------------
# Journalisation
# RGPD : seules les features numériques (distance, durée, arrêts, CO2) sont
# enregistrées dans les logs. Aucune donnée personnelle identifiable (IP,
# nom, email, identifiant) n'est collectée ni stockée.
# ---------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("obrail")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _fh = logging.FileHandler("logs/api.log", encoding="utf-8")
    _fh.setFormatter(_formatter)
    logger.addHandler(_fh)
    _ch = logging.StreamHandler()
    _ch.setFormatter(_formatter)
    logger.addHandler(_ch)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ObRail Europe — API de prédiction ferroviaire",
    description=(
        "API REST du projet ObRail Europe. "
        "Deux enjeux exposés :\n\n"
        "- **Substitution avion/train** : classification binaire — la liaison "
        "est-elle candidate à remplacer un vol aérien ?\n"
        "- **Émissions CO2 futures** : régression — estimation des émissions "
        "kgCO2 selon un scénario de développement du réseau."
    ),
    version="2.0.0",
    contact={"name": "ObRail Europe", "email": "data@obrail.eu"},
)

# ---------------------------------------------------------------------------
# Chargement des modèles au démarrage (une seule fois)
# ---------------------------------------------------------------------------

try:
    _model_substitution = joblib.load("models/classification_substitution_avion.joblib")
    _encoders = joblib.load("models/encoders.joblib")
    _substitution_ok = True
    logger.info("Modèle classification_substitution_avion chargé avec succès")
except Exception as e:
    _substitution_ok = False
    _substitution_error = str(e)
    logger.error(f"Échec chargement modèle classification_substitution_avion : {e}")

try:
    # Pipeline complet sklearn (ColumnTransformer + XGBoost) — pas d'encodeur séparé
    _model_co2 = joblib.load("models/regression_co2.joblib")
    _co2_ok = True
    logger.info("Modèle regression_co2 chargé avec succès")
except Exception as e:
    _co2_ok = False
    _co2_error = str(e)
    logger.error(f"Échec chargement modèle regression_co2 : {e}")

# ---------------------------------------------------------------------------
# Schémas Pydantic — Classification substitution
# ---------------------------------------------------------------------------

class LiaisonSubstitutionInput(BaseModel):
    """Caractéristiques d'une liaison ferroviaire pour la classification substitution."""

    distance_km: float = Field(
        ..., example=1200.0,
        description="Distance géographique de la liaison en km"
    )
    duration_minutes: float = Field(
        ..., example=360.0,
        description="Durée du trajet en minutes"
    )
    n_stops: int = Field(
        ..., example=2,
        description="Nombre d'arrêts intermédiaires"
    )
    co2_estime: float = Field(
        ..., example=450000.0,
        description="Émissions CO2 estimées en gCO2"
    )
    consommation_totale: float = Field(
        ..., example=20000.0,
        description="Consommation énergétique totale en kWh"
    )
    type_train: str = Field(
        ..., example="electric",
        description="Type de traction : 'electric' ou 'diesel'"
    )
    country: str = Field(
        ..., example="FR",
        description="Code pays : 'FR', 'ES', 'IT' ou 'DE'"
    )


class SubstitutionOutput(BaseModel):
    """Résultat de la classification substitution avion/train."""

    substitution_avion: int = Field(description="1 = substituable, 0 = non substituable")
    probabilite: float = Field(description="Probabilité d'être substituable (entre 0 et 1)")
    label: str = Field(description="Libellé lisible de la prédiction")


# ---------------------------------------------------------------------------
# Schémas Pydantic — Régression CO2
# ---------------------------------------------------------------------------

class LiaisonCO2Input(BaseModel):
    """Caractéristiques d'une liaison ferroviaire et scénario pour l'estimation CO2."""

    distance_km: float = Field(
        ..., example=400.0,
        description="Distance géographique de la liaison en km"
    )
    duration_minutes: float = Field(
        ..., example=120.0,
        description="Durée du trajet en minutes"
    )
    n_stops: int = Field(
        ..., example=2,
        description="Nombre d'arrêts intermédiaires"
    )
    consommation_energy: float = Field(
        ..., example=10.0,
        description="Consommation énergétique en kWh/km"
    )
    gco2_per_kwh: float = Field(
        ..., example=21.7,
        description="Facteur carbone du pays en gCO2/kWh"
    )
    consommation_totale: float = Field(
        ..., example=4000.0,
        description="Consommation totale actuelle en kWh"
    )
    type_train: str = Field(
        ..., example="diesel",
        description="Type de traction : 'electric' ou 'diesel'"
    )
    scenario: str = Field(
        ..., example="diesel_50_electrique",
        description=(
            "Scénario de développement du réseau. "
            "Valeurs : 'reference', 'diesel_50_electrique', "
            "'conso_moins_15', 'distance_moins_10'"
        )
    )


class CO2Output(BaseModel):
    """Résultat de l'estimation des émissions CO2 futures."""

    scenario: str = Field(description="Scénario appliqué")
    co2_estime_kg: float = Field(description="Émissions CO2 estimées en kgCO2")
    label: str = Field(description="Libellé lisible du résultat")


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------

def _build_substitution_df(liaison: LiaisonSubstitutionInput) -> pd.DataFrame:
    """Encode et construit le DataFrame d'entrée pour le modèle de classification."""
    type_train_enc = _encoders["le_type_train"].transform([liaison.type_train])[0]
    country_enc = _encoders["le_country"].transform([liaison.country])[0]
    return pd.DataFrame([{
        "distance_km":        liaison.distance_km,
        "duration_minutes":   liaison.duration_minutes,
        "n_stops":            liaison.n_stops,
        "co2_estime":         liaison.co2_estime,
        "consommation_totale": liaison.consommation_totale,
        "type_train":         type_train_enc,
        "country":            country_enc,
    }])


def _build_co2_df(liaison: LiaisonCO2Input) -> pd.DataFrame:
    """Calcule les features dérivées et construit le DataFrame pour le Pipeline CO2."""
    # Features dérivées (identiques à predict_co2.py)
    vitesse = (liaison.distance_km / (liaison.duration_minutes / 60)
               if liaison.duration_minutes > 0 else 0.0)
    co2_par_km = ((liaison.consommation_energy * liaison.gco2_per_kwh) / liaison.distance_km
                  if liaison.distance_km > 0 else 0.0)
    is_diesel   = 1 if liaison.type_train == "diesel" else 0
    is_electric = 1 if liaison.type_train == "electric" else 0

    # Adaptation selon le scénario
    distance_scenario_km  = liaison.distance_km
    consommation_scenario = liaison.consommation_totale

    if liaison.scenario == "diesel_50_electrique" and liaison.type_train == "diesel":
        consommation_scenario *= 0.50
    elif liaison.scenario == "conso_moins_15":
        consommation_scenario *= 0.85
    elif liaison.scenario == "distance_moins_10":
        distance_scenario_km  *= 0.90
        consommation_scenario *= 0.90
    # "reference" : aucune modification

    return pd.DataFrame([{
        "distance_scenario_km":  distance_scenario_km,
        "duration_minutes":      liaison.duration_minutes,
        "n_stops":               liaison.n_stops,
        "consommation_energy":   liaison.consommation_energy,
        "gco2_per_kwh":          liaison.gco2_per_kwh,
        "consommation_scenario": consommation_scenario,
        "vitesse_moyenne_kmh":   vitesse,
        "co2_par_km":            co2_par_km,
        "is_diesel":             is_diesel,
        "is_electric":           is_electric,
        "type_train":            liaison.type_train,
        "scenario":              liaison.scenario,
    }])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Général"])
def root():
    """Message d'accueil — liste les routes disponibles."""
    return {
        "message": "ObRail Europe — API de prédiction ferroviaire",
        "version": "2.0.0",
        "routes": {
            "GET  /health":               "Statut des modèles chargés",
            "POST /predict/substitution": "Classification substitution avion/train",
            "POST /predict/co2":          "Estimation CO2 futur selon scénario",
            "GET  /docs":                 "Documentation interactive (Swagger)",
        },
    }


@app.get("/health", tags=["Général"])
def health():
    """Vérifie que l'API et les deux modèles sont opérationnels."""
    debut = time.time()
    resultat = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "modeles": {
            "classification_substitution": "ok" if _substitution_ok else f"erreur : {_substitution_error}",
            "regression_co2":              "ok" if _co2_ok          else f"erreur : {_co2_error}",
        },
    }
    duree_ms = round((time.time() - debut) * 1000, 2)
    logger.info(
        f"REQUETE route=/health "
        f"statut_clf={_substitution_ok} statut_co2={_co2_ok} "
        f"duree_ms={duree_ms}"
    )
    return resultat


@app.post("/predict", response_model=SubstitutionOutput, tags=["Classification"])
def predict_compat(liaison: LiaisonSubstitutionInput):
    """
    [Compatibilité v1] Prédit si une liaison est candidate à la substitution avion/train.
    Identique à POST /predict/substitution — conservée pour ne pas casser les intégrations existantes.
    """
    debut = time.time()
    try:
        result = _predict_substitution_logic(liaison)
        duree_ms = round((time.time() - debut) * 1000, 2)
        logger.info(
            f"REQUETE route=/predict "
            f"distance_km={liaison.distance_km} duration_minutes={liaison.duration_minutes} "
            f"n_stops={liaison.n_stops} co2_estime={liaison.co2_estime} "
            f"consommation_totale={liaison.consommation_totale} "
            f"resultat={result.substitution_avion} proba={result.probabilite} "
            f"duree_ms={duree_ms}"
        )
        return result
    except HTTPException as e:
        duree_ms = round((time.time() - debut) * 1000, 2)
        logger.error(f"ERREUR route=/predict status={e.status_code} duree_ms={duree_ms}")
        raise


@app.post("/predict/substitution", response_model=SubstitutionOutput, tags=["Classification"])
def predict_substitution(liaison: LiaisonSubstitutionInput):
    """
    Prédit si une liaison ferroviaire est candidate à remplacer un vol aérien.

    - **1** : liaison substituable (distance 300-1500 km, durée < 8h)
    - **0** : liaison non substituable
    """
    debut = time.time()
    try:
        result = _predict_substitution_logic(liaison)
        duree_ms = round((time.time() - debut) * 1000, 2)
        logger.info(
            f"REQUETE route=/predict/substitution "
            f"distance_km={liaison.distance_km} duration_minutes={liaison.duration_minutes} "
            f"n_stops={liaison.n_stops} co2_estime={liaison.co2_estime} "
            f"consommation_totale={liaison.consommation_totale} "
            f"resultat={result.substitution_avion} proba={result.probabilite} "
            f"duree_ms={duree_ms}"
        )
        return result
    except HTTPException as e:
        duree_ms = round((time.time() - debut) * 1000, 2)
        logger.error(f"ERREUR route=/predict/substitution status={e.status_code} duree_ms={duree_ms}")
        raise


def _predict_substitution_logic(liaison: LiaisonSubstitutionInput) -> SubstitutionOutput:
    """Logique commune aux routes /predict et /predict/substitution."""
    if not _substitution_ok:
        raise HTTPException(status_code=503, detail=f"Modèle de classification indisponible : {_substitution_error}")
    try:
        X = _build_substitution_df(liaison)
        prediction = int(_model_substitution.predict(X)[0])
        proba = float(_model_substitution.predict_proba(X)[0][1])
        label = "Substituable à l'avion" if prediction == 1 else "Non substituable"
        return SubstitutionOutput(
            substitution_avion=prediction,
            probabilite=round(proba, 4),
            label=label,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Valeur invalide : {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")


@app.post("/predict/co2", response_model=CO2Output, tags=["Régression CO2"])
def predict_co2(liaison: LiaisonCO2Input):
    """
    Estime les émissions CO2 futures d'une liaison ferroviaire selon un scénario.

    Scénarios disponibles :
    - **reference** : pas de changement — baseline actuel
    - **diesel_50_electrique** : trains diesel à -50% d'émissions
    - **conso_moins_15** : consommation réduite de 15% sur tous les trains
    - **distance_moins_10** : trajets raccourcis de 10%

    Retourne le CO2 estimé en **kgCO2**.
    """
    debut = time.time()

    if not _co2_ok:
        duree_ms = round((time.time() - debut) * 1000, 2)
        logger.error(f"ERREUR route=/predict/co2 status=503 detail=Modèle_CO2_indisponible duree_ms={duree_ms}")
        raise HTTPException(status_code=503, detail=f"Modèle de régression CO2 indisponible : {_co2_error}")

    scenarios_valides = {"reference", "diesel_50_electrique", "conso_moins_15", "distance_moins_10"}
    if liaison.scenario not in scenarios_valides:
        duree_ms = round((time.time() - debut) * 1000, 2)
        logger.error(
            f"ERREUR route=/predict/co2 status=422 "
            f"detail=Scénario_invalide:{liaison.scenario} duree_ms={duree_ms}"
        )
        raise HTTPException(
            status_code=422,
            detail=f"Scénario '{liaison.scenario}' inconnu. Valeurs acceptées : {sorted(scenarios_valides)}"
        )

    try:
        X = _build_co2_df(liaison)
        co2_predit = float(_model_co2.predict(X)[0])
        result = CO2Output(
            scenario=liaison.scenario,
            co2_estime_kg=round(co2_predit, 4),
            label=f"CO2 estimé : {co2_predit:.2f} kgCO2 (scénario : {liaison.scenario})",
        )
        duree_ms = round((time.time() - debut) * 1000, 2)
        logger.info(
            f"REQUETE route=/predict/co2 "
            f"distance_km={liaison.distance_km} duration_minutes={liaison.duration_minutes} "
            f"n_stops={liaison.n_stops} consommation_energy={liaison.consommation_energy} "
            f"gco2_per_kwh={liaison.gco2_per_kwh} consommation_totale={liaison.consommation_totale} "
            f"scenario={liaison.scenario} co2_estime_kg={result.co2_estime_kg} "
            f"duree_ms={duree_ms}"
        )
        return result
    except Exception as e:
        duree_ms = round((time.time() - debut) * 1000, 2)
        logger.error(
            f"ERREUR route=/predict/co2 type={type(e).__name__} "
            f"detail={e} duree_ms={duree_ms}"
        )
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction CO2 : {e}")
