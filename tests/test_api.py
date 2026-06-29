"""
test_api.py — Tests pytest pour l'API ObRail Europe
Couvre les routes /health, /predict/substitution et /predict/co2.
"""

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health():
    """Vérifie que GET /health retourne 200 et confirme que les 2 modèles sont actifs."""
    reponse = client.get("/health")
    assert reponse.status_code == 200
    data = reponse.json()
    assert data["modeles"]["classification_substitution"] == "ok"
    assert data["modeles"]["regression_co2"] == "ok"


def test_predict_substitution_substituable():
    """Vérifie que la liaison Paris→Marseille est classifiée comme substituable à l'avion (attendu : 1)."""
    payload = {
        "distance_km": 800,
        "duration_minutes": 195,
        "n_stops": 3,
        "co2_estime": 450000,
        "consommation_totale": 16000,
        "type_train": "electric",
        "country": "FR",
    }
    reponse = client.post("/predict/substitution", json=payload)
    assert reponse.status_code == 200
    assert reponse.json()["substitution_avion"] == 1


def test_predict_substitution_non_substituable():
    """Vérifie qu'une liaison longue distance (5800 km) est classifiée non substituable (attendu : 0)."""
    payload = {
        "distance_km": 5800,
        "duration_minutes": 800,
        "n_stops": 1,
        "co2_estime": 900000,
        "consommation_totale": 90000,
        "type_train": "electric",
        "country": "FR",
    }
    reponse = client.post("/predict/substitution", json=payload)
    assert reponse.status_code == 200
    assert reponse.json()["substitution_avion"] == 0


def test_predict_co2_reference():
    """Vérifie que /predict/co2 en scénario 'reference' retourne un float positif."""
    payload = {
        "distance_km": 400,
        "duration_minutes": 120,
        "n_stops": 2,
        "consommation_energy": 10.0,
        "gco2_per_kwh": 21.7,
        "consommation_totale": 4000,
        "type_train": "diesel",
        "scenario": "reference",
    }
    reponse = client.post("/predict/co2", json=payload)
    assert reponse.status_code == 200
    data = reponse.json()
    assert isinstance(data["co2_estime_kg"], float)
    assert data["co2_estime_kg"] > 0


def test_predict_co2_electrification():
    """Vérifie que le scénario 'diesel_50_electrique' produit moins d'émissions que le scénario 'reference'."""
    payload_base = {
        "distance_km": 400,
        "duration_minutes": 120,
        "n_stops": 2,
        "consommation_energy": 10.0,
        "gco2_per_kwh": 21.7,
        "consommation_totale": 4000,
        "type_train": "diesel",
    }
    reponse_ref = client.post("/predict/co2", json={**payload_base, "scenario": "reference"})
    reponse_elec = client.post("/predict/co2", json={**payload_base, "scenario": "diesel_50_electrique"})

    assert reponse_ref.status_code == 200
    assert reponse_elec.status_code == 200

    co2_reference = reponse_ref.json()["co2_estime_kg"]
    co2_electrification = reponse_elec.json()["co2_estime_kg"]
    assert co2_electrification < co2_reference
