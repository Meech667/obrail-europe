"""
predict_co2.py - Pipeline de prediction minimal pour l'estimation CO2 futur
Usage : python predict_co2.py
"""
import joblib
import pandas as pd
import numpy as np


def predict_co2(distance_km, duration_minutes, n_stops,
                consommation_energy, gco2_per_kwh, consommation_totale,
                type_train, scenario):
    """
    Predit les emissions CO2 futures d'une liaison ferroviaire selon un scenario.
    Scenarios disponibles : 'reference', 'diesel_50_electrique',
                            'conso_moins_15', 'distance_moins_10'
    Retourne la valeur estimee en kgCO2.
    """
    model = joblib.load('models/regression_co2.joblib')

    # Features derivees
    vitesse_moyenne_kmh = distance_km / (duration_minutes / 60) if duration_minutes > 0 else 0
    co2_par_km_base = (consommation_energy * gco2_per_kwh) / distance_km if distance_km > 0 else 0
    is_diesel   = 1 if type_train == 'diesel' else 0
    is_electric = 1 if type_train == 'electric' else 0

    # Adaptation selon le scenario
    distance_scenario_km  = distance_km
    consommation_scenario = consommation_totale

    if scenario == 'diesel_50_electrique' and type_train == 'diesel':
        consommation_scenario *= 0.50
    elif scenario == 'conso_moins_15':
        consommation_scenario *= 0.85
    elif scenario == 'distance_moins_10':
        distance_scenario_km  *= 0.90
        consommation_scenario *= 0.90

    X = pd.DataFrame([{
        'distance_scenario_km':  distance_scenario_km,
        'duration_minutes':      duration_minutes,
        'n_stops':               n_stops,
        'consommation_energy':   consommation_energy,
        'gco2_per_kwh':          gco2_per_kwh,
        'consommation_scenario': consommation_scenario,
        'vitesse_moyenne_kmh':   vitesse_moyenne_kmh,
        'co2_par_km':            co2_par_km_base,
        'is_diesel':             is_diesel,
        'is_electric':           is_electric,
        'type_train':            type_train,
        'scenario':              scenario
    }])

    co2_predit = model.predict(X)[0]
    print(f"Scenario     : {scenario}")
    print(f"CO2 estime   : {co2_predit:.4f} kgCO2")
    return co2_predit


if __name__ == '__main__':
    # Exemple : Paris -> Lyon, train diesel, scenario diesel_50_electrique
    predict_co2(
        distance_km=400,
        duration_minutes=120,
        n_stops=2,
        consommation_energy=10.0,
        gco2_per_kwh=21.7,
        consommation_totale=4000.0,
        type_train='diesel',
        scenario='diesel_50_electrique'
    )
