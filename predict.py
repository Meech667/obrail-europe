"""
predict.py — Pipeline de prédiction minimal pour la substitution avion/train
Usage : python predict.py
"""
import joblib
import pandas as pd


def predict_substitution(distance_km, duration_minutes, n_stops,
                          co2_estime, consommation_totale,
                          type_train, country):
    """
    Prédit si une liaison ferroviaire est candidate à la substitution avion.
    Retourne 1 (substituable) ou 0 (non substituable).
    """
    model = joblib.load('models/classification_substitution_avion.joblib')
    encoders = joblib.load('models/encoders.joblib')

    # Encodage des variables catégorielles
    type_train_enc = encoders['le_type_train'].transform([type_train])[0]
    country_enc = encoders['le_country'].transform([country])[0]

    # Construction du vecteur de features dans le bon ordre
    X = pd.DataFrame([{
        'distance_km': distance_km,
        'duration_minutes': duration_minutes,
        'n_stops': n_stops,
        'co2_estime': co2_estime,
        'consommation_totale': consommation_totale,
        'type_train': type_train_enc,
        'country': country_enc
    }])

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    label = "Substituable à l'avion" if prediction == 1 else "Non substituable"

    print(f"Résultat : {label}")
    print(f"Probabilité de substitution : {proba:.2%}")
    return prediction


if __name__ == '__main__':
    # Exemple : Paris → Madrid, 1200 km, 9h45 (585 min), 3 arrêts
    predict_substitution(
        distance_km=1200,
        duration_minutes=585,
        n_stops=3,
        co2_estime=521280,
        consommation_totale=24000,
        type_train='electric',
        country='FR'
    )
