"""
conftest.py — Configuration pytest pour ObRail Europe
Vérifie la disponibilité des modèles joblib avant chaque test.
"""

import os
import sys
import pytest

# Racine du projet (répertoire parent de tests/)
RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Assure que api.py est importable depuis les fichiers de test
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

# Modèles requis pour que l'API fonctionne correctement
MODELES_REQUIS = [
    os.path.join(RACINE, "models", "classification_substitution_avion.joblib"),
    os.path.join(RACINE, "models", "encoders.joblib"),
    os.path.join(RACINE, "models", "regression_co2.joblib"),
]


@pytest.fixture(autouse=True)
def verifier_modeles():
    """Vérifie la présence des fichiers modèles avant chaque test ; skip si absents."""
    manquants = [os.path.basename(m) for m in MODELES_REQUIS if not os.path.exists(m)]
    if manquants:
        pytest.skip(f"Modèles non disponibles : {', '.join(manquants)}")
