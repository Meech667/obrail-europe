# Journal des Incidents — ObRail Europe

Suivi des incidents rencontrés pendant le développement et la mise en production.
Chaque incident documente : symptôme, cause racine, solution et mesure préventive.

---

## Incident 1 — `encoders.joblib` manquant au démarrage de l'API

**Date :** Pendant le développement (phase d'intégration API)
**Sévérité :** Critique — API non fonctionnelle au démarrage
**Route concernée :** POST /predict/substitution, POST /predict

### Symptôme

L'API plante au démarrage avec l'erreur suivante dans les logs :

```
ERROR - Échec chargement modèle classification_substitution_avion :
[Errno 2] No such file or directory: 'models/encoders.joblib'
```

Toutes les routes de classification retournent ensuite HTTP 503.

### Cause racine

Le modèle de classification `classification_substitution_avion.joblib`
utilise des `LabelEncoder` scikit-learn (pour les colonnes `type_train` et `country`)
sauvegardés dans un fichier séparé `encoders.joblib`.

Ce fichier n'était pas présent dans le répertoire `models/` lors du premier déploiement :
seul le fichier du modèle avait été copié, sans ses encodeurs.

### Solution appliquée

Ajout du chargement conditionnel avec `try/except` dans `api.py` :

```python
try:
    _model_substitution = joblib.load("models/classification_substitution_avion.joblib")
    _encoders = joblib.load("models/encoders.joblib")
    _substitution_ok = True
    logger.info("Modèle classification_substitution_avion chargé avec succès")
except Exception as e:
    _substitution_ok = False
    _substitution_error = str(e)
    logger.error(f"Échec chargement modèle classification_substitution_avion : {e}")
```

L'API démarre désormais même sans les modèles. La route `/health` indique
l'état exact de chaque modèle, et les routes de prédiction retournent
un message d'erreur explicite (HTTP 503) si le modèle est absent.

### Mesure préventive

La route `/health` vérifie l'état de chaque modèle en temps réel :

```json
{
  "status": "ok",
  "modeles": {
    "classification_substitution": "ok",
    "regression_co2": "ok"
  }
}
```

En production, ajouter une alerte Prometheus si
`classification_substitution != "ok"` persiste plus de 60 secondes après le démarrage.

---

## Incident 2 — Cross-validation absente du notebook régression

**Date :** Pendant la préparation de la soutenance
**Sévérité :** Moyen — résultat non démontrable lors de la présentation
**Fichier concerné :** `01_regression_co2.ipynb`

### Symptôme

Le notebook `01_regression_co2.ipynb` ne contenait pas de cellule de cross-validation
5 folds. Le cahier des charges exige une validation croisée pour démontrer
la stabilité des modèles sur des jeux de données non vus.

### Cause racine

Lors du développement assisté, la cellule `cross_val_score` avait été
générée uniquement dans le notebook de classification
(`02_classification_substitution_avion.ipynb`) et non reportée dans
le notebook de régression.

Le notebook de régression contenait GridSearchCV (optimisation des hyperparamètres)
mais pas la cross-validation finale qui mesure la variance du modèle.

### Solution appliquée

Ajout manuel d'une cellule de cross-validation dans `01_regression_co2.ipynb` :

```python
from sklearn.model_selection import cross_val_score

# Cross-validation 5 folds sur le meilleur modèle (XGBoost)
cv_scores = cross_val_score(
    best_model, X_train, y_train,
    cv=5, scoring='r2', n_jobs=-1
)
print(f"R² moyen (CV 5-folds) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

Résultat obtenu : **R² = 1.000 ± 0.000**, confirmant la stabilité parfaite
du modèle XGBoost sur ce jeu de données.

### Mesure préventive

Checklist de vérification systématique des deux notebooks avant chaque soutenance :

**`01_regression_co2.ipynb`**
- [ ] Section EDA avec visualisations (distribution CO2, corrélations)
- [ ] 3 modèles testés : Régression Linéaire, Random Forest, XGBoost
- [ ] GridSearchCV pour chaque modèle
- [ ] Cross-validation 5 folds (`cross_val_score`)
- [ ] Évaluation finale sur le jeu de test (R², MAE, RMSE)
- [ ] Sauvegarde du modèle avec `joblib.dump`

**`02_classification_substitution_avion.ipynb`**
- [ ] Section EDA avec visualisations (distribution classes, déséquilibre)
- [ ] 3 modèles testés : Logistic Regression, Random Forest, XGBoost
- [ ] GridSearchCV pour chaque modèle
- [ ] Cross-validation 5 folds (`cross_val_score`)
- [ ] Matrice de confusion et courbe ROC
- [ ] Sauvegarde du modèle et des encodeurs avec `joblib.dump`
