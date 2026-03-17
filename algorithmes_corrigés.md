# Rapport Essentiel — Algorithmes de Prédiction Supervisée

> **Liste d'algorithmes choisis :** Logistique, Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes, KNN, Linear Regression, Ridge, Lasso, Elastic Net, SVR, MLP.  
> **Auteur :** A. LArhlimi (Généré)  
> **Date :** Mars 2026

---

## Table des matières

| # | Algorithme | Type |
|---|-----------|------|
| 1 | Régression Logistique | Classification |
| 2 | Decision Tree | Classification / Régression |
| 3 | Random Forest | Classification / Régression |
| 4 | Gradient Boosting | Classification / Régression |
| 5 | SVM (Support Vector Machine) | Classification |
| 6 | Naive Bayes | Classification |
| 7 | KNN (K-Nearest Neighbors) | Classification / Régression |
| 8 | Linear Regression (OLS) | Régression |
| 9 | Ridge Regression | Régression (Régularisation L2) |
| 10| Lasso Regression | Régression (Régularisation L1) |
| 11| Elastic Net | Régression (L1 + L2) |
| 12| SVR (Support Vector Regression)| Régression |
| 13| MLP (Multi-Layer Perceptron) | Réseau de Neurones |

---

## SECTION A — CLASSIFICATION

---

### 1. Régression Logistique

**Principe :** Modélise la probabilité d'appartenance à une classe (souvent binaire) via la fonction sigmoïde ou softmax.
**Forces :** Très interprétable (les coefficients indiquent l'importance et le sens d'une variable). Rapide.
**Faiblesses :** Ne capture que les relations linéaires.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline([
    ("sc", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
])
# Entraînement : lr.fit(X_train, y_train)
# Prédiction probabilités : lr.predict_proba(X_test)
```

---

### 2. Decision Tree (Arbre de Décision)

**Principe :** Sépare l'espace des données par des règles de type "Si/Sinon" successives.
**Forces :** Très interprétable, facile à visualiser, ne nécessite pas de normalisation.
**Faiblesses :** Tendance très forte au surapprentissage (Overfitting) si `max_depth` n'est pas limité.

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=4,         # Limite fondamentale pour éviter le surapprentissage
    min_samples_split=5, 
    random_state=42
)
# Importance des features : dt.feature_importances_
```

---

### 3. Random Forest (Forêt Aléatoire)

**Principe :** Crée une "forêt" de multiples arbres de décision entraînés sur des sous-échantillons aléatoires (bootstraps). La prédiction finale est un vote de la majorité.
**Forces :** Très performant "out of the box". Ne surapprend pas (très peu de variance). Gère les relations non-linéaires.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,      # Nombre d'arbres
    max_features="sqrt",   # Nb de features à évaluer à chaque séparation
    random_state=42, 
    n_jobs=-1              # Utilise tous les coeurs du CPU
)
```

---

### 4. Gradient Boosting (GBM / XGBoost / LightGBM)

**Principe :** Crée des arbres de décision séquentiellement, où chaque nouvel arbre essaie de corriger les erreurs (résidus) de l'arbre précédent.
**Forces :** Souvent le meilleur algorithme possible sur des données tabulaires (Kaggle).
**Faiblesses :** Plus lent à entraîner que Random Forest. Sensible aux hyperparamètres.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=200,   
    learning_rate=0.1,  # Vitesse d'apprentissage (réduire la taille des corrections)
    max_depth=3,        # Arbres très petits
    random_state=42
)
```

---

### 5. SVM (Machine à Vecteurs de Support)

**Principe :** Cherche l'hyperplan qui sépare les classes avec la marge maximale. Utilise l'"Astuce du Noyau" (Kernel Trick) pour gérer les données non-linéaires.
**Forces :** Excellent en haute dimension ou quand nombre de dimensions > nombre d'échantillons.
**Faiblesses :** Lent sur les très grands datasets (>100k lignes). Normalisation indispensable.

```python
from sklearn.svm import SVC

svm = Pipeline([
    ("sc", StandardScaler()),  # OBLIGATOIRE
    ("svm", SVC(kernel="rbf", C=1.0, probability=True))
])
```

---

### 6. Naive Bayes

**Principe :** Modèle probabiliste basé sur le Théorème de Bayes, en faisant l'hypothèse "naïve" que toutes les features sont indépendantes (ce qui n'est jamais vrai en réalité).
**Forces :** Extrêmement rapide. Excellent pour la classification de texte (NLP).

```python
from sklearn.naive_bayes import GaussianNB # Pour variables continues

gnb = GaussianNB() 
# gnb.fit(X_train, y_train)
```

---

### 7. KNN (K Plus Proches Voisins)

**Principe :** Mémorise simplement les données. Pour prédire un point, trouve les $K$ points d'entraînement les plus proches et prend un vote majoritaire.
**Forces :** Aucune phase d'entraînement mathématique. Compréhensible.
**Faiblesses :** La phase de prédiction est très lente (calcule la distance avec TOUT le dataset). Fléau de la dimension (devient mauvais en très haute dimension).

```python
from sklearn.neighbors import KNeighborsClassifier

knn = Pipeline([
    ("sc", StandardScaler()), # OBLIGATOIRE (utilise des distances)
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
```

---

## SECTION B — RÉGRESSION & RÉSEAUX

---

### 8. Linear Regression (MCO - Moindres Carrés Ordinaires)

**Principe :** Cherche la ligne droite (ou hyperplan) qui minimise la somme des carrés des erreurs (résidus).
**Le Modèle de base** absolu pour la régression.

```python
from sklearn.linear_model import LinearRegression

lr = Pipeline([
    ("sc", StandardScaler()), 
    ("ols", LinearRegression())
])
# Évaluation : r2_score(y_test, predictions), mean_squared_error()
```

---

### 9. Ridge Regression (Régularisation L2)

**Principe :** Une régression linéaire qui ajoute une pénalité (Alpha) à l'erreur pour forcer les coefficients à être les plus petits possibles.
**Pourquoi l'utiliser ?** Empêche le surapprentissage de la régression linéaire et gère la colinéarité (si deux variables sont très corrélées).

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0) # Si alpha=0, alors Ridge devient une LinearRegression classique
```

---

### 10. Lasso Regression (Régularisation L1)

**Principe :** Comme la Ridge, mais la pénalité force certains coefficients à devenir **strictement égaux à zéro**.
**Pourquoi l'utiliser ?** Si vous avez 500 variables et que vous voulez que le modèle sélectionne automatiquement les 20 plus importantes (Sélection de variables intégrée).

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, max_iter=5000)
# lasso.coef_ aura beaucoup de zéros
```

---

### 11. Elastic Net

**Principe :** C'est le compromis parfait. Il combine la régularisation **L1 (Lasso)** et **L2 (Ridge)** en même temps.

```python
from sklearn.linear_model import ElasticNet

# l1_ratio=0.5 signifie 50% de Lasso et 50% de Ridge
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
```

---

### 12. SVR (Support Vector Regression)

**Principe :** Cherche à faire passer un "tube" (de marge epsilon) contenant le maximum de points. Ne pénalise pas les erreurs tant qu'elles restent dans le tube.
Très puissant pour la régression non-linéaire grâce aux kernels (ex: RBF).

```python
from sklearn.svm import SVR

svr = Pipeline([
    ("sc", StandardScaler()), # OBLIGATOIRE
    ("svr", SVR(kernel="rbf", C=100, epsilon=0.1))
])
```

---

### 13. MLP (Multi-Layer Perceptron / Réseau de Neurones)

**Principe :** Un réseau de neurones artificiels dense (Feed-Forward). Constitué d'une couche d'entrée, de couches cachées (neurones) avec activation non-linéaire (ex: ReLU), et d'une sortie.
**Forces :** Capable d'apprendre n'importe quelle fonction continue (Approximateur Universel).
**Faiblesses :** Boîte noire (aucune interprétabilité). Niveau avancé requis pour paramétrer (architecture, learning_rate, batch_size, etc.).

```python
# Peut être utilisé en Classification (MLPClassifier) ou Régression (MLPRegressor)
from sklearn.neural_network import MLPClassifier

mlp = Pipeline([
    ("sc", StandardScaler()), # OBLIGATOIRE
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), # 3 couches de 128, 64 et 32 neurones
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    ))
])
```

---
*Fin du rapport de synthèse sur les 13 algorithmes sélectionnés.*
