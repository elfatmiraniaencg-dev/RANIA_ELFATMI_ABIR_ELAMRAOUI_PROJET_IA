# 📊 Rapport de Modélisation Prédictive et Classification Algorithmique

## 📝 Présentation du Projet
Ce dépôt constitue une étude approfondie des capacités de l'apprentissage automatique (**Machine Learning**) appliqué à deux domaines distincts : la finance boursière avec le cas de **Maroc Telecom (IAM)** et la classification de données biologiques et botaniques. L'approche adoptée privilégie la précision statistique et la robustesse des modèles à travers des pipelines de traitement de données rigoureux et une sélection méticuleuse d'algorithmes de pointe.

---

## 📉 Partie 1 : Prédiction Boursière — Dossier IAM S8

Cette section se concentre sur l'analyse et la prédiction du cours de clôture de l'action Maroc Telecom. L'enjeu majeur ici est de transformer une série temporelle brute en un signal prédictible en tenant compte de la volatilité intrinsèque du marché marocain.

### 🧪 Ingénierie des Caractéristiques (Feature Engineering)
Le pipeline de données ne se limite pas aux prix d'ouverture et de fermeture ; il intègre des indicateurs mathématiques avancés via la bibliothèque `ta` (Technical Analysis) :

* **Indice de Force Relative (RSI) :** Cet oscillateur calcule le rapport entre les hausses et les baisses moyennes sur une période donnée pour déterminer si l'action est en zone de surachat ou de survente, fournissant ainsi une variable cruciale pour capter les retournements de tendance.
* **Moyennes Mobiles Exponentielles (EMA) :** Contrairement aux moyennes simples, les EMA accordent plus de poids aux données récentes, permettant au modèle de réagir plus rapidement aux nouvelles fluctuations du marché.
* **Bandes de Bollinger :** Elles permettent d'intégrer la notion de volatilité en mesurant l'écart-type autour d'une moyenne mobile, ce qui aide le modèle à identifier les phases de compression ou d'explosion des prix.

### 💻 Analyse du Modèle Champion : Gradient Boosting
Le modèle retenu pour cette mission est le **Gradient Boosting Regressor**. Cet algorithme fonctionne par une construction itérative d'arbres de décision où chaque nouvel arbre est spécifiquement entraîné pour prédire et corriger l'erreur résiduelle (le gradient de la fonction de perte) commise par l'ensemble des arbres précédents.

$$Loss = \sum (y_i - \hat{y}_i)^2$$

```python
# Implémentation du modèle de régression par Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

# Initialisation du modèle avec des paramètres favorisant la convergence
model_iam = GradientBoostingRegressor(
    n_estimators=100,  # Le modèle va construire 100 arbres successifs
    learning_rate=0.1, # Facteur d'échelle appliqué à chaque arbre
    max_depth=3,       # Profondeur limitée pour assurer une généralisation optimale
    random_state=42    # Garantit la reproductibilité des résultats
)
model_iam.fit(X_train, y_train)
