# 🫀 Clustering de Patients Cardiaques

## Objectif du projet
Ce projet a pour but d’**identifier des profils de patients à risque de maladie cardiaque** à l’aide de **méthodes de clustering non supervisées**.  
À partir d’un jeu de données médicales, plusieurs approches ont été comparées pour regrouper les individus selon leurs similarités cliniques.

---

## ⚙️ Méthodes utilisées
Trois algorithmes de clustering ont été appliqués :

- **K-Means** → approche basée sur la distance moyenne (Nathan)  
- **Gaussian Mixture Models (GMM)** → approche probabiliste (Amir)  
- **DBSCAN** → approche par densité, avec réduction de dimension (Joubrane)  

Des techniques de **réduction de dimension** comme **PCA**, **t-SNE** et **Isomap** ont également été employées pour faciliter la visualisation et améliorer la détection des structures.


## 👥 Répartition du travail
| Membre     | Algorithme étudié         
|-------------|---------------------------|
| **Nathan**  | K-Means                   | 
| **Amir**    | Gaussian Mixture Models   | 
| **Joubrane**| DBSCAN                    | 

Joubrane s'est occupé de l'exploration des données.
Nous nous sommes tous occupé de la partie traitement des données (Transformation [Scale, PCA, t-SNE])

---

Vous trouverez le rapport du projet dans le dépôt ainsi que 3 notebook pour chaque algorithme utilisé
