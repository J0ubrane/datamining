import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE







def showPCA2dim(X) : 

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 2. Analyse de la Variance Expliquée
    explained_variance = pca.explained_variance_ratio_
    variance_cumulee = sum(explained_variance)

    print("--- Analyse de la Variance Expliquée ---")
    print(f"Variance expliquée par la Composante Principale 1 (CP1): {explained_variance[0]*100:.1f} %")
    print(f"Variance expliquée par la Composante Principale 2 (CP2): {explained_variance[1]*100:.1f} %")
    print(f"Variance cumulée (CP1 + CP2): {variance_cumulee*100:.1f} %")
    print("-" * 60)


    # 3. Création du Plot (Nuage de Points Non-Clustérisés)
    plt.figure(figsize=(10, 7))

    # Afficher tous les points en une seule couleur (pas de cluster pour l'instant)
    plt.scatter(
        X_pca[:, 0], 
        X_pca[:, 1], 
        marker='o', 
        s=30,           # Taille des marqueurs
        c='teal',       # Couleur uniforme
        alpha=0.6       # Transparence
    )

    plt.title('Visualisation 2D des Données Quantitatives via PCA', fontsize=16)
    plt.xlabel(f'Composante Principale 1 ({explained_variance[0]*100:.1f} % de variance)', fontsize=12)
    plt.ylabel(f'Composante Principale 2 ({explained_variance[1]*100:.1f} % de variance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return X_pca

def showTSNE(X_reduced, p=30):
    tsne = TSNE(n_components=2, perplexity=p, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_reduced)

    # 2. Création du Plot avec Coloration par Cible
    plt.figure(figsize=(10, 7))


    scatter = plt.scatter(
        X_tsne[:, 0], 
        X_tsne[:, 1], 
        cmap='coolwarm',    # Palette de couleurs (Bleu pour 0, Rouge pour 1)
        s=40,               # Taille des marqueurs
        alpha=0.7,
        edgecolors='w',
        linewidths=0.5
    )


    plt.title('Visualisation 2D des Données via t-SNE', fontsize=16)
    plt.xlabel('Composante t-SNE 1', fontsize=12)
    plt.ylabel('Composante t-SNE 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show() 

    return X_tsne



def scaleData(X, quant_col, cat_col=None) : 
    if cat_col != None : 
        X[cat_col] = X[cat_col].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), quant_col)
        ],
        remainder='passthrough'
    )

    # 3. Application de la transformation
    X_scaled_selective = preprocessor.fit_transform(X)

    return X_scaled_selective


def reducedPcaWith90(X) : 
    pca_90_percent = PCA(n_components=0.9)

    # 2. Application de la transformation
    X_reduced = pca_90_percent.fit_transform(X)

    # 3. Récupération des informations
    n_components_needed = pca_90_percent.n_components_
    variance_explained = sum(pca_90_percent.explained_variance_ratio_)


    # 4. Affichage des résultats
    print("--- Réduction de Dimension (PCA pour 90% de Variance) ---")
    print(f"Nombre initial de features : {X.shape[1]}")
    print(f"Nombre de Composantes Principales (CP) retenues pour 90% : {n_components_needed}")
    print(f"Variance cumulée expliquée : {variance_explained*100:.2f} %")
    print("-" * 60)
    print(f"Forme du nouvel ensemble de features : {X_reduced.shape}")
    print("\nPremières lignes de l'espace réduit (X_reduced) :")
    print(X_reduced[:5])

    return X_reduced


def gridSearchDBScan(X_reduced, eps_to_test, min_samples_to_test) : 
    best_params = None 
    best_score = -1 
    best_n_clusters = 0
    best_noise = 0

    for eps_test in eps_to_test: 
        for min_samples_value in min_samples_to_test:
            
            dbscan_test = DBSCAN(eps=eps_test, min_samples=min_samples_value)
            # Utilisation de X_reduced !
            clusters_test = dbscan_test.fit_predict(X_reduced)
            
            # Calcul des métriques
            n_noise = np.sum(clusters_test == -1)
            # Nombre de clusters (hors bruit -1)
            n_clusters = len(np.unique(clusters_test)) - (1 if -1 in clusters_test else 0)
            
            current_score = np.nan
            
            if n_clusters >= 2:
                # Filtrer les points du cluster -1 pour le calcul de la Silhouette
                X_filtered = X_reduced[clusters_test != -1]
                clusters_filtered = clusters_test[clusters_test != -1]
                
                # S'assurer qu'il reste au moins deux classes distinctes dans l'ensemble filtré
                if len(np.unique(clusters_filtered)) >= 2:
                    current_score = silhouette_score(X_filtered, clusters_filtered)
                
                # Mise à jour des meilleurs paramètres
                if current_score > best_score:
                    best_score = current_score
                    best_params = (eps_test, min_samples_value)
                    best_n_clusters = n_clusters
                    best_noise = n_noise

            # Affichage du résultat de cet essai
            print(f"Eps: {eps_test:.1f}, MinPts: {min_samples_value}")
            print(f"  Bruit: {n_noise} ({n_noise / len(X_reduced) * 100:.1f} %)")
            print(f"  Clusters: {n_clusters}, Score Silhouette: {round(current_score, 4) if not np.isnan(current_score) else 'N/A'}")
            
    print("-" * 60)
    print("--- MEILLEURS RÉSULTATS ---")
    if best_params:
        print(f"Meilleurs Paramètres: Eps={best_params[0]}, MinPts={best_params[1]}")
        print(f"Meilleur Score de Silhouette: {best_score:.4f}")
        print(f"Nombre de Clusters trouvés: {best_n_clusters}")
        print(f"Bruit: {best_noise}")

    else:
        print("Aucune combinaison n'a produit au moins 2 clusters avec un score valide.")


    return best_params



def cluster_summary(cluster_final, list_disease) :

    df_analysis = pd.DataFrame({
        'Cluster_Label': cluster_final,
        'HeartDisease': list_disease
    })

    # 2. Calcul des Statistiques par Cluster
    cluster_summary = df_analysis.groupby('Cluster_Label')['HeartDisease'].agg(
        # Compte le nombre total de points dans le cluster
        Total_Points='count',
        # Compte le nombre de patients avec HeartDisease (HeartDisease = 1)
        Nb_Malades='sum',
        # Calcule le taux de HeartDisease (Moyenne = Taux pour 0/1)
        Taux_Maladie='mean'
    )

    # 3. Formatage et Affichage
    cluster_summary['Taux_Maladie'] = (cluster_summary['Taux_Maladie'] * 100).round(2).astype(str) + '%'

    print(cluster_summary)

    # Interprétation Rapide du Bruit (-1)
    noise_row = cluster_summary.loc[-1] if -1 in cluster_summary.index else None
    if noise_row is not None:
        print(f"Interprétation du Bruit (-1) : {noise_row['Nb_Malades']} patients ({noise_row['Taux_Maladie']}) parmi les points de bruit sont malades.")


def dbscanOnTSNE(X_tsne, eps, min_samples) :
    # 3.5 et 10
    dbscan_tsne = DBSCAN(eps=eps, min_samples=min_samples)
    clusters_tsne = dbscan_tsne.fit_predict(X_tsne)

    # 2. Analyse des résultats
    n_noise = np.sum(clusters_tsne == -1)
    n_clusters = len(np.unique(clusters_tsne)) - (1 if -1 in clusters_tsne else 0)


    silhouette_avg = 'N/A'

    if n_clusters >= 2:
        X_filtered = X_tsne[clusters_tsne != -1]
        clusters_filtered = clusters_tsne[clusters_tsne != -1]
        
        if len(np.unique(clusters_filtered)) >= 2:
            silhouette_avg = silhouette_score(X_filtered, clusters_filtered)
            silhouette_avg = round(silhouette_avg, 4)


    print("--- Résultat DBSCAN sur t-SNE ---")
    print(f"Paramètres utilisés: Eps={eps}, MinPts={min_samples}")
    print(f"Clusters trouvés: {n_clusters}")
    print(f"Points de bruit: {n_noise} ({n_noise / len(X_tsne) * 100:.2f} %)\n")
    print(f"Score silhouette : {silhouette_avg}")



    # 3. Visualisation des Clusters sur la t-SNE
    plt.figure(figsize=(10, 7))

    # Définir les étiquettes de cluster uniques (y compris -1 pour le bruit)
    unique_labels = np.unique(clusters_tsne)
    # Utiliser une colormap pour attribuer des couleurs différentes à chaque cluster
    colors = plt.cm.get_cmap('Spectral', len(unique_labels))

    for k in unique_labels:
        class_member_mask = (clusters_tsne == k)
        xy = X_tsne[class_member_mask]
        
        if k == -1:
            # Le Bruit est affiché en gris, plus petit et plus transparent
            color = 'gray'
            marker_size = 20
            alpha = 0.4
            label = 'Bruit (-1)'
        else:
            # Les Clusters sont colorés
            color = colors(k)
            marker_size = 50
            alpha = 0.8
            label = f'Cluster {k}'

        plt.scatter(
            xy[:, 0], 
            xy[:, 1], 
            marker='o', 
            s=marker_size, 
            c=color, 
            alpha=alpha, 
            label=label
        )

        

    plt.title('DBSCAN Appliqué à la Visualisation t-SNE', fontsize=16)
    plt.xlabel('Composante t-SNE 1', fontsize=12)
    plt.ylabel('Composante t-SNE 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show() 


    return clusters_tsne