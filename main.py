import pandas as pd
#sert le calcule numerique (matrice) 
import numpy as np
#gestion de visuel du maths 
import matplotlib.pyplot as plt
import seaborn as sns
#outils pour le marchine lerning  tout les formules des reg linaire. train_test_split qui separt les jeux de données en 80% train et 20% test pour verifier qui ne triche pas
from sklearn.model_selection import train_test_split
#standardiser : x - moyenne x (center) / ecart type de x (centrée reduite )
from sklearn.preprocessing import StandardScaler
#class de logistic reg
from sklearn.linear_model import LogisticRegression
#score metrics
from sklearn.metrics import accuracy_score

#============ importation des données
path = "googleplaystore_merged.csv"
raw_df = pd.read_csv(path, sep=",",decimal=".")

#============ copie du dataset brut
jeuxvideos_df = raw_df
jeuxvideos_df.head()

#============ vérification des types
jeuxvideos_df.dtypes

#============ afficher la dimension
print(jeuxvideos_df.shape)

# #============ importation des données
# pathr = "googleplaystore_user_reviews.csv"
# raw_dfr = pd.read_csv(pathr, sep=",")
# raw_dfr = raw_dfr.drop(['Translated_Review'], axis=1)

# #============ copie du dataset brut
# jeuxvideos_dfr = raw_dfr
# jeuxvideos_dfr.head()

# #============ vérification des types
# jeuxvideos_dfr.dtypes

# #============ afficher la dimension
# print(jeuxvideos_dfr.shape)

# # Supprimer les doublons
# jeuxvideos_df = jeuxvideos_df.drop_duplicates()

# # Vérifier la dimension après la suppression des doublons
# print(jeuxvideos_df.shape)

# # Nom du nouveau fichier CSV
# nouveau_nom_fichier = "googleplaystore_sans_doublons.csv"

# # Enregistrer le DataFrame sans doublons dans un nouveau fichier CSV
# jeuxvideos_df.to_csv(nouveau_nom_fichier, index=False)

# # Afficher un message de confirmation
# print(f"Le fichier {nouveau_nom_fichier} a été créé avec succès.")

# # Grouper les données par application et compter les occurrences de chaque sentiment
# sentiment_counts = raw_dfr.groupby('App')['Sentiment'].value_counts().unstack().fillna(0)

# # Compter le nombre de valeurs NaN par application
# nan_counts = raw_dfr.groupby('App')['Sentiment'].apply(lambda x: x.isna().sum())

# # Ajouter les valeurs NaN au DataFrame des comptages de sentiments
# sentiment_counts['NaN'] = nan_counts

# # Trouver la colonne avec la valeur la plus élevée par ligne
# max_column = sentiment_counts.idxmax(axis=1)

# # Ajouter le montant de la colonne 'NaN' à la colonne avec la valeur la plus élevée
# for app, col in max_column.items():
#     sentiment_counts.loc[app, col] += sentiment_counts.loc[app, 'NaN']

# # Supprimer la colonne 'NaN'
# sentiment_counts.drop(columns=['NaN'], inplace=True)

# # Calculer la moyenne de la polarité du sentiment pour chaque application, en excluant les valeurs NaN
# polarity_mean = raw_dfr.groupby('App')['Sentiment_Polarity'].mean()

# # Calculer la moyenne de la subjectivité du sentiment pour chaque application, en excluant les valeurs NaN
# subjectivity_mean = raw_dfr.groupby('App')['Sentiment_Subjectivity'].mean()

# # Fusionner les deux DataFrames
# result = pd.concat([sentiment_counts, polarity_mean, subjectivity_mean], axis=1)

# # Renommer les colonnes
# result.rename(columns={'Sentiment_Polarity': 'AVG_Pol', 'Sentiment_Subjectivity': 'AVG_Sub'}, inplace=True)

# # Renommer les colonnes pour plus de clarté
# # result.columns.name = 'Sentiment'

# # Afficher le tableau résultant
# print(result)

# # Fusionner les DataFrames sur la colonne 'App'
# merged_df = pd.merge(jeuxvideos_df, result, on='App', how='left')

# # Enregistrer le nouveau DataFrame fusionné dans un fichier CSV
# nouveau_nom_fichier = "googleplaystore_merged.csv"
# merged_df.to_csv(nouveau_nom_fichier, index=False)

# # Afficher un message de confirmation
# print(f"Le fichier {nouveau_nom_fichier} a été créé avec succès.")

# Supprimer les doublons basés sur le nom de l'application
jeuxvideos_df = jeuxvideos_df.drop_duplicates(subset='App')

# Vérifier la dimension après la suppression des doublons
print("Dimension après suppression des doublons:", jeuxvideos_df.shape)

# Enregistrer le DataFrame sans doublons dans le même fichier CSV
jeuxvideos_df.to_csv(path, index=False)

# # Calculer la moyenne de la colonne "Rating" pour chaque catégorie en excluant les valeurs manquantes
# rating_mean_by_category = jeuxvideos_df[jeuxvideos_df['Rating'].notna()].groupby('Category')['Rating'].mean()

# # Remplacer les valeurs manquantes dans la colonne "Rating" par la moyenne de la catégorie correspondante
# for category, rating_mean in rating_mean_by_category.items():
#     jeuxvideos_df.loc[(jeuxvideos_df['Category'] == category) & (jeuxvideos_df['Rating'].isna()), 'Rating'] = rating_mean

# # Arrondir les valeurs de la colonne "Rating" à un chiffre après la virgule
# jeuxvideos_df['Rating'] = jeuxvideos_df['Rating'].round(1)

# # Enregistrer le DataFrame mis à jour dans le même fichier CSV
# jeuxvideos_df.to_csv(path, index=False)

# # Afficher un message de confirmation
# print("Les valeurs manquantes dans la colonne 'Rating' ont été remplacées avec succès, et les valeurs de la colonne 'Rating' ont été arrondies à un chiffre après la virgule.")


# # Convertir les valeurs de la colonne "Size" en kilooctets
# def convert_to_kilobytes(size):
#     if 'M' in size:
#         # Supprimer le 'M' et convertir en kilooctets
#         return float(size.replace('M', '')) * 1024
#     elif 'k' in size:
#         # Supprimer le 'k' et garder en kilooctets
#         return float(size.replace('k', ''))
#     else:
#         return None  # Pour les valeurs inconnues

# # Appliquer la fonction de conversion à la colonne "Size"
# jeuxvideos_df['Size'] = jeuxvideos_df['Size'].apply(convert_to_kilobytes)

# # Afficher un message de confirmation
# print("Les valeurs de la colonne 'Size' ont été converties en kilooctets avec succès.")

# # Enregistrer le DataFrame mis à jour dans le même fichier CSV
# jeuxvideos_df.to_csv(path, index=False)

# # Afficher un message de confirmation
# print("Le fichier googleplaystore_merged.csv a été mis à jour avec succès.")

# # Supprimer tous les caractères "+" de la colonne "Installs"
# jeuxvideos_df['Installs'] = jeuxvideos_df['Installs'].str.replace('+', '')

# # Afficher un message de confirmation
# print("Les caractères '+' ont été supprimés de la colonne 'Installs' avec succès.")

# # Enregistrer le DataFrame mis à jour dans le même fichier CSV
# jeuxvideos_df.to_csv(path, index=False)

# Définir une fonction pour supprimer les virgules
def remove_commas(value):
    # Vérifier si la valeur est une chaîne de caractères et contient des virgules
    if isinstance(value, str) and ',' in value:
        # Supprimer les virgules
        return int(value.replace(',', ''))
    else:
        return value

# Appliquer la fonction à la colonne "Installs"
jeuxvideos_df['Installs'] = jeuxvideos_df['Installs'].apply(remove_commas)

# Afficher un message de confirmation
print("Les virgules ont été supprimées de la colonne 'Installs'.")

# Enregistrer le DataFrame mis à jour dans le même fichier CSV
jeuxvideos_df.to_csv(path, index=False)

# Afficher un message de confirmation
print("Le fichier googleplaystore_merged.csv a été mis à jour avec succès.")

# Enregistrer le DataFrame mis à jour dans le même fichier CSV
jeuxvideos_df.to_csv(path, index=False)

# Afficher un message de confirmation
print("Le fichier googleplaystore_merged.csv a été mis à jour avec succès.")