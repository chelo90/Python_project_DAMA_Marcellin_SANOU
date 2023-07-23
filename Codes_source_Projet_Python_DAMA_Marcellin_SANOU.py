#!/usr/bin/env python
# coding: utf-8

# ## Intitulé du projet : Analyse des données de santé « Cas du Paludisme en Afrique »

# ### Contenu du Dataset « Africa Malaria » :
# L'ensemble de données "Africa Malaria" comprend des données sur les pays africains de 2007 à 2017, avec les caractéristiques suivantes :
# - Code de pays ISO-3 unique : Chaque pays est identifié par un code de pays ISO-3, qui est un code standardisé utilisé pour représenter les pays dans les données internationales.
# - Latitude et longitude : Pour chaque pays, l'ensemble de données fournit également les coordonnées de latitude et de longitude, qui donnent la position géographique approximative du pays.
# - Cas de paludisme signalés : L'ensemble de données comprend des informations sur les cas de paludisme signalés dans chaque pays et chaque année. Ces données peuvent inclure le nombre total de cas, le nombre de cas selon le sexe, l'âge ou d'autres caractéristiques démographiques, ainsi que la gravité des cas signalés.
# - Mesures préventives : L'ensemble de données fournit également des données sur les mesures préventives prises pour lutter contre le paludisme dans chaque pays. Cela peut inclure des informations sur les campagnes de sensibilisation, les programmes de distribution de moustiquaires, les traitements médicaux administrés, etc.

# ### 1. Importation des bibliothèques pandas, numpy et matplotlib, pour manipuler les données et effectuer des analyses statistiques

# In[101]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### 2. Chargement du fichier du Dataset dans un dataframe pandas
# Lien de téléchargement du Dataset : https://www.kaggle.com/datasets/lydia70/malaria-in-africa

# In[102]:


df = pd.read_csv("DatasetAfricaMalaria.csv")


# ### 3. Nettoyage et structuration des données pour l'analyse

# #### 3.1. Copie du Dataset

# In[103]:


dfc = df.copy()


# #### 3.2. Affichage des 5 premières lignes du Dataset

# In[104]:


dfc.head()


# #### 3.3. Affichage des 5 dernières lignes du Dataset

# In[105]:


dfc.tail()


# #### 3.4. Examen des différentes colonnes et types de données dans le Dataset

# In[106]:


dfc.info()


# ##### Nombre d'index dans notre Dataset

# In[107]:


dfc.index


# ##### Conversion du type de données de la colonne "Malaria cases reported" en nombre entier

# In[108]:


dfc['Malaria cases reported']


# In[109]:


dfc['Malaria cases reported'] = dfc['Malaria cases reported'].astype('Int64')


# In[110]:


dfc['Malaria cases reported']


# ##### Renommons certaines colonnes de notre Dataset

# In[111]:


### Listons les noms de colonnes du Dataset avant renommage
list(dfc)


# #### Renommage des colonnes du Dataset

# In[112]:


dfc.rename(columns = {'Children with fever receiving antimalarial drugs (% of children under age 5 with fever)':'% of children under age 5 with fever receiving antimalarial drugs', 'Intermittent preventive treatment (IPT) of malaria in pregnancy (% of pregnant women)':'% of pregnant women using Intermittent preventive treatment (IPT) of malaria in pregnancy', 'People using safely managed drinking water services (% of population)':'% of population using safely managed drinking water services', 'People using safely managed drinking water services, rural (% of rural population)':'% of rural population using safely managed drinking water services', 'People using safely managed drinking water services, urban (% of urban population)':'% of urban population using safely managed drinking water services', 'People using safely managed sanitation services (% of population)':'% of population using safely managed sanitation services', 'People using safely managed sanitation services, rural (% of rural population)':'% of rural population using safely managed sanitation services', 'People using safely managed sanitation services, urban  (% of urban population)':'% of urban population using safely managed sanitation services', 'People using at least basic drinking water services (% of population)':'% of population using at least basic drinking water services', 'People using at least basic drinking water services, rural (% of rural population)':'% of rural population using at least basic drinking water services', 'People using at least basic drinking water services, urban (% of urban population)':'% of urban population using at least basic drinking water services', 'People using at least basic sanitation services (% of population)':' % of population using at least basic sanitation services', 'People using at least basic sanitation services, rural (% of rural population)':'% of rural population using at least basic sanitation services', 'People using at least basic sanitation services, urban  (% of urban population)':'% of urban population using at least basic sanitation services', 'geometry':'Localisation'}, inplace = True)


# In[113]:


list(dfc)


# ### 4. Analyse exploratoire des données

# ####  4.1. Statistiques descriptives du Dataset

# In[114]:


df.describe()


# #### 4.2. Analyses statistiques pour identifier les tendances des maladies, les facteurs de risque

# #### * Tendances annuelles des cas de paludisme, classées par année du nombre de cas le plus élevé de paludisme au nombre de cas le plus faible

# In[115]:


som_tendances_annuelles = dfc.groupby("Year")["Malaria cases reported"].sum().sort_values(ascending=False)


# In[117]:


som_tendances_annuelles


# In[118]:


moy_tendances_annuelles = dfc.groupby("Year")["Malaria cases reported"].mean().sort_values(ascending=False)


# In[119]:


moy_tendances_annuelles


# #### * Tendances par pays des cas de paludisme de 2007-2017, classé par pays ayant le nombre de cas de paludisme le plus élevé au nombre de cas le plus faible

# In[120]:


som_tendances_pays = dfc.groupby("Country Name")["Malaria cases reported"].sum().sort_values(ascending=False)


# In[121]:


som_tendances_pays


# In[122]:


moy_tendances_pays = dfc.groupby("Country Name")["Malaria cases reported"].mean().sort_values(ascending=False)


# In[123]:


moy_tendances_pays


# ##### * Visualisation des résultats des tendances annuelles

# In[135]:


som_tendances_annuelles.plot(kind="line")
plt.xlabel("Year")
plt.ylabel("Malaria cases reported")
plt.title("Figure 01: Tendances annuelles des cas de paludisme en Afrique")
plt.show()


# ### Lecture de la Figure 01 : la tendance est que le nombre de cas de paludisme en Afrique n'a fait qu'augmenter d'année en année entre 2007 et 2017

# #### * Tendances des cas de paludisme en fonction de l'utilisation de moustiquaires imprégnées d'insecticide (% de la population de moins de 5 ans)

# In[131]:


som_tendances_treated_bed_nets = dfc.groupby("Use of insecticide-treated bed nets (% of under-5 population)")["Malaria cases reported"].sum()


# In[132]:


som_tendances_treated_bed_nets


# #### * Visualisation de la tendance des cas de paludisme en Afrique pour enfants de -5 ans dormant sous moustiquaires imprégnés

# In[136]:


som_tendances_treated_bed_nets.plot(kind="line")
plt.xlabel("Use of insecticide-treated bed nets (% of under-5 population)")
plt.ylabel("Malaria cases reported")
plt.title("Figure 02 : Tendances des cas de paludisme en Afrique pour enfants de -5 ans dormant sous moustiquaires imprégnés")
plt.show()


# ### Lecture de la Figure 02 : la tendance est que plus les enfants de moins de 5 ans dorment sous moustiquaires imprégnés, moins ils ont le paludisme.

# #### 4.3. Corrélations entre les variables pour analyser les relations entre les facteurs de risque et les cas de paludisme

# ##### Definition : La relation statistique entre deux variables est appelée leur corrélation.

# In[138]:


correlation_matrix = dfc.corr()


# In[139]:


correlation_matrix


# #### Analyse des corrélations et identification des facteurs de risque potentiels en examinant les valeurs de corrélation

# ##### * Corrélations entre les cas de paludisme et les variables démographiques

# ###### - Avec Incidence du paludisme (pour 1 000 habitants à risque)

# In[140]:


corr_cases_population_1 = correlation_matrix["Malaria cases reported"]["Incidence of malaria (per 1,000 population at risk)"]


# In[141]:


corr_cases_population_1


# ##### - Avec % d'enfants de moins de 5 ans ayant de la fièvre recevant des médicaments antipaludiques

# In[142]:


corr_cases_population_2 = correlation_matrix["Malaria cases reported"]["% of children under age 5 with fever receiving antimalarial drugs"]


# In[143]:


corr_cases_population_2


# ###### - Avec l'utilisation de moustiquaires imprégnées d'insecticide (% de la population de moins de 5 ans)

# In[144]:


corr_cases_population_3 = correlation_matrix["Malaria cases reported"]["Use of insecticide-treated bed nets (% of under-5 population)"]


# In[145]:


corr_cases_population_3


# ###### - Avec la Population urbaine (% de la population totale)

# In[146]:


corr_cases_population_4 = correlation_matrix["Malaria cases reported"]["Urban population growth (annual %)"]


# In[147]:


corr_cases_population_4


# ##### Forte Corrélation entre "% of rural population using safely managed sanitation services" et "% of rural population using safely managed drinking water services"

# In[148]:


corr_cases_population_5 = correlation_matrix["% of rural population using safely managed sanitation services"]["% of rural population using safely managed drinking water services"]


# In[149]:


corr_cases_population_5


# #### Machine Learning : Modélisation des données pour identifier des modèles et des relations importantes dans les données

# In[184]:


## Utilisons la bibliothèque scikit-learn pour effectuer une régression linéaire et importons le modèle de régression linéaire
from sklearn.linear_model import LinearRegression
## Le module Impute de Sklearn (scikit-learn) permet de nettoyer notre dataset des valeurs manquantes qui le composes.
## SimpleImputer remplace toute valeur manquante par une statistique ou une constante donnée
from sklearn.impute import SimpleImputer


# In[201]:


## X représente des variables indépendantes : dans notre cas "% d'enfants de moins de 5 ans dormant sous moustiquaires imprégnées" et "% de femmes enceintes utilisant le traitement préventif intermittent (TPI) du paludisme pendant la grossesse"
X = dfc[["Use of insecticide-treated bed nets (% of under-5 population)", "% of pregnant women using Intermittent preventive treatment (IPT) of malaria in pregnancy"]]
## y représente la variable cible : dans notre cas "Year (Année)"
y = dfc["Year"]


# In[203]:


## Utilisons SimpleImputer pour remplacer toute valeur manquante par une statistique ou une constante donnée. Dans notre cas, la moyenne des valeurs non manquantes
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
## Transformons dans les variables indépendantes contenues dans X, les valeurs manquantes par np.nan
X_imputed = imputer.fit_transform(X)


# In[206]:


## Utilisons SimpleImputer pour remplacer toute valeur manquante par une statistique ou une constante donnée. Dans notre cas, la moyenne des valeurs non manquantes
imputer_y = SimpleImputer(strategy='mean')
## Appelons la méthode `reshape` sur `y.values`, pour remodeler le tableau en un tableau 2D avec une seule colonne, 
## ce qui est le format d'entrée attendu pour la méthode `fit_transform` de la classe `SimpleImputer`.
y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1))


# In[207]:


## Instancions ensuite un objet modèle de régression linéaire en utilisant la classe LinearRegression().
model = LinearRegression()
## À l'aide de la méthode fit(), entraînons le modèle en utilisant les données X et y.
model.fit(X_imputed, y_imputed)
## Utilisons la méthode predict() pour effectuer des prédictions sur les données d'entraînement X et stockez les prédictions 
## résultantes dans la variable predictions.
predictions = model.predict(X_imputed)


# #### Machine Learning : Evaluation du modèle de Regression Linéaire

# In[210]:


## Importation de la fonction "mean_squared_error" du module "sklearn.metrics". Cela permettra de calculer l'erreur 
## quadratique moyenne ou Mean Squared Error(MSE).
from sklearn.metrics import mean_squared_error

## Calcul de l'erreur quadratique moyenne (MSE) entre les valeurs réelles 'y' et les valeurs prédites 'predictions' 
## en utilisant la fonction 'mean_squared_error'.
mse = mean_squared_error(y, predictions)

## Calcul de la racine carrée de l'erreur quadratique moyenne (MSE) en utilisant la fonction `np.sqrt` de numpy.
rmse = np.sqrt(mse)

## Affichage de la valeur du RMSE en utilisant la fonction 'print'
## Habituellement, un score RMSE inférieur à 180 est considéré comme un bon score pour un algorithme qui fonctionne modérément 
## ou bien. Dans le cas où la valeur RMSE dépasse 180, nous devons effectuer une sélection de caractéristiques et un réglage 
## des paramètres hyper sur les paramètres du modèle.
print("RMSE:", rmse)


# #### Conclusion RMSE : Le résultat du RMSE est inférieur à 180, donc notre algorithme de Machine Learning fonctionne bien

# #### Machine Learning : Communication des résultats

# #### Visualisation des résultats

# In[214]:


plt.scatter(X['Use of insecticide-treated bed nets (% of under-5 population)'], y)
plt.plot(X['Use of insecticide-treated bed nets (% of under-5 population)'], predictions, color='red')
plt.xlabel('Use of insecticide-treated bed nets (% of under-5 population)')
plt.ylabel('Year')
plt.title("Figure 03 : Visualisation des résultats de notre modèle de Regression Linéaire")
plt.show()


# ### Lecture de la Figure 03 : le taux d'enfants de moins de 5 ans dormant sous moustiquaires imprégnés, continuera à augmenter d'année en année afin de réduire le nombre de cas de paludisme.

# In[ ]:




