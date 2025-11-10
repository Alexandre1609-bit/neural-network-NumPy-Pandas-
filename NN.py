import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#Import et nettoyage du DataSet 
data = pd.read_csv(r"ds\Iris.csv")

data.drop(['Id'], axis='columns',inplace=True)
species_dic = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
data['Species'] = data['Species'].apply(lambda x : species_dic[x])
print(data.head())
data = np.array(data) 


#Préparation des données
m, n = data.shape
np.random.shuffle(data)

data_dev = data[100:150].T
Y_dev = data_dev[4]
X_dev = data_dev[:4, :]

data_train = data[0:100].T
Y_train = data_train[4]
X_train = data_train[:4, :]


#Préparation du réseau
def init_params():

    
    W1 = np.random.randn(10, 4) * 0.01  
    b1 = np.zeros(10, 1)
   
    
    W2 = np.random.randn(3, 10) * 0.01
    b2 = np.zeros(3,1)
    return W1, b1, W2, b2


    #(W1 @ X_train) cf. allusion diagnostiqueur : 
    '''10 diagnostiqueurs dans une pièce qui se disent :
     "Je regarde les 4 symptômes. D'après mes connaissances (mes 4 poids), le score de cette fleur est [valeur]."

    La réponse Z1 des neurones sera un tableau (10, 100) car 1 ligne/neurone et chaque neurone a 100 résultats (car 100 fleurs à analyser)
    A ça on ajoute le biais b1 (humeur ou l'a priori) de chaque diagnostiqueur
    Ex : Diagnostiqueur 7 est naturellement "pessimiste". il aura un biais (b1[7]) très négatif
    Même si le score des symptômes (W @ X) est un peu positif, son biais négatif le tirera vers le bas.
    À l'inverse, un Diagnostiqueur 2 "optimiste" (biais b1[2]) aura tendance à donner des scores élevés, même si les symptômes sont faibles.
    Le biais ajoute de la fléxibilité au neurone (Facile ou non à activer)'''
   
   #Pourquoi W2 = (3, 10) et non (10, 3) : 
    '''La shape de mon W2 sera de (3, 10) pourquoi ? Car c'est la sortie des résulats de ma première "couche" de neuronnes.
    La première couche se base sur les 4 caractéristiques de chaque iris, ma deuxième couche, elle 
    sort les "résultats" donc elle fait un choix parmis les 3 iris du DataSet, d'où le (3, 10)
    (3, 10) et non (10, 3) CAR l'objectif de la 2nd couche est de calculer Z2 = (W2 * A1) + b2
    A1 est le résultat de la première couche Z1. Z1 (donc A1) a été calculé par W1 (10, 4) @ X (4, 100) la forme de A1 est donc de (10, 100) 10 neurones, 100 exemples. C'est le point de départ.
    
    Notre sortie Z2 sera le résultat de notre deuxième "couche" avec 3 neurones de sortie (Un pour chaque classe d'iris)
    et il doit le faire pour 100 exemples. Donc la forme de Z2 sera de (3, 100). C'est notre objectif
    
    On doit trouver W2 pour que notre équation soit vraie : 
    Shape de Z2 = Shape de W2 @ Shape de A1. Donc (3, 100) = (??, ??) @ (10, 100)
    Pour mutliplier (A, B) @ (B, C) la dimension intérieur de B doit être identique. Le résultat sera (A, C)
    
    Si on applique ça à notre équation : (??, ??) @ (10, 100)
    La dimension de W2 doit correspondre à la première dimension de A1. Donc W2 = (??, 10)
    le résultat sera (??, 100). On veut un résultat de forme (3, 100)
    Si le résultat est de (??, 100) et qu'on veut (3, 100) alors "??" doit être 3.
    DONC la seule forme possible de W2 est de (3, 10)'''


    #Activation (Rendre le tout "non-linéaire"(ReLu? SoftMax?)) :
    

