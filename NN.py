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


    #Activation (Rendre le tout "non-linéaire"(ReLu? SoftMax?, Sigmoïde?, initialisation de Z1, Z2... ?? )) :
    '''Fonction d'activation. ReLu regarde si un résultat est < 0 ou > 0. 
    Si Z est négatif (-1.5) alors ReLu dira "Ce n'est pas assez fort, j'ignore" et renverra un 0, le neurone sera "éteint.
    Dans le cas contraire, si Z est positif (3.8) alors ReLu dit "Ce signal est important, je le laisse passer tel quel. 
    Il renvoie 3.8, le neurone est "allumé".'''

def relu(Z):  #Erreur, pas mettre relu(Z1) mais plutôt un nom générique comme relu(Z), pareil pour softmax.
    ReLu = np.maximum(0.0, Z)
    return ReLu

def softmax(Z):
     numerateur = np.exp(Z)
     denominateur = np.sum(np.exp(Z), axis=0, keepdims=True)
     sm = numerateur / denominateur
     return sm


     '''Pour softmax : C'est une des fonctions d'activation standard pour la couche finale d'un problème de classification multi-classes (Ici 3 iris).
    On peut voir cette fonction comme un "traducteur" qui convertit les scores bruts en bulletin de confiance.
    La couche finale calcule ses scores bruts Z2, pour une fleurs ces scores (logits) pourraient être par exemple :
    [Score Setosa : 1.2], [Score Versicolor ; -0.8], [Score Virginica : 4.1].
    Le problème est que nous ne savons pas interpreter ces scores.
    4.1 est élevé mais est-ce 50% de confiance, 90%, 100% ?? Et comment gérer les scores négatifs ??
    
    -1"Exponentielle"
    
    C'est ici que softmax intervient. La fonction rend tous les scores positifs en utilisant (np.exp()).
    Donc nos scores deviennent [3.32, 0.45, 60.34]. On voit que l'écart entre le score le plus haut et les autres est énorme.

    -2"Normalisation"

    Ensuite la fonction divise chaque score par la somme de tous les scores (3.32 + 0.45 + 60.34 = 64.11)
    On obtient : 
    Setosa : 3.32 / 64.11 = 0.05 (soit 5%)
    Versicolor : 0.45 / 64.11 = 0.007 (soit 0.7%)
    Virginica : 60.34 / 64.11 = 0.94 (soit 94%)

    Les resultats nous montrent une prédiction claire : [0.05, 0.007, 0.94].
    C'est un vecteur de probabilité dont la somme fait 1 !

    On devient donc en mesure d'interpréter les résultats (pourcentages de confiance, pas de score bruts.)
    C'est aussi "soft", un Hard Max aurait donné [0, 0, 1]. Avec Softmax c'est plus nuancé.
    On voit qu'il y a 94% de chance que ce soit Virginica mais qu'il y ait aussi 5% de chance que ce soit Setosa.
    Cette nuance est vitale pour l'apprentissage. (Rétropropagation)

    '''

     #Forward Propagation (Forward Pass?)
     '''Se renseigner sur le sujet.'''

def forward_prop(X, W1, b1, W2, b2):
    
    Z1 = W1 @ X + b1 #Erreur : pas * mais @ car c'est l'opérateur de NumpY pour la multiplication matricielle
    A1 = relu(Z1)
    
    Z2 = W2 @ A1 + b2 #Erreur : Pas  Z2 = W2 @ X_train + b2 car Z2 prend les résultat de Z1 (donc A1), c'est une chaîne !
    A2 = softmax(Z2)

    return A1, A2, Z1, Z2

#Prochaine étape : Fonction de coût : (Loss function)
'''Se renseigner dessus. Analogie du "Juge".
Cette fonction va regarder la prédiction A2 et la comparer au vraies réponses (Y_train) afin de donner une "note d'erreur" (le "coût" ou "loss")'''