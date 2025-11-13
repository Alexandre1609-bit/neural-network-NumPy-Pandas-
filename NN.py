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


#Préparation du réseau (index 1)
def Init_params():

    W1 = np.random.randn(10, 4) * 0.01  
    b1 = np.zeros(10, 1)
   
    W2 = np.random.randn(3, 10) * 0.01
    b2 = np.zeros(3,1)
    return W1, b1, W2, b2


    
    

#Activation (index 2) (Rendre le tout "non-linéaire"(ReLu? SoftMax?, Sigmoïde?, initialisation de Z1, Z2... ?? )) :
def Relu(Z):  #nb: pas mettre Relu(Z1) mais plutôt un nom générique comme Relu(Z), pareil pour softmax.
    ReLu = np.maximum(0.0, Z)
    return ReLu


def Softmax(Z):
     Numerateur = np.exp(Z)
     Denominateur = np.sum(np.exp(Z), axis=0, keepdims=True)
     Sm = Numerateur / Denominateur
     return Sm




#Forward Propagation (index 3) (Forward pass, Propagation avant)
def forward_prop(X, W1, b1, W2, b2):
    
    Z1 = W1 @ X + b1 #nb: pas "*" mais "@" car c'est l'opérateur de NumpY pour la multiplication matricielle
    A1 = Relu(Z1)
    
    Z2 = W2 @ A1 + b2 #nb: Pas  "Z2 = W2 @ X_train + b2" car "Z"2 prend les résultat de "Z1" (donc "A1"), c'est une chaîne !
    A2 = Softmax(Z2)

    return A1, A2, Z1, Z2

   

#Fonction de coût (index 4) (Loss function)
def one_hot(Y): #Categorical Cross-Entropy (toujours utilisé avec Softmax ? ) 

    m = Y.size
    columns_index = np.arange(m)

    Y_one_hot = np.zeros(( 3, m))
    Y_one_hot[Y, columns_index] = 1 

    return Y_one_hot
    

def compute_loss(Y_one_hot, A2):

    m = A2.shape[1]
    loss = -1 / m * np.sum((Y_one_hot * np.log(A2 +1e-9)))
    return loss



#Backward propagation (Backward pass, Rétropropagation)
def backward_prop():




    ### INDEX ###

    '''
    1:
        #(W1 @ X_train) cf. allusion diagnostiqueur : 
        10 diagnostiqueurs dans une pièce qui se disent :
        "Je regarde les 4 symptômes. D'après mes connaissances (mes 4 poids), le score de cette fleur est [valeur]."

        La réponse Z1 des neurones sera un tableau (10, 100) car 1 ligne/neurone et chaque neurone a 100 résultats (car 100 fleurs à analyser)
        A ça on ajoute le biais b1 (humeur ou l'a priori) de chaque diagnostiqueur
        Ex : Diagnostiqueur 7 est naturellement "pessimiste". il aura un biais (b1[7]) très négatif
        Même si le score des symptômes (W @ X) est un peu positif, son biais négatif le tirera vers le bas.
        À l'inverse, un Diagnostiqueur 2 "optimiste" (biais b1[2]) aura tendance à donner des scores élevés, même si les symptômes sont faibles.
        Le biais ajoute de la fléxibilité au neurone (Facile ou non à activer)
    
        #Pourquoi W2 = (3, 10) et non (10, 3) : 
        La shape de mon W2 sera de (3, 10) pourquoi ? Car c'est la sortie des résulats de ma première "couche" de neuronnes.
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
        DONC la seule forme possible de W2 est de (3, 10)
    
    
    2:
        Fonction d'activation. ReLu regarde si un résultat est < 0 ou > 0. 
        Si Z est négatif (-1.5) alors ReLu dira "Ce n'est pas assez fort, j'ignore" et renverra un 0, le neurone sera "éteint.
        Dans le cas contraire, si Z est positif (3.8) alors ReLu dit "Ce signal est important, je le laisse passer tel quel. 
        Il renvoie 3.8, le neurone est "allumé".
        
        Pour softmax : C'est une des fonctions d'activation standard pour la couche finale d'un problème de classification multi-classes (Ici 3 iris).
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
    
        
    3: 
        Explication de la partie "Forward propagation : 

        "def forward_prop(X, W1, b1, W2, b2):" : 
            On défini notre "chaîne de montagne" : 
            X : la matière première (les données, X_train (4x100))
            W1, b1 : les "pièces" de la machine n°1 (couche 1)
            W2, b2 : les "pièces" de la machine n°2 (couche 2)

        "Z1 = W1 @ X + b1" :
            On calcule le "score brut" de la première couche 
            "Vote pondéré" Les 10 neurones (lignes de W1) "regardent" les 4 caractéristiques de X et votent
            "W1 @ X" : c'est le vote "multiplication matricielle).
            " + b1" : C'est le "biais" ou l'a priori" de chaque neurone qui est ajouté au score.
            "Z1" : Le score "brut" (non activé) de la couche 1. (forme 10, 100)

        "A1 = Relu(Z1)" :
            Appliquer la non-linéarité (l'activation)
            C'est "l'interrupteur". On prend les scores bruts Z1 et on les passe dans la fonction "Relu".
            Tous les scores négatifs dans Z1 sont mis à 0 (l'interrupteur est "off). Tous les scores positifs restent inschangés (l'interrupteur est "On").

        "Z2 = W2 @ X + b2" : 
            Calculer le "score brut" de la couche final.
            Le "vote final". Les 3 neurones de sortie (ligne de W2) "regardent" les 10 signaux de la couche précédente (A1) et votent.
            (Attention, l'entrée ici est A1, pas X, la chaîne continue.)
            "Z2" : le "score final brut" pour les 3 classes. Sa forme est (3, 100).

        "A2 = Softmax(Z2)" :
            Convertir les scores finaux en probabilités. 
            C'est comme un "traducteur de confiance" On prend les scores finaux bruts Z2 (ex : [1.2, -09, 4.5]) et on les passe dans la fonction "Softmax".
            La fonction "Softmax" "écrase" les scores et les transforme en pourcentages dont somme fait 1 (ex [0.05, 0.01, 0.94]) 
            "A2" : le "produit fini". C'est la prédiction officielle du réseau. Sa forme est (3, 100).

        
    4:
        Cette fonction va regarder la prédiction A2 et la comparer au vraies réponses (Y_train) afin de donner une "note d'erreur" (le "coût" ou "loss")
        Transformer un simple vecteur d'indices (comme [0, 2, 1]) en une matrice complète ("traduction" de one-hot).
        On imagine un tableau d'affichage électronique vide (Y_one_hot). 
        Ce tableau a 3 lignes (une pour chaque classe : 0,1,2),
        Et "m" colonnes (une pour chaque exemple, 100 ici). 
        Y (ex : [0, 2, 1, ...]) correspond à ma liste d'instruction, me disant quelle lumière allumer dans chaque colonne.
        
        Explication du code : 
        "m =Y.size" : Combien j'ai d'instructions ? Y est ma liste d'instruction ([0, 2, 1, ...]), Y.size demande à numPy : "Combien y a-t-il d'éléments dans ce tableau ?".
        Si Y est mon Y_train (100 exemples, m vaut 100. C'est le nombre de colonnes dont le tableau d'affichage aura besoin.
        
        "colums_index = np.arrange(m)" : 
        Créer une liste de tous les numéros de colonne. np.arrange(100) crée un tableau compteur : [0, 1, 2, 3, 4, ..., 99].
        C'est très important pour l'étape de la Loss function. Avec ça on a deux listes de même taille :
        - "Y" (les lignes où allumer) : [0, 2, 1, 0, ..., 2]
        - "colums_index" (les colonnes où allumer) : [0, 1, 2, 3, ..., 99]

        "Y_one_hot = np.zeros(( 3, m))" :
        Construction du tableau d'affichage, en laissant toutes les lumières éteintes.
        On crée une "toile vierge". C'est une matrice de 3 lignes et m (100) colonnes, remplie de zéros.
        [[0., 0., 0., ..., 0.],  <-- Ligne 0
        [0., 0., 0., ..., 0.],  <-- Ligne 1
        [0., 0., 0., ..., 0.]]  <-- Ligne 2
        
        "Y_one_hot[Y, columns_index] = 1" : 
        Signal qu'on donne à numPy pour lui dire "d'allumer les lumières" en donnant les coordonées exactes.
        C'est de "l'indexation avancée" (fancy indexing). On donne à numPy deux listes (au lieu d'un nombre) entre les crochets.
        NumPy va les "zipper" (appairer) ensemble pour créer les coordonnées (ligne, colonnes) :
        1ère lumière : 
            Ligne : Y[0] (qui vaut 0)
            Colonne :  colums_index[0] (qui vaut 0)
            NumPy allume un 1 à la coordonnée (0, 0)
        
        2ème lumière :
            Ligne : Y[1] (qui vaut 2)
            Colonne : columns_index[1] (qui vaut 1)
            NumPy allume un 1 à la coordonnée (2, 1)

        3ème lumière : 
            Ligne : Y[2] (qui vaut 1)
            Colonne : columns_index[2] (qui vaut 2)
            NumPy allume un 1 à la coordonée (1, 2)

        En résultat notre tableau de zéros est modifié "en place" et ressemble à ceci (3 premiers exemples) :
            [[1., 0., 0., ..., 0.],  <-- Ligne 0
            [0., 0., 1., ..., 0.],  <-- Ligne 1
            [0., 1., 0., ..., 0.]]  <-- Ligne 2
        
        "return Y_one_hot" : 
        Le tableau est prêt, on le renvoie.
        La fonction renvoie la nouvelle matrice (3, 100) pour qu'on puisse l'utiliser après.
    '''



    
    ### Sources ###

    """
        https://youtu.be/w8yWXqWQYmU
        https://www.digitalocean.com/community/tutorials/relu-function-in-python
        https://numpy.org/devdocs/index.html
        https://www.ibm.com/think/topics/loss-function
        https://en.wikipedia.org/wiki/Softmax_function#:~:text=The%20softmax%20function%2C%20also%20known,used%20in%20multinomial%20logistic%20regression.
        https://www.geeksforgeeks.org/deep-learning/the-role-of-weights-and-bias-in-neural-networks/
        https://en.wikipedia.org/wiki/Neural_network_(machine_learning)
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
        https://medium.com/@amit25173/pandas-map-vs-apply-practical-guide-51f046a15cd9#:~:text=Think%20of%20map()%20as,want%20to%20square%20each%20one.
        https://www.ibm.com/think/topics/backpropagation
        https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/
        http://neuralnetworksanddeeplearning.com/index.html
        https://www.ibm.com/think/topics/backpropagation#:~:text=Backpropagation%20is%20a%20machine%20learning,(AI)%20%E2%80%9Clearn.%E2%80%9D
        Gemini AI pour les analogies utilisées dans les explications. (Elles me sont utiles pour une meilleure compréhension.)
    """


