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
    b1 = np.zeros((10, 1))
   
    W2 = np.random.randn(3, 10) * 0.01
    b2 = np.zeros((3,1))
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
    Y = Y.astype(int)

    m = Y.size
    columns_index = np.arange(m)

    Y_one_hot = np.zeros(( 3, m))
    Y_one_hot[Y, columns_index] = 1 

    return Y_one_hot
    

def compute_loss(Y_one_hot, A2):

    m = A2.shape[1] #nb: "m" est un tuple, une liste et non un array, donc "A2.shape[1] prendra 100 et non 3 (3, 100)"
    loss = -1 / m * np.sum((Y_one_hot * np.log(A2 +1e-9)))
    return loss


#Backward propagation (index 5) (Backward pass, Rétropropagation)
def backward_prop(A1, A2, Y_one_hot, W1, W2, b1, b2, Z1, X):
    m = A2.shape[1]

    dZ2 = A2 - Y_one_hot
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dW2 = dZ2 @ A1.T / m

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (Z1 > 0)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    dW1 = dZ1 @ X.T / m

    return dW1, db1, dW2, db2


#Update_parameters (Index 6)
def update_parameters(dW1, db1, dW2, db2, W1, b1, W2, b2, learning_rate): 

    updateW1 = W1 - (learning_rate * dW1)
    updateW2 = W2 - (learning_rate * dW2)

    updateb1 = b1 - (learning_rate * db1)
    updateb2 = b2 - (learning_rate * db2)

    return updateW1, updateW2, updateb1, updateb2

#Assemblage (Index 7)
def gradient_descent(X, Y, learning_rate, iterations):
    
    #On initialise
    W1, b1, W2, b2 = Init_params()
    
    #On prépare le "corrigé" Y
    Y_one_hot = one_hot(Y)

    #Mise en place de la boucle
    for i in range(iterations):
        
        #1- Forward : On lance la première étape et on récupère les résultats
        A1, A2, Z1, Z2 = forward_prop(X, W1, b1, W2, b2)
        
        #2- Backward : On trouve les "coupables" (gradients)
        dW1, db1, dW2, db2 = backward_prop(A1, A2, Y_one_hot, W1, W2, b1, b2, Z1, X)
        
        #3- Update : On corrige les erreurs
        W1, W2, b1, b2 = update_parameters(dW1, db1, dW2, db2, W1, b1, W2, b2, learning_rate)
        
        #4- Affichage : Tous les 100 tours on regarde le socre.
        if i % 100 == 0:
            loss = compute_loss(Y_one_hot, A2)
            print(f"Iteration {i} : Loss = {loss}")

   
    return W1, b1, W2, b2

#Connaître la précision du model
def get_predictions(A2):
    # On transforme les probas [0.1, 0.8, 0.1] en décision [1]
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    # On compare la décision avec la réalité et fait la moyenne
    print(predictions, Y)#nb: Optionnel : pour voir les devinettes vs réalité
    return np.sum(predictions == Y) / Y.size

print("Début de l'entraînement...")

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, learning_rate=0.1, iterations=1000)

print("Fin de l'entraînement !")


# 1- Prédiction finale avec les machines entraînées (W1, b1...)
Z1, A1, Z2, A2 = forward_prop(X_train, W1, b1, W2, b2)

# 2- Traduction des probas en classes (0, 1 ou 2)
predictions = get_predictions(A2)

# 3. Calcule de la note finale
accuracy = get_accuracy(predictions, Y_train)
print(f"Précision de l'entraînement : {accuracy * 100}%")

print("Test sur des données inconnues...")

#1- On utilise X_dev ici pour passer le "vrai" examen
#nb: O,n utilise les W et b qu'on vient d'entraîner
Z1_dev, A1_dev, Z2_dev, A2_dev = forward_prop(X_dev, W1, b1, W2, b2)

#2- On récupère les réponses
dev_prediction = get_predictions(A2_dev)

#On compare avec le "corrigé"
dev_accuracy = get_accuracy(dev_prediction, Y_dev)

print(f"Précision sur le jeu de test : {dev_accuracy * 100}%")



#Prochain objectif : adapter et améliorer ce réseau de neurones pour le faire fonctionner avec le "Pima Indians Diabetes Dataset"

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

    
    5:
        Découvrir qui est responsable de l'erreur finale (le "coût") et comment les corriger.
        Explication de ma fonction "backward_prop" ligne par ligne.
        
        La couche de sortie (couche 2) : 

            "def backward_prop(A1, A2, Y_one_hot, W1, W2, Z1, X)":
            On rassemble le "dossier", on a besoin de trouver les coupables et pour ça
            on a besoin de tous les éléments: 
            A2 (prédiction) et Y_one_hot (vérité) : Pour trouver l'erreur de départ.
            A1, Z1, X : Les "archives" de ce qui est entré et sorti de chaque machine.
            W2 : Le "plan" de la machine 2, pour voir comment l'erreur l'a traversée.
            (Ici W1, b1, b2 sont inutile dans le calcul mais on les passe souvent dans une implémentation par classe).

            "m = A2.shape[1]" : 
            On compte le nombre de "dossiers" (exemples). On en a besoin pour faire la moyenne des responsabilités à la fin.

            "dZ2 = A2 - Y_one_hot" : 
            C'est l'équivalent de : "dZ2 = erreur_brute_couche_finale".
            C'est le point de départ. C'est la combinaison de "Softmax" + l'entropie croisée. Le "rapport d'erreur" (dZ2) est
            simplement la différence entre la prédiction (A2) et la vérité (Y_one_hot).
            Par exemple : Si A2 = [0.1, 0.2, 0.7] et Y_one_hot = [0, 0, 1], alors dZ2 = [0.1, 0.2, -0.3]
            Traduction : Il y a 0.1 de trop dans la classe 0, 0.2 de trop dans la classe 1, et il nous a manqué 0.3 dans la classe 2.

            "db2 = num.sum(dZ2, axis=1, keepdims=True) / m" :
            C'est l'équivalent de : "db2 = rapport_responsabilité_du_biais_b2".
            La biais b2 a affecté tous les exemples de la même manière. Pour trouver sa responsabilité, on fait la moyenne de toutes
            les erreus (dZ2) horizontalement (axis=1, à travers les 100 exemples) pour chacun des 3 neurones.
            ("keepdimes=True" est crucial. Il garde la forme (3,1)(une "colonne") au lieux de (3, )(une ligne), ce qui est vital pour la maj.)

            "dW2 = dZ2 @ A1.T / m" : 
            C'est l'équivalent de : "db2 = rapport_responsabilité_des_poids_W2".
            Ici on "blâme" W2. La responsabilité de W2 est plus complexe, elle dépend de deux choses : 
            1 : L'erreur qu'il a aidé à produire (dZ2).
            2 : Le signal qui est entré dans la machine (A1). (Un singal d'entrée A1 fort aura eu plus d'impact sur l'erreur).
            Le produit matriciel "dZ2 @ A1.T" est la formule qui "croise" l'erreur de sortie avec le signal d'entrée.
            On divise par "m" pour faire la moyenne.

        Remonter à la couoche cachée (couche 1) : 

            "dA1 = W2.T @ dZ2" : 
            C'est l'équivalent de : "dA1 = erreur_transmise_à_la_couche_1".
            C'est le "rapport de blâme transmis". On a fini "l'enquête" de la couche 2. On remonte la chaîne.
            On prend l'erreur dZ2 t on la fait passer à l'envers à travers le "plan" de W2 (en utilisant sa transposée W2.T).
            On obtient dA1, qui est le "blâme" tel qu'il arrive à la sortie de la couche 1.

            "dZ1 = dA1 * (Z1 > 0) : 
            C'est l'équivalent de : dZ1 = erreur_brute_couche_1 (après "l'interrupteur" ReLu)
            On fait passer le blâme à travers "l'interrupteur ReLu".
            C'est la dérivée de ReLu. On regarde les "archives" Z1 (Le signal avant ReLU).
            "Z1 > 0" est une "carte" qui vaut 1 là ou l'interrupteur étati "On" (positif) et 0 là où il était "Off" (négatif)
            "dA1 * ..." On multiplie le blâme dA1 par cette carte.
            Résultat : Si l'interrupteur était "Off" (0), le blâme est bloqué (dA1 * 0 = 0). S'il était "On" (1), le blâme passe (dA1 * 1 = dA1).

            "db1 = np.sum(dZ1, axis=1, keepdims=True) / m" : 
            C'est l'équivalent de : "db1 = rapport_responsabilite_du_biais_b1".
            C'est identique à db2. On blâme "lA Priori" de la couche 1.
            On fait la moyenne de l'erreur dZ1 horizontalement (axis=1) pour chacun des 10 neurones.

            "dW1 = dZ1 @ X.T / m" : 
            On blâme W1.
            C'est identique à dW2. On croise l'erreur de la couche "dZ1" avec la "Matière première" (X) qui est entrée dans la "machine" à l'origine.

            "return dW1, db1, dW2, db2" :
            C'est l'équivalent de "return rapports_de_responsabilité".
            "L'enquête" est terminée. On dépose le rapport final, qui contient les "ordres de correction" (d)
            pour chaqye "machine" et "biais".


    6:
        Analogie : Un randonneur dans le Brouillard :

            On est perdu sur une montagne en pleine nuit. Notre altitude corespond à "L'erreur" (Loss), plus nous sommes haut, plus l'erreur est grande.
            Notre but est de descendre tout en bas, dans la vallée (là ou l'erreur est proche de 0).
            Notre position : Ce sont nos paramètres actuels (W1, b1...). On ne vois pas la vallée, on ne peux que regarder sous nos pieds.

        1: Le gradient (dW, db) = La pente :

            L'étape précédente (backward_prop) a calculé dW et db. Mathématiquement, le gradient indique la direction de la montée la plus rapide.
            Il nous dit: "Si tu vas par là, tu vas monter très vite".

        2: Le signe "-" = La direction : 

            Puisque le gradient indique la montée, et qu'on veut descendre (réduire l'erreur), on doit aller dans le sens opposé.
            C'est pour cela qu'on fait une soustraction (W - ...)

        3: Le learning Rate = La taille du pas : 

            C'est la longueur de la jambe du randonneur. 
            Si le pas est trop grand : On risque de "sauter" par-dessus la valée et d'atterir sur la montagne d'en face. (l'erreur augmente).

            Si le pas est trop petit : On va mettre des années à descendre (l'apprentissage est trop lent.)
            
        Explication Technique : 

            "updateW1 = W1 - (learning_rate * dW1)" :
            "dW1" : C'est le "rapport de responsabilité".
                Si dW1 est grand (ex:5.0), cela veut dire que ce poids a une énorme influence sur l'erruer. Il faut le changer beaucoup.
                
                Si dW1 est petit (ex:0.001), ce poids est presque parfait, on n'y touche pas.

            "learningèrate * dW1" : 
            C'est la correction réelle. On prend l'avis du gradient (dW1) mais on le "calme" un peu en le multipliant par un petit chiffre (ex:0.01)
            pour ne pas casser la machine en changeant les réglages trop brutalement.

            "W1 - ..." : 
            C'est la mise à jour. On prend l'ancien réglage et on applique la correction.

        Pour résumer cette fonction applique la règle de l'apprentissage : "Regarde où est l'erreur, et fait un petit pas dans la direction opposée".
        Répété 1000, 10 000 fois permet au réseau de trouver la solution parfaite.

    7:
        Analogie "Le cycle d'apprentissage"
        On imagine un élève (le réseau) qui révise pour un examen : 
        - Il a des conaissances de bases floues (Initialisation)
        - Il passe un examen blanc (Forward)
        - Il compare ses réponses au corrigé pour voir ses erreurs (Loss)
        - Il analyse ses erreurs pour comprendre ce qu'il a mal compris (Backward)
        - Il met à jour ses erreurs pour comprendre ce qu'il a mal compris (Update)
        - Il recommence 1000 fois (Boucle)

        
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
        https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
        Gemini AI pour les analogies utilisées dans les explications. (Elles me sont utiles pour une meilleure compréhension.)
"""


