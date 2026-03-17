import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#Import et nettoyage du DataSet 
data = pd.read_csv(r"ds/Iris.csv")

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

    history_loss = []

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
            history_loss.append(loss)
            print(f"Iteration {i} : Loss = {loss}")

   
    return W1, b1, W2, b2, history_loss

#Connaître la précision du model
def get_predictions(A2):
    # On transforme les probas [0.1, 0.8, 0.1] en décision [1]
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    # On compare la décision avec la réalité et fait la moyenne
    print(predictions, Y)#nb: Optionnel : pour voir les devinettes vs réalité
    return np.sum(predictions == Y) / Y.size

print("Début de l'entraînement...")

W1, b1, W2, b2, history_loss = gradient_descent(X_train, Y_train, learning_rate=0.1, iterations=1000)

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

plt.plot(history_loss)
plt.title("Évolution de l'erreur (Loss) pendant l'entraînement")
plt.xlabel("Centaines d'itérations")
plt.ylabel("Erreur")
plt.savefig("courbe_apprentissage.png")
print("Graphique sauvegardé sous le nom 'courbe_apprentissage.png'")


