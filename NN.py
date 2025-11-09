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
    return W1, b1

    #(W1 @ X_train) cf: allusion diagnostiqueur 
    '''10 diagnostiqueurs dans une pièce qui se disent :
     "Je regarde les 4 symptômes. D'après mes connaissances (mes 4 poids), le score de cette fleur est [valeur]."

    La réponse Z1 des neurones sera un tableau (10, 100) car 1 ligne/neurone et chaque neurone a 100 résultats
    A ça on ajoute le biais b1 (humeur ou l'a priori) de chaque diagnostiqueur
    Ex : Diagnostiqueur 7 est naturellement "pessimiste". il aura un biais (b1[7]) très négatif
    Même si le score des symptômes (W @ X) est un peu positif, son biais négatif le tirera vers le bas.
    À l'inverse, un Diagnostiqueur 2 "optimiste" (biais b1[2]) aura tendance à donner des scores élevés, même si les symptômes sont faibles.
    Le biais ajoute de la fléxibilité au neurone (Facile ou non à activer)'''
   
    