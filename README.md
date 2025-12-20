Réseau de Neurones Artificiel "From Scratch" (Iris Dataset)

  Ce projet est une implémentation complète d'un Réseau de Neurones Artificiel (ANN) codé entièrement à la main en Python, utilisant uniquement NumPy pour les calculs matriciels.
  Aucune librairie de Deep Learning "boîte noire" (comme TensorFlow, PyTorch ou Keras) n'a été utilisée. L'objectif était de déconstruire et de coder les mathématiques fondamentales derrière l'intelligence artificielle : la propagation avant, le calcul de l'erreur (Loss), la rétropropagation (Backpropagation) et la descente de gradient.
  
  
Objectif 

  Classer les fleurs d'Iris en 3 espèces (Setosa, Versicolor, Virginica) en fonction de 4 caractéristiques physiques (longueur/largeur des pétales et sépales)


Technologies Utilisées

  Python 3.x
  NumPy : Pour l'algèbre linéaire, les manipulations matricielles et les calculs de gradients.
  Pandas : Pour le chargement, le nettoyage et la préparation du Dataset (.csv).


Architecture du Réseau

  Le modèle est un Perceptron Multicouche (MLP) avec l'architecture suivante :
  Couche d'Entrée (Input Layer) : 4 neurones (correspondant aux 4 features du dataset).
  Couche Cachée (Hidden Layer) : 10 neurones.
    Fonction d'activation : ReLU (Rectified Linear Unit).
  Couche de Sortie (Output Layer) : 3 neurones (correspondant aux 3 classes d'Iris).
    Fonction d'activation : Softmax (pour obtenir des probabilités).
    
  Formules Clés :
    Fonction de Coût (Loss) : Entropie Croisée Catégorielle (Categorical Cross-Entropy).
    Optimiseur : Descente de Gradient (Batch Gradient Descent).


Structure du Code

  Le code est structuré de manière modulaire pour imiter le fonctionnement d'un framework : 
  init_params() : Initialisation aléatoire des poids ($W$) et des biais ($b$).
  forward_prop() : Qui calcule les prédictions ($Z$, $A$).
  backward_prop() : Qui calcule les gradients ($dW$, $db$) via la règle de dérivation en chaîne.
  update_parameters() : La mise à jour des poids selon le Learning Rate.
  gradient_descent() : La boucle d'entraînement principale.


Installation et Utilisation

```bash
Cloner le projet :
  git clone https://github.com/Alexandre1609-bit/neural-network-NumPy-Pandas-
  
Installer les dépendances :

  pip install numpy pandas matplotlib
  
Lancer l'entraînement :
  python NN.py
```
  
Résultats

Le modèle atteint une précision remarquable après 1000 itérations :
  Précision Entraînement : ~98% - 100%
  Précision Test (Généralisation) : ~96% - 98%
  Loss final : < 0.1

Cela démontre que le modèle a correctement "appris" la logique de classification sans surapprentissage (overfitting) majeur.


Ce que j'ai appris

  Ce projet m'a permis de comprendre en profondeur l'importance des dimensions matricielles $(N_{out}, N_{in})$ dans les réseaux de neurones. 
  Le fonctionnement de la Rétropropagation (Backpropagation) et comment l'erreur se propage à travers les couches.
  Le rôle des fonctions d'activation non-linéaires (ReLU, Softmax).
  La manipulation avancée de données avec NumPy (Broadcasting, Indexation, Vectorisation).
  
  
Projet réalisé dans le cadre d'un auto-apprentissage guidé sur les fondements du Deep Learning.
