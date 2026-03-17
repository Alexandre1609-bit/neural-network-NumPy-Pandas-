# Condensé de mes notes sur le projet (annexes) et des sources utilisées :

> // à remettre au propre

---

## Index

---

### 1. Les poids, biais et dimensions des couches

`#(W1 @ X_train)` cf. allusion diagnostiqueur :

10 diagnostiqueurs dans une pièce qui se disent :
_"Je regarde les 4 symptômes. D'après mes connaissances (mes 4 poids), le score de cette fleur est [valeur]."_

La réponse Z1 des neurones sera un tableau `(10, 100)` car 1 ligne/neurone et chaque neurone a 100 résultats (car 100 fleurs à analyser).
À ça on ajoute le biais `b1` (humeur ou l'a priori) de chaque diagnostiqueur.

- **Ex :** Diagnostiqueur 7 est naturellement "pessimiste". Il aura un biais (`b1[7]`) très négatif. Même si le score des symptômes (`W @ X`) est un peu positif, son biais négatif le tirera vers le bas.
- À l'inverse, un Diagnostiqueur 2 "optimiste" (biais `b1[2]`) aura tendance à donner des scores élevés, même si les symptômes sont faibles.
- Le biais ajoute de la flexibilité au neurone (Facile ou non à activer).

---

#### Pourquoi W2 = (3, 10) et non (10, 3) ?

La shape de mon `W2` sera de `(3, 10)`. Pourquoi ? Car c'est la sortie des résultats de ma première "couche" de neurones.
La première couche se base sur les 4 caractéristiques de chaque iris. Ma deuxième couche, elle, sort les "résultats" donc elle fait un choix parmi les 3 iris du DataSet, d'où le `(3, 10)`.

`(3, 10)` et non `(10, 3)` CAR l'objectif de la 2ème couche est de calculer `Z2 = (W2 * A1) + b2`.

`A1` est le résultat de la première couche Z1. Z1 (donc A1) a été calculé par `W1 (10, 4) @ X (4, 100)` — la forme de A1 est donc de `(10, 100)` : 10 neurones, 100 exemples. C'est le point de départ.

Notre sortie `Z2` sera le résultat de notre deuxième "couche" avec 3 neurones de sortie (un pour chaque classe d'iris), et il doit le faire pour 100 exemples. Donc la forme de `Z2` sera de `(3, 100)`. C'est notre objectif.

On doit trouver `W2` pour que notre équation soit vraie :

> Shape de Z2 = Shape de W2 @ Shape de A1 → `(3, 100) = (??, ??) @ (10, 100)`

Pour multiplier `(A, B) @ (B, C)`, la dimension intérieure de B doit être identique. Le résultat sera `(A, C)`.

Si on applique ça à notre équation : `(??, ??) @ (10, 100)`.
La dimension de W2 doit correspondre à la première dimension de A1. Donc `W2 = (??, 10)`.
Le résultat sera `(??, 100)`. On veut un résultat de forme `(3, 100)`.
Si le résultat est de `(??, 100)` et qu'on veut `(3, 100)` alors `??` doit être `3`.

**DONC la seule forme possible de W2 est de `(3, 10)`.**

---

### 2. Les fonctions d'activation : ReLU et Softmax

#### ReLU

La fonction ReLU regarde si un résultat est `< 0` ou `> 0`.

- Si Z est **négatif** (ex : `-1.5`) → ReLU dira _"Ce n'est pas assez fort, j'ignore"_ et renverra un `0`. Le neurone sera **"éteint"**.
- Si Z est **positif** (ex : `3.8`) → ReLU dit _"Ce signal est important, je le laisse passer tel quel"_. Il renvoie `3.8`, le neurone est **"allumé"**.

#### Softmax

C'est une des fonctions d'activation standard pour la couche finale d'un problème de classification multi-classes (ici 3 iris). On peut voir cette fonction comme un "traducteur" qui convertit les scores bruts en bulletin de confiance.

La couche finale calcule ses scores bruts `Z2`. Pour une fleur, ces scores (logits) pourraient être par exemple :
`[Score Setosa : 1.2]`, `[Score Versicolor : -0.8]`, `[Score Virginica : 4.1]`.

Le problème est que nous ne savons pas interpréter ces scores. `4.1` est élevé mais est-ce 50% de confiance, 90%, 100% ? Et comment gérer les scores négatifs ?

**Étape 1 — "Exponentielle"**

C'est ici que Softmax intervient. La fonction rend tous les scores positifs en utilisant `np.exp()`.
Donc nos scores deviennent `[3.32, 0.45, 60.34]`. On voit que l'écart entre le score le plus haut et les autres est énorme.

**Étape 2 — "Normalisation"**

Ensuite la fonction divise chaque score par la somme de tous les scores (`3.32 + 0.45 + 60.34 = 64.11`). On obtient :

| Classe     | Calcul        | Résultat |
| ---------- | ------------- | -------- |
| Setosa     | 3.32 / 64.11  | 5%       |
| Versicolor | 0.45 / 64.11  | 0.7%     |
| Virginica  | 60.34 / 64.11 | 94%      |

Les résultats nous montrent une prédiction claire : `[0.05, 0.007, 0.94]`. C'est un vecteur de probabilité dont la somme fait **1** !

On devient donc en mesure d'interpréter les résultats (pourcentages de confiance, pas des scores bruts).
C'est aussi "soft" : un Hard Max aurait donné `[0, 0, 1]`. Avec Softmax c'est plus nuancé.
On voit qu'il y a 94% de chance que ce soit Virginica mais qu'il y ait aussi 5% de chance que ce soit Setosa.
**Cette nuance est vitale pour l'apprentissage. (Rétropropagation)**

---

### 3. La Forward Propagation

```python
def forward_prop(X, W1, b1, W2, b2):
```

On définit notre "chaîne de montagne" :

- `X` : la matière première (les données, X_train `(4x100)`)
- `W1, b1` : les "pièces" de la machine n°1 (couche 1)
- `W2, b2` : les "pièces" de la machine n°2 (couche 2)

---

```python
Z1 = W1 @ X + b1
```

On calcule le "score brut" de la première couche — "Vote pondéré" :

- Les 10 neurones (lignes de W1) "regardent" les 4 caractéristiques de X et votent.
- `W1 @ X` : c'est le vote (multiplication matricielle).
- `+ b1` : c'est le "biais" ou l'"a priori" de chaque neurone qui est ajouté au score.
- `Z1` : le score "brut" (non activé) de la couche 1. (forme `10, 100`)

---

```python
A1 = Relu(Z1)
```

Appliquer la non-linéarité (l'activation) — "L'interrupteur" :

- On prend les scores bruts `Z1` et on les passe dans la fonction ReLU.
- Tous les scores négatifs dans `Z1` sont mis à `0` (l'interrupteur est **"off"**).
- Tous les scores positifs restent inchangés (l'interrupteur est **"On"**).

---

```python
Z2 = W2 @ A1 + b2
```

> ⚠️ Attention, l'entrée ici est `A1`, pas `X`. La chaîne continue.

Calculer le "score brut" de la couche finale — "Le vote final" :

- Les 3 neurones de sortie (lignes de W2) "regardent" les 10 signaux de la couche précédente (`A1`) et votent.
- `Z2` : le "score final brut" pour les 3 classes. Sa forme est `(3, 100)`.

---

```python
A2 = Softmax(Z2)
```

Convertir les scores finaux en probabilités — "Traducteur de confiance" :

- On prend les scores finaux bruts `Z2` (ex : `[1.2, -0.9, 4.5]`) et on les passe dans la fonction Softmax.
- La fonction "écrase" les scores et les transforme en pourcentages dont la somme fait 1 (ex : `[0.05, 0.01, 0.94]`).
- `A2` : le "produit fini". C'est la prédiction officielle du réseau. Sa forme est `(3, 100)`.

---

### 4. La Loss Function et le One-Hot Encoding

Cette fonction va regarder la prédiction `A2` et la comparer aux vraies réponses (`Y_train`) afin de donner une "note d'erreur" (le "coût" ou "loss").

On transforme un simple vecteur d'indices (comme `[0, 2, 1]`) en une matrice complète — "traduction" en One-Hot.

On imagine un **tableau d'affichage électronique vide** (`Y_one_hot`) :

- Ce tableau a **3 lignes** (une pour chaque classe : 0, 1, 2).
- Et **m colonnes** (une pour chaque exemple, 100 ici).
- `Y` (ex : `[0, 2, 1, ...]`) correspond à ma liste d'instructions, me disant quelle lumière allumer dans chaque colonne.

---

```python
m = Y.size
```

Combien j'ai d'instructions ? `Y` est ma liste d'instructions (`[0, 2, 1, ...]`). `Y.size` demande à NumPy : _"Combien y a-t-il d'éléments dans ce tableau ?"_. Si `Y` est mon `Y_train` (100 exemples), `m` vaut 100. C'est le nombre de colonnes dont le tableau d'affichage aura besoin.

---

```python
columns_index = np.arange(m)
```

Créer une liste de tous les numéros de colonne. `np.arange(100)` crée un tableau compteur : `[0, 1, 2, 3, 4, ..., 99]`.
C'est très important pour l'étape de la Loss function. Avec ça on a deux listes de même taille :

- `Y` (les lignes où allumer) : `[0, 2, 1, 0, ..., 2]`
- `columns_index` (les colonnes où allumer) : `[0, 1, 2, 3, ..., 99]`

---

```python
Y_one_hot = np.zeros((3, m))
```

Construction du tableau d'affichage, en laissant toutes les lumières éteintes. On crée une "toile vierge" :

```
[[0., 0., 0., ..., 0.],  <-- Ligne 0
 [0., 0., 0., ..., 0.],  <-- Ligne 1
 [0., 0., 0., ..., 0.]]  <-- Ligne 2
```

---

```python
Y_one_hot[Y, columns_index] = 1
```

Signal qu'on donne à NumPy pour lui dire "d'allumer les lumières" en donnant les coordonnées exactes.
C'est de **l'indexation avancée** (fancy indexing). On donne à NumPy deux listes (au lieu d'un nombre) entre les crochets. NumPy va les "zipper" (appairer) ensemble pour créer les coordonnées `(ligne, colonne)` :

| Lumière | Ligne (`Y[i]`) | Colonne (`columns_index[i]`) | Coordonnée allumée |
| ------- | -------------- | ---------------------------- | ------------------ |
| 1ère    | `Y[0]` = 0     | `columns_index[0]` = 0       | `(0, 0)`           |
| 2ème    | `Y[1]` = 2     | `columns_index[1]` = 1       | `(2, 1)`           |
| 3ème    | `Y[2]` = 1     | `columns_index[2]` = 2       | `(1, 2)`           |

En résultat, notre tableau de zéros est modifié "en place" et ressemble à ceci (3 premiers exemples) :

```
[[1., 0., 0., ..., 0.],  <-- Ligne 0
 [0., 0., 1., ..., 0.],  <-- Ligne 1
 [0., 1., 0., ..., 0.]]  <-- Ligne 2
```

---

```python
return Y_one_hot
```

Le tableau est prêt, on le renvoie. La fonction renvoie la nouvelle matrice `(3, 100)` pour qu'on puisse l'utiliser après.

---

### 5. La Backward Propagation

Découvrir qui est responsable de l'erreur finale (le "coût") et comment les corriger.

#### La couche de sortie (couche 2)

```python
def backward_prop(A1, A2, Y_one_hot, W1, W2, Z1, X):
```

On rassemble le "dossier" — on a besoin de trouver les coupables et pour ça on a besoin de tous les éléments :

- `A2` (prédiction) et `Y_one_hot` (vérité) : Pour trouver l'erreur de départ.
- `A1`, `Z1`, `X` : Les "archives" de ce qui est entré et sorti de chaque machine.
- `W2` : Le "plan" de la machine 2, pour voir comment l'erreur l'a traversée.

> _(Ici `W1`, `b1`, `b2` sont inutiles dans le calcul mais on les passe souvent dans une implémentation par classe)._

---

```python
m = A2.shape[1]
```

On compte le nombre de "dossiers" (exemples). On en a besoin pour faire la moyenne des responsabilités à la fin.

---

```python
dZ2 = A2 - Y_one_hot
```

**≡ `dZ2 = erreur_brute_couche_finale`**

C'est le point de départ. C'est la combinaison de Softmax + l'entropie croisée. Le "rapport d'erreur" (`dZ2`) est simplement la différence entre la prédiction (`A2`) et la vérité (`Y_one_hot`).

> **Ex :** Si `A2 = [0.1, 0.2, 0.7]` et `Y_one_hot = [0, 0, 1]`, alors `dZ2 = [0.1, 0.2, -0.3]`
> → Il y a `0.1` de trop dans la classe 0, `0.2` de trop dans la classe 1, et il nous a manqué `0.3` dans la classe 2.

---

```python
db2 = np.sum(dZ2, axis=1, keepdims=True) / m
```

**≡ `db2 = rapport_responsabilité_du_biais_b2`**

Le biais `b2` a affecté tous les exemples de la même manière. Pour trouver sa responsabilité, on fait la moyenne de toutes les erreurs (`dZ2`) horizontalement (`axis=1`, à travers les 100 exemples) pour chacun des 3 neurones.

> `keepdims=True` est crucial. Il garde la forme `(3,1)` (une "colonne") au lieu de `(3,)` (une ligne), ce qui est vital pour la mise à jour.

---

```python
dW2 = dZ2 @ A1.T / m
```

**≡ `dW2 = rapport_responsabilité_des_poids_W2`**

Ici on "blâme" `W2`. La responsabilité de `W2` est plus complexe, elle dépend de deux choses :

1. L'erreur qu'il a aidé à produire (`dZ2`).
2. Le signal qui est entré dans la machine (`A1`). _(Un signal d'entrée A1 fort aura eu plus d'impact sur l'erreur)._

Le produit matriciel `dZ2 @ A1.T` est la formule qui "croise" l'erreur de sortie avec le signal d'entrée. On divise par `m` pour faire la moyenne.

---

#### Remonter à la couche cachée (couche 1)

```python
dA1 = W2.T @ dZ2
```

**≡ `dA1 = erreur_transmise_à_la_couche_1`**

C'est le "rapport de blâme transmis". On a fini "l'enquête" de la couche 2. On remonte la chaîne. On prend l'erreur `dZ2` et on la fait passer à l'envers à travers le "plan" de `W2` (en utilisant sa transposée `W2.T`). On obtient `dA1`, qui est le "blâme" tel qu'il arrive à la sortie de la couche 1.

---

```python
dZ1 = dA1 * (Z1 > 0)
```

**≡ `dZ1 = erreur_brute_couche_1` (après "l'interrupteur" ReLU)**

On fait passer le blâme à travers "l'interrupteur ReLU". C'est la **dérivée de ReLU**.

- On regarde les "archives" `Z1` (le signal avant ReLU).
- `Z1 > 0` est une "carte" qui vaut `1` là où l'interrupteur était **"On"** (positif) et `0` là où il était **"Off"** (négatif).
- `dA1 * ...` : on multiplie le blâme `dA1` par cette carte.
  - Si l'interrupteur était **"Off"** (`0`) → le blâme est bloqué (`dA1 * 0 = 0`).
  - S'il était **"On"** (`1`) → le blâme passe (`dA1 * 1 = dA1`).

---

```python
db1 = np.sum(dZ1, axis=1, keepdims=True) / m
```

**≡ `db1 = rapport_responsabilite_du_biais_b1`**

C'est identique à `db2`. On blâme "l'A Priori" de la couche 1. On fait la moyenne de l'erreur `dZ1` horizontalement (`axis=1`) pour chacun des 10 neurones.

---

```python
dW1 = dZ1 @ X.T / m
```

On blâme `W1`. C'est identique à `dW2`. On croise l'erreur de la couche (`dZ1`) avec la "matière première" (`X`) qui est entrée dans la "machine" à l'origine.

---

```python
return dW1, db1, dW2, db2
```

**≡ `return rapports_de_responsabilité`**

"L'enquête" est terminée. On dépose le rapport final, qui contient les "ordres de correction" pour chaque "machine" et "biais".

---

### 6. La Gradient Descent (Mise à jour des paramètres)

#### Analogie : Un randonneur dans le brouillard

On est perdu sur une montagne en pleine nuit. Notre altitude correspond à "L'erreur" (Loss) — plus nous sommes haut, plus l'erreur est grande. Notre but est de descendre tout en bas, dans la vallée (là où l'erreur est proche de 0). Notre position : ce sont nos paramètres actuels (`W1`, `b1`...). On ne voit pas la vallée, on ne peut que regarder sous nos pieds.

---

**1. Le gradient (`dW`, `db`) = La pente**

L'étape précédente (`backward_prop`) a calculé `dW` et `db`. Mathématiquement, le gradient indique la direction de la montée la plus rapide. Il nous dit : _"Si tu vas par là, tu vas monter très vite"._

**2. Le signe `"-"` = La direction**

Puisque le gradient indique la montée, et qu'on veut **descendre** (réduire l'erreur), on doit aller dans le sens opposé. C'est pour cela qu'on fait une soustraction (`W - ...`).

**3. Le Learning Rate = La taille du pas**

C'est la longueur de la jambe du randonneur.

- Si le pas est **trop grand** → On risque de "sauter" par-dessus la vallée et d'atterrir sur la montagne d'en face. (L'erreur augmente).
- Si le pas est **trop petit** → On va mettre des années à descendre. (L'apprentissage est trop lent).

---

#### Explication Technique

```python
updated_W1 = W1 - (learning_rate * dW1)
```

- **`dW1`** : C'est le "rapport de responsabilité".
  - Si `dW1` est grand (ex : `5.0`) → ce poids a une énorme influence sur l'erreur. Il faut le changer beaucoup.
  - Si `dW1` est petit (ex : `0.001`) → ce poids est presque parfait, on n'y touche pas.
- **`learning_rate * dW1`** : C'est la correction réelle. On prend l'avis du gradient (`dW1`) mais on le "calme" un peu en le multipliant par un petit chiffre (ex : `0.01`) pour ne pas casser la machine en changeant les réglages trop brutalement.
- **`W1 - ...`** : C'est la mise à jour. On prend l'ancien réglage et on applique la correction.

> **Résumé :** Cette fonction applique la règle de l'apprentissage : _"Regarde où est l'erreur, et fait un petit pas dans la direction opposée."_ Répété 1 000 ou 10 000 fois, cela permet au réseau de trouver la solution parfaite.

---

### 7. Le Cycle Complet d'Apprentissage

#### Analogie : "Le cycle d'apprentissage"

On imagine un élève (le réseau) qui révise pour un examen :

1. **Initialisation** → Il a des connaissances de base floues.
2. **Forward** → Il passe un examen blanc.
3. **Loss** → Il compare ses réponses au corrigé pour voir ses erreurs.
4. **Backward** → Il analyse ses erreurs pour comprendre ce qu'il a mal compris.
5. **Update** → Il met à jour ses connaissances.
6. **Boucle** → Il recommence 1 000 fois.

---

## Sources

- https://youtu.be/w8yWXqWQYmU
- https://www.digitalocean.com/community/tutorials/relu-function-in-python
- https://numpy.org/devdocs/index.html
- https://www.ibm.com/think/topics/loss-function
- https://en.wikipedia.org/wiki/Softmax_function
- https://www.geeksforgeeks.org/deep-learning/the-role-of-weights-and-bias-in-neural-networks/
- https://en.wikipedia.org/wiki/Neural_network_(machine_learning)
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
- https://medium.com/@amit25173/pandas-map-vs-apply-practical-guide-51f046a15cd9
- https://www.ibm.com/think/topics/backpropagation
- https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/
- http://neuralnetworksanddeeplearning.com/index.html
- https://www.ibm.com/think/topics/backpropagation
- https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/
- Gemini AI pour les analogies utilisées dans les explications. (Elles me sont utiles pour une meilleure compréhension.)
