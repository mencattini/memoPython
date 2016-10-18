# Mémo Python

## IPython
Utilisation de ```pylab```:
```
ipython3 --pylab
```

Lancement d'un script depuis le shell:
```
In [1]: %run mon_script.py
```

Recharger une lib après l'avoir modifier:
```
reload(my_lib)
```

Calculer le temps d'exécution d'une instruction:
```
x = 'folder'
y = 'fol'

%timeit x.startswith(y) # = 1000000 loops, best of 3: 210 ns per loop

%timeit x[3:] == y # 1000000 loops, best of 3: 184 ns per loop
```
On remarque que la deuxième solution est en moyenne meilleure.

## Numpy
Somme d'un tableau de booléen:
```
tab = [True, True, True, False, True, False]
np.sum(tab) # = 4
```

Change le type d'un ```array``` si possible:
```
array = np.array(["4.5","3.4","3"]) # 	= array(['4.5', '3.4', '3'], dtype='<U3')
array.astype(np.float64) # 				= array([ 4.5,  3.4,  3. ])
```

Utilisation de condition avec ```np.where```
```
array = np.random.rand(3,3)
#array([[ 0.71406375,  0.74411441,  0.16653242],
#     	[ 0.56306589,  0.14198397,  0.4568308 ],
#      	[ 0.01144967,  0.24321215,  0.39831054]])

np.where(array > 0.5, 1, 0)
#array([[1, 1, 0],
#      	[1, 0, 0],
#       [0, 0, 0]])

np.where(array > 0.5, 1, array)
#array([[ 1.        ,  1.        ,  0.16653242],
#       [ 1.        ,  0.14198397,  0.4568308 ],
#      [ 0.01144967,  0.24321215,  0.39831054]])
```


## Matplotlib
On peut ajouter du code **Latex** dans les axes et les légendes:
```
import matplotlib.pyplot as plt
import numpy as np

m = np.random.rand(10,10)

plt.imshow(m)
plt.xlabel("valeur $x$")
plt.ylabel("valeur $y$")
plt.title("Exemple avec $m_{ij} \in [0,1[ \ \forall i,j$")
plt.show()
```
Donne :
![exemple 1.1](figures/exemple1.1.png)


## Performances
Pour concaténer des ```string```:
```
strings = [str(ele) for ele in ["hello", " world", " my", " my", " name", " is", " Romain"]
res = "".join(strings) # res = "hello my my name is Romain"
```

Utilisations de ```Counter``` pour compter des mots, avec réutilisation de ``res``:
```
from collections import Counter
import re

dict = Counter(re.findall('\w+', strings))
# dict = {"hello": 1, "world": 1, "my":2 ...
```

Boucle création  et compte d'un dictionnaire:
```
for key in dictionnary:
#try except if more fast than if-then
	try:
            dictionnary[key] += 1
	except KeyError:
            dictionnary[key] = 0
```

Profiler son code:
- il faut : ```pip3 line_profiler```
- ajouter dans le fichier ```import line_profiler```
- ajouter un décorateur ```@profile``` en dessus de la fonction à profiler
- lancer le script avec ```kernprof -l -v mon_script.py```

Voici un exemple de sortie:

    0         Line     Hits  Time  Per Hit   % Time  Line Contents

    11                                           @profile
    12                                           def compute_prior(folder):
    13                                               """
    14                                               Given a folder, we compute the prior of neg and pos
    15                                               folder = "./movie-reviews-en/train/"
    16                                               """
    17                                               # we compute the number of positive reviews
    18         3         1719    573.0     52.9      number_positive = len([f for f in listdir(folder + "pos/")])
    19                                               # then the negative
    20         3         1512    504.0     46.6      number_negative = len([f for f in listdir(folder + "neg/")])
    21                                               # we add it and we have the total
    22         3            6      2.0      0.2      total = number_positive + number_negative
    23                                               # we devide to have the probabilites
    24         3            6      2.0      0.2      number_positive /= total
    25         3            1      0.3      0.0      number_negative /= total
    26                                               # we return this three numbers
    27         3            3      1.0      0.1      return [number_positive, number_negative, total]

