import numpy as np
import matplotlib . pyplot as plt
import time
from sklearn import cluster, metrics
from scipy . io import arff

path = './artificial/'
databrut = arff . loadarff ( open ( path + "spiral.arff" , 'r') )
datanp = np.array([ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ])
# Affichage en 2D
# Extraire chaque valeur de features pour en faire une liste
# Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
f0 = datanp [ : ,0 ] # tous les elements de la premiere colonne
f1 = datanp [ : ,1 ] # tous les elements de la deuxieme colonne
plt . scatter ( f0 , f1 , s = 8 )
plt . title ( " Donnees initiales " )
plt . show ()
#
# Les donnees sont dans datanp ( 2 dimensions )
# f0 : valeurs sur la premiere dimension
3
# f1 : valeur sur la deuxieme dimension
#
print ( " Appel KMeans pour une valeur fixee de k " )
tps1 = time . time ()
best_score =0.0
score =0.0
k = 1
y_score = []
x_k= []

for i in range(40):
    k +=1
    model = cluster . KMeans ( n_clusters =k , init = 'k-means++')
    model . fit (datanp)
    tps2 = time . time ()
    labels = model . labels_
    iteration = model . n_iter_
    score = metrics.silhouette_score(datanp,labels)
    if(score >= best_score):
        best_score = score
        best_k = k
        plt . scatter ( f0 , f1 , c = labels , s = 8 )
        plt . title ( " Donnees apres clustering Kmeans " )
        plt . show ()
        print ( " nb clusters = " ,k , " , nb iter = " , iteration , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ,"ms" )



    y_score.append(score)
    x_k.append(k)

    #plt . scatter ( f0 , f1 , c = labels , s = 8 )
    #plt . title ( " Donnees apres clustering Kmeans " )
    #plt . show ()
    #print ( " nb clusters = " ,k , " , nb iter = " , iteration , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ,"ms" )

plt.figure(figsize=(10,6))
plt.plot (x_k, y_score, marker = 'o', linestyle='-', color='b', label='Score en fonction du nombre de clusters k')
plt.xticks(x_k)
plt.show()
print(" best cluster : ",best_score, " best k: ", best_k)

