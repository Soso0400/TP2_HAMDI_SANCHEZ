import numpy as np
import matplotlib . pyplot as plt
import time
from sklearn import cluster, metrics
from scipy . io import arff
import scipy . cluster . hierarchy as shc

path = './artificial/'
databrut = arff . loadarff ( open ( path + "D31.arff" , 'r') )
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

clusters = ['single','average','complete','ward']
thresholds =[0.1,1,10,50,100] 
for name_cluster in clusters:
    best_score = 0.0
    y_score =[]
    x_k =[] 
    for dist in thresholds:
        # Donnees dans datanp
        print ( " Dendrogramme ", name_cluster, " donnees initiales " )
        linked_mat = shc . linkage ( datanp , name_cluster)
        plt . figure ( figsize = ( 12 , 12 ) )
        shc . dendrogram ( linked_mat ,orientation = 'top' ,
        distance_sort ='descending',
        show_leaf_counts = False )
        plt . show ()
        # set distance_threshold ( 0 ensures we compute the full tree )
        tps1 = time . time ()
        model = cluster . AgglomerativeClustering ( distance_threshold = dist ,linkage = name_cluster , n_clusters = None )
        model = model . fit ( datanp )
        tps2 = time . time ()
        labels = model . labels_
        k = model . n_clusters_
        leaves = model . n_leaves_
        if k>1 :
            score = metrics.davies_bouldin_score(datanp,labels)
            print("score DB=",score)
            score = metrics.silhouette_score(datanp,labels)
            print("score SS=",score)

            # Affichage clustering
            plt . scatter ( f0 , f1 , c = labels , s = 8 )
            plt . title ( " Resultat du clustering " )
            plt . show ()
            print ( " nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

    # best_score = 0.0
    # y_score =[]
    # x_k =[] 
    #     # set the number of clusters
    # for k in range(2,20):
    #     tps1 = time . time ()
    #     model = cluster . AgglomerativeClustering ( linkage = name_cluster, n_clusters = k )
    #     model = model . fit ( datanp )
    #     tps2 = time . time ()
    #     labels = model . labels_
    #     kres = model . n_clusters_
    #     leaves = model . n_leaves_

    #     score = metrics.silhouette_score(datanp,labels)
    #     if (best_score<score):
    #         best_score = score
    #         print("best_score SS= ",best_score)

    #         # Affichage clustering
    #         plt . scatter ( f0 , f1 , c = labels , s = 8 )
    #         plt . title ( " Resultat du clustering " )
    #         plt . show ()
    #         print ( " nb clusters = " ,k , " , nb feuilles = " , leaves ," runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
        
        
            y_score.append(score)
            x_k.append(k)
    plt.figure(figsize=(10,6))
    plt.plot (x_k, y_score, marker = 'o', linestyle='-', color='b', label='Score en fonction du nombre de clusters k')
    plt.xticks(x_k)
    plt.show()
    

