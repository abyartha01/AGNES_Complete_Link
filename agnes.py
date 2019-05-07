import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.spatial import distance

df = pd.read_csv("Dataset.csv")
dist_matrix = np.zeros((df.shape[0], df.shape[0]))
level = []
k = 0
level.append(k)
sequence_no = 1

def plot_graph():
    df = pd.read_csv('Dataset.csv')
    df.plot(kind='scatter', x='X_value', y='Y_value', color='red')
    plt.show()
    
def init_dist_matrix():
    global dist_matrix
    row_count = 0
    
    for i in range(df.shape[0]):
        for j in range(row_count):
            dist_matrix[i, j] = distance.euclidean(df.iloc[i, [0, 1]], df.iloc[j, [0, 1]])
            
        row_count = row_count + 1
        
    for i in range(df.shape[0]):
        for j in range(df.shape[0]): 
            if dist_matrix[i, j] == 0:
                dist_matrix[i, j] = float('NaN')

# Begin with the disjoint clustering having level L(0) = 0 and sequence number m = 0.     
# Find the least distance pair of clusters in the current clustering, say pair (r), (s), according to d[(r),(s)] = min d[(i),(j)]   where the minimum is over all pairs of clusters in the current clustering.

# Update the distance matrix, D, by deleting the rows and columns corresponding to clusters (r) and (s) and adding a row and column corresponding to the newly formed cluster. The distance between the new cluster, denoted (r,s) and old cluster(k) is defined in this way: d[(k), (r,s)] = min (d[(k),(r)], d[(k),(s)]).

def agglomerativeclustering():

    global dist_matrix, level,sequence_no, k
    
    min_dist = np.nanmin(dist_matrix)
    min_col, min_row = np.where(dist_matrix == min_dist)
    min_coor = (min_row[0], min_col[0])
    new_row = np.zeros(df.shape[0])
    dist_matrix = np.insert(arr=dist_matrix, obj=dist_matrix.shape[0], values=0, axis=0)
    new_cluster_index=dist_matrix.shape[0] - 1
    for i in range(df.shape[0]):
        dist_matrix[new_cluster_index][i] = np.nanmax([dist_matrix[min_coor[0]][i],dist_matrix[min_coor[1]][i],dist_matrix[i][min_coor[0]],dist_matrix[i][min_coor[1]]])
    
    print(dist_matrix)
init_dist_matrix()
agglomerativeclustering()

    



