import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance

df = pd.read_csv("Dataset.csv")
dist_matrix = np.zeros((df.shape[0], df.shape[0]))
level = []
level.append(0)
sequence_no = 1
clusters = {}

def plot_graph_init():
    df = pd.read_csv('Dataset.csv')
    df.plot(kind='scatter', x='X_value', y='Y_value', color='red')
    plt.show()
    
def set_NaN():
    
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]): 
            if dist_matrix[i, j] == 0:
                dist_matrix[i, j] = float('NaN')
    
def init_dist_matrix():
    global dist_matrix
    row_count = 0
    
    for i in range(df.shape[0]):
        for j in range(row_count):
            dist_matrix[i, j] = distance.euclidean(df.iloc[i, [0, 1]], df.iloc[j, [0, 1]])
            
        row_count = row_count + 1
        
    set_NaN()

# Begin with the disjoint clustering having level L(0) = 0 and sequence number m = 0.     
# Find the least distance pair of clusters in the current clustering, say pair (r), (s), according to d[(r),(s)] = min d[(i),(j)]   where the minimum is over all pairs of clusters in the current clustering.

# Update the distance matrix, D, by deleting the rows and columns corresponding to clusters (r) and (s) and adding a row and column corresponding to the newly formed cluster. The distance between the new cluster, denoted (r,s) and old cluster(k) is defined in this way: d[(k), (r,s)] = min (d[(k),(r)], d[(k),(s)]).

def update_clusters(min_coor):
    global clusters, dist_matrix
    
    if not bool(clusters):
        clusters[dist_matrix.shape[0]-1] = min_coor
        
    else:
        keys = list(clusters.keys())
        for i in keys:
            if i >= min_coor[1] and i <= min_coor[0]:
                new_key = i-1
                old_key = i
                clusters[new_key] = clusters.pop(old_key)
                
            elif i >= min_coor[0]:
                new_key = i-2
                old_key = i
                clusters[new_key] = clusters.pop(old_key)
                
        clusters[dist_matrix.shape[0]-1] = min_coor
        

def agglomerative_clustering():
    global dist_matrix, level, sequence_no
    
    #while(dist_matrix.shape != (1, 1)):
        min_dist = np.nanmin(dist_matrix)
        min_row, min_col = np.where(dist_matrix == min_dist)
        min_coor = (min_row[0], min_col[0])

        dist_matrix = np.insert(arr=dist_matrix, obj=dist_matrix.shape[0], values=0, axis=0)
        dist_matrix = np.insert(arr=dist_matrix, obj=dist_matrix.shape[1], values=0, axis=1)
    
        new_cluster_row = dist_matrix.shape[0] - 1
        new_cluster_col = dist_matrix.shape[1] - 1
    
        for i in range(dist_matrix.shape[0]):
            dist_matrix[new_cluster_row][i] = np.nanmax([dist_matrix[min_coor[0]][i], dist_matrix[min_coor[1]][i],  dist_matrix[i][min_coor[0]], dist_matrix[i][min_coor[1]]])
        
        dist_matrix = np.delete(dist_matrix, min_coor[0], 0)
        dist_matrix = np.delete(dist_matrix, min_coor[1], 0)
        dist_matrix = np.delete(dist_matrix, min_coor[0], 1)
        dist_matrix = np.delete(dist_matrix, min_coor[1], 1)
        sequence_no = sequence_no + 1
        level.append(min_dist)
    
        set_NaN()
    
        update_clusters(min_coor)
    
init_dist_matrix()
agglomerative_clustering()

    



