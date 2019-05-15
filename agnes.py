import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance

df = pd.read_csv("Dataset2.csv")
dist_matrix = np.zeros((df.shape[0], df.shape[0]))
level = {}
sequence_no = 1
clusters = {}
for i in range(dist_matrix.shape[0]):
    clusters[i] = [i]
keys = list(clusters.keys())
final_clusters = {}


def plot_graph_init():
    df.plot(kind='scatter', x='X_value', y='Y_value', color='black')
    plt.title('Visualising Dataset')
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
    global clusters, dist_matrix, keys
    
    temp_list = clusters[min_coor[1]].copy()

    for i in keys:
        if i < min_coor[0] and i >= min_coor[1]:
            if len(clusters[i]) == 1:
                if abs(min_coor[0]-min_coor[1]) == 1:
                    clusters[i] = clusters[i+2].copy()
                else:
                    clusters[i] = clusters[i+1].copy()
            else:
                if abs(min_coor[0]-min_coor[1]) == 1:
                    clusters[i].extend(clusters[min_coor[1]])
                else:
                    clusters[i].extend(clusters[min_coor[1]])        

        elif i >= min_coor[0] and (i <= dist_matrix.shape[0]-2 or dist_matrix.shape == (2, 2)):
            if len(clusters[i]) == 1 and i+2 <= dist_matrix.shape[0]:
                clusters[i] = clusters[i+2].copy()
            else:
                for item in temp_list:
                    clusters[i].append(item)
               
    if(dist_matrix.shape != (2, 2)):
        clusters[dist_matrix.shape[0]-1] = list(min_coor)
        

def update_level(min_coor, min_dist):
    global level, clusters, dist_matrix, sequence_no
    
    if min_dist in level:
        min_dist = min_dist + 0.0000000001
        
    level[min_dist] = [clusters[min_coor[0]][i] for i in range(len(clusters[min_coor[0]]))]
    temp_list = [clusters[min_coor[1]][i] for i in range(len(clusters[min_coor[1]]))]
    for item in temp_list:
        level[min_dist].append(item)

def agglomerative_clustering():
    global dist_matrix, level, sequence_no
    
    while(dist_matrix.shape != (1, 1)):
        min_dist = np.nanmin(dist_matrix)
        min_row, min_col = np.where(dist_matrix == min_dist)
        min_coor = (min_row[0], min_col[0])

        dist_matrix = np.insert(arr=dist_matrix, obj=dist_matrix.shape[0], values=0, axis=0)
        dist_matrix = np.insert(arr=dist_matrix, obj=dist_matrix.shape[1], values=0, axis=1)
    
        new_cluster_row = dist_matrix.shape[0] - 1
    
        for i in range(dist_matrix.shape[0]):
            if i == new_cluster_row:
                dist_matrix[new_cluster_row][i] = 0
                continue
        
            dist_matrix[new_cluster_row][i] = np.nanmax([dist_matrix[min_coor[0]][i], dist_matrix[min_coor[1]][i],  dist_matrix[i][min_coor[0]], dist_matrix[i][min_coor[1]]])
        
        dist_matrix = np.delete(dist_matrix, min_coor[0], 0)
        dist_matrix = np.delete(dist_matrix, min_coor[1], 0)
        dist_matrix = np.delete(dist_matrix, min_coor[0], 1)
        dist_matrix = np.delete(dist_matrix, min_coor[1], 1)
    
        set_NaN()

        update_level(min_coor, min_dist)
        sequence_no = sequence_no + 1
        update_clusters(min_coor)
    

def determine_clusters():
    global level, threshold, df, final_clusters
    temp = []
    sorted_keys = sorted(level)
    sorted_keys.reverse()
    
    for i in sorted_keys:
        if i < threshold:
            key = sorted_keys.index(i)
            break
            
    j = 1
    try:
        while(len(temp) != df.shape[0]):
            final_clusters[j] = level[sorted_keys[key]]
            temp.extend(level[sorted_keys[key]])  
            j = j + 1
            key = key + 1
    except:
        for i in range(df.shape[0]):
            if i not in temp:
                final_clusters[j] = [i]
                j = j + 1
    
    
def plot_graph_final():
    global final_clusters, df
    colors = ['red', 'blue', 'green', 'yellow', 'pink', 'orange', 'brown']
    points = []
    
#    fig = plt.figure(figsize=(5, 5))
    
    for i in range(len(final_clusters)):
        points.extend(final_clusters[i+1])
        #plt.scatter(final_clusters[i+1], color=colors[i])
        for j in points:
            plt.scatter(x=list(df.iloc[j, [0]]), y=list(df.iloc[j, [1]]) ,color=colors[i])
            
        points = []
        
    plt.xlim(df['X_value'].min()-1, df['X_value'].max()+1)
    plt.ylim(df['Y_value'].min()-1, df['Y_value'].max()+1)
    plt.xlabel('X_value')
    plt.ylabel('Y_value')
    plt.title('Visualising Clusters')
    
    plt.show()


plot_graph_init()
init_dist_matrix()
agglomerative_clustering()

threshold = float(input("Enter the threshold value you want: "))

determine_clusters()
plot_graph_final()


