import matplotlib.pyplot as plt  
import pandas as pd  
import time
import numpy as np  
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance


df = pd.read_csv('t5.8k.csv', delimiter=" ")  
data = df.iloc[:].values

cluster_points_x = {}
cluster_points_y = {}
coor_x = []
coor_y = []

def plot_graph_init():
    df.plot(kind='scatter', x='X_value', y='Y_value', color='black')
    plt.title('Visualising Dataset')
    plt.show()


plot_graph_init()
time.sleep(600)
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')  
cluster.fit_predict(data)  
plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')  
plt.show()
for i in range(cluster.n_clusters):
    for j in range(len(cluster.labels_)):
        if(cluster.labels_[j] == i):
            coor_x.append(df.iloc[j, [0]].X_value)
            coor_y.append(df.iloc[j, [1]].Y_value)
        else:
            continue
    
    cluster_points_x[i] = coor_x
    cluster_points_y[i] = coor_y
    coor_x = []
    coor_y = []

def calculatehomogenity():
    global cluster_points_x,cluster_points_y
    distances = {}
    dist = []
    homogenity = []
    for i in range(len(cluster_points_x)):
        centroid_x = sum(cluster_points_x[i])/len(cluster_points_x[i])
        centroid_y = sum(cluster_points_y[i])/len(cluster_points_y[i])
        centroid = (centroid_x,centroid_y)
        for j in range(len(cluster_points_x[i])):
            point = (cluster_points_x[i][j],cluster_points_y[i][j])
            dist.append(distance.euclidean(point,centroid))
        
        distances[i] = dist
        dist = []        
    for i in range(len(distances)):
        homogenity.append(sum(distances[i])/len(distances[i]))

    print("The homogenity matrix is: ", homogenity)

def calculatesavgseparation():
    global cluster_points_x,cluster_points_y
    cluster_pts = []
    centers = {}
    list1 = []
    list2 = []
    for i in range(len(cluster_points_x)):
        centers[i] = ((sum(cluster_points_x[i])/len(cluster_points_x[i])),(sum(cluster_points_y[i])/len(cluster_points_y[i])))
    for i in range(len(cluster_points_x)):
        for j in range(len(cluster_points_x)):
            if(i == j):
                continue
            else:
                req = len(cluster_points_x[i])*len(cluster_points_x[j])*distance.euclidean(centers[i],centers[j])
                list1.append(req)
                req2 = len(cluster_points_x[i])*len(cluster_points_x[j])
                list2.append(req2)
    
    avg_separation = (1/(sum(list2)/2)) * sum(list1)
    print("The average separation is: ", avg_separation)

calculatehomogenity()
calculatesavgseparation()