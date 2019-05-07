import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.spatial import distance

df = pd.read_csv("Dataset.csv")
dist_matrix = np.zeros((df.shape[0], df.shape[0]))

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
        
init_dist_matrix()