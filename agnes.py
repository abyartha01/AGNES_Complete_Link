import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Dataset.csv')
df.plot(kind='scatter',x='X_value',y='Y_value',color='red')
plt.show()