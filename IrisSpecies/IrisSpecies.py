import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

irisDataSetTotal =  pd.read_csv('dataSet/Iris.csv', sep="," )

print (irisDataSetTotal.head())

colnames = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
color_wheel = {'Iris-setosa': "#FF0000", 'Iris-versicolor': "#00FF00", 'Iris-virginica': "#0000FF"}


colors = irisDataSetTotal["Species"].map(lambda x: color_wheel.get(x))
scatter_matrix(irisDataSetTotal[colnames], color = colors,  diagonal='kde', marker="*")

plt.show()

print( 'fin')