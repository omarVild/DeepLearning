import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

irisDataSetTotal =  pd.read_csv('dataSet/Iris.csv', sep="," )

print (irisDataSetTotal.head())

colnames = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
color_wheel = {'Iris-setosa': "#FF0000", 'Iris-versicolor': "#00FF00", 'Iris-virginica': "#0000FF"}


colors = irisDataSetTotal["Species"].map(lambda x: color_wheel.get(x))
scatter_matrix(irisDataSetTotal[colnames], color = colors,  diagonal='kde', marker="*")

plt.show()

x_train, x_test, y_train, y_test = train_test_split(irisDataSetTotal[colnames], irisDataSetTotal['Species'], test_size=0.3)
print('***************************')
print(x_train)