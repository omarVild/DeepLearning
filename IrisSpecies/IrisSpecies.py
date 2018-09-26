import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

irisDataSetTotal =  pd.read_csv('dataSet/Iris.csv', sep="," )

print (irisDataSetTotal.head())

colnames = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

irisDataSet = pd.read_csv('dataSet/Iris.csv', skipinitialspace=True, usecols=colnames)

print(irisDataSet)

scatter_matrix(irisDataSet, alpha=0.2, diagonal='kde')

plt.show()

print( 'fin')