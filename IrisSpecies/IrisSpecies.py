import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

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


input_size = 4
nets_size_layer_1 = 10
nets_size_layer_2 = 10
nets_size_layer_3 = 10

output_size = 4

inputs = tf.placeholder(tf.float32,[None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

################################LAYER_1#################################
nets_1 = tf.get_variable("nets_1", [input_size, nets_size_layer_1] )
biases_net_1 = tf.get_variable("biases_1", [nets_size_layer_1])
############################OUTPUT_LAYER_1##############################
outputs_1 = tf.nn.relu( tf.matmul(inputs, nets_1) + biases_net_1)


################################LAYER_2#################################
nets_2 = tf.get_variable("nets_2", [nets_size_layer_1, nets_size_layer_2])
biases_2 = tf.get_variable("biases_2", [nets_size_layer_2])
############################OUTPUT_LAYER_2##############################
outputs_2 = tf.nn.relu( tf.matmul(outputs_1, nets_2) + biases_2 )






