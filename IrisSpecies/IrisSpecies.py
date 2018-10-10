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
weights_1 = tf.get_variable("weights_1", [input_size, nets_size_layer_1] )
biases_net_1 = tf.get_variable("biases_1", [nets_size_layer_1])
nets_1 = tf.matmul(inputs, weights_1) + biases_net_1
############################OUTPUT_LAYER_1##############################
outputs_1 = tf.nn.relu( nets_1)


################################LAYER_2#################################
weights_2 = tf.get_variable("weights_2", [nets_size_layer_1, nets_size_layer_2])
biases_net_2 = tf.get_variable("biases_2", [nets_size_layer_2])
nets_2 = tf.matmul(outputs_1, weights_2) + biases_net_2
############################OUTPUT_LAYER_2##############################
outputs_2 = tf.nn.relu( nets_2 )



################################LAYER_3#################################
weights_3 = tf.get_variable("weights_3", [nets_size_layer_2, nets_size_layer_3])
biases_net_3 = tf.get_variable("biases_3", [nets_size_layer_3])
nets_3 = tf.matmul(outputs_2, weights_3) + biases_net_3


loss = tf.nn.softmax_cross_entropy_with_logits(logits=nets_3, labels=targets)





