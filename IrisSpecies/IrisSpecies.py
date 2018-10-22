import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

irisDataSetTotal =  pd.read_csv('dataSet/Iris.csv', sep="," )

print (irisDataSetTotal.head())

colnames = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']


one_hot = [[1,0,0],[0,1,0],[0,0,1]]
color_wheel = ["#FF0000", "#00FF00", "#0000FF"]
colors = irisDataSetTotal["Species"].map(lambda x: color_wheel[x-1])
scatter_matrix(irisDataSetTotal[colnames], color = colors,  diagonal='kde', marker="*")
label_targets = irisDataSetTotal["Species"].map(lambda x: one_hot[x-1])
#print("************label_targets*************************")
#print(label_targets)
#plt.show()

x_train, x_test, y_train, y_test = train_test_split(irisDataSetTotal[colnames], label_targets, test_size=0.3)


input_size = 4
nets_size_layer_1 = 5
nets_size_layer_2 = 5
output_size = 3


inputs = tf.placeholder(tf.float32,  shape=[None, input_size])
targets = tf.placeholder(tf.float32,  shape=[None, output_size])

################################LAYER_1#################################
weights_layer_1 = tf.get_variable("weights__layer_1", [input_size, nets_size_layer_1] )
biases_layer_1 = tf.get_variable("biases_1", [nets_size_layer_1])
nets_layer_1 = tf.matmul(inputs, weights_layer_1) + biases_layer_1
outputs_1 = tf.nn.relu( nets_layer_1)


################################LAYER_2#################################
weights_layer_2 = tf.get_variable("weights_layer_2", [nets_size_layer_1, nets_size_layer_2])
biases_layer_2 = tf.get_variable("biases_2", [nets_size_layer_2])
nets_layer_2 = tf.matmul(outputs_1, weights_layer_2) + biases_layer_2
outputs_2 = tf.nn.relu( nets_layer_2 )


################################LAYER_3#################################
weights_layer_3 = tf.get_variable("weights_layer_3", [nets_size_layer_2, output_size])
biases_net_3 = tf.get_variable("biases_3", [output_size])
nets_layer_3 = tf.matmul(outputs_2, weights_layer_3) + biases_net_3



###############################ACYIVATION FUNCTION######################
loss = tf.nn.softmax_cross_entropy_with_logits(logits=nets_layer_3, labels=targets)
cross_entropy = tf.reduce_mean(loss)


#############################OPTIMIZATION ALGORITHM#####################
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
cost= optimizer.minimize(cross_entropy)



x_train, x_test, y_train, y_test = train_test_split(irisDataSetTotal[colnames].as_matrix(), irisDataSetTotal['Species'], test_size=0.3)


#print('*************X train**************')
#print(x_train)

epochs_nums = 10000
step = 1
display_step=100

init = tf.global_variables_initializer()


#Start training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs_nums):
        for i in range(len(x_train)):
            print("*************train input:" , i ," *********************")
            #print("train input shape:", x_train[i].shape)
            print("train input values: ",  x_train[i])
            print("expected crude value: " , y_train[i:i+1].values[0])
            print("expected value: " , one_hot[y_train[i:i+1].values[0] -1])
            y_trainTMP = [one_hot[y_train[i:i+1].values[0] -1]]
            print("")
            print("")
            
            
            sess.run(cost, feed_dict={inputs: x_train[i:i+1], targets: y_trainTMP })
            
            
            
            #if step % display_step == 0 or step == 1:
                
              

