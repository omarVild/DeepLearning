import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

irisDataSetTotal =  pd.read_csv('dataSet/Iris.csv', sep="," )

print (irisDataSetTotal.head())

colnames = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
color_wheel = ["#FF0000", "#00FF00", "#0000FF"]
one_hot = [[1,0,0],[0,1,0],[0,0,1]]

colors = irisDataSetTotal["Species"].map(lambda x: color_wheel[x-1])
scatter_matrix(irisDataSetTotal[colnames], color = colors,  diagonal='kde', marker="*")


label_targets = irisDataSetTotal["Species"].map(lambda x: one_hot[x-1])

#print("************label_targets*************************")
#print(label_targets)

#plt.show()

x_train, x_test, y_train, y_test = train_test_split(irisDataSetTotal[colnames], label_targets, test_size=0.3)
#print('***************************')
#print(x_train)


input_size = 4
nets_size_layer_1 = 3
nets_size_layer_2 = 3
nets_size_layer_3 = 3

output_size = 3

inputs = tf.placeholder(tf.float32,  shape=[None, input_size])
targets = tf.placeholder(tf.float32,  shape=[None, output_size])

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
cross_entropy = tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
cost= optimizer.minimize(cross_entropy)

x_train, x_test, y_train, y_test = train_test_split(irisDataSetTotal[colnames].as_matrix(), irisDataSetTotal['Species'], test_size=0.3)

#print('*************X train**************')
#print(x_train)

epochs_nums = 10000
step = 1
display_step=100

init = tf.global_variables_initializer()


out_equals_target = tf.equal(tf.argmax(nets_3, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

one_hot= [[1,0,0],[0,1,0],[0,0,1]]

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)


    for epoch in range(epochs_nums):
        for i in range(len(x_train)):
            print("*************x_train[i: i+1]******************")
            print( x_train[i: i+1].shape)
            print( x_train[i: i+1])
            
            
            print("*******************************+")
            print(y_train[i: i + 1].shape)
            print(y_train[i: i + 1])
            print(y_train[i: i + 1].values)
            print("*******************************+")
            print(one_hot[y_train[i: i + 1].values[0]-1])
            y_trainTMP = [one_hot[y_train[i: i + 1].values[0]-1]]
            print(y_trainTMP)
            print("*******************************+")
            
            
            sess.run(cost, feed_dict={inputs: x_train[i: i+1], targets: y_trainTMP })
            
            #if step % display_step == 0 or step == 1:
                
              

