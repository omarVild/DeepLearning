import tensorflow as tf
import pandas as pd

#this is a very basic example of a linear neuron, 
#the idea is to try to explain the most basic concept of a neural network
#https://en.wikipedia.org/wiki/Perceptron

##################prices##########################
#popcorn= $55   soda=$12    nachos=$23


porcornsDataSet =  pd.read_csv('movieTheaterDataSet.csv', sep="," )
colnames = ['popcorn','sodas','nachos']

#print(porcornsDataSet.head)

popcorn = tf.placeholder(tf.float32, shape = [1,1])
nachos = tf.placeholder(tf.float32, shape = [1,1])
sodas = tf.placeholder(tf.float32, shape = [1,1])

total_target = tf.placeholder(tf.float32,  shape=[1,1])

price_popcorn = tf.get_variable("price_popcorn", [1, 1] )
price_nachos = tf.get_variable("price_nachos", [1,1])
price_soda = tf.get_variable("price_soda", [1,1])

#####################porcorns#################################
total_popcorn = tf.matmul(popcorn, price_popcorn)
outputs_porcorns = tf.nn.relu( total_popcorn )

########################nachos#################################
total_nachos = tf.matmul(nachos, price_nachos) 
outputs_nachos = tf.nn.relu( total_nachos )

######################"#Sodas#################################
total_sodas = tf.matmul(sodas, price_soda) 
outputs_sodas = tf.nn.relu( total_sodas )

total_price_st = tf.add(total_popcorn,total_nachos)
total_price = tf.add(total_price_st,total_sodas)


squared_total = tf.square(total_target-total_price) 
loss = tf.reduce_sum(squared_total)

optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(loss)


input_test = porcornsDataSet[colnames].as_matrix()
target_test = porcornsDataSet['target'].as_matrix()




init = tf.global_variables_initializer()

max_epochs = 7


with tf.Session() as sess:
    sess.run(init)
    for epoch_counter in range(max_epochs):
        for i in range( 144 ):
            #print('*************i:' , i)
            #print(input_test[i])
            porcorn_test = [[input_test[i][0]]]
            soda_test = [[input_test[i][1]]]
            nachos_test = [[input_test[i][2]]]
            target_tmp = [[target_test[i]]]
            
            #print('******************************')
            #print('porcorn_test:', porcorn_test)
            #print('soda_test:', soda_test)
            #print('nachos_test:', nachos_test)
            
            #print('target_tmp:', target_tmp)
            
            sess.run(train, feed_dict={popcorn:porcorn_test, nachos:nachos_test, sodas:soda_test, total_target: target_tmp })
    
    predited_price_popcorn =  sess.run(price_popcorn)
    print('predited_price_popcorn:', predited_price_popcorn) 
    
    predited_price_sodas =  sess.run(price_soda)
    print('predited_price_sodas:', predited_price_sodas)
    
    predited_price_nachos =  sess.run(price_nachos)
    print('predited_price_nachos:', predited_price_nachos)
    
    #1    1    2    113
    total_price_predicted = sess.run(total_price, feed_dict={popcorn:[[1]], sodas:[[1]], nachos:[[2]]  })
    print(total_price_predicted)
    
        #1    2    3    148
    total_price_predicted = sess.run(total_price, feed_dict={popcorn:[[1]], sodas:[[2]], nachos:[[3]]  })
    print(total_price_predicted)
    
    
    
    
    
    
    
    
