import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random


#http://www.connellybarnes.com/work/class/2016/deep_learning_graphics/proj1/
ticTacToeDataSet =  pd.read_csv('tictac_single.csv', sep =',')
colnames = ['x0','x1','x2','x3','x4','x5','x6','x7','x8']
targets = ['y']

x_train, x_test, y_train, y_test = train_test_split(ticTacToeDataSet[colnames], ticTacToeDataSet[targets], test_size=0.2)


mlp = MLPClassifier(hidden_layer_sizes=(100,100,50),  max_iter=2000, activation= 'relu', learning_rate_init=0.001)
mlp.fit(x_train, y_train.values.ravel())

predicted_values = mlp.predict(x_test)
accuracy= accuracy_score(y_test, predicted_values)
print("accuracy:", accuracy)

test_value =  random.randint(0, y_test['y'].count())
predicted_value = mlp.predict(x_test[test_value:test_value+1])

print("TicTacToe values:\n", x_test[test_value:test_value+1])
print("predicted_value:", predicted_value)
print("Real value: ", y_test[test_value:test_value+1].values.ravel())

print('end')
