import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#https://www.kaggle.com/fivethirtyeight/the-ultimate-halloween-candy-power-ranking/
halloweenCandyDataSet =  pd.read_csv('HalloweenCandy.csv', sep =',')
colnames = ['fruity','caramel','peanutyalmondy','nougat','crispedricewafer','hard','bar','pluribus','sugarpercent','pricepercent','winpercent']
targets = ['chocolate']

x_train, x_test, y_train, y_test = train_test_split(halloweenCandyDataSet[colnames], halloweenCandyDataSet[targets], test_size=0.3)

mlp = MLPClassifier(max_iter=2000, learning_rate_init=0.001, solver='sgd')
mlp.fit(x_train, y_train.values.ravel())

predicted_values = mlp.predict(x_test)
accuracy= accuracy_score(y_test, predicted_values)
print("accuracy:", accuracy)

predicted_value = mlp.predict(x_test[0:1])
print("predicted_value:", predicted_value)
print("Real value: ", y_test[0:1].values.ravel())

print('end')