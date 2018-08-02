import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


#autos.csv https://www.kaggle.com/orgesleka/used-cars-database

columns = ['price', 'kilometer', 'powerPS', 'yearOfRegistration']
cars = pd.read_csv('../autos.csv', usecols=columns)

print "------------------------------"
print cars.loc[cars['yearOfRegistration'].idxmax()]
print "------------------------------"

print "------------------------------"
print cars.loc[cars['price'].idxmax()]
print "------------------------------"

#Filters
cars = cars.query('(price<100000) & (price>5000)')
cars = cars.query('(yearOfRegistration<2018) & (yearOfRegistration>1900)')


cars = cars.sort_values(['yearOfRegistration'], ascending=False) 
print cars.head(10)


cars.plot(kind="scatter", x="price", y="yearOfRegistration", alpha=0.8)
scatter_matrix(cars[columns], figsize=(12,8))
plt.show()