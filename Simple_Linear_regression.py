# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# read csv file
dataset = pd.read_csv('Salary_Data.csv')

# slicing the dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

# spliting the dataset into traning and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)


# train the model(fit)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# check the prediction on X_test base
y_pred = regressor.predict(X_test)

# R2_score value
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(score)



# plotting the traning set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Traning set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()



# plotting the test set result

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

###########################  Prediction ###########################
regressor.predict([[4000]])

# save model
joblib.dump(regressor, 'Simple_regression')
 
#load the model from disk
KNN_model = joblib.load('Simple_regression')