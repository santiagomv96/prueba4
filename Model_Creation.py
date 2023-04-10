import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib




# Get Mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Resahpe 2x2 the dataset 
X_train = X_train.reshape(X_train.shape[0] , X_train.shape[1]* X_train.shape[2])  
X_test = X_test.reshape(X_test.shape[0] , X_test.shape[1]* X_test.shape[2])  

#build the logistic regression
regressor = LogisticRegression(random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

joblib.dump(regressor, "modelo_entrenado.pkl")