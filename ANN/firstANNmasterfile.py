# artificial neuaral network



# installing theano
"""
- the theano library is an open source numerical computations library built off of numpy
- it runs on both gpu and cpu (gpu is more powerful)
"""

# installing tensorflow
"""
- this library was developed by google brain
- used mostly for research and development in deep learning
"""

# installing keras
"""
- used for deep learning with little code
- based on (runs on) theano and tensorflow
- built on top of tensorflow
- developed by machine learning scientist at google
"""

# part 1 - Data Preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

# importing the dataset
df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3 : 13].values
y = df.iloc[:, 13].values

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X1 = LabelEncoder() # encoding categorical variables
X[:, 1] = labelEncoder_X1.fit_transform(X[:, 1])
labelEncoder_X2 = LabelEncoder()
X[:, 2] = labelEncoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) # creating dummy variables because not ordinal
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # removing first column so there are only two variables rather than three


# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # scaling data
X_test = sc.transform(X_test)

# making the ANN

# importing keras and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the ANN
classifier = Sequential() #this is a class -- oop

# adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11)) # average between number of nodes in input and output layers, uniform check, activation function, input dim is number of independent variables

# adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu")) # average between number of nodes in input and output layers, uniform check, activation function, input dim is number of independent variables, rectifier activation function

# adding ouput layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
"""
changing units because output is categorical so only one output node, need to change activation function and want probabilities for the
outcome, switch to sigmoid function would need to change to softmax if more than 1 option for categorical variable."""

# compiling the ANNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
"""
-using algorithm to find the best weights
-we are using optimizer adam
-this is for stochastic gradient descent
-specifying loss function that will be used with SGC
-here loss function is same as logistic regression (which we know because of what we are trying to predict)
-if more than two categories would be different loss function
-metrics is what we use to evaluate model, this is a list variable
"""

# fitting ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
"""
-no rule of thumb really for choosing number of epochs and batch size
"""

# making predictions
y_pred = classifier.predict(X_test, )
y_pred = (y_pred > 0.5)
"""threshold tells us if it should be predicted at 1 or 0, because this method gives probabilities"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[1,1] + cm[0,1] + cm[1,0])
print(f"accuracy: {accuracy}")

# predict
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
"""use double pair of brackets for horizantol array"""
new_prediction = (new_prediction > 0.5)
print(f"prediction: {new_prediction}")

# evaluating the ann -- kfold crossvalidation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifer():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = build_classifer, batch_size = 10, nb_epoch = 100) #creating object
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

"""cv is number of folds in cross-validation, 10 is usually good, n_jobs set at -1 will allow all cpu's to be used at same time"""


"""this function builds ANN classifier just as i did above this should build architecture, not fit the data"""
