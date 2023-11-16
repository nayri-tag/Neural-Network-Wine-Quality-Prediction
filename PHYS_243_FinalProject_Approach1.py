#!/usr/bin/env python
# coding: utf-8

# # Steven Acevez 862397425

# # Kenny Courser 861136315 

# # Nayri Tagmazian 861260091

# # Physics 243 Final Project Code

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, SparseCategoricalAccuracy

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('WineQT.csv')
df.head()


# In[3]:


df = df.drop('Id', axis=1)


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df['quality'].value_counts()


# In[8]:


X = df.drop(columns=['quality'])
y = df['quality']


# In[9]:


# Create a SMOTE instance
smote = SMOTE(random_state=42)

# Apply SMOTE to the training data
X_sm, y_sm = smote.fit_resample(X, y)


# In[10]:


y_sm.shape


# In[11]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[21]:


# Define the build function for the KerasClassifier
def build_classifier(layers, hidden_units, activation, optimizer, metrics):
    classifier = Sequential()
    classifier.add(Dense(units=hidden_units, activation=activation, input_dim=X_train.shape[1]))
    for _ in range(layers - 1):
        classifier.add(Dense(units=hidden_units, activation=activation))
    classifier.add(Dense(units=len(y.unique()), activation='softmax'))
    classifier.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=metrics)
    return classifier

# Create a KerasClassifier with build_classifier function
classifier = KerasClassifier(build_fn=build_classifier)

# Define hyperparameters grid for GridSearchCV
param_grid = {
    'layers': [1, 2, 3],
    'hidden_units': [16, 32, 64, 128, 256],
    'activation': ['relu', 'sigmoid', 'tanh', 'linear'],
    'batch_size': [16, 32, 64, 128, 256],
    'optimizer': ['adam', 'sgd', 'adagrad', 'adadelta', 'rmsprop'],
    'metrics' : [Accuracy(), CategoricalAccuracy(), SparseCategoricalAccuracy()]
}

# Perform grid search
grid_search = RandomizedSearchCV(estimator=classifier, param_distributions=param_grid, scoring='accuracy', cv=3)
grid_search = grid_search.fit(X_train_scaled, y_train)

# Get the best model and its accuracy
best_model = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

print("Best Model Hyperparameters:")
print(best_model.get_params())
print("Best Accuracy:", best_accuracy)


# In[22]:


# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# Feature Importance Analysis:
# For simplicity, feature weights are calculated from the final layer of the neural network.
feature_weights = best_model.model.layers[-1].get_weights()[0]
feature_importances = np.sum(np.abs(feature_weights), axis=1)
sorted_features = np.argsort(feature_importances)[::-1]

print("Feature Importance Ranking:")
for idx, feature_name in enumerate(X.columns):
    print(f"{feature_name}: {feature_importances[idx]}")

