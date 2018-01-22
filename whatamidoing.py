# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:43:52 2018

@author: Dillon

Just trying to figure out what I am doing with this dataset

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv("exoTrain.csv")

########################################################################################

# Visualize the flux data for the first star vector

flux_star_1 = dataset_train.iloc[0,1:].values # Confirmed to have an exoplanet
t = np.arange(0, flux_star_1.size)

plt.plot(t, flux_star_1, 'r')
plt.title("Star One Flux")
plt.xlabel("Time Elapsed (unitless)")
plt.ylabel("Flux Values")
plt.show()

flux_star_2 = dataset_train.iloc[40,1:].values # Unconfirmed if has one or not
plt.plot(t, flux_star_2, 'b')
plt.title("Star Two Flux (No Exo)")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.show()

########################################################################################

# First test: Assume All flux are independent and lead to the rating of 2 or 1

x = dataset_train.iloc[:,1:].values # Should be all the flux values for every star
y = dataset_train.iloc[:,0].values # Should be the ratings for every star

# Split the data into training and validation

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25, random_state = 0)

#########################################################################################

# Attempt 1: Use a decision tree classifier

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state = 0)
tree.fit(x_train, y_train)

print("Accuracy on training: {:.3f}".format(tree.score(x_train, y_train)))
print("Accuracy on validation: {:.3f}".format(tree.score(x_val, y_val)))


new_data = pd.read_csv("exoTest.csv")
x_new = new_data.iloc[:, 1:].values
y_new = new_data.iloc[:,0].values

print("Accuracy on test: {:.3f}".format(tree.score(x_new, y_new)))

# Visualize the tree

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot",class_names=["exo", "non"], impurity= False, filled = True)

import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

##########################################################################################




