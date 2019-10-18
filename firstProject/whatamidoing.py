# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:43:52 2018

@author: Dillon

Just trying to figure out what I am doing with this dataset

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import random

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
# look more into tree.dot
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

##########################################################################################

# Attempting to smooth the data from the first curve

lowess = sm.nonparametric.lowess(flux_star_1, t, frac=0.1)
lowess_2 = sm.nonparametric.lowess(flux_star_2, t, frac=0.1)

# Without smoothing

plt.subplot(2,2,1)
plt.plot(t, flux_star_1, 'blue')
plt.title("Unsmoothed light curve (Exo)")
plt.xlabel("Time")
plt.ylabel("Flux")

# With Smoothing

plt.subplot(2,2,2)
plt.plot(lowess[:,0], lowess[:,1],'blue')
plt.title("Smoothed Light Curve (Exo)")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

# Without smoothing for Star 40 (No Exoplanet)

plt.subplot(2,2,3)
plt.plot(t, flux_star_2, 'red')
plt.title("Unsmoothed Light Curve (Non Exo)")
plt.xlabel("Time")
plt.ylabel("Flux")

# With smoothing for Star 40

plt.subplot(2,2,4)
plt.plot(lowess_2[:,0], lowess_2[:,1], 'red')
plt.title("Smoothed Light Curve (Non Exo)")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

plt.tight_layout()
plt.show()

##########################################################################################

# Plot 4 random exoplanet indices and 4 non-exoplanet indices

randExo = []
randNonExo = []
for i in range(4):
    randExo.append(random.randrange(1,36,1))
    randNonExo.append(random.randrange(37,5087,1))

smoothed_exos = [] # Initial run gave me randExo = [9,15,14,10]
for val in randExo:
    smoothed_exos.append(sm.nonparametric.lowess(dataset_train.iloc[val,1:].values,t,frac=0.1))

smoothed_nonExos = [] # Initial run gave me randNonExo = [4975, 256, 2701, 4027]
for val in randNonExo:
    smoothed_nonExos.append(sm.nonparametric.lowess(dataset_train.iloc[val,1:].values, t, frac=0.1))

# 4 Stars with exoplanets

plt.subplot(2,4,1)
plt.plot(smoothed_exos[0][:,0],smoothed_exos[0][:,1], 'blue')
plt.title("Smoothed LC for Star 10")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

plt.subplot(2,4,2)
plt.plot(smoothed_exos[1][:,0],smoothed_exos[1][:,1], 'blue')
plt.title("Smoothed LC for Star 16")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

plt.subplot(2,4,3)
plt.plot(smoothed_exos[2][:,0],smoothed_exos[2][:,1], 'blue')
plt.title("Smoothed LC for Star 15")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

plt.subplot(2,4,4)
plt.plot(smoothed_exos[3][:,0],smoothed_exos[3][:,1], 'blue')
plt.title("Smoothed LC for Star 11")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

# 4 stars without? Or unconfirmed...

plt.subplot(2,4,5)
plt.plot(smoothed_nonExos[0][:,0],smoothed_nonExos[0][:,1], 'blue')
plt.title("Smoothed LC for Star 4976")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

plt.subplot(2,4,6)
plt.plot(smoothed_nonExos[1][:,0],smoothed_nonExos[1][:,1], 'blue')
plt.title("Smoothed LC for Star 257")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

plt.subplot(2,4,7)
plt.plot(smoothed_nonExos[2][:,0],smoothed_nonExos[2][:,1], 'blue')
plt.title("Smoothed LC for Star 2702")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

plt.subplot(2,4,8)
plt.plot(smoothed_nonExos[3][:,0],smoothed_nonExos[3][:,1], 'blue')
plt.title("Smoothed LC for Star 4028")
plt.xlabel("Time")
plt.ylabel("Smoothed Flux")

plt.show()

##########################################################################################




