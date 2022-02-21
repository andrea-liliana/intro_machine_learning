#%%
"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms

"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score #to verify results
from sklearn.model_selection import GridSearchCV 

from data import make_data1, make_data2
from plot import plot_boundary

from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from itertools import product


import plotly.graph_objects as go
import plotly.io as pio





def print_scores_knn(x_LS,y_LS,x_TS,y_TS,number):

    score_values_LS=[]
    score_values_TS=[]
    neighbors=[]
    
    for i in range (1,7):
        
        if i==1:
            ng=1
        elif i==2:
            ng=5
        elif i==3:
            ng=10
        elif i==4:
            ng=75
        elif i==5: 
            ng=100
        elif i==6:
            ng=150
     
        neighbors.append(ng)
        
        knn_test= KNeighborsClassifier(n_neighbors=ng) #if nothing max_depth='none'
        
        #train model
        knn_test = knn_test.fit(x_LS, y_LS)

        
        score_test=KNeighborsClassifier.score(knn_test,x_TS,y_TS,)
        score_learning=KNeighborsClassifier.score(knn_test,x_LS,y_LS,)
        
        score_values_LS.append(score_learning)
        score_values_TS.append(score_test)
        

        
        neigh=str(ng)

        plot_boundary('knn'+number+'_'+neigh,knn_test, x_LS, y_LS, mesh_step_size=0.1, title="", inline_plotting=False)
        
        
    print('Numbers of neighbours',neighbors)
    print('Score values for the learning sample',score_values_LS)
    print('Score values for the test sample',score_values_TS) 
    print(" ")
    
           :
    return knn_test  

# Calculate test accuracies 
# Returns the plot of accuracy vs number of neighbors

def Test_Accuracy(k_range, n, x_LS, y_LS, x_TS):
    k_range = list(range(1,n,1))
    
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_LS, y_LS)
        y_pred = knn.predict(x_TS)
        scores.append(accuracy_score(y_TS, y_pred))
    plt.plot(k_range, scores, 'o' )
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    opt_neig = np.argmax(scores) +1 
    print(opt_neig)
    return plt.show(), opt_neig 



def MakeTable(x,y):
    pio.renderers.default = 'browser'

    fig = go.Figure(data=[go.Table(header=dict(values=x), cells=dict(values=y))  ])
    return fig.show()

def Mean(lst): 
    return sum(lst) / len(lst) 




if __name__ == "__main__":
    pass # Make your experiments here
    plt.rcParams.update({'font.size': 20})

# (1) Decision Boundary for {1,5,10,75,100,150}

# in X both inputs , in Y binary output ( 0 or 1)
    [x_LS1,y_LS1,x_TS1,y_TS1]=make_data1(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)

    print("For the first data set:")
    model1=print_scores_knn(x_LS1,y_LS1,x_TS1,y_TS1,'1')

    [x_LS2,y_LS2,x_TS2,y_TS2]=make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)
    print(" ")

    print("For the second data set:")
    model2=print_scores_knn(x_LS2,y_LS2,x_TS2,y_TS2,'2')


# (2) 5-K fold cross validation for the second data set

[x_LS,y_LS,x_TS,y_TS]=make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)
x_LS_splitted  = np.split(x_LS, 5)
x_TS_splitted  = np.split(x_TS, 5)
y_LS_splitted  = np.split(y_LS, 5) 
y_TS_splitted  = np.split(y_TS, 5)
print(x_LS_splitted[0])



drop = 1
array =[]
for i in range(5):
    if i!= drop:
        for e in x_LS_splitted[i]:
            array.append(e)


# Creating odd list K for KNN
neighbors = [1,5,10,75,100,150]
kfold = [1,2,3,4,5]

# Cross Validation :

optimal_value = []
mean_score = []
y_prediction = []
#for K in neighbors:
for i in range(5):
    # Save the cross validation error
    cv = [ ]

    X_train = []
    y_train = []
    X_test = []
    y_test = []    
    for l in range(5):
        if l!= i:
            for a,b,c,d in zip(x_LS_splitted[l] , y_LS_splitted[l] , x_TS_splitted[l] , y_TS_splitted[l]) :
                X_train.append(a)
                y_train.append(b)               
                X_test.append(c)
                y_test.append(d)    

    #for i in kfold :  

    scores = []

    for K in neighbors:
        knn = KNeighborsClassifier(n_neighbors = K)


        knn.fit( X_train, y_train )
        y_pred = knn.predict(X_test) 

        acc = accuracy_score(y_test,y_pred)

        #y_prediction[i] = y_prediction.append( knn.predict(x_TS_splitted[i]))
              
        print("Accuracy:{}".format(acc), "for ", K, "neighbors")

        scores.append(acc)
        #score of cross validation  
    

    print("Mean accuracy : ",np.mean(scores))
    mean_score.append(np.mean(scores))
    print("The optimal number of neighbors is {}".format(neighbors[np.argmax(scores) ]), "in the fold ", i)
    optimal_value.append(neighbors[np.argmax(scores) ])


x = ['Fold','Optimal value of n_neighbors', 'Mean score']
y = [1,2,3,4,5],optimal_value ,mean_score
MakeTable(x,y)


################################################################################################

# Make the test with the actual function and compare results 

[x_LS,y_LS,x_TS,y_TS]=make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
k_range = [1,5,10,75,100,150]
param_grid = dict( n_neighbors=k_range)
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(x_LS, y_LS)
#check top performing n_neighbors value
print(knn_gscv.best_params_)
#check mean score for the top performing value of n_neighbors
print(knn_gscv.best_score_)

#################################################################################################

# Graph  
# Try K=1 through K=25 and record testing accuracy
k_range =  [1,5,10,75,100,150]
scores = []
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_LS, y_LS)
    y_pred = knn.predict(x_TS)
    scores.append(accuracy_score(y_TS, y_pred))
print(scores)

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

# (3) curve of test accuracies (on a TS of size500) for every possible number of neighbors for LS of sizes
# {50,200,250,500}

#Data Set 1 

[x_LS,y_LS,x_TS,y_TS]=make_data1(n_ts=500, n_ls=50, noise=0, plot=False, random_state=0)
k_range1 =  list(range(1,50,1))
n = 50
Test_Accuracy(k_range1, n, x_LS, y_LS, x_TS)



[x_LS ,y_LS,x_TS,y_TS]=make_data1(n_ts=500, n_ls=200, noise=0, plot=False, random_state=0)
k_range1 =  list(range(1,200,1))
n =   200
Test_Accuracy(k_range1, n, x_LS, y_LS, x_TS)



[x_LS ,y_LS,x_TS,y_TS]=make_data1(n_ts=500, n_ls=250, noise=0, plot=False, random_state=0)
k_range1 =  list(range(1,250,1))
n = 250
Test_Accuracy(k_range1, n, x_LS, y_LS, x_TS)



[x_LS,y_LS,x_TS,y_TS]=make_data1(n_ts=500, n_ls=500, noise=0, plot=False, random_state=0)
k_range1 =  list(range(1,250,1))
n = 250
Test_Accuracy(k_range1, n, x_LS, y_LS, x_TS)
# Error taking 500, the max is 250 


#Data Set 2 

[x_LS,y_LS,x_TS,y_TS]=make_data2(n_ts=500, n_ls=50, noise=0, plot=False, random_state=0)
k_range2 =  list(range(1,50,1))
n = 50
Test_Accuracy(k_range2, n, x_LS, y_LS, x_TS)


[x_LS,y_LS,x_TS,y_TS]=make_data2(n_ts=500, n_ls=200, noise=0, plot=False, random_state=0)
k_range2 =  list(range(1,200,1))
n = 200
Test_Accuracy(k_range2, n, x_LS, y_LS, x_TS)


[x_LS,y_LS,x_TS,y_TS]=make_data2(n_ts=500, n_ls=250, noise=0, plot=False, random_state=0)
k_range2 =  list(range(1,250,1))
n = 200
Test_Accuracy(k_range2, n, x_LS, y_LS, x_TS)


[x_LS,y_LS,x_TS,y_TS]=make_data2(n_ts=500, n_ls=500, noise=0, plot=False, random_state=0)
k_range2 =  list(range(1,250,1))
n = 200
Test_Accuracy(k_range2, n, x_LS, y_LS, x_TS)



#Optimal value of neighbors respect to the LS size illustration
neighbors = [1,1,1,1]
ls_size = [50, 200, 250, 500]

plt.plot(ls_size, neighbors, 'o' )
plt.ylabel('Optimal value of neighbors')
plt.xlabel('LS Size')
plt.show() 

neighbors = [1,5,1,1]
ls_size = [50, 200, 250, 500]

plt.plot(ls_size, neighbors, 'o' )
plt.ylabel('Optimal value of neighbors')
plt.xlabel('LS Size')
plt.show()

   
