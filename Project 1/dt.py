"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz 

from data import make_data1, make_data2
from plot import plot_boundary


from matplotlib import pyplot as plt
import graphviz
import plotly.graph_objects as go
import plotly.io as pio

# (Question 1)

def print_scores_tree(x_LS,y_LS,x_TS,y_TS,number):
   # score_fitting=[]
    score_values_LS=[]
    score_values_TS=[]
    dep=[]
    
    for i in range (1,6):
        
        if i==1:
            nd=1
        elif i==2:
            nd=2
        elif i==3:
            nd=4
        elif i==4:
            nd=8
        elif i==5: 
            nd=None
     
        dep.append(nd)
        
        tree_test= tree.DecisionTreeClassifier(max_depth=nd) #if nothing max_depth='none'
        
        #train model
        tree_test = tree_test.fit(x_LS, y_LS)
        #tree.plot_tree(tree_test)
        
          #proportion=True -> in percentage , fontsize=...
        plt.figure(figsize=(25,15))
        #tree.plot_tree(tree_test,class_names=['circle1','circle2'],filled=True,rounded=True)
              
        #export_graphviz(tree_test,out_file='test_projet1_tree.dot',filled=True,rounded=True)

        
        score_test=DecisionTreeClassifier.score(tree_test,x_TS,y_TS,)
        score_learning=DecisionTreeClassifier.score(tree_test,x_LS,y_LS,)
        
       
        score_values_LS.append(1-score_learning)
        score_values_TS.append(1-score_test)
        
        
        dmax=DecisionTreeClassifier.get_depth(tree_test)
        depth_model=str(dmax)

        depp=str(nd)
        
        name=str(nd)
    
        plot_boundary('Data'+number+'_'+name,tree_test, x_LS, y_LS, mesh_step_size=0.1, title="", inline_plotting=False)

    
    print('Values of the maximum depth',dep)
    print('Score values for the learning sample',score_values_LS)
    print('Score values for the test sample',score_values_TS) 

    plt.figure(figsize=(25,15))
    plt.title("Error in function of the depth")
    plt.xlabel("Depth")
    plt.ylabel("Error")
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)
    plt.plot([1,2,4,8,dmax],score_values_LS, label='learning sample')
    plt.plot([1,2,4,8,dmax],score_values_TS, label='test sample')
    plt.legend()
    
    
   
    
    return tree_test
    
# (Question 2)

def test_generations(dataset):
         
    meanG=[]
    stdG=[]
    
    MatrixDepth=[]
    MeanGenerations=[]
    StdGenerations=[]
    

    
    for d in range(1,6):  #depth
        if d==1:
            depth=1
        elif d==2:
            depth=2
        elif d==3:
            depth=4
        elif d==4:
            depth=8
        elif d==5: 
            depth=None    
            
        
        score_generations=[]
        for j in range(1,6): #generations
            if j==1:
                gen=0
            elif j==2:
                gen=1
            elif j==3:
                gen=2
            elif j==4:
                gen=3
            elif j==5:
                gen=4
    
            
            if dataset==1:               
                [x_LS,y_LS,x_TS,y_TS]=make_data1(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=gen)
            elif dataset==2:
                [x_LS,y_LS,x_TS,y_TS]=make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=gen) 
                
                              
            tree_test= tree.DecisionTreeClassifier(max_depth=depth) #if nothing max_depth='none'
            
            #train model
            tree_test = tree_test.fit(x_LS, y_LS)
            #plt.figure(figsize=(25,15))
            #tree.plot_tree(tree_test,class_names=['circle1','circle2'],filled=True,rounded=True)
            
            score_test=DecisionTreeClassifier.score(tree_test,x_TS,y_TS,)
            score_generations.append(score_test)
        
        mean=np.mean(score_generations)
        std=np.std(score_generations)       
        dep_gen=DecisionTreeClassifier.get_depth(tree_test)
        mean = "%.5f" % mean 
        std = "%.5f" % std 
   
    
        MatrixDepth.append(depth)
        MeanGenerations.append(mean)
        StdGenerations.append(std)
    
    return MeanGenerations,StdGenerations,dep_gen   #mean, std
        


def MakeTable(x,y):
    pio.renderers.default = 'browser'

    fig = go.Figure(data=[go.Table(header=dict(values=x),
                     cells=dict(values=y))
                         ])
    fig.show()


if __name__ == "__main__":
    pass # Make your experiments here

plt.rcParams.update({'font.size': 45})

# Question 1) depth :

[x_LS1,y_LS1,x_TS1,y_TS1]=make_data1(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)

print("For the first data set:")
model1=print_scores_tree(x_LS1,y_LS1,x_TS1,y_TS1,'1')

[x_LS2,y_LS2,x_TS2,y_TS2]=make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)



print("For the second data set:")
model2=print_scores_tree(x_LS2,y_LS2,x_TS2,y_TS2,'2')
print(" ")

# Question 2) Generations of dataset

[mean1,std1,dep_gen1]=test_generations(dataset=1)
[mean2,std2,dep_gen2]=test_generations(dataset=2)



x = ['Maximum Depth', 'Mean dataset1', 'Std dataset1', 'Mean dataset2', 'Std dataset2']
y = [1,2,4,8,'None'],mean1,std1,mean2,std2

#MakeTable(x,y)






