"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_data1, make_data2
from plot import plot_boundary

from data import make_data1,make_data2
from plot import plot_boundary, plot_boundary_extended

from matplotlib import pyplot as plt

class residual_fitting(BaseEstimator, ClassifierMixin):

        

    def fit(self,X, y):
        """Fit a Residual fitting model using the training set (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.  
        """
        ''' Removed for extended
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")
'''
        #algorithm : 
        n=len(y)
        attributes_max=len(X[0])
        
        w=np.zeros((attributes_max+1))   #size of the number of attributes +1
        w0=np.mean(y)
        w[0]=w0

        Dim=np.ones((n))
        
        a=X               
        sumK=0
           
        for k in range(1,attributes_max+2): # for the two attributes,  k=1 k=2 k=3 " 
            
            if k>1:
                sumK=sumK + w[k-2]*a[:,k-2] #start at k=2 and first value position 0
                                    
            delta_ky= y[:] - w[0]*Dim - sumK
            
               
            if k>1:

                w[k-1]= np.corrcoef(a[:,k-2],delta_ky[:])[0,1] * np.std(delta_ky[:])
                                       
        #print(w)
        #v=54
        #print(w[0]+w[1]*a[v,0]+w[2]*a[v,1]-y[v])
        self.w = w
             
        return self

    def predict(self, X):
        """Predict class for X.
        
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # ====================
        #print(model.w)
        w=model.w

        n=len(X)
        Dim=np.ones((n))
        ln=len(X[0])
        wx=0
        
        for j in range(ln):        # model: y_estimated= w0 + wi*xi
            wx=w[j+1]*X[:,j]+wx
            
        y_estimated=w[0]*Dim + wx
                
        model.y_estimated= y_estimated
        
        prediction=np.zeros((n))
        for i in range(n):
            if y_estimated[i]>0.5:               
                prediction[i] =1  #class 0 or 1
            elif y_estimated[i]<=0.5:
                prediction[i]=0
        
        model.prediction = prediction
          
      
        return prediction,y_estimated

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
            
        """

        # ====================
        
        prediction,y_estimated=residual_fitting().predict(X)
        
        size=len(X)
        probas=np.zeros((size,2))
        
        for f in range(size):  
            if y_estimated[f]==0.5:
                probas[f,0]=0.5
                probas[f,1]=0.5                
            elif y_estimated[f] <= 0:
                probas[f,0]=1  
                probas[f,1]=0  
            elif y_estimated[f] >= 1:
                probas[f,0]=0
                probas[f,1]=1
            elif 0.5 > y_estimated[f] > 0: #ex: 0.4
                probas[f,0]= 1-y_estimated[f]  #class0 = 0.6
                probas[f,1]= y_estimated[f]    #class1 = 0.4 
                
            elif 1 > y_estimated[f] > 0.5: #ex : 0.9
                probas[f,0]= 1- y_estimated[f]   #class0: 0.1
                probas[f,1]= y_estimated[f]  #class1=0.9

           
        model.predict_proba=probas
        
        
        # ====================

        return probas
    
 
if __name__ == "__main__":
    plt.rcParams.update({'font.size': 30})

    'Choose data set  1 or 2 :'
    [x_ls,y_ls,x_ts,y_ts]=make_data2()

    
    '''        
    "Question3:"
    model=residual_fitting().fit(X=x_ls, y=y_ls)  
    prediction1,y_estimated1=residual_fitting().predict(X=x_ts)   
          
    'test set accuracies:'
    #to know how much data are misclassified:
    diff= y_ts - prediction1
    
    count=0
    T=len(x_ls)
    for i in range(T):
        if diff[i] == 0:
            count=count+1
    
    accuracy=(count/T)*100
    print('The accuracy on data set 1 is: ',accuracy)

       
    residual_fitting().predict_proba(X=x_ts)  
       
    
    plot_boundary('residu',residual_fitting(), x_ts, y_ts, mesh_step_size=0.1, title="", inline_plotting=True)
    
        
    
    '''   
     
    #Question 4: X extended


    x1x1=x_ls[:,0]*x_ls[:,0]
    x2x2=x_ls[:,1]*x_ls[:,1]
    x1x2=x_ls[:,0]*x_ls[:,1]

    x2_ls=np.zeros((250,5))
    x2_ls[:,0]=x_ls[:,0]
    x2_ls[:,1]=x_ls[:,1]
    x2_ls[:,2]=x1x1
    x2_ls[:,3]=x2x2
    x2_ls[:,4]=x1x2
    
 
    model=residual_fitting().fit(X=x2_ls, y=y_ls)  
    prediction2,y_estimated2=residual_fitting().predict(X=x_ts)
    
    
    'test set accuracies:'
    #to know how much data are misclassified:
    diff= y_ts - prediction2
    
    count=0
    T=len(x_ts)
    for i in range(T):
        if diff[i] == 0:
            count=count+1
    
    accuracy=(count/T)*100
    print('The accuracy is: ',accuracy)

   
    # need :  model.predict_probas
    plot_boundary_extended('',residual_fitting(), x_ts, y_ts, mesh_step_size=0.1, title="", inline_plotting=True)   
    
