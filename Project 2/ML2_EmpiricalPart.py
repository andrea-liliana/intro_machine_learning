# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:01:06 2020

"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

def expo(x,a):
    result=x**a
    return result

"Generation learning sample sets"

def generation_LS(nb_LS,length,noisy=True):
    '''
    nb_LS #number of learning sample
    length
    noisy  to have the function without noise for the approximation in
    the 2.d question
    '''
    x_min=0
    x_max=2
    step=(x_max-x_min)/length  #size of each learning sample
    
    x_LS=np.zeros((length))
    y_LS=np.zeros((length,nb_LS))
    
    for k in range(1,length):
        x1=x_LS[k-1]
        x_LS[k]=x1+step
    
    
    for j in range(nb_LS): 
    
            
        sample=np.zeros((length,2))
        x=0
        for i in range(length):
            x=x_LS[i]
            
            if noisy==False:
                noise=0
            else:   
                v=np.sqrt(0.1)
                noise=np.random.normal(loc=0.0,scale=v) #loi normale(0,0.1)
            
            y= -expo(x,3) + 3*expo(x,2) - 2*x + 1 + noise

            sample[i,1]=y

        y_LS[:,j]=sample[:,1]

    return x_LS,y_LS


def bayes(x_LS,y_LS):
    nb_sample=len(y_LS[0])
    length=len(x_LS)
    
    y_bayes=np.zeros((length))
    residual_error=np.zeros((length))
    
    for i in range(length):
        y_line=np.zeros((nb_sample))
        
        for j in range(nb_sample):
            
            y_line[j]=y_LS[i,j]

            
        y_bayes[i]= np.mean(y_line)
        residual_error[i]=np.var(y_line)
            
    return y_bayes, residual_error

def estimation(x_LS,y_LS,m):
    length=len(x_LS)
    nb_sample=len(y_LS[0])    
    
    algo=LinearRegression()
    
    if m==0:
        x_LS=x_LS.reshape(-1, 1) #to say it only has one feature
    
    y_estimated_all=np.zeros((length,nb_sample))
    
    for i in range(nb_sample):
                
        model=algo.fit(x_LS,y_LS[:,i])       
        y_estimated_all[:,i]=model.predict(x_LS)
        
    y_estimated=np.zeros((length))
    variance_LS=np.zeros((length))
    
    for j in range(length):
        y_estimated[j]=np.mean(y_estimated_all[j,:])
        variance_LS[j]=np.var(y_estimated_all[j,:])
        

    return y_estimated, variance_LS
    
def Ridge_estimation(x_LS,y_LS,lbd):
    length=len(x_LS)
    nb_sample=len(y_LS[0])    
    
    algo=Ridge(alpha=lbd)
    
    y_estimated_all=np.zeros((length,nb_sample))
    
    for i in range(nb_sample):
                
        model=algo.fit(x_LS,y_LS[:,i])
        y_estimated_all[:,i]=model.predict(x_LS)
        
    y_estimated=np.zeros((length))
    variance_LS=np.zeros((length))
    
    for j in range(length):
        y_estimated[j]=np.mean(y_estimated_all[j,:])
        variance_LS[j]=np.var(y_estimated_all[j,:])
        

    return y_estimated, variance_LS


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 18})
    
    ''' Question c'''
       
    x_LS1,y_LS1=generation_LS(5000,2000)  
    x1,y1=generation_LS(1,2000,noisy=False)
    
    y_bayes, residual_error=bayes(x_LS1,y_LS1)  
    plt.plot(x_LS1,y_bayes)
    plt.xlabel("X")
    plt.ylabel("Bayes model/f(x)")
    plt.plot(x1,y1)
    plt.legend(["Bayes model", "F(x)"])
    plt.show()
    plt.plot(x_LS1,residual_error)
    plt.xlabel("X")
    plt.ylabel("Residual Error")
    plt.show()
    
    
    '''Question d'''
    x,y=generation_LS(1,30,noisy=False)  #f(x) (approximation)

    x_LS,y_LS=generation_LS(1000,30) #1000 30
    length=len(x_LS)
    
    y_estimated=np.zeros((length,6))
    variance_LS=np.zeros((length,6))
    BiasSquared=np.zeros((length,6))
    ExpectedError=np.zeros((length,6))
    
    for m in range(6):
        x_LS_degree = np.zeros((length,m+1))  
        # ! verify if not a problem here a column of 1 for the degree 0 but for higher degree okay to keep?
        x_LS_degree[:,0]=np.ones((length))
        
        for d in range(1,m+1):

            x_LS_degree[:,d]=expo(x_LS,d)

        y_estimated_degree,variance_LS_degree=estimation(x_LS=x_LS_degree,y_LS=y_LS,m=m) 

        y_estimated[:,m]=y_estimated_degree[:]
        variance_LS[:,m]=variance_LS_degree[:]
        
        for n in range(length):
            BiasSquared[n,m]= (y[n]-y_estimated_degree[n])**2
            ExpectedError[n,m]=residual_error[n] + BiasSquared[n,m] + variance_LS_degree[n]
            

    plt.plot(x_LS,y_estimated)
    plt.xlabel("X")
    plt.ylabel("Estimated model")
    plt.legend(["degree 0", "degree 1","degree 2","degree 3","degree 4","degree 5"],loc='upper center', bbox_to_anchor=(1.25, 0.9), shadow=True, ncol=1) 
    plt.show()

    plt.plot(x_LS,variance_LS)
    plt.xlabel("X")
    plt.ylabel("Variance")
    plt.legend(["degree 0", "degree 1","degree 2","degree 3","degree 4","degree 5"],loc='upper center', bbox_to_anchor=(1.25, 0.9), shadow=True, ncol=1) 
    plt.show()
    
    plt.plot(x_LS,BiasSquared)
    plt.xlabel("X")
    plt.ylabel("Squared Bias")
    plt.legend(["degree 0", "degree 1","degree 2","degree 3","degree 4","degree 5"],loc='upper center', bbox_to_anchor=(1.25, 0.9), shadow=True, ncol=1) 
    plt.show()

    plt.plot(x_LS,ExpectedError)
    plt.xlabel("X")
    plt.ylabel("Expected Error")
    plt.legend(["degree 0", "degree 1","degree 2","degree 3","degree 4","degree 5"],loc='upper center', bbox_to_anchor=(1.25, 0.9), shadow=True, ncol=1) 
    plt.show()
          
        
    '''Question e''' #plot mean values in function of complexity m 

    degree=[0,1,2,3,4,5]
    MeanSquaredBias=np.zeros(6)
    MeanVariance=np.zeros(6)
    MeanExpectedError=np.zeros(6)
    
    for deg in range(6):
        
        MeanSquaredBias[deg]=np.mean(BiasSquared[:,deg])
        MeanVariance[deg]=np.mean(variance_LS[:,deg])
        MeanExpectedError[deg]=np.mean(ExpectedError[:,deg])
    
    
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(degree,MeanSquaredBias, label='Squared Bias')
    ax.plot(degree,MeanVariance, label='Variance')
    ax.plot(degree,MeanExpectedError, label='Expected Error')
    ax.plot()
    ax.legend(loc='upper center', bbox_to_anchor=(1.31, 0.7), shadow=True, ncol=1)
    plt.xlabel('Degree')
    plt.ylabel('Error')
    plt.show()

        
    '''Question f''' #Ridge: bias^2, variance, error effect of  λ ∈ [0.0, 2.0] (regularisation level)
    
    m=5   
    x_LS_ridge=np.zeros((length,m+1))
    
    for d in range(1,m+1):
        x_LS_ridge[:,d]=expo(x_LS,d)
    
    lamd=np.zeros(20)
    y_estimated_R=np.zeros((length,5))
    variance_LS_R=np.zeros((length,5))
    BiasSquaredRidge=np.zeros((length,5))
    ExpectedErrorRidge=np.zeros((length,5))
    
    for fac in range(5):
        if fac==0:
            lbd=0
        elif fac==1:
            lbd=0.5
        elif fac==2:
            lbd=1
        elif fac==3:
            lbd=1.5
        elif fac==4:
            lbd=2
        
        y_estimated_Ridge,variance_LS_Ridge=Ridge_estimation(x_LS=x_LS_ridge,y_LS=y_LS,lbd=lbd)
         
        y_estimated_R[:,fac]=y_estimated_Ridge[:]
        variance_LS_R[:,fac]=variance_LS_Ridge[:]
        
        for n in range(length):
            BiasSquaredRidge[n,fac]= (y[n]-y_estimated_Ridge[n])**2
            ExpectedErrorRidge[n,fac]=residual_error[n] + BiasSquaredRidge[n,fac] + variance_LS_Ridge[n]
            

    plt.plot(x_LS,y_estimated_R)
    plt.xlabel("X")
    plt.ylabel("Estimated model")
    plt.legend(["lambda=0", "lambda=0.5","lambda=1","lambda=1.5","lambda=2"],loc='upper center', bbox_to_anchor=(1.28, 0.8), shadow=True, ncol=1) 
    plt.show()

    plt.plot(x_LS,variance_LS_R)
    plt.xlabel("X")
    plt.ylabel("Variance")
    plt.legend(["lambda=0", "lambda=0.5","lambda=1","lambda=1.5","lambda=2"],loc='upper center') 
    plt.show()
    
    plt.plot(x_LS,BiasSquaredRidge)
    plt.xlabel("X")
    plt.ylabel("Squared Bias")
    plt.legend(["lambda=0", "lambda=0.5","lambda=1","lambda=1.5","lambda=2"],loc='upper right') 
    plt.show()

    plt.plot(x_LS,ExpectedErrorRidge)
    plt.xlabel("X")
    plt.ylabel("Expected Error")
    plt.legend(["lambda=0", "lambda=0.5","lambda=1","lambda=1.5","lambda=2"],loc='upper center') 
    plt.show()
        
    lamd=[0,0.5,1,1.5,2]
    MeanSquaredBiasRidge=np.zeros(5)
    MeanVarianceRidge=np.zeros(5)
    MeanExpectedErrorRidge=np.zeros(5)
    
    for lambd in range(5):
        
        MeanSquaredBiasRidge[lambd]=np.mean(BiasSquaredRidge[:,lambd])
        MeanVarianceRidge[lambd]=np.mean(variance_LS_R[:,lambd])
        MeanExpectedErrorRidge[lambd]=np.mean(ExpectedErrorRidge[:,lambd])
    

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(lamd,MeanSquaredBiasRidge, label='Squared Bias')
    ax.plot(lamd,MeanVarianceRidge, label='Variance')
    ax.plot(lamd,MeanExpectedErrorRidge, label='Expected Error')
    ax.plot()
    ax.legend()
    plt.xlabel('Regularisation Level')
    plt.ylabel('Error')
    plt.show()
 
