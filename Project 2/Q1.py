import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import get_test_data
from sklearn.utils import check_random_state


def generate_samples_class1(nb_points):
    rho=0.75
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]   
    
    x=np.zeros((nb_points,2))
    y=np.ones(nb_points)
    for i in range(nb_points):
        rho=0.75        
        X=np.random.multivariate_normal(mean, cov)
        x[i,0] = X[0]
        x[i,1] = X[1]
             
    return x,y         

def generate_samples_class2(nb_points):
    rho=-0.75
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]   

    x=np.zeros((nb_points,2))
    y=np.zeros(nb_points)
    for i in range(nb_points):
        y[i]=-1
        X=np.random.multivariate_normal(mean, cov)
        x[i,0] = X[0]
        x[i,1] = X[1]

    return x,y

def multivariate_gaussian(pos, mu, Sigma):


    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N



if __name__ == '__main__':

    '''Plot data'''
    x1,y1=generate_samples_class1(1000)
    x2,y2=generate_samples_class2(1000)    
        
    plt.figure()
    plt.scatter(x1[:,0],x1[:,1],label='y=1')
    plt.scatter(x2[:,0],x2[:,1],label='y=2')
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.legend(loc='upper center', bbox_to_anchor=(1.1, 0.5), shadow=True, ncol=1)
    plt.show()
    
    '''Plot gaussians'''

    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)
    
    # Mean vector and covariance matrix
    mu = np.array([0., 0.])
    Sigma = np.array([[ 1. , 0.75], [0.75,  1]])
    
    mu2 = np.array([0.,0.])
    Sigma2 = np.array([[ 1. , -0.75], [-0.75,  1]])
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    Z = multivariate_gaussian(pos, mu, Sigma)
    z2= multivariate_gaussian(pos, mu2, Sigma2)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis, color='r')
    
    ax.plot_surface(X, Y, z2, rstride=3, cstride=3, linewidth=0, antialiased=False, cmap='coolwarm', color='g')
    ax.set(xlabel='x0', ylabel='x1')
    ax.set_zlim(0,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)
    
    fig = plt.show()




