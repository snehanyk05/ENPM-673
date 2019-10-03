# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:50:02 2019

@author: Sneha
"""
#Gaussian Function

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
def em_implementation(X,N,d,r):
    
    m =  X.shape[0]
    values = np.random.permutation(m);
    mu = X[values[0:N]]+r;
    Mu=mu.reshape(-1, 1)
#    print('Mu',Mu.shape)
    sigma = np.zeros((1,1,N));
    
    for j in range(0,N):
      
        sigma[:,:,j]=(np.cov(X)+np.eye(d,d)*r)
       
    phi = np.ones(N)
  
    pdf = np.zeros((X.shape[0],N));
    X=X.reshape(-1,1)
    for k in range(0,1000):
    
        for j in range(0,N):
#            print(mu[j])
           pdf[:,j] = gaussian_nd(X, Mu[j,:], sigma[:,:,j]);
         
        temp_pdf = pdf * phi;
        
        sum1=np.sum(temp_pdf, 1)
        sum1=sum1.reshape(-1,1)
       
        weights = np.divide(temp_pdf,sum1)
        mu_previous = Mu; 
        
        for j in range(0,N):
           
            phi[j] = np.mean(weights[:, j], 0);
            Mu[j,:] = np.dot(np.transpose(weights[:, j,np.newaxis]),X)
            Mu[j,:] = Mu[j,:] / np.sum(weights[:, j], 0)
            
            
            temp = np.zeros((d, d))+np.eye(d,d)*r;
            Xm = X - Mu[j, :]
            for i in range (0,m):
                temp = temp + (weights[i, j] * (np.transpose(Xm[i, :]) * Xm[i, :]))
            
            sigma[:,:,j] = temp / np.sum(weights[:, j])
            
        if (np.array_equal(Mu,mu_previous)):
            break;
    
#    temp_pdf = pdf .* phi;
#    weights = temp_pdf./sum(temp_pdf, 2);
#    mu_previous = mu;  
    return (Mu,sigma)
def gaussian_nd(X, mu, Sigma):
    
    n = X.shape[1]
    
    diff = X - mu
    diff=diff.reshape(-1,1)
    
    N1 = 1/(np.sqrt(((2 * np.pi)**n)*np.linalg.det(Sigma))) * np.exp( -1/2 * np.sum((((diff)/Sigma)*(diff)),axis=1))
#    N1=N1.reshape(-1,1)
   
    return N1
def gaussian_1D(x,mu,Sigma):
    N1 = 1/(Sigma*np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * Sigma**2))
    
    return N1

def gaussian_1d(x,mu,Sigma):
    N1= (2 * Sigma**2)
#    N1 = 1/(Sigma*np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * Sigma**2))
    
    return N1

mean1 = [0] 
mean2 = [5]
mean3 = [10]
sigma1 = 3
sigma2 = 2
sigma3 = 1.5

data = np.arange(-10,20,0.1)
#print(data)
norm1 = gaussian_1D(data,mean1,sigma1);
plt.plot(data,norm1,'r',lw=2)

norm2 = gaussian_1D(data,mean2,sigma2);
plt.plot(data,norm2,'g',lw=2)

norm3 = gaussian_1D(data,mean3,sigma3);
plt.plot(data,norm3,'b',lw=2)
plt.show()

sample1 = np.random.normal(mean1, sigma1, 10)
sample2 = np.random.normal(mean2, sigma2, 10)
sample3 = np.random.normal(mean3, sigma3, 10)
#D = np.array([-0.90316267,0.07807467,-6.15308237,-0.66100194,4.63738257,-7.66972735,-0.54345368,-0.41701947,0.53412763,-5.95980806,5.38564362,6.59367813,4.51250495,6.30280297,5.21618673,6.47747097,5.70761478,4.21495775,5.18369714,3.91917485,9.25093624,11.65903423,10.45799807,11.18982658,10.75598797,8.86847202,9.81451286,11.22035528,11.6376232,9.98589465])

D = sample1
D=np.append(D,np.append(sample2,sample3))



nmodes = 3;
GMModel=GaussianMixture(n_components = nmodes, covariance_type = 'diag')
X=D.reshape(-1, 1)
GMModel = GMModel.fit((X))

#GMModel = fitgmdist(D,3);
mean_new = GMModel.means_;
sigma_new_t=GMModel.covariances_
sigma_new = np.array([GMModel.covariances_[0], GMModel.covariances_[1], GMModel.covariances_[2]]);
gmm1 = gaussian_1D(data,mean_new[0],sigma_new[0]);
gmm2 = gaussian_1D(data,mean_new[1],sigma_new[1]);
gmm3 = gaussian_1D(data,mean_new[2],sigma_new[2]);

data1=data[ np.newaxis,:]
plt.plot(data,gmm1,'y',lw=2)

plt.plot(data,gmm2,'r',lw=2)

plt.plot(data,gmm3,'b',lw=2)
plt.show()
#
mean_new_1,sigma_new_1=em_implementation(D,3,1,10**-2)

sigma=np.transpose(sigma_new_1[0])
gmm1 = gaussian_1D(data,mean_new_1[0],sigma[0]);
gmm2 = gaussian_1D(data,mean_new_1[1],sigma[1]);
gmm3 = gaussian_1D(data,mean_new_1[2],sigma[2]);
#
#
plt.plot(data,gmm1,'y',lw=2)

plt.plot(data,gmm2,'r',lw=2)

plt.plot(data,gmm3,'b',lw=2)

plt.show()


#4D Gaussians
nmodes = 4;
GMModel=GaussianMixture(n_components = nmodes, covariance_type = 'diag')
X=D.reshape(-1, 1)
GMModel = GMModel.fit((X))

#GMModel = fitgmdist(D,3);
mean_new = GMModel.means_;
sigma_new_t=GMModel.covariances_
sigma_new = np.array([GMModel.covariances_[0], GMModel.covariances_[1], GMModel.covariances_[2], GMModel.covariances_[3]]);
gmm1 = gaussian_1D(data,mean_new[0],sigma_new[0]);
gmm2 = gaussian_1D(data,mean_new[1],sigma_new[1]);
gmm3 = gaussian_1D(data,mean_new[2],sigma_new[2]);
gmm4 = gaussian_1D(data,mean_new[3],sigma_new[3]);
data1=data[ np.newaxis,:]
plt.plot(data,gmm1,'y',lw=2)

plt.plot(data,gmm2,'r',lw=2)

plt.plot(data,gmm3,'b',lw=2)

plt.plot(data,gmm4,'g',lw=2)
plt.show()
#
mean_new_1,sigma_new_1=em_implementation(D,4,1,10**-2)

sigma=np.transpose(sigma_new_1[0])
gmm1 = gaussian_1D(data,mean_new_1[0],sigma[0]);
gmm2 = gaussian_1D(data,mean_new_1[1],sigma[1]);
gmm3 = gaussian_1D(data,mean_new_1[2],sigma[2]);
gmm4 = gaussian_1D(data,mean_new_1[3],sigma[3]);
#
#
plt.plot(data,gmm1,'y',lw=2)

plt.plot(data,gmm2,'r',lw=2)

plt.plot(data,gmm3,'b',lw=2)
plt.plot(data,gmm4,'g',lw=2)

plt.show()

#norm2 = gaussian_1D(data,mean2,sigma2);
#plot(data,norm2,'g','LineWidth',2);
#
#norm3 = gaussian_1D(data,mean3,sigma3);
#plot(data,norm3,'b','LineWidth',2);
