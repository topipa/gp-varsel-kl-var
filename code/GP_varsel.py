""" 
    This file contains functions KLrel and VARrel that compute
    relevance estimates for covariates in Gaussian process models.
    The KL and VAR methods are from https://arxiv.org/abs/1712.08048
    They take a GPy model and the data matrix X as parameters.
    
    Topi Paananen (topi.paananen@aalto.fi)
    18/04/2018
"""

import GPy
import numpy as np
import scipy as scp

def KLrel(X,model,delta):
    """Computes relevance estimates for each covariate using the KL method based on the data matrix X and a GPy model.
    The parameter delta defines the amount of perturbation used."""
    n = X.shape[0]
    p = X.shape[1]
    relevances = np.zeros(p)
    jitter = 1e-16

    # perturbation
    deltax = np.linspace(-delta,delta,3)

    # loop through the data points X
    for j in range(0, n):

        x_n = np.reshape(np.repeat(X[j,:],3),(p,3))

        # loop through covariates
        for dim in range(0, p):
            
            # perturb x_n
            x_n[dim,:] = x_n[dim,:] + deltax
            
            # compute the mean and variance of the predictive distribution
            preddeltamean,preddeltavar = GPy.models.GPRegression.predict(model,x_n.T,full_cov=False)
            mean_orig = np.asmatrix(np.repeat(preddeltamean[1],3)).T
            var_orig = np.asmatrix(np.repeat(preddeltavar[1],3)).T

            # compute the relevance estimate at x_n
            KLsqrt = np.sqrt(0.5*(var_orig/preddeltavar + np.multiply((preddeltamean.reshape(3,1)-mean_orig),(preddeltamean.reshape(3,1)-mean_orig))/preddeltavar - 1 + np.log((preddeltavar/var_orig))) + jitter)
            relevances[dim] += 0.5*(KLsqrt[0] + KLsqrt[2])/delta
            
            # remove the perturbation
            x_n[dim,:] = x_n[dim,:] - deltax
    return relevances/n


def VARrel(X,model,nquadr): # now stops jittering when cond_2 < 100
    """Computes relevance estimates for each covariate using the VAR method based on the data matrix X and a GPy model.
    The parameter nquadr defines the number of quadrature points to use in Gauss-Hermite quadrature integration."""
    n = X.shape[0]
    p = X.shape[1]
    relevances = np.zeros(p)
    [points,weights] = np.polynomial.hermite.hermgauss(nquadr)
    jitter = 1e-9


    # full covariance matrix of X plus small jitter on diagonal
    fullcov = np.cov(X,rowvar=False) + jitter*np.eye(p)

    # if condition number is high, add a diagonal term until it goes below 100
    # this is a bit heuristic
    while (np.linalg.cond(fullcov,p=2) > 100):
        jitter = jitter*10
        fullcov = fullcov + jitter*np.eye(p)

    # Cholesky decomposition of the covariance matrix
    cholfull = scp.linalg.cholesky(fullcov,lower=True)

    # loop through covariates
    for j in range(0, p):

        # remove j'th covariate
        jvals = X[:,j]
        nojvals = np.delete(X,(j), axis=1)
        jmean = jvals.mean() 
        nojmean = nojvals.mean(axis=0)

        jcov = fullcov[j,j]
        jnojcov = fullcov[j,:]
        jnojcov = np.delete(jnojcov,(j),axis=0)
        jnojcov = jnojcov.reshape(1,p-1)
        
        # Cholesky decomposition of the submatrix
        cholsub = cholsubmatrix(cholfull,j)
        meanfactor = scp.linalg.cho_solve((cholsub,True),jnojcov.T).T
        intcov = jcov - np.dot(  meanfactor,jnojcov.T)

        # loop through data points
        for k in range(0, n):

            nojtark = nojvals[k,:]
            intmean = jmean + np.dot(meanfactor,(nojtark - nojmean))
            fcalcpoints = np.repeat(X[k,:],nquadr).reshape(p,nquadr).T
            fcalcpoints[:,j] = np.sqrt(2)*np.sqrt(intcov)*points + intmean

            predmean,predvar = GPy.models.GPRegression.predict(model,fcalcpoints,full_cov=False)
            fsquare = predmean*predmean

            # Gauss-Hermite quadrature integration
            relevances[j] += np.dot(fsquare.T,weights)/np.sqrt(np.pi) - np.dot(predmean.T,weights)*np.dot(predmean.T,weights)/np.pi

    return relevances/n


# Auxiliary functions

def cholr1update(L,x):
    """Computes a rank-1 update to a Cholesky decomposition"""
    n = np.shape(x)[0]
    for k in np.arange(0,n):
        r = np.sqrt(L[k,k]*L[k,k] + x[k]*x[k])
        c = r / L[k,k]
        s = x[k] / L[k,k]
        L[k,k] = r
        L[k+1:,k] = (L[k+1:,k] + s * x[k+1:]) / c
        x[k+1:] = c * x[k+1:] - s * L[k+1:,k]
    return L

def cholsubmatrix(L,index):
    """Computes the Cholesky decomposition for a submatrix (matrix with one row and one column removed) using the Cholesky of the full matrix"""
    n = np.shape(L)[0]
    chol = np.zeros((n-1,n-1)) 
    chol[:index,:index] = L[:index,:index]
    if (index < n-1):
        chol[index:,:index] = L[index+1:,:index]
        chol[index:,index:] = cholr1update(L[index+1:,index+1:],L[index+1:,index])
    return chol
