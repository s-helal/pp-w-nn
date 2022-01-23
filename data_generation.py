# import packages
import numpy as np
import scipy
from scipy.stats import norm 
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
import math

## INVERSE TRANSFORM 

# generate sample from normal distribution using inverse transform sampling 
def InverseTransform(mu, sigma, n_samples):

  # start counter
  tic = time.perf_counter() 

  # simulate uniformly randomly distributed data of size n_samples
  u = np.random.uniform(0, 1, n_samples)

  # perform inverse transform sampling, scale by sigma, and translate by mu
  x = norm.ppf(u) * sigma + mu # quantile of standard normal distribution, scaled

  # end counter
  toc = time.perf_counter()

  # plot histogram of generated data
  plt.hist(x)
  plt.show()

  # plot qqplot of generated data
  sm.qqplot(x) #defaut comparison distribution is standard normal
  plt.show()

  # extract mean and standard deviation from generated data
  mu_new, sigma_new = norm.fit(x)

  # print given and generated parameters
  print("univariate inverse transform took {} seconds".format(toc - tic))
  print("given: mu = {}, sigma = {}".format(mu, sigma)) # print given parameters
  print("generated: mu = {}, sigma = {}".format(mu_new, sigma_new)) # print generated parameters

  return x

## BOX-MULLER

# generate sample from normal distribution using Box-Muller Transform
def BoxMuller(mu,sigma,n,plot=False):
  
  # start counter
  tic = time.perf_counter() 

  # simulate theta and r
  u1, u2 = np.random.uniform(0,1,n), np.random.uniform(0,1,n)
  r = np.sqrt(-2*np.log(1-u1))
  theta = 2*np.pi*u2

  # calculcate x coordinate
  x = np.cos(theta)*r*sigma + mu

  # end counter
  toc = time.perf_counter() 

  if plot == True:

    # plot histogram of generated data
    plt.hist(x,density=True)
    plt.show()

    # plot qqplot of generated data
    sm.qqplot(x)
    plt.show()

    # mean and standard deviation from generated data
    mu_new, sigma_new = norm.fit(x)
    
    print("univariate Box Muller transform took {} seconds".format(toc - tic))
    print("given: mu = {}, sigma = {}".format(mu, sigma)) # print given parameters
    print("generated: mu = {}, sigma = {}".format(mu_new, sigma_new)) # print generated parameters

  return x

def BoxMullerAlt(n,mu=0,sigma=1):

  # start counter
  tic = time.perf_counter() 

  # simulate random uniformly distributed data of size n
  u1, u2 = np.random.uniform(0,1,n), np.random.uniform(0,1,n)

  # apply inverse cdf of exponential to u1
  s = -np.log(1-u1)  

  # transform u2 to theta
  theta = 2*np.pi*u2

  # convert s to r
  r = np.sqrt(2*s)

  # generate z1,z2 from r and theta
  z1, z2 = (r*np.cos(theta)*sigma + mu), (r*np.sin(theta)*sigma + mu)

  # end counter
  toc = time.perf_counter() 

  # plot histogram and scatterplot
  fig = plt.figure(figsize=plt.figaspect(0.4))
  ax = fig.add_subplot(1,2,1)
  ax.hist(z1)
  ax.hist(z2)
  ax = fig.add_subplot(1,2,2)
  ax.scatter(z1,z2,s=0.5)

  # plot qqplot of generated data
  sm.qqplot(z1)
  plt.show()
  sm.qqplot(z2)
  plt.show()

  # extract mean and standard deviation from generated data
  mu_1, sigma_1 = norm.fit(z1)
  mu_2,sigma_2 = norm.fit(z2)

  # print given and generated parameters
  print("univariate alternative Box Muller transform took {} seconds".format(toc - tic))
  print("given: mu = {}, sigma = {}".format(mu, sigma)) # print given parameters
  print("generated z1: mu = {}, sigma = {}".format(mu_1, sigma_1)) # print generated parameters for z1
  print("generated z2: mu = {}, sigma = {}".format(mu_2, sigma_2)) # for z2


## CENTRAL LIMIT THEOREM

def CentralLimitTheorem(mu,sigma,n_means,n_samples):
  
  # start counter
  tic = time.perf_counter() 

  # generate data
  matrix = np.random.uniform(0,1,size=(n_means,n_samples))
  
  # calculcate sample means
  sampleMeans = matrix.sum(axis=0)/n_samples

  # standardize
  mean = np.mean(sampleMeans)
  sd = np.std(sampleMeans)
  standard = (sampleMeans - mean)/sd

  # scale by sigma, translate by mu
  normal = standard*sigma+mu
  
  # end counter
  toc = time.perf_counter() 

  # plot histogram of generated data
  plt.hist(normal)
  plt.show()

  # plot qqplot of generated data
  sm.qqplot(normal)
  plt.show()

  # mean and standard deviation from generated data
  mu_new = np.mean(normal)
  sigma_new = np.std(normal)

  # print given and generated parameters
  print("univariate CLT took {} seconds".format(toc - tic))
  print("given: mu = {}, sigma = {}".format(mu, sigma)) # print given parameters
  print("generated: mu = {}, sigma = {}".format(mu_new, sigma_new)) # print generated parameters


## MULTIVARIATE INVERSE TRANSFORM

np.random.seed(000)

# helper function to determine if inputted cov_matrix is valid covariance matrix
def is_cov_mat(x,tol=1e-8):

  # check if matrix is symmetric
  if np.array_equal(x,x.T): 

    # check if diagonal entries are non-negative  
    if any(n < 0 for n in x.diagonal()):
      return False
    
    # check if matrix is positive semi-definite
    else:      
      E = np.linalg.eigvalsh(x)
      return np.all(E > -tol)   
  else:
     return False

# generate sample from multivariate normal distribution using Inverse Transform Sampling
def multiNormalSample_ITS(mu, cov_matrix, n_samples, n_dim):
  if is_cov_mat(cov_matrix):
    # start counter
    tic = time.perf_counter() 
  
    # initialize sample matrix as zeros
    z = np.zeros((n_dim,n_samples)) 

    # populate matrix independently with values from the univariate standard normal
    for d in list(range(0,n_dim)):
      u = np.random.uniform(0, 1, n_samples)
      z[d] = norm.ppf(u)
    
    # compute square root of covariance matrix through eigenvalue decomposition
    eigval, eigvec = np.linalg.eig(cov)
    print(eigval)
    print(eigvec)
    eigval_sqrt = np.diag(np.sqrt(eigval))
    print(eigval_sqrt)
    cov_sqrt = np.matmul(np.matmul(eigvec, eigval_sqrt),eigvec.T)

    # compute 
    x = np.matmul(cov_sqrt,z) + mu 
  
    # end counter
    toc = time.perf_counter()
  
    # print time taken
    print("multivariate sampling took {} seconds".format(toc - tic)) 
    
    # print eigenvalues
    print("the eigenvalues of the covariance matrix were {}".format(eigval)) 

    return x

  else:
    return("Inputted covariance matrix is invalid")


## MULTIVARIATE BOX-MULLER

# generate sample from multivariate normal distribution using Box Muller Transform
def multiNormalSample_BM(mu, cov_matrix, n_samples, n_dim):
  if is_cov_mat(cov_matrix):
    # start counter
    tic = time.perf_counter() 
  
    # initialize sample matrix as zeros
    z = np.zeros((n_dim,n_samples)) 

    # populate matrix independently with values from the univariate standard normal
    for d in list(range(0,n_dim)):
      # simulate theta and r
      theta = np.random.uniform(0,2*np.pi,n_samples)
      r = np.sqrt(2*np.random.exponential(1,n_samples))

     # calculcate x coordinate
      z[d] = np.cos(theta)*r
    
    # compute square root of covariance matrix through eigenvalue decomposition
    eigval, eigvec = np.linalg.eig(cov_matrix)
    eigval_sqrt = np.diag(eigval*1/2)
    cov_sqrt = np.matmul(np.matmul(eigvec, eigval_sqrt),eigvec.T)

    # print eigenvalues
    # print("eigenvalues of covariance matrix:", eigval)

    # compute 
    x = np.matmul(cov_sqrt,z) + mu 
  
    # end counter
    toc = time.perf_counter()
  
    # print time take
    # print("multivariate sampling took {} seconds".format(toc - tic)) 
    
    return x
  
  else:
    return("Inputted covariance matrix is invalid")


## MAKE BLOBS

def make_n_blobs(n_samples=100,n_features=2,centers=4,mu_coef=20,sigma_coef=4,user_mu=[],weights=[0.1,0.2,0.3,0.4]):
  # divide total number of samples by number of blobs (no remainder)
  n_samples //= centers # each blob will have this many data points

  # check if user has provided array of mean vectors
  manual_mu = False
  if np.shape(user_mu) == (centers,n_features): # (num blobs x num dimensions)
    # will use user inputted mu
    manual_mu = True
  else:
    # initialize mu
    mu = np.random.randn(n_features,1)

  tic = time.perf_counter() # start timer

  # initialize data matrix
  data = np.zeros((n_features,n_samples * centers))
  data_labels = np.zeros((n_samples * centers, 1))

  # plt.figure()
  # loop through blobs
  for i in range(centers):
    # generate a random invertible matrix
    A = np.random.randn(n_features,n_features) # generate a random matrix

    # compute eigenvalue decomoposition of symmetric matrix
    A_symmetric = A @ A.transpose()
    D, U = np.linalg.eig(A_symmetric)

    # generate eigenvalues close to each other    
    eig_starter = max(2,np.random.randn()) # CONTROLS COVARIANCE
    eigs = np.zeros(n_features) + eig_starter
    for j in range(1,n_features):
      adder = np.random.rand() / sigma_coef
      eigs[j] = eig_starter * (1 + adder)
    eigs = np.array(list(eigs))
    
    # generate matrix with desired eigenvalues
    new_A = U @ np.diag(eigs) @ np.linalg.inv(U)

    # generate positive semidefinite covariance matrix
    cov_matrix = new_A @ new_A.transpose()

    # generate mean vector
    if manual_mu == True:
      mu = np.array(user_mu[i]).reshape((n_features,1)) # use user inputted mu
    else:
      mu += mu_coef * np.random.rand(n_features,1) # CONTROLS MEAN

    # generate data
    X = multiNormalSample_BM(mu, cov_matrix, n_samples, n_features) # use Box Muller    
    data[:,n_samples*i:n_samples*(i+1)] = X
    data_labels[n_samples*i:n_samples*(i+1)] = i

  toc = time.perf_counter() # end timer

  print("from scratch make blobs took {} seconds".format(toc - tic))

  # output full data matrix
  return(data, data_labels)


  ## MAKE BLOBS - MIXTURE MODEL

def mixture_make_blobs(n_samples=100,n_features=2,centers=4,
                       mu_coef=20,sigma_coef=4,mu_vec=[],
                       bins=[0.3,0.5,0.8,1]):
  # if user has not provided valid array of mean vectors, initialize
  manual_mu = True
  if np.shape(mu_vec) != (centers,n_features): # (num blobs x num dimensions)
    manual_mu = False
    mu_vec = []
    mu = np.random.randn(n_features,1)

  tic = time.perf_counter() # start timer

  # initialize covariance vector
  cov_vec = []

  # initialize data matrix
  data = np.zeros((n_features,n_samples))
  data_labels = np.zeros((n_samples, 1))

  # loop through blobs to create means and covariance
  for i in range(centers):
    # generate a random invertible matrix
    A = np.random.randn(n_features,n_features) # generate a random matrix

    # compute eigenvalue decomoposition of symmetric matrix
    A_symmetric = A @ A.transpose()
    D, U = np.linalg.eig(A_symmetric)

    # generate eigenvalues close to each other    
    eig_starter = np.random.uniform(1,2) # CONTROLS COVARIANCE
    eigs = np.zeros(n_features) + eig_starter
    for j in range(1,n_features):
      adder = np.random.rand() / sigma_coef
      eigs[j] = eig_starter * (1 + adder)
    eigs = np.array(list(eigs))
    
    # generate matrix with desired eigenvalues
    new_A = U @ np.diag(eigs) @ np.linalg.inv(U)

    # generate positive semidefinite covariance matrix
    cov_matrix = new_A @ new_A.transpose()
    cov_vec.append(cov_matrix)

    # generate mean vector
    if manual_mu == False:
      mu += mu_coef * np.random.rand(n_features,1) # CONTROLS MEAN
      mu_vec.append(mu.tolist())

  # loop through samples and draw from mixture distribution
  for j in range(n_samples):
    # generate random number in [0,1]
    alpha = np.random.uniform()

    # identify the index of weighted bin in which alpha belongs
    k = np.digitize(alpha,bins)

    # use mean and covariance from the chosen index
    mu = mu_vec[k]
    cov_matrix = cov_vec[k]

    # generate data
    X = multiNormalSample_BM(mu, cov_matrix, 1, n_features).flatten() # use Box Muller   
    data[:,j] = X
    data_labels[j] = k

  toc = time.perf_counter() # end timer

  print("mixture make blobs took {} seconds".format(toc - tic))
  # output full data matrix
  return(data, data_labels)


## GENERATE UNIFORM DATA ON SPHERE
def unif_sphere(n_samples=1000,n_features=3,radius=1,plot=False):
  # generate random normal data
  X = np.zeros((n_features,n_samples))

  for i in range(n_features):
    x=BoxMuller(mu=0, sigma=1, n=n_samples,plot=False).tolist()
    X[i,:] = x

  for i in range(n_samples):
    X[:,i] = X[:,i]/np.linalg.norm(X[:,i]) * radius
 
  if plot != False:
    for j in range(n_features):
      sm.qqplot(X[j,:], dist=scipy.stats.uniform)

  # plot
  if plot != False:
    if n_features == 2:
      plt.scatter(X[0,:],X[1,:])
    elif n_features == 3:
      fig = plt.figure(figsize=plt.figaspect(1))
      ax = fig.add_subplot(projection='3d')
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      ax.set_zlabel('z')
      ax.scatter(X[0,:],X[1,:],X[2,:])
      ax.view_init(0,45)
    else:
      print("no visualization available")

  return(X)

def uniform_sphere_polar(n_samples = 1000, r = 1,plot=False):

  phi = np.random.uniform(0,np.pi,n_samples)
  theta = np.random.uniform(0,2*np.pi,n_samples) 

  x=r*np.cos(phi)*np.sin(theta)
  y=r*np.sin(phi)*np.sin(theta)
  z=r*np.cos(theta)

  if plot != False:

    sm.qqplot(x, dist=scipy.stats.uniform)
    sm.qqplot(y, dist=scipy.stats.uniform)
    sm.qqplot(z, dist=scipy.stats.uniform)

    fig = plt.figure(figsize=plt.figaspect(1))
    ax=fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(x,y,z)
    ax.view_init(0,45)
  
  return(x,y,z)

def uniform_circle(n_samples = 1000, r =1,plot=False):
  theta = np.random.uniform(0,2*np.pi,n_samples)
  x = r*np.sin(theta)
  y=r*np.cos(theta)

  # sm.qqplot(x, dist=scipy.stats.uniform)

  # sm.qqplot(y, dist=scipy.stats.uniform)

  if plot != False:
    plt.figure()
    plt.scatter(x,y)
    plt.scatter(theta)
  
  return(x,y)

def uniform_sphere_from_normal(n_samples = 1000, r = 1,plot=False):

  x=BoxMuller(mu=0, sigma=1, n=n_samples,plot=False)
  y=BoxMuller(mu=0, sigma=1, n=n_samples,plot=False)
  z=BoxMuller(mu=0, sigma=1, n=n_samples,plot=False)

  for i in range(n_samples):
    v=[x[i],y[i],z[i]]
    x[i]=x[i]/np.linalg.norm(v)
    y[i]=y[i]/np.linalg.norm(v)
    z[i]=z[i]/np.linalg.norm(v)

  x,y,z = x*r, y*r, z*r

  if plot != False:
    fig = plt.figure(figsize=plt.figaspect(1))
    ax=fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(x,y,z)
    ax.view_init(0,45)

  return(x,y,z)


## GENERATE UNIFORM DATA ON CYLINDER

def unif_cylinder(n_samples=1000,radius=1,height=100,n_features=3):
  # initialize data matrix
  X = np.zeros((n_features,n_samples))

  circ_dim = n_features - 1
  # generate a 2-d uniformly random circles
  X[:circ_dim,:] = unif_sphere(n_samples=n_samples,n_features=circ_dim,radius=1)
  
  # add uniformly random values of the third dimension to circles
  X[circ_dim,:] = np.random.uniform(0,height,n_samples).tolist()

#   fig = plt.figure(figsize=plt.figaspect(0.5))
#   ax = fig.add_subplot(1,2,2,projection='3d')
#   ax.set_xlabel('x')
#   ax.set_ylabel('y')
#   ax.set_zlabel('z')
#   ax.scatter(X[0,:],X[1,:],X[2,:])
#   ax.view_init(90,0)

#   ax = fig.add_subplot(1,2,1,projection='3d')
#   ax.set_xlabel('x')
#   ax.set_ylabel('y')
#   ax.set_zlabel('z')
#   ax.scatter(X[0,:],X[1,:],X[2,:])
#   ax.view_init(45,45)
  return(X)

## GENERATE UNIFORM DATA ON SQUARE

def uniform_square(side_length=1,n_samples=1000,center=[0,0],dim=2,angle_degree = 45):
    
    angle = angle_degree* np.pi/180 #degrees to radians 
    dist_from_center = side_length/2
    samples_per_side = math.floor(n_samples/4)
    
    top_x  = np.random.uniform(0,side_length,samples_per_side) - dist_from_center
    top_y = np.repeat(dist_from_center,samples_per_side)
    top = np.asarray((top_x,top_y)).transpose()
    
    bottom_x  = np.random.uniform(0,side_length,samples_per_side) - dist_from_center
    bottom_y = np.repeat(-dist_from_center,samples_per_side)
    bottom = np.asarray((bottom_x,bottom_y)).transpose()
    
    left_y  = np.random.uniform(0,side_length,samples_per_side) - dist_from_center
    left_x = np.repeat(-dist_from_center,samples_per_side)
    left = np.asarray((left_x,left_y)).transpose()
    
    right_y = np.random.uniform(0,side_length,samples_per_side) - dist_from_center
    right_x = np.repeat(dist_from_center,samples_per_side)
    right = np.asarray((right_x,right_y)).transpose()
    
    X = np.concatenate((top,bottom,left,right))

    #rotate the square
    for i in range(n_samples):
        point = X[i]
        x = point[0]
        y = point[1]
        #rotation matrix
        X[i][0] = (x*np.cos(angle)) - (y*np.sin(angle))
        X[i][1] = (x*np.sin(angle)) + (y*np.cos(angle))

    #shift shape from the center
    X += center  

    #plot 
    plt.scatter(X[:,0],X[:,1])
    plt.axis("equal")


def square(side_length=1,n_samples=1000,dim1=True, dim2= True, dim3 = True, adder = 0,plot=False):

  X = np.zeros((3,n_samples)) + adder

  if dim1 == True:
    X[0,:] = np.random.uniform(0,side_length,n_samples) 
  if dim2 == True:  
    X[1,:] = np.random.uniform(0,side_length,n_samples)
  if dim3 == True:
    X[2,:] = np.random.uniform(0,side_length,n_samples)


  if plot != False:
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(X[0,:],X[1,:],X[2,:])
    ax.view_init(45,45)

  return(X)


## GENERATE UNIFORM DATA ON CUBE
## FILLED CUBE

def cube(side_length=1,n_samples=10000):

  X = np.zeros((3,n_samples))

  X[0,:] = np.random.uniform(0,side_length,n_samples) 
  X[1,:] = np.random.uniform(0,side_length,n_samples) 
  X[2,:] = np.random.uniform(0,side_length,n_samples) 

  fig = plt.figure(figsize=plt.figaspect(1))
  ax = fig.add_subplot(projection='3d')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.scatter(X[0,:],X[1,:],X[2,:])
  ax.view_init(45,45)

## EMPTY CUBE

def empty_cube(side_length=1,n_samples=10000,center=[0,0,0],dim=2,angle_degree = 45,plot=True):
    
    center = np.array(center).reshape(3,1)
    angle = angle_degree* np.pi/180 #degrees to radians 
    dist_from_center = side_length/2
    samples_per_side = math.floor(n_samples/6)

 
    side1 = square(n_samples=samples_per_side, dim1=False) - dist_from_center
    side2 = square(n_samples=samples_per_side, dim1=False, adder = side_length) - dist_from_center

    side3 = square(n_samples=samples_per_side, dim2=False) - dist_from_center
    side4 = square(n_samples=samples_per_side, dim2=False, adder = side_length) - dist_from_center

    side5 = square(n_samples=samples_per_side, dim3=False) - dist_from_center
    side6 = square(n_samples=samples_per_side, dim3=False, adder = side_length) - dist_from_center

    #verify cube
    X_hollow = np.concatenate((side1,side2,side3,side4),axis=1) + center

    X = np.concatenate((side1,side2,side3,side4,side5,side6),axis=1) + center

    if plot != False:
      fig = plt.figure(figsize=plt.figaspect(0.5))
      ax = fig.add_subplot(1,2,1,projection='3d')
      ax.scatter(X[0,:],X[1,:],X[2,:])
      ax.view_init(45,45)

      ax = fig.add_subplot(1,2,2,projection='3d')
      ax.scatter(X_hollow[0,:],X_hollow[1,:],X_hollow[2,:])
      ax.view_init(90,0)
    
    return(X)