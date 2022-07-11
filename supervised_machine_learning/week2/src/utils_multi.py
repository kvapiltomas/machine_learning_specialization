import numpy as np
import copy
import math
#import matplotlib.pyplot as plt

def load_house_data():
    data = np.loadtxt('data/houses.txt', delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X,y

def compute_gradient_matrix(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X : (array_like Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) Values of parameters of the model      
      b : (scalar )                Values of parameter of the model      
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
                                  
    """
    m, n = X.shape
    f_wb = X * w + b
    err = f_wb - y
    dj_dw = (1/m) * (X.T  err)
    dj_db = (1/m) * np.sum(err)

    return dj_db, dj_dw

def compute_cost(X, y, w, b):
    """
    compute cost

    Args:
        X (ndarray) : Shape (m,n) matrix of examples with multiple features
        y (ndarray) : shape (n,) target values
        w (ndarray) : Shape (n)   parameters for prediction   
        b (scalar) : parameter  for prediction   
    Returns
        cost (scalar) : cost
    """
    m = X.shape
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2*m)
    return(np.squeeze(cost))    


def gradient_descent_houses(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking num_iters gradient steps with learning rate alpha.

    Args:
        X (array_like Shape (m,n)) : matrix of examples
        y (array_like Shape (m,)) : target value of each example
        w_in (array_like Shape (n,)) : Initial values of parameters of the model
        b_in (scalar) : Initial value of parameter of the model
        cost_function : function to compute cost
        gradient_function : function to compute the gradient
        alpha (float) : Learning rate
        num_iters (int) : number of iterations to run gradient descent
    Returns
        w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
        b : (scalar)                Updated value of parameter of the model after
          running gradient descent    
    """
    m = len(X)
    hist = {}
    hist["cost"] = []; hist["params"] = []; hist["grads"]=[]; hist["iter"]=[];

    w = copy.deepcopy(w_in)
    b = b_in
    save_interval = np.ceil(num_iters/10000)

    print(f"Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ")
    print(f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|")

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J,w,b at each save interval for graphing
        if i == 0 or i % save_interval == 0:     
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w,b])
            hist["grads"].append([dj_dw,dj_db])
            hist["iter"].append(i)

        if i% math.ceil(num_iters/10) == 0:
            #print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, y, w, b)
            print(f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")
       
    return w, b, hist


def run_gradient_descent(X, y, iterations=1000, alpha = 1e-6):
    m, n = X.shape
    initial_w = np.zeros(n)
    initial_b = 0
    w_out, b_out, hist_out = gradient_descent_houses(X, y, initial_w, initial_b, 
                                                    compute_cost, compute_gradient_matrix,
                                                    alpha, iterations)
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")
    
    return(w_out, b_out, hist_out)                                                
