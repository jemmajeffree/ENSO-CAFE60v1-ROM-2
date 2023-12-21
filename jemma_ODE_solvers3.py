import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.optimize
#Last updated 25th Sep 2021

def analytic_2D_ODE_IVP(y0, A, dt):
    '''The analytic solution to a 2D ODE system with dy/dt = Ay
    Returns y values for a step forwards of dt from each y0
    '''
    def real_r():
        t0_eig = np.array((( (-a12),   (-a12)  ),
                         ( (mu-eta), (mu+eta) )))
        k = np.linalg.lstsq(t0_eig,y0,rcond=None)[0]
        x1 = k[0]*np.exp((a11-mu+eta)*dt)*(-a12)   + k[1]*np.exp((a11-mu-eta)*dt)*(-a12)
        x2 = k[0]*np.exp((a11-mu+eta)*dt)*(mu-eta) + k[1]*np.exp((a11-mu-eta)*dt)*(mu+eta) 
        return np.array((x1,x2))    
    def repeated_r():
        t0_eig = np.array(((  (-a12),   (0)  ),
                           (  (mu),     (-1) )))        
        
        k = np.linalg.lstsq(t0_eig,y0,rcond=None)[0]
        x1 = np.exp((a11-mu)*dt)*(  k[0]*(-a12) + k[1]*(-a12*dt)  )
        x2 = np.exp((a11-mu)*dt)*(  k[0]*mu     + k[1]*(-1+mu*dt)  )
        return np.array((x1,x2))
    def complex_r():
        t0_eig = np.array(((  (-a12),  0 ),
                         (  (mu),     (-eta) )))        
        
        k = np.linalg.lstsq(t0_eig,y0,rcond=None)[0]
        x1 = np.exp((a11-mu)*dt)*(  k[0]*-a12*np.cos(eta*dt)                   + k[1]*-a12*np.sin(eta*dt)                    ) 
        x2 = np.exp((a11-mu)*dt)*(  k[0]*(mu*np.cos(eta*dt)+eta*np.sin(eta*dt)) + k[1]*(mu*np.sin(eta*dt)-eta*np.cos(eta*dt))  )
        return np.array((x1,x2))
    
    #Unpack constants
    a11 = A[0,0] #That's gonna cause trouble unless I'm careful
    a12 = A[0,1]
    mu = (A[0,0]-A[1,1])/2
    eta = np.sqrt(np.abs(mu**2+A[0,1]*A[1,0]))
    
    #Determine system type
    radicand = mu**2+A[0,1]*A[1,0]
    #print(scipy.linalg.eig(A.reshape((2,2)))[0])
    if np.abs(radicand)<10**(-6) and radicand !=0 :
        pass
        #warnings.warn("You may run into errors with floating point arithmatic here - is it supposed to be repeated?")
    if radicand < -10**(-6): #Anything close to 0 is being classed as zero - executive decision (to help with floating point calculation error)
        return complex_r()
    elif radicand > 10**(-6):
        return real_r()
    else:
        return repeated_r()
    
def fast_2D_A(y,dt):
    
    ''' Finds the coefficient matrix for a linear ODE system from approximations of 
    dy/dt (linear fit between the points on either side, which happens to be the same 
    as a quadratic fit). Not especially accurate and tends to lag, but a useful first-order
    approximation 
    y has form 2, N'''
    
    dy = (y[:,2:]-y[:,:-2])/(dt*2)
    return np.linalg.lstsq(y.T[1:-1],dy.T, rcond=None)[0].T
    
    
def timeseries_optimiser(flat_A,Y,dt):
    ''' A function to be passed to scipy.optimise, which estimates the best
    A for a least squares fit for a RK4 solved system with timestep h (for the least squares)
    and data to interpolate Y, evenly spaced with vector of times t (floats)
    '''
    
    A = np.reshape(flat_A,(Y.shape[0],Y.shape[0]))
    
    return np.sum((Y[:,1:] - analytic_2D_ODE_IVP(Y[:,:-1],A,dt))**2)


def paired_optimiser(flat_A,Y0,Y1,dt):
    ''' A function to be passed to scipy.optimise, 
    Returns the sum of residuals squared for predictions forward from Y0 to Y1
    Y0, Y1 have shape (2, N)
    flat_A is (4,)
    dt is a scalar, probably 1 (I've never tested what happens when it isn't)
    '''

    assert Y0.shape == Y1.shape
    
    A = np.reshape(flat_A,(2,2))
    return np.sum((Y1-analytic_2D_ODE_IVP(Y0,A,dt))**2)

def paired_optimiser_exp(flat_A,Y0,Y1,dt):
    ''' A function to be passed to scipy.optimise, 
    Returns the sum of residuals squared for predictions forward from Y0 to Y1
    Y0, Y1 have shape (2, N)
    flat_A is (4,)
    dt is a scalar, probably 1 (I've never tested what happens when it isn't)
    Should give the same answer as above, but uses a different analytic solution
    '''

    assert Y0.shape == Y1.shape
    
    A = np.reshape(flat_A,(2,2))
    return np.sum((Y1-np.matmul(scipy.linalg.expm(A*dt),Y0))**2)
