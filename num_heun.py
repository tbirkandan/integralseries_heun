###############################
# For theory, please see:
# P.-L. Giscard, A. Tamar,
# arXiv:2010.03919 [math-ph]
# 
# Code by T. Birkandan
# birkandant@itu.edu.tr
# Python 3.8.5
###############################

##########################################
# import the modules numpy, scipy and matplotlib:
import numpy as np
from scipy.integrate import quad
from matplotlib import pyplot as plt

##########################################
# Integrand for H0=H0p
def Iij(zeta,Zi,Zj,alpha,beta,q,t,gamma,delta,epsilon):
    
    fun1=(zeta**gamma)*((zeta-1)**delta)*((t-zeta)**epsilon)*np.exp(zeta)
    fun2=((Zj)**gamma)*((Zj-1)**delta)*((t-Zj)**epsilon)
    fun3=(q-alpha*beta*zeta)/((zeta-1)*(zeta)*(zeta-t))
    fun4=(epsilon/(t-zeta))+(gamma/zeta)+(delta/(zeta-1))+1
    integrand=(fun1/fun2)*(fun3-fun4)
    
    return integrand

##########################################
# Primary procedure
def procedure1(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon):
    
    if (H0==0 and H0p==0):
        d=int((b-a)/smdelta) # Number of points
        Z=np.linspace(a,b,num=d) # Equidistant d points between a and b
        H=np.zeros([d,d]) # dxd zero matrix
    elif (H0==0):
        Z,H=procedure3(a,b,z0,smdelta,
                       H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
    elif (H0==H0p):
        Z,H=procedure2(a,b,z0,smdelta,
                       H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
    else:
        Z,H1=procedure2(a,b,z0,smdelta,
                        H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
        Z,H2=procedure3(a,b,z0,smdelta,
                        H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
        H=H1+H2
    
    return (Z,H)

##########################################
# Procedure for H0=H0p
def procedure2(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon):
    
    d=int(((b-a)/smdelta)+1) # Number of points
    Z=np.linspace(a,b,num=d) # Equidistant d points between a and b
    H=np.zeros([d,d]) # dxd zero matrix
    # Upper triangular matrix with 0 if i>j, 1 elsewhere:
    T=np.triu(np.ones([d,d]), k=0) 
    Id=np.identity(d) # dxd identity matrix
    
    K1=np.zeros([d,d]) # dxd zero matrix
    for i in range(d):
        for j in range(i,d):
            # Integration of Iij using quad from scipy module:
            myIij=quad(Iij,Z[i],Z[j]
                       ,args=(Z[i],Z[j],alpha,beta,q,t,gamma,delta,epsilon))
            K1[i,j]=1+np.exp(-Z[j])*myIij[0]
    
    R=np.linalg.inv(Id-smdelta*K1)-Id
    R=np.dot(R,T) # Matrix multiplication
    H=H0*(1+R[0]) # not sure!
    
    return (Z,H)

##########################################
# Procedure for H0=0
def procedure3(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon):
    
    d=int(((b-a)/smdelta)+1) # Number of points
    Z=np.linspace(a,b,num=d) # Equidistant d points between a and b
    H=np.zeros([d,d]) # dxd zero matrix
    Id=np.identity(d) # dxd identity matrix
    
    T=np.triu(np.ones([d,d]), k=0)
    for i in range(d):
        for j in range(i,d):
            T[i,j]=np.exp((j-i)*smdelta)-1
    
    K2=np.zeros([d,d])
    for i in range(d):
        for j in range(i,d):
            fun1=(q-alpha*beta*Z[j])/((Z[j]-1)*(Z[j])*(Z[j]-t))
            fun2=(epsilon/(t-Z[j]))+(gamma/Z[j])+(delta/(Z[j]-1))+1
            K2[i,j]=np.exp(Z[j]-Z[i])*(fun1-fun2)-fun1
    
    R=np.dot(np.linalg.inv(Id-K2*smdelta),T)
    H=(H0p-H0)*R[0]
    
    return (Z,H)

##########################################
# Main program

# Initial location, interval, $\Delta$:
z0=6.0
a=z0
b=26.0
smdelta=0.05 # $\Delta$

# Heun parameters:
alpha=1.0
beta=3.0/2.0
gamma=2.0
delta=7.0
epsilon=-1.0
q=1.0
t=4.0 # Location of the regular singularity other than {0,1,oo}

# Initial conditions:
H0=1.0 # Ini. cond. for the function
H0p=1.0 # Ini. cond. fot the 1st derivative of the function
#H0p=q/(gamma*t) # May be used for HG (from the series solution).

# Call the main procedure:
Zresult,Hresult=procedure1(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)

# Plot the result:
plt.plot(Zresult,Hresult)
plt.xlabel('$z$') 
plt.ylabel('$H_G(z)$') 
plt.show()
