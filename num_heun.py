###############################
# For theory, please see:
# P.-L. Giscard, A. Tamar,
# arXiv:2010.03919 [math-ph]
# 
# Code by T. Birkandan
# birkandant@itu.edu.tr
###############################

import numpy as np
from matplotlib import pyplot as plt

##########################################
# Integral calculation needed for H0=H0p
def Iij(Zi,Zj,alpha,beta,q,t,gamma,delta,epsilon,smdelta):
    
    zeta=(Zi+Zj)/2.0
    fun1=(zeta**gamma)*((zeta-1)**delta)*((t-zeta)**epsilon)*np.exp(zeta)
    fun2=((Zj)**gamma)*((Zj-1)**delta)*((t-Zj)**epsilon)
    fun3=(q-alpha*beta*zeta)/((zeta-1)*(zeta)*(zeta-t))
    fun4=(epsilon/(t-zeta))+(gamma/zeta)+(delta/(zeta-1))+1
    integrand=(fun1/fun2)*(fun3-fun4)
    
    return(np.exp(-Zj)*(smdelta*integrand))

##########################################
# Primary procedure
def procedure1(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon):
    
    if (H0==0 and H0p==0):
        d=int((b-a)/smdelta)
        Z=np.linspace(a,b,num=d)
        H=np.zeros_like(Z)
    elif (H0==0):
        Z,H=procedure3(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
    elif (H0==H0p):
        Z,H=procedure2(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
    else:
        Z,H1=procedure2(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
        Z,H2=procedure3(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
        H=H1+H2
    
    return (Z,H)

##########################################
# Procedure for H0=H0p
def procedure2(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon):
    
    d=int(((b-a)/smdelta)+1)
    Z=np.linspace(a,b,num=d)
    H=np.zeros_like(Z)
    T=np.triu(np.ones([d,d]), k=0)
    Id=np.identity(d)
    
    K1=np.zeros_like(Id)
    for i in range(d):
        for j in range(i,d):
            K1[i,j]=1+Iij(Z[i],Z[j],alpha,beta,q,t,gamma,delta,epsilon,smdelta)
    
    R=np.linalg.inv(Id-K1*smdelta)-Id
    R=np.dot(R,T)
    H=H0*(1+R[0]) # not sure!
    
    return (Z,H)

##########################################
# Procedure for H0=0
def procedure3(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon):
    
    d=int(((b-a)/smdelta)+1)
    Z=np.linspace(a,b,num=d)
    H=np.zeros_like(Z)
    Id=np.identity(d)
    
    T=np.triu(np.ones([d,d]), k=0)
    for i in range(d):
        for j in range(i,d):
            T[i,j]=np.exp((j-i)*smdelta)-1
    
    K2=np.zeros_like(Id)
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
a=6
b=25
z0=6
smdelta=0.05
H0=1
H0p=1
alpha=1
beta=3/2.0
q=1
t=4
gamma=2
delta=7
#epsilon=alpha+beta+1-gamma-delta
epsilon=-1

Z1,H1=procedure1(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon)
#print(Z1)
print("***********")
#print(H1)

plt.plot(Z1,H1)
plt.show()
