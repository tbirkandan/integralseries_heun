###############################
# For theory, please see:
# P.-L. Giscard, A. Tamar,
# arXiv:2010.03919 [math-ph]
# 
# Code by T. Birkandan & P.-L. Giscard
# birkandant@itu.edu.tr
#
# Language: Python 3.8.5
###############################

##########################################
# import the modules numpy, scipy and matplotlib:
import numpy as np
from scipy.integrate import cumtrapz
from scipy import linalg
from matplotlib import pyplot as plt
import time

##########################################
# Primary procedure
def procedure1(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d):
   
   Z = np.linspace(a,b,num=d) # Equidistant d points between a and b
 
   if (H0==0 and H0p==0):
        H = np.zeros([d,d]) # dxd zero matrix

   elif (H0==0):
        Z,H = procedure3(a,b,z0,smdelta,
                       H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,Z)

   elif (H0==H0p):
        Z,H = procedure2(a,b,z0,smdelta,
                       H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,Z)

   else:
        Z,H1 = procedure2(a,b,z0,smdelta,
                        H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,Z)
        Z,H2 = procedure3(a,b,z0,smdelta,
                        H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,Z)
        H = H1+H2
    
   return (Z,H)

##########################################
# Procedure, relevant whenever H0 is non-zero. Sole procedure when H0=H0p
def procedure2(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,Z):
    
    # Calculation of K1:
    e = np.zeros(d,dtype=complex) # d zero list
    fun1 = np.zeros(d,dtype=complex)
   
    #fun34 = np.array([(q-alpha*beta*Z[0])/((Z[0]-1)*(Z[0])*(Z[0]-t)) - ((epsilon/(Z[0]-t))+(gamma/Z[0])+(delta/(Z[0]-1))+1), 0],dtype=complex)
  
    # Integration of modified Iij now only over successive small intervals. Approximation through trapezoidal rule
    fun1 = np.exp(Z)*(Z**gamma)*((Z-1)**delta)*((t-Z)**epsilon)
    fun2 = (q-alpha*beta*Z)/((Z-1)*(Z)*(Z-t)) - ( (epsilon/(Z-t))+(gamma/Z)+(delta/(Z-1))+1 )

    fun1x2 = fun1*fun2
    e = (fun1x2[1:d]+fun1x2[0:d-1])/2 # Trapezoidal integration over the infinitesimal interval from (i-1)dt to idt. There should be a *dt but it is put later on to avoid numerical stability issues
    e = np.insert(e, 0, 0, axis=0)
    
    k1 = np.cumsum(e) # Relies on integral linearity to get the integral over larger intervals, i.e. Int_a^b = Int_a^c + Int_c^b
    k1x, k1y = np.meshgrid(k1, k1, sparse=False)
    K1 = k1x - k1y
    
    f1x, _ = np.meshgrid(fun1, fun1, sparse=False)
    K1 = K1/f1x
    #for j in range(1,d):
    #      K1[:,j]=K1[:,j]/fun1[j]        
    
    K1 = smdelta * K1 + 1 # add 1 to all elements on and above the diagonal, mathematically this is adding a Heaviside function.

    # Calculation of H:
    Id = np.identity(d) # dxd identity matrix
    M = Id - smdelta * K1 # Possible numerical pre-conditioning before inversion: Id + np.exp(-smdelta * K1) - 1, does not significantly impact the accuracy here so left out
    b = np.zeros(d,dtype=complex)
    b[0]=1

    G1 = linalg.solve_triangular(M,b,trans=1) # Extracts only the first column of the resolvent of K1
    G1[0] = 2*smdelta

    R1 = cumtrapz(G1,initial = 0) # G1 is the Green's function. The solution is its integral
    H = H0*(1+R1) # Here the 1+ comes from the Dirac delta distribution, written explicitely to avoid numerical issues it would otherwise raise
    
    return (Z,H)

##########################################
# Procedure : relevant whenever H0p - H0 is not zero. Sole procedure when H0=0
def procedure3(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,Z):
    
    
    # Calculation of K2:
    K2=np.zeros([d,d],dtype=complex)
  
    fun1 = (q-alpha*beta*Z)/((Z-1)*(Z)*(Z-t))
    fun1e = np.exp(Z)*fun1
    fun2  = np.exp(Z)*(-(epsilon/(t-Z))+(gamma/Z)+(delta/(Z-1))+1)

    for i in range(d):
        K2[i] = np.exp(-Z[i])*(fun1e-fun2)-fun1
      

    # Calculation of H:
    Id = np.identity(d) # dxd identity matrix
    M = Id-smdelta*K2 #np.triu(Id + np.exp(- smdelta * K2) - T) # Numerical pre-conditioning before inversion, may not be needed
    
    b = np.zeros(d,dtype=complex)
    b[0] = 1
    G2 = linalg.solve_triangular(M,b,trans=1) # This is the first column of thew *-resolvent, i.e. (1-K2)^*-1(z,z0)
    G2[0] = G2[0]-1 # Effective removal of the Dirac delta distribution corresponding to K2^*0
    
    eG2 = np.exp(-Z)*G2     # Builds up exp(-zeta)G2(ZRangezeta,z0)
    sumeG2 = np.cumsum(eG2)  # Integration of the above with respect to zeta
    resG2 = np.exp(Z)*sumeG2 # Multiply result by exp(z)

    R2 = np.exp(Z-z0) - 1 + resG2 - cumtrapz(G2, initial = G2[0]) # This is exp(z-z0)-1+int_z0^z (exp(z-zeta)-1)G2(zeta,z0) dzeta
    
    H = (H0p-H0) * R2 
    
    return (Z,H)

##########################################
# Main program

start_time = time.time()

# Initial point z0, interval (a,b), step size smdelta:
z0 = 5.0
a = z0
b = 10.0
smdelta = 0.0025 # Must NOT be 1
d=int(((b-a)/smdelta)+1)

print('Number of points', d)

# Heun parameters:
# Mathematical warning : the smallest interval containing a,b, and z0 must not contain any singularity of the Heun function
alpha = 1.3+1j*0
beta = 0.12+1j*0
gamma = -0.14+1j*0
delta = 4.32+1j*0
epsilon = 1.0+alpha+beta-gamma-delta
q = -0.2+1j*0
t = 4.3+1j*0 


# Initial conditions:
H0 = 1.02272 - 1j*0.0503288 # Ini. cond. for the function
H0p = 0.0438095 + 1j*0.0182909 # Ini. cond. for the 1st derivative of the function

# Call the main procedure:
Zresult,Hresult = procedure1(a,b,z0,smdelta,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d)

print("--- %s seconds ---" % (time.time() - start_time))

# Plot the result:
plt.figure()
plt.plot(Zresult,Hresult.real)
plt.xlabel('$z$') 
plt.ylabel('$Re(H_G(z))$') 
plt.show()

plt.figure()
plt.plot(Zresult,Hresult.imag)
plt.xlabel('$z$') 
plt.ylabel('$Im(H_G(z))$') 
plt.show()



