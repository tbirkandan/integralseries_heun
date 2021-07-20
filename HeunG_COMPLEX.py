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


####################################################################
########################### PREPARATIONS ###########################
####################################################################
# SubDivide : procedure preparing the computations
def SubDivide(A,B,NumberPoints,N2,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon):

   CUTS = np.linspace(A,B,num = NumberPoints + 1) # Time cuts: points in time where one subintervals finishes and the other starts, here A and B are the real parts of the coordinates of the interval
   
   NumForward = 0
   NumBack = 0
   
   for i in range(NumberPoints):
      if z0.real > CUTS[i+1]:  # Indicates the 'arrow of time', meaning that is after whole interval [CUTS[i], CUTS[i+1]]
         NumBack += 1      # Counts the number of backward intervals
      elif z0.real >= CUTS[i]:  # Indicates the 'arrow of time', meaning z0 is in this interval
         PosZ0 = i         # Records the location of the initial condition
      else:                # Indicates the 'arrow of time', meaning on this interval z<z0
         NumForward += 1   # Counts the number of forward intervals 

   # The first interval to be treated is that containing the initial condition
   a = CUTS[PosZ0]
   b = CUTS[PosZ0+1]
   
   if a == z0.real:   
      Z, smdelta = np.linspace(z0.real, b, num = N2, retstep=True) # Equidistant N2 points between a and b
      Z = Z + z0.imag*1j # Add back imaginary part, path is parallel to real axis
      H = procedure1(Z,smdelta,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,N2,1)
      smdelta2 = smdelta
      smdelta1 = smdelta
   
   elif b == z0.real:
      
      Z, smdelta = np.linspace(a, z0.real, num = N2, retstep=True) # Equidistant N2 points between a and b
      Z = Z + z0.imag*1j # Add back imaginary part, path is parallel to real axis
      Z = Z[::-1] # Reverse z direction
      H = procedure1(Z,smdelta,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,N2,-1)
      Z = Z[::-1] # Reverse z direction
      H = H [::-1] # Reverse h direction
      smdelta2 = smdelta
      smdelta1 = smdelta

   else:
      d = max(2, int(N2 * abs(z0.real-a)/(b-a)))
      dd = max(2,N2 - d)

      # Backward in first interval from z0 to a < z0 
      Z1, smdelta1 = np.linspace(a, z0.real, num = d, retstep=True) 
      Z1 = Z1 + z0.imag*1j # Add back imaginary part, path is parallel to real axis
      Z1 = Z1[::-1] # Reverse z direction
      H1 = procedure1(Z1,smdelta1,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,-1)
      Z1 = Z1[::-1] # Reverse z direction
      H1 = H1[::-1] # Reverse h direction
     
      # Forward in first interval from z0 to b > z0
      Z2, smdelta2 = np.linspace(z0.real, b, num = dd, retstep=True)
      Z2 = Z2 + z0.imag*1j # Add back imaginary part, path is parallel to real axis
      H2 = procedure1(Z2,smdelta2,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,dd,1)
      
      Z = np.concatenate([Z1,Z2])
      H = np.concatenate([H1,H2])

   # From the above, extract boundary conditions at the boundary of interval
   h = len(H)
   
   PosCondForward = Z[h-2].real
   H0Forward = H[h-2]
   H0pForward = (H[h-1]-H[h-3])/(2.0*smdelta2)
   Z = Z[:-1] # Removal of the last element, this is because it is the penultimate element which is the last one with a correctly approximated derivative
   H = H[:-1]
   
   PosCondBack = Z[1].real 
   H0Back = H[1]
   H0pBack = (H[2]-H[0])/(2.0*smdelta1)

   # At this point we have the Heun function on the interval containing z0. Now we can move forward from there
   if NumForward > 0:

      for i in range(1,NumForward+1):
           a = PosCondForward # CUTS[PosZ0 + i]
           b = CUTS[PosZ0 + i + 1]
              
           Zi, smdelta = np.linspace(a, b, num = N2, retstep=True) # Equidistant d points between a and b
           Zi = Zi + z0.imag*1j
           Hi = procedure1(Zi,smdelta,PosCondForward+z0.imag*1j,H0Forward,H0pForward,alpha,beta,q,t,gamma,delta,epsilon,N2,1) 

           # Update the boundary values for next interval
           h = len(Hi)
           CondForward = Zi[h-2].real 
           orward = Hi[h-2]
           Forward = (Hi[h-1]-Hi[h-3])/(2.0*smdelta)  
           Zi = Zi[:-1] # Removal of the last element, this is because it is the penultimate element which is the last one with a correctly approximated derivative
           Hi = Hi[:-1] 
           Z = np.concatenate([Z,Zi])
           H = np.concatenate([H,Hi])

   # At this point we have the Heun function for z > z0. Now we can move backward from there to z < z0
   if NumBack > 0:

      for i in range(1,NumBack+1):
           b = PosCondBack # CUTS[PosZ0 - i + 1]
           a = CUTS[PosZ0 - i]

           Zi, smdelta = np.linspace(a, b, num = N2, retstep=True) # Equidistant d points between a and b
           Zi = Zi + z0.imag*1j
           Zi = Zi[::-1] # Reverse z direction
           Hi = procedure1(Zi,smdelta,PosCondBack+z0.imag*1j,H0Back,H0pBack,alpha,beta,q,t,gamma,delta,epsilon,N2,-1) 
           Zi = Zi[::-1] # Reverse z direction
           Hi = Hi[::-1] # Reverse h direction
           
           # Update the boundary values for next interval
           PosCondBack = Zi[1].real
           H0Back = Hi[1]
           H0pBack = (Hi[2]-Hi[0])/(2.0*smdelta)
           Zi = Zi[1:] # Removal of the first element, this is because it is the second element which is the first one with a correctly approximated derivative
           Hi = Hi[1:]
           Z = np.concatenate([Zi,Z])
           H = np.concatenate([Hi,H])
    
   return Z, H

##########################################
# Procedure 1 : tests whether one of the two pieces of the Heun computation can be avoided due to peculiar boundary conditions
def procedure1(Z,smdelta,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,direction):
              
   if (H0==0 and H0p==0):
        H = np.zeros(d) # d zero vector

   elif (H0==0):
        H = procedure3(Z,smdelta,
                       z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,direction)

   elif (H0==H0p):
        H = procedure2(Z,smdelta,
                       z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,direction)

   else:
        H1 = procedure2(Z,smdelta,
                        z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,direction)
        H2 = procedure3(Z,smdelta,
                        z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,direction)
        H = H1+H2
    
   return H


####################################################################
############# MATHEMATICAL HEART OF THE CODE #######################
####################################################################
# Procedure 2 : relevant whenever H0 is non-zero. Sole procedure when H0=H0p
def procedure2(Z,smdelta,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,direction):

    # Calculation of K1:

    # Integration of modified Iij now only over successive small intervals. Approximation through trapezoidal rule
    fun1 = np.exp(Z)*(Z**gamma)*((Z-1)**delta)*((t-Z)**epsilon)
    fun2 = (q-alpha*beta*Z)/((Z-1)*(Z)*(Z-t)) - ( (epsilon/(Z-t))+(gamma/Z)+(delta/(Z-1))+1 )

    fun1x2 = fun1*fun2
    e = (fun1x2[1:d]+fun1x2[0:d-1])/2 # Trapezoidal integration over the infinitesimal interval from (i-1)dt to idt. There should be a *dt but it is put later on to avoid numerical stability issues

    e = np.insert(e, 0, 0, axis=0)    # Adds back the first point
    k1 = np.cumsum(e)                 # Relies on integral linearity to get the integral over larger intervals, i.e. Int_a^b = Int_a^c + Int_c^b
    k1x, k1y = np.meshgrid(k1, k1, sparse=False)
    K1 = k1x - k1y

    f1x, _ = np.meshgrid(fun1, fun1, sparse=False)
    K1 = K1/f1x                       # K1 is now a matrix, entry i,j of which is K1(t_i,t_j)

    K1 = direction * smdelta * K1 + 1 # Add 1 to all elements on and above the diagonal, mathematically this is adding a Heaviside function.

    # Calculation of contribution of G1 to H:
    Id = np.identity(d)               # dxd identity matrix
    M = direction * K1 # Possible numerical pre-conditioning before inversion: Id + np.exp(-smdelta * K1) - 1, does not significantly impact the accuracy here so left out
    b = np.zeros(d, dtype = complex)
    b[0] = 1

    dM = np.diag(np.diag(M))
    Mat1 = Id - M*smdelta + dM*smdelta/2.0

    G1 = linalg.solve_triangular(Mat1,b,trans=1) # This is the first column of thew *-resolvent, i.e. (1-K2)^*-1(z,z0).
                                                 # Here the *-resolvent is taken with respect to the trapezoidal rule of integration
    G1[0] = 0

    R1 = (G1[1:d]+G1[0:d-1])/2.0       # G1 is the Green's function. The solution is its integral
    R1 = np.insert(R1, 0, 0, axis=0)
    R1 = np.cumsum(R1)
    H = H0*(1+R1)                      # Here the 1+ comes from the Dirac delta distribution, written explicitely to avoid numerical issues it would otherwise raise
    
    return H

##########################################
# Procedure 3 : relevant whenever H0p - H0 is not zero. Sole procedure when H0=0z,
def procedure3(Z,smdelta,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon,d,direction):

    # Calculation of K2:
    K2 = np.zeros([d,d], dtype = np.clongfloat)
  
    fun1 = (q-alpha*beta*Z)/((Z-1)*(Z)*(Z-t))
    fun1e = np.exp(Z)*fun1
    fun2  = np.exp(Z)*(-(epsilon/(t-Z))+(gamma/Z)+(delta/(Z-1))+1)

    for i in range(d):
        K2[i] = np.exp(-Z[i])*(fun1e-fun2)-fun1
      
    # Calculation of contribution of G2 to H:
    Id = np.identity(d) # dxd identity matrix
    M = direction*K2 #np.triu(Id + np.exp(- smdelta * K2) - T) # Numerical pre-conditioning before inversion, may not be needed
   
    b = np.zeros(d, dtype = np.clongfloat)
    b[0] = 1

    dM = np.diag(np.diag(M))
    Mat1 = Id - M*smdelta + dM*smdelta/2.0
 

    G2 = linalg.solve_triangular(Mat1,b,trans=1, check_finite=False) # This is the first column of thew *-resolvent, i.e. (1-K2)^*-1(z,z0). 
                                                 # Here the *-resolvent is taken with respect to the trapezoidal rule of integration
    G2[0] = 2*(G2[0]-1)  # Effective removal of the Dirac delta distribution corresponding to K2^*0
    
    eG2 = np.exp(-Z)*G2     # Builds up exp(-zeta)G2(zeta,z0)
    
    sumeG2 = (eG2[1:d]+eG2[0:d-1])/2.0   # Integration of eG2 with respect to zeta
    sumeG2 = np.insert(sumeG2, 0, 0, axis=0)
    sumeG2 = np.cumsum(sumeG2)
 
    resG2 = np.exp(Z)*sumeG2 # Multiply result by exp(z)
   
    intG2 = (G2[1:d]+G2[0:d-1])/2.0   # Integration of G2
    intG2 = np.insert(intG2, 0, 0, axis=0)
    intG2 = np.cumsum(intG2)

    R2 = np.exp(Z-z0) - 1  + resG2 - intG2 # This is exp(z-z0)-1+int_z0^z (exp(z-zeta)-1)G2(zeta,z0) dzeta
   
    H = (H0p-H0) * R2

    return H
###################################################################



####################################################################
########################### MAIN PROGRAM ###########################
####################################################################
######################## HEUN PARAMETERS 
### Time parameters :
# Real parts of the interval required [A,B], computations will be parallel to real axis with imaginary part that of z0
A = -0.1
B = 3

### Heun parameters :
t = 1+0.01*1j
q = -1.0+1j*0

alpha = 1.0+1j*0
beta = -1.5+1j*0
gamma = -0.14+1j*0
delta = 4.32+1j*0
epsilon = 1.0 + alpha + beta - gamma - delta # Mathematica's convention

######################## BEGINNING OF PROCEDURE 
start_time = time.time()
N1 = 100 # Total number of subintervals in [A,B], at least 2 for simulating HeunG
N2 = 5000 # Path-sum points per subinterval

N1left = max(1,int(N1*abs(A)/abs(B-A))) # Number of subintervals in [A,0]
N1right = max(1,N1-N1left) # Number of subintervals in [0,B]

## z0 must lie somewhere on the line [A,B] parallel to real axis.
z0 = -0.001+5*0.000001*1j 
## Initial conditions close to 0
H0 = 1 + q*z0/(t*gamma)-z0**2*((alpha*beta+(q*(-1-q-alpha-beta+delta-t*(gamma+delta)))/(t*gamma))/(2*t*(1+gamma)))-z0**3*(((q*(1+alpha)*(1+beta))/(t*gamma)-((alpha*beta+(q*(-1-q-alpha-beta+delta-t*(gamma+delta)))/(t*gamma))*(-q-2*(2+alpha+beta-delta+t*(1+gamma+delta))))/(2*t*(1+gamma)))/(3*t*(2+gamma)))
H0p = q/(t*gamma) - z0/(t*(1+gamma))*( alpha*beta + q/(t*gamma)*(-1-q-alpha-beta+delta-t*(gamma+delta)) )-z0**2*(((q*(1+alpha)*(1+beta))/(t*gamma)-((alpha*beta+(q*(-1-q-alpha-beta+delta-t*(gamma+delta)))/(t*gamma))*(-q-2*(2+alpha+beta-delta+t*(1+gamma+delta))))/(2*t*(1+gamma)))/(t*(2+gamma)))
Z1,H1 = SubDivide(A.real,z0.real,N1left,N2,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon) # Backward from 0

z0 = 0.001+5*0.001*1j 
# Initial conditions close to 0
H0 = 1 + q*z0/(t*gamma)-z0**2*((alpha*beta+(q*(-1-q-alpha-beta+delta-t*(gamma+delta)))/(t*gamma))/(2*t*(1+gamma)))-0*z0**3*(((q*(1+alpha)*(1+beta))/(t*gamma)-((alpha*beta+(q*(-1-q-alpha-beta+delta-t*(gamma+delta)))/(t*gamma))*(-q-2*(2+alpha+beta-delta+t*(1+gamma+delta))))/(2*t*(1+gamma)))/(3*t*(2+gamma)))
H0p = q/(t*gamma) - z0/(t*(1+gamma))*( alpha*beta + q/(t*gamma)*(-1-q-alpha-beta+delta-t*(gamma+delta)) )-z0**2*(((q*(1+alpha)*(1+beta))/(t*gamma)-((alpha*beta+(q*(-1-q-alpha-beta+delta-t*(gamma+delta)))/(t*gamma))*(-q-2*(2+alpha+beta-delta+t*(1+gamma+delta))))/(2*t*(1+gamma)))/(t*(2+gamma)))

Z2,H2 = SubDivide(z0.real,B.real,N1right,N2,z0,H0,H0p,alpha,beta,q,t,gamma,delta,epsilon) # Forward from 0

Z = np.concatenate([Z1,Z2])
H = np.concatenate([H1,H2])
          

########################## OUTPUTS 
print("--- %s seconds ---" % (time.time() - start_time))
print('Number of points', len(Z))

# Plot the result:
plt.figure()
plt.plot(Z.real, H.real, color='b')
plt.xlabel('$Re(z)$') 
plt.ylabel('$Re(H_G(z))$') 
plt.show()

plt.figure()
plt.plot(Z.real, H.imag, color='r')
plt.xlabel('$Re(z)$') 
plt.ylabel('$Im(H_G(z))$') 
plt.show()

###Example print values of H_G as close as possible from values z=z1 and z=z2
z1=B
z2=0.8+0.5*0.00001*1j 

Z1=abs(Z-z1)
mZ1 = min(abs(Z-z1))
Z2=abs(Z-z2)
mZ2= min(abs(Z-z2))

i1=np.where(np.ravel(Z1)==mZ1)
i2=np.where(np.ravel(Z2)==mZ2)

print(Z[i1])
print(H[i1])

print(Z[i2])
print(H[i2])

