# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:29:59 2023

@author: Colton and Nikhil
"""
# =============================================================================
# X direction is along the grain boundry (oxygen to aluminum or vice versa)
# Y direction is across the grain boundry/bulk
# =============================================================================

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%% DIFFERENCE SCHEMES

def diffusion(x, h, D): #diffusion in x direction
    uiPlus1 = np.copy(x)
    uiMinus1 = np.copy(x)
    metalBoundryTotal = np.sum(x[:,len(x[0,:])-1]) #addition of concentrations at metal boundry
    for i in range(len(x[:,0])):
        uiPlus1[i,:] = np.append(x[i,1:], x[i,len(x[0,:])-1]) # THIS IS FOR NO DIFFUSION INTO METAL
        #uiPlus1[i,:] = np.append(x[i,1:],metalBoundryTotal/len(x[:,0])) #this boundry condiditon is eq. 23 (unweighted average atm)
        uiMinus1[i,:] = np.append(1,x[i,:-1])
    return (np.multiply(D,(uiPlus1 + -2*x + uiMinus1) / h**2))

def periodicdiffusion(x, h, D): #diffusion in y direction
    uiPlus1 = np.copy(x)
    uiMinus1 = np.copy(x)
    for i in range(len(x[0,:])):
        uiPlus1[:,i] = np.append(x[1:,i],x[:1,i])
        uiMinus1[:,i] = np.append(x[-1:,i],x[:-1,i])
    return (np.multiply(D,(uiPlus1 + -2*x + uiMinus1) / h**2))

def mirrordiffusion(x, h, D): #for y direction half boundry
    uiPlus1 = np.copy(x)
    uiMinus1 = np.copy(x)
    for i in range(len(x[0,:])):
        uiPlus1[:,i] = np.append(x[1:,i],x[len(x[:,0])-1,i])
        uiMinus1[:,i] = np.append(x[0,i],x[:-1,i])
    return (np.multiply(D,(uiPlus1 + -2*x + uiMinus1) / h**2))

def isotopeDiffusion(x, co, h, Do, L, fl, Omega, cop):
    uiPlus1 = np.copy(x)
    uiMinus1 = np.copy(x)
    coPlus1 = np.copy(co)
    coMinus1 = np.copy(co)
    metalBoundryTotal = np.sum(x[:,-1]) #addition of concentrations at metal boundry
    coBoundaryTotal = np.sum(co[:,-1])
    for i in range(len(x[:,0])):
        uiPlus1[i,:] = np.append(x[i,1:],x[i,len(x[0,:])-1]) # THIS IS FOR NO DIFFUSION INTO METAL
        # uiPlus1[i,:] = np.append(x[i,1:],metalBoundryTotal/len(x[:,0])) #this boundry condiditon is eq. 23 (unweighted average atm)
        uiMinus1[i,:] = np.append(0,x[i,:-1])
        # coPlus1[i,:] = np.append(co[i,1:],coBoundaryTotal/len(co[:,0])) 
        coPlus1[i,:] = np.append(co[i,1:],co[i,len(co[0,:])-1]) 
        coMinus1[i,:] = np.append(1,co[i,:-1])

    return (np.multiply(Do,(uiPlus1 + -2*x + uiMinus1) / h**2)) \
    - ((1/cop)*np.multiply(co,(np.multiply(Do,(uiPlus1 + -2*x + uiMinus1) / h**2)))) \
    - ((1/cop))*np.multiply(Do,np.multiply((coPlus1-coMinus1)/2*h,(uiPlus1 - uiMinus1) / 2*h))

def mirrorIsotopeDiffusion(x, co, h, Do, L, fl, Omega, cop):
    uiPlus1 = np.copy(x)
    uiMinus1 = np.copy(x)
    coPlus1 = np.copy(co)
    coMinus1 = np.copy(co)
    for i in range(len(x[0,:])):
        uiPlus1[:,i] = np.append(x[1:,i],x[len(x[:,0])-1,i])
        uiMinus1[:,i] = np.append(x[0,i],x[:-1,i])
        coPlus1[:,i] = np.append(co[1:,i],co[len(co[:,0])-1,i])
        coMinus1[:,i] = np.append(co[0,i],co[:-1,i])

    return (np.multiply(Do,(uiPlus1 + -2*x + uiMinus1) / h**2)) \
    - ((1/cop)*np.multiply(co,(np.multiply(Do,(uiPlus1 + -2*x + uiMinus1) / h**2))))\
    - ((1/cop))*np.multiply(Do,np.multiply((coPlus1-coMinus1)/2*h,(uiPlus1 - uiMinus1) / 2*h))\
    


#%% RUNGE KUTTA METHODS

def RK4(x, dt, dx, D):
    fa = diffusion(x, dx, D)
    fb = diffusion(x + fa*dt/2, dx, D)
    fc = diffusion(x + fb*dt/2, dx, D)
    fd = diffusion(x + fc*dt, dx, D)
    return 1/6 *(fa + 2*fb + 2*fc + fd) * dt

def periodicRK4(x, dt, dx, D):
    fa = periodicdiffusion(x, dx, D)
    fb = periodicdiffusion(x + fa*dt/2, dx, D)
    fc = periodicdiffusion(x + fb*dt/2, dx, D)
    fd = periodicdiffusion(x + fc*dt, dx, D)
    return 1/6 *(fa + 2*fb + 2*fc + fd) * dt

def mirrorRK4(x, dt, dx, D):
    fa = mirrordiffusion(x, dx, D)
    fb = mirrordiffusion(x + fa*dt/2, dx, D)
    fc = mirrordiffusion(x + fb*dt/2, dx, D)
    fd = mirrordiffusion(x + fc*dt, dx, D)
    return 1/6 *(fa + 2*fb + 2*fc + fd) * dt

def isotopeRK4(x, co, h, Do, L, fl, Omega, cop, dt):
    fa = isotopeDiffusion(x, co, h, Do, L, fl, Omega, cop)
    fb = isotopeDiffusion(x + fa*dt/2, co, h, Do, L, fl, Omega, cop)
    fc = isotopeDiffusion(x + fb*dt/2, co, h, Do, L, fl, Omega, cop)
    fd = isotopeDiffusion(x + fc*dt, co, h, Do, L, fl, Omega, cop)
    return 1/6 *(fa + 2*fb + 2*fc + fd) * dt

def mirrorIsotopeRK4(x, co, h, Do, L, fl, Omega, cop, dt):
    fa = mirrorIsotopeDiffusion(x, co, h, Do, L, fl, Omega, cop)
    fb = mirrorIsotopeDiffusion(x + fa*dt/2, co, h, Do, L, fl, Omega, cop)
    fc = mirrorIsotopeDiffusion(x + fb*dt/2, co, h, Do, L, fl, Omega, cop)
    fd = mirrorIsotopeDiffusion(x + fc*dt, co, h, Do, L, fl, Omega, cop)
    return 1/6 *(fa + 2*fb + 2*fc + fd) * dt

#EULER METHODS

def euler(x, dt, dx, D):
    return((dt * diffusion(x, dx, D)))

def mirroreuler(x, dt, dx, D):
    return((dt * mirrordiffusion(x, dx, D)))

def periodiceuler(yn, stepsize, t, h):
    return(yn + (stepsize * periodicdiffusion(t, yn, h)))


def isotopeeuler(x, co, h, Do, L, fl, Omega, cop, dt):
    return (dt*(isotopeDiffusion(x, co, h, Do, L, fl, Omega, cop)))

def mirrorIsotopeeuler(x, co, h, Do, L, fl, Omega, cop, dt):
    return (dt*(mirrorIsotopeDiffusion(x, co, h, Do, L, fl, Omega, cop)))


#%% EXTRA BOUNDRY CONDITIONS
def metalBoundry(x, D, i):
    return

#%% Initializing

def initial_left(x): #unsaturated oxygen
    stemp = (int((ymax-ymin)/dx)+1,int((xmax-xmin)/dx)+1)
    y = np.zeros(stemp)
    return(y)

def initalOxygen(x):
    stemp = (int((ymax-ymin)/dx)+1,int((xmax-xmin)/dx)+1)
    y = np.ones(stemp)
    return(y*(1-(4*10e-4)))

#Setting step size

dx=0.1 #SPATIAL STEP SIZE
dy=0.1 #the program runs on a square grid everything is set to dh. this will need to be changed to change the step size in grain boundry
h=0.1
xmin=0 
xmax=1 #INITIAL RANGE BETWEEN OXYGEN AND ALLUMINIUM
ymin=0
ymax=0.3 #DISTANCE IN BULK DIRECTION
dt=0.001 #TEMPORAL STEP SIZE

cop=1
L=1
fl=1
Omega=1


s=(int((ymax-ymin)/dx)+1,int((xmax-xmin)/dx)+1) #number of nodes (used in defining D matrix)
x = np.linspace(int(xmin),xmax,int((xmax-xmin)/dx)+1) #this just defines the axis for plots
y = np.linspace(int(xmin),int(xmax),int((xmax-xmin)/dx)+1) #this is only used in 3d plotting

# MATRIX OF DIFFUSION COEFICIENTS
D = np.ones(s)
for i in range (1,int((ymax-ymin)/dx)+1):
    D[i,:] = 0.0001
D[:,-1]=1

#Setting the starting conditions
u = initial_left(x)
#v = initial_right(x)
o18 = initial_left(x)
oTotal = initalOxygen(x)


print('x=',x)
plt.figure(1)
plt.plot(x,u[0,:])
print('y=', u[0,:])
plt.show()

#%% THIS LOOP IS FOR 1D only
# for i in range (0, 20000000):
#     u = RK4(u, dt, h, D, 1)
#     # v = RK4(v, dt, h, D/2, 2)
#     maxes.append(np.max(u))
#     times.append(i*dt)
#     if i%500 == 0:
#         # plt.plot(x,v, label = 't = '+str(i*0.001))
#         plt.plot(x,u, label = 't = '+str(i*0.001))
#         plt.xlim(-1,20)
#         plt.ylim(0,100)
#         plt.axvline(x = 0, color = 'b')
#         plt.axvline(x = xmax, color = 'b')
#         plt.show()
#     if u[len(u)-1] > 30:
#         u = np.append(u,0)
#         # v = np.append(0,v)
#         xmax = xmax + h
#         x = np.append(x,xmax+h)
#     # if v[0] > 25:
#     #     u = np.append(u,0)
#     #     v = np.append(0,v)
#     #     xmax = xmax + h
#     #     x = np.append(x,xmax+h)


#%%Keeping track of tracer in new nodes
tracer = []

#%%This loop is for 3D
for i in range (0,200000000):
    oTotal = oTotal + euler(oTotal, dt, dx, D) + mirroreuler(oTotal, dt, dx, D)
    o18 = o18 + isotopeeuler(o18, oTotal, dx, D, L, fl, Omega, cop, dt) + mirrorIsotopeeuler(o18, oTotal, dx, D, L, fl, Omega, cop, dt)
    oTotal[:,0] = 1
    o18[:,0] = 1
    
    # if i%50 == 0: THIS IS FOR A 3D PLOT
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     ax.contour3D(x, y, u, 50, cmap='coolwarm')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #     ax.set_title('t='+ str(i))
    #     plt.show()
        
    if i%100000 == 0: #2D projections
        plt.close()
        plt.plot(x, oTotal[0,:], label = 'Total O gb')
        plt.plot(x,o18[0,:], label = 'O18% gb')
        plt.plot(x,o18[1,:], label = 'O18% bulk1')
        plt.plot(x,o18[2,:], label = 'O18% bulk2')
        plt.title('Tracer Diffusion in Grain Boundary and Bulk')
        plt.xlabel('Position')
        plt.ylabel('Concentration')
        plt.axvline(x = 0, color = 'b')
        plt.axvline(x = xmax, color = 'b')
        plt.ylim(0,1.05)
        plt.legend()
        plt.savefig(str(i)+'.png')
        plt.show(block=False)
        print(o18)
        print(oTotal)
    
    if np.sum(oTotal[:,-1]) > 3-3*(4*10e-4): #Moving boundry
        plt.close()
        plt.plot(x, oTotal[0,:], label = 'Total O gb')
        plt.plot(x,o18[0,:], label = 'O18% gb')
        plt.plot(x,o18[2,:], label = 'O18% bulk')
        plt.xlabel('Position')
        plt.ylabel('Concentration')
        plt.axvline(x = 0, color = 'b')
        plt.axvline(x = xmax, color = 'b')
        plt.ylim(0,1.05)
        plt.legend()
        plt.savefig('New nodes/'+str(i)+'.png')
        plt.show(block=False)
    
        oTotal = np.c_[oTotal,(1-10e-4)*np.zeros(len(oTotal[:,0]))]
        o18 = np.c_[o18,o18[:,-1]]
        xmax = xmax + dx
        x = np.append(x,xmax)
        D[:,-1]=0.0001
        D[0,-1]=1
        D = np.c_[D,[1, 1, 1]]
        # D = np.c_[D,[1, 0.001, 0.001]]
        
        print('at', i,': O18 Concentration is: ', o18[0,-1], ' at GB ')
        tracer.append(o18[0,-1])
        print(o18[0,:])
        np.savetxt('nodes.csv', tracer, delimiter= ', ', fmt= '%s ')

        
        
        
#%%
plt.close()
plt.plot(np.linspace(0,len(tracer), len(tracer)), tracer)
plt.ylabel('concentration of O18')
plt.xlabel('node #')
plt.title('concentration of tracer in new nodes')
plt.savefig('newO18InBulk.png')
np.savetxt('nodes.csv', tracer, delimiter=", ", fmt='% s')
