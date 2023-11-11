"""IMPORTS"""
from random import random as randex
from math import sqrt,cos,sin,pi
from matplotlib import pylab as plt
from numpy import mean

"""PARAMETERS"""

#Influencing computation time
P    =  1000	#(2^Pmax = Number of citizens                      [10^3-10^6]
rep  =    10    #Repetition for same P
#City
Nc   =    400 	#Number of locations                               [100]
L    =     10.0	#(Km): city size                                   [10]
#Car mobility
Vmax =     35.0	#(Km/h): free traveling speed                      [35]
t0   =      3.0	#Inverse Value of Time [Note: Vmax*t0 = l ~ 100Km  [3]
c    =      10.0#Capacity in Bureau of Public Roads function       [1000]
mi   =      3.0 #Exponent in Bureau of Public Roads function       [3]
#Subway mobility
Ns   =     20   #Number of metro stations
wd   =     0.5  #(Km - walking distance)
Vsub =     20.0 #(Km/h) Average metro-travel speed
Vwalk=     5.0  #(Km/h) Walking speed





def calculateGoodNc(aMi):
    const = 0.25
    Pstar = c*(Vmax*t0/L/Nc)**(1/aMi)
    k = 1+(P/Pstar/const)**(mi/(1+aMi))	
    return k


""" Check Saturation Problems"""

x = 1+[i/100.0 for i in range(1,300)]
y = [calculateGoodNc(i) for i in x]

plt.plot(x,y)


"""CODE"""

def countNotZero(x):
    C = 0
    for i in x:
        if i>0:
            C+=1
    return C
    
def metroTime(Ox,Oy,Dx,Dy,Sx,Sy,Vsub,Vwalk):
    
    #X and Y Distances...
    dOx = []
    dDx = []
    dOy = []
    dDy = []

    for x in Sx:
        dOx.append(Ox-x)
        dDx.append(Dx-x)
        
    for y in Sy:
        dOy.append(Oy-y)
        dDy.append(Dy-y)
        
    
    #... added up
    d1 = []
    d2 = []
    for i in range(len(Sx)):
        d1.append(sqrt(dOx[i]**2 + dOy[i]**2))
        d2.append(sqrt(dDx[i]**2 + dDy[i]**2))
        
    #Find best itinerary
    dw1 = min(d1)
    i1 = d1.index(dw1)
    dw2 = min(d2)
    i2 = d2.index(dw2)
    
    #Compute travel time
    dw = dw1+dw2
    ds = sqrt((Sx[i1]-Sx[i2])**2 + (Sy[i1]-Sy[i2])**2 )

    return ds/Vsub + dw/Vwalk
            



def fillAcity(Nc,P,L,Vmax,t0,c,mi,Ns,Vsub,Vwalk,wd):
    #Create Locations
    #[ASSUMPTION: Uniform Distribution w/ a natural central place]
    #[DEVELOPEMENT: Locations should become objects]
    
    
    Xl = [0] 	#Place 1 is at city center
    Yl = [0]
    U =  [1] 	#Maximum Utility
    Tc = [0] 	#Car Inflow location
    T = [0] 	#Total Inflow Location
    Vc = [Vmax]	#Traveling speed
    
    for i in range(1,Nc):
        Xl.append(L*(rand()-0.5))
        Yl.append(L*(rand()-0.5))
        U.append(rand())         #This is \eta
        Tc.append(0)
        T.append(0)
        Vc.append(Vmax)           #They are not really using a speed interpretation
        
    #A random place at wd from city center
    r = rand()*wd
    th = 2*pi*rand()   
    Xs = [r*cos(th)]
    Ys = [r*sin(th)]
    for i in range(1,Ns):
        Xs.append(L*(rand()-0.5))
        Ys.append(L*(rand()-0.5))
    
    
    #Person association to Places
    #[ASSUMPTION: Uniform Distribution, Uncorrelated w/ locations], No Fractal Dimension]
    
    """ MY COMMENT: this progressive insertion of people is out from wardrop quilibrium """
    
    Xp = []
    Yp = [] 
    Pstar = 0
    for i in range(P):
        Xp.append(L*(rand()-0.5)) 	
        Yp.append(L*(rand()-0.5))
    
        #Compute distances and times
        Z = []
        pickCar = []
        for j in range(Nc):
            d = sqrt((Xp[i]-Xl[j])**2 + (Yp[i]-Yl[j])**2 )
   	    tc = d/Vc[j]                  #[DEVELOPMENT: here place code for subways]
   	    
   	    tm = metroTime(Xp[i],Yp[i],Xl[j],Yl[j],Xs,Ys,Vsub,Vwalk)
   	                
            t = min(tc,tm)
            
   	    Z.append(U[j] - t/t0)       #[ASSUMPTION: Linear Value of Time]
   	    pickCar.append(t == tc)
    
   	
        #Find optimal working places
        wp = Z.index(max(Z))                #[ASSUMPTION: 100% optimal choice]

        #Critical value for populaton
        if wp > 0 and Pstar == 0:
            Pstar = i
            
        #Update chosen location Inflow and Speed
        T[wp]+=1

        if pickCar[wp]:
            Tc[wp]+=1
            Vc[wp] = Vmax /(1+(Tc[wp]/c)**mi)	#[ASSUMPTION: Bureau of Public Roads congestion, No correlation between d and v]
     
    return [countNotZero(T),Pstar]

""" THERE IS A PROBLEM IN THE EVALUATION IF Pstar == 0"""

""" Population growth"""


Pmax =9
def populationGrowth():
    Pvec = [2**i for i  in range(1,Pmax+1)]
            
    k = []
    Pstar = []
    for Pi in Pvec:
        print Pi
        krow = []
        Psrow = []
        for i in range(rep):
            output = fillAcity(Nc,Pi,L,Vmax,t0,c,mi,Ns,Vsub,Vwalk,wd)
            krow.append(output[0])
            Psrow.append(output[1])
        k.append(krow)
        Pstar.append(Psrow)

    k_m = [ mean(x) for x in k]
    Pstar_m = [ mean(x) for x in Pstar]

    return k_m


"""Subway Growth"""

Smax = 8

def subwayGrowth():
    Svec = [2**i for i  in range(1,Smax+1)]
            
    k = []
    Pstar = []
    for Si in Svec:
        print Si
        krow = []
        Psrow = []
        for i in range(rep):
            output = fillAcity(Nc,P,L,Vmax,t0,c,mi,Si,Vsub,Vwalk,wd)
            krow.append(output[0])
            Psrow.append(output[1])
        k.append(krow)
        Pstar.append(Psrow)
    
    
    k_m = [ mean(x) for x in k]
    

    return k_m

"""Dependance upon mi """



def muDependence():
    mu_vec = [1.5+i*0.5 for i  in range(0,5)]

    k = []
    Pstar = []
    for mu_i in mu_vec:
        print mu_i
        if calculateGoodNc(mu_i) > Nc:
		print "Saturation Warning: Nc="+str(Nc)+" safeK: "+str(calculateGoodNc(mu_i))        
        krow = []
        Psrow = []
        for i in range(rep):
            output = fillAcity(Nc,P,L,Vmax,t0,c,mu_i,Ns,Vsub,Vwalk,wd)
            krow.append(output[0])
            Psrow.append(output[1])
        k.append(krow)
        Pstar.append(Psrow)
    
    
    k_m = [ mean(x) for x in k]
 
    return k_m
"""TO SEE Pstar WE NEED TO AVERAGE ON MANY RUNS""" 
"""TO STUDY K we can superpose all growths"""
    
#subway alternative: t = d/Vsub --> pick the min between two traveling times
#put Ns station in the system: there should be a "walking distance (wd)" threshold, and there should exist a station at walking distance from Location 1 (center)
"""MY COMMENT: subways might have a maximum capacity"""
#key parameters: rho_r = Ns/A , pi wd^2 rho_r = n_s  --> there has to be a transition: ns >= 1 never develop a multicentric city 


"""MAIN"""

"""
x = [2**i for i  in range(1,Smax+1)]
y = subwayGrowth()

plt.close()
plt.loglog(x,y,'o',x,[i**-.33*55 for i in x])
plt.show()

"""

"""
x = [2**i for i  in range(1,Pmax+1)]
y = populationGrowth()

plt.close()
plt.loglog(x,y,'o',x,[(i/x[0])**(mi/(mi+1))*y[0] for i in x])
plt.show()
"""

x = [1.5+i*0.5 for i  in range(0,5)]
y = muDependence()

plt.close()
plt.loglog(x,y,'o',x,[(i/x[0])**(mi/(mi+1))*y[0] for i in x])
plt.show()

