import numpy as np
import scipy as sp
from scipy.constants import * #this gives directly mu_0 and epsilon_0
from matplotlib import pyplot as plt, cm, image as mpimg

ysize = 200
xsize = 200
deltaT = 10/(2.4*(10**9))#1e-8
Time = 426 #total time
Io = [1.,1.,1.] #=[Ix, Iy, Iz] current density of source
n_0=1.0 #air
n_1=2.55 #wall
mu_0=4*np.pi*(10**-7)
mu_1 = 1*mu_0
#==========define functions=========#
def getMag(list): #get the Magnitude
    return (list[:,:,0]**2+list[:,:,1]**2+list[:,:,2]**2)
  
def curl(field, length):
    #These return the x,y and z components of the field as a 2D array
    fx = field[:,:,0]
    fy = field[:,:,1]
    fz = field[:,:,2]
  
    #These calculate the gradient of these arrays at each point. They return two arrays - one for the gradient along the row index
    # and one along the column
    dfx = np.gradient(fx)
    dfy = np.gradient(fy)
    dfz = np.gradient(fz)
  
    dfx_dx = dfx[0]
    dfx_dy = dfx[1]
  
    dfy_dx = dfy[0]
    dfy_dy = dfy[1]
      
    dfz_dx = dfz[0]
    dfz_dy = dfz[1]
  
    #The curl for each component is then calculated using the necessary derivatives. Note that there are no dz components
    curl_list = [dfz_dy,-dfz_dx,dfy_dx-dfx_dy]
  
    c_field = np.ndarray(shape=(xsize,ysize,3))
    #The array of the curl is then restructured to give each component of the curl at each point in space, much like the input
    # vector field.
    for i in range(xsize):
        for j in range(ysize):
            c_field[i][j] = [curl_list[0][i][j],curl_list[1][i][j],curl_list[2][i][j]]
  
    return c_field
  
def updateFields(eField,hField,source,tStep=deltaT):
    '''
    This updates the electric and magnetic fields using an FDTD scheme to step forward in time. The curl of each field
    is calculated at each point and is used to update the fields at every point in the array. This returns the updated fields.
    '''
    curlE = curl(eField,xsize)
    curlH = curl(hField,xsize)
  
    eField += (tStep/epsilon_0)*(curlH-source)
    hField -= (tStep/mu_0)*(curlE)
  
    return eField,hField
  
def altC(list,t): #this is for alternating current
    for q in range (2):
        list[q]*np.cos(2*sp.pi*2.4*(10**9)*t)
    return list
  
#===========initialize===========#
E_field=np.ndarray(shape=(xsize,ysize,3))
H_field=np.ndarray(shape=(xsize,ysize,3))
I_grid = np.ndarray(shape=(xsize,ysize,3))
for i in range(xsize):  ##i is row (starts from 0)
    for j in range(ysize):  ##j is colm (starts from 0)
        for q in range(3): ##q is x/y/z component (q=0/1/2)
            E_field[i][j][q]=0.0 #makes E_field start with all zeros!
            H_field[i][j][q] = 0.0
            I_grid[i][j][q] = 0.0

Fields = []
Fields = updateFields(E_field,H_field,I_grid)


def reflect_perp(list, angle): #n_refl is fixed as WALL; change if doing per PIXEL
    for x in list:
        x+=x*(n_0*np.cos(angle)-(mu_0/mu_1)*np.sqrt(n_1**2-(n_0*np.sin(angle))**2))
    return list  #reflecting one elmt at a time.
    
def reflect_par(list, angle):
    for x in list:
        x+=x*((mu_0/mu_1)*(n_1**2)*np.cos(angle)-n_0*np.sqrt(n_1**2-(n_0*np.sin(angle))**2))/((mu_0/mu_1)*(n_1**2)*np.cos(angle)+n_0*np.sqrt(n_1**2-(n_0*np.sin(angle))**2))
    return list

def get_i_horiz(list):
    if np.array_equal(list[:][:][1],0.): 
        i_horiz=np.pi/2.
    else: 
        i_horiz=np.arctan(list[:][:][0]/list[:][:][1])
    return i_horiz
 
#=========plot the figure=======#
plt.ion()
plt.show()
for t in range(Time+1):
    # Reassign the value of the source at the points where it should be non-zero
    I_grid[(xsize/2)-1][(ysize/2)-1] = altC(Io,t)
    I_grid[20][30] = altC(Io,t)
    I_grid[30][50] = altC(Io,t)
      
    Fields = updateFields(Fields[0],Fields[1],I_grid)
    E_field=Fields[0]
    B_field=Fields[1]
    
###HORIZONTAL
    i_horiz=np.arctan(E_field[:][:][0]/E_field[:][:][1])
    if np.array_equal(E_field[:][:][1],0.): 
        i_horiz=np.pi/2.
    else:
        pass
    
    #Wall 1: E_field[0][:][:]
        E_field[0][:][1]=reflect_perp(E_field[0][:][1],i_horiz) 
        E_field[0][:][0]=reflect_par(E_field[0][:][0], i_horiz) 
        E_field[0][:][2]=reflect_par(E_field[0][:][2], i_horiz) 
        
    #Wall 2: E_field[ysize-1][:][:]
        E_field[ysize-1][:][1]=reflect_perp(E_field[ysize-1][:][1],i_horiz) 
        E_field[ysize-1][:][0]=reflect_par(E_field[ysize-1][:][0], i_horiz) 
        E_field[ysize-1][:][2]=reflect_par(E_field[ysize-1][:][2], i_horiz) 

    #Wall 1: B_field[0][:][:]
        B_field[0][:][1]=reflect_perp(B_field[0][:][1],i_horiz) 
        B_field[0][:][0]=reflect_par(B_field[0][:][0], i_horiz) 
        B_field[0][:][2]=reflect_par(B_field[0][:][2], i_horiz) 
        
    #Wall 2: B_field[ysize-1][:][:]
        B_field[ysize-1][:][1]=reflect_perp(B_field[ysize-1][:][1],i_horiz) 
        B_field[ysize-1][:][0]=reflect_par(B_field[ysize-1][:][0], i_horiz) 
        B_field[ysize-1][:][2]=reflect_par(B_field[ysize-1][:][2], i_horiz) 

###VERTICAL
    i_vert=np.arctan(E_field[:][:][1]/E_field[:][:][0])
    if np.array_equal(E_field[:][:][0],0.): 
        i_vert=np.pi/2.
    else:
        pass

    #Wall 3: E_field[:][0][:]
        E_field[:][0][0]=reflect_perp(E_field[:][0][0],i_vert) 
        E_field[:][0][1]=reflect_par(E_field[:][0][1], i_vert) 
        E_field[:][0][2]=reflect_par(E_field[:][0][2], i_vert) 

    #Wall 4: E_field[:][xsize-1][:]
        E_field[:][xsize-1][0]=reflect_perp(E_field[:][xsize-1][0],i_vert) 
        E_field[:][xsize-1][1]=reflect_par(E_field[:][xsize-1][1], i_vert) 
        E_field[:][xsize-1][2]=reflect_par(E_field[:][xsize-1][2], i_vert) 

    #Wall 3: B_field[:][0][:]
        B_field[:][0][0]=reflect_perp(B_field[:][0][0],i_vert) 
        B_field[:][0][1]=reflect_par(B_field[:][0][1], i_vert) 
        B_field[:][0][2]=reflect_par(B_field[:][0][2], i_vert) 
        
    #Wall 4: B_field[:][xsize-1][:]
        B_field[:][xsize-1][0]=reflect_perp(B_field[:][xsize-1][0],i_vert) 
        B_field[:][xsize-1][1]=reflect_par(B_field[:][xsize-1][1], i_vert) 
        B_field[:][xsize-1][2]=reflect_par(B_field[:][xsize-1][2], i_vert) 

  
    plt.figure(1)

    plt.title('t= %s'%(t))
    i,j = np.meshgrid(np.arange(xsize), np.arange(ysize))
    #Z = np.log(getMag(E_field)+getMag(B_field))
    Z = getMag(E_field)+getMag(B_field)
    Z = np.ma.masked_where(Z==0., Z)
    plt.imshow(Z,origin="lower",interpolation="nearest")
    plt.draw()
    #plt.savefig('image_'+str(t)+'.png')
  
plt.colorbar()
plt.ioff()
  
plt.show()
