# C. Akshay Santoshi
# CS21BTECH11012
# To find the mode of the given data

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from statistics import median
import pandas as pd
from scipy.ndimage import shift

#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *

#for creating lines
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
 
I = np.eye(2)
e2 = I[0:,1]

#Input parameters from excel file 
df= pd.read_excel(r'C:\Users\aksha\Documents\Book1.xlsx')
print(df)
dst = df.to_numpy()[:,1:]
nvalues = np.size(dst[0])-1

#Creating numpy matrix of the given data
A = np.block([[dst[0]],[dst[1]]])
A = np.array(A.tolist(), dtype=float)
Amax = np.amax(A[1])

#Locating the index for the mode class
imed = np.where(A[1]==Amax)
imed = imed[0]

P = np.array([A[0,imed],A[1,imed]])
Q = np.array([A[0,imed-1],A[1,imed-1]])
R = np.array([A[0,imed-1],A[1,imed]])
S = np.array([A[0,imed],A[1,imed+1]])

#Finding the mode 
n1 = omat@(P-Q)
n2 = omat@(R-S)


#Computing the mode
M = line_intersect(n1.T,P,n2.T,R)
Mx = np.array([M[0],0])
print(Mx)


#Generating PQ and RS
xPQ = line_gen(P,Q)
xRS = line_gen(R,S)

#generating the mode line
xM = line_gen(M,Mx)


#Plotting the bar graph for the data
plt.bar(A[0,:],A[1,:],width=10)

#Plotting the lines PQ and RS
plt.plot(xPQ[0,:],xPQ[1,:],color='red')#,label='$Diameter$')
plt.plot(xRS[0,:],xRS[1,:],color='orange')#,label='$Diameter$')

#Plotting the Mode line
plt.plot(xM[0,:],xM[1,:],color='yellow')#,label='$Diameter$')


#Labeling the coordinates
tri_coords = np.block([[P.T],[Q.T],[R.T],[S.T],[M],[Mx]]).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q','R','S','M','Mx']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.show()