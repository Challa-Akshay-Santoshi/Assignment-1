import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from statistics import median
import pandas as pd
from scipy.ndimage import shift

fig,ax = plt.subplots(1,1)
s= pd.read_excel(r'C:\Users\aksha\Documents\Book2.xlsx')
print(s)
s1 = df.to_numpy()
X = np.block([s1[0]])
X = np.array(X.tolist(), dtype=float)

df= pd.read_excel(r'C:\Users\aksha\Documents\Book1.xlsx')
print(df)
dst = df.to_numpy()
A = np.block([[dst[0]],[dst[1]]])
A = np.array(A.tolist(), dtype=float)
Amax = np.amax(A[1])
imed = np.where(A[1]==Amax)
imed = imed[0]

ax.hist(X[0], A[0], ec="black")
ax.set_title("histogram ")
ax.set_xticks(A[0])
ax.set_xlabel('Runs scored')
ax.set_ylabel('No. of batsmen')

P = np.array([A[0,imed],A[1,imed]])
Q = np.array([A[0,imed-1],A[1,imed-1]])
R = np.array([A[0,imed-1],A[1,imed]])
S = np.array([A[0,imed],A[1,imed+1]])

# to draw line segments
x_values = [P[0], Q[0]]
y_values = [P[1], Q[1]]
plt.plot(x_values, y_values, "yellow")

x_values = [R[0], S[0]]
y_values = [R[1], S[1]]
plt.plot(x_values, y_values, "yellow")

# annotating the points
plt.annotate("P",P,xytext=(1,3),textcoords="offset points")
plt.annotate("Q",Q,xytext=(-10,2),textcoords="offset points")
plt.annotate("R",R,xytext=(-5,3),textcoords="offset points")
plt.annotate("S",S,xytext=(0,1),textcoords="offset points")

m1=0.014
m2=-0.009
b1=-52
b2=54
xi = (b1-b2) / (m2-m1)
yi = m1 * xi + b1

# plotting mode point
M=np.array([xi,yi])
plt.annotate("M",M,xytext=(-12,0),textcoords="offset points")

x_values = [xi, xi]
y_values = [yi, 0]
plt.plot(x_values, y_values, "orange")


plt.show()
