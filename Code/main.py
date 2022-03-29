from matplotlib import pyplot as plt
import numpy as np

# drawing histogram
fig,ax = plt.subplots(1,1)
a = np.array([3001, 3002, 3003, 3004, 4001, 4002, 4003, 4004, 4005, 4006, 4007,
 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 5001, 5002, 
 5003, 5004, 5005, 5006, 5007, 5008, 5009, 6001, 6002, 6003, 6004, 6005, 6006, 
 7001, 7002, 7003, 7004, 7005, 7006, 7007, 8001, 8002, 9001, 9002, 9003, 9004])
ax.hist(a, bins = [3000,4000,5000,6000,7000,8000, 9000, 10000], ec="black")
ax.set_title("histogram ")
ax.set_xticks([3000,4000,5000,6000,7000,8000, 9000, 10000])
ax.set_xlabel('Runs scored')
ax.set_ylabel('No. of batsmen')

#plotting the points
P=np.array([5000,18])
Q=np.array([4000,4])
R=np.array([4000,18])
S=np.array([5000,9])

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