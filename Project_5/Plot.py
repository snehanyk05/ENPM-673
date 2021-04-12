import matplotlib.pyplot as plt
import numpy as np
f = open('x.txt','r')
x = f.read().splitlines()
f1 = open('z.txt','r')
z = f1.read().splitlines()
X = []
Z = []
for i in range(len(x)):
    m = float(x[i])
    X.append((-m)-sum(m)+50)
    n = float(z[i])
    Z.append((n)-sum(n)+60)


plt.title('Camera Trajectory')
plt.plot(X,Z)
plt.xlabel('Motion in x-direction')
plt.ylabel('Motion in z-direction')
plt.show()
