from Functions import *
from Ransac_8pnts import get_RANSAC
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib','qt')

imgfiles = 'data/'
img1 = cv2.imread(imgfiles+'0.png',cv2.IMREAD_GRAYSCALE)

R_t = np.matrix(np.identity(3))
t_t = np.matrix([[0],[0],[0]])
x = [0]
z = [0]

k=0
for i in range(19,30):
    img2 = cv2.imread(imgfiles+'%i.png'%i,cv2.IMREAD_GRAYSCALE)
    
    set1,set2 = feature_match(img1,img2)
#    print(ss[k])
    F,inliers1,inliers2= get_RANSAC(ss[k],rr[k],set1,set2,300)
#    F,inliers1,inliers2 = f_matrix(set1,set2)
#    ss.append(set1)
#    rr.append(set2)
    E = essential_matrix(F)
    print('E',E)
    Rset,Cset = get_pose(E)
    Xset = []
    Xset_new = []
    for j in range(0,4):
        X,X_new,X1 = linear_triangulation(Cset[j],Rset[j],inliers1,inliers2)
        Xset.append(X)
        Xset_new.append(X_new)
    print('Xset',Xset)
    print('Xset_new',Xset_new)
    R,t = correct_pose(Cset,Rset,Xset,Xset_new)
    print('R', R,'t',t)
    t_t = t_t+(R_t*t)
    R_t = (R_t*R)
    print('R_t', R_t,'t_t',t_t)
#    print('t',t_t[0])
    x.append(float(t_t[0]))
    z.append(float(t_t[2]))
    print(i)
    k+=1
#    img1 = img2
    #plotting the data
#plt.axis([-100,1100,-300,1000])
plt.pause(0.01)
plt.title('Camera Trajectory')
plt.plot(x,z,'r-')
plt.xlabel('Motion in x-direction')
plt.ylabel('Motion in y-direction')