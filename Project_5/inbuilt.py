from Functions import *
from Ransac_8pnts import get_RANSAC
import cv2
import numpy as np
import matplotlib.pyplot as plt

imgfiles = 'data/'
img1 = cv2.imread(imgfiles+'0.png',cv2.IMREAD_GRAYSCALE)


R_t = np.identity(3)
t_t = np.zeros((3,1))
x = [0]
z = [0]
for i in range(19,30):
    img2 = cv2.imread(imgfiles+'%i.png'%i,cv2.IMREAD_GRAYSCALE)
    
    set1,set2 = feature_match(img1,img2)
#    x1 = [set1[i][0] for i in range(len(set1))]
#    y1 =[set1[i][1] for i in range(len(set1))]
#    x2 = [set2[i][0] for i in range(len(set2))]
#    y2 = [set2[i][1] for i in range(len(set2))]
#    points1 = np.column_stack((x1,y1))
#    points2 = np.column_stack((x2,y2))
    F,inliers1,inliers2 = get_RANSAC(set1,set2,800)
#    print(inliers1)
    points1 = inliers1
    points2 = inliers2
    E,mask= cv2.findEssentialMat(points1,points2)
    points,R,t,mask = cv2.recoverPose(E,points1,points2)
    
    t_t = t_t+np.dot(R_t,t)
    R_t = np.dot(R_t,R)
    x.append(float(t_t[0]))
    z.append(float(t_t[2]))
    print(i)
    img1 = img2
    #plotting the data

plt.pause(0.01)
plt.title('Camera Trajectory')
plt.plot(x,z,'r-')
plt.xlabel('Motion in x-direction')
plt.ylabel('Motion in z-direction')
