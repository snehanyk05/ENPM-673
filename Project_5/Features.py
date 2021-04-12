import cv2 
import random
import numpy as np
from Ransac_8pnts import get_RANSAC
img1 = cv2.imread('data/109.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/120.png',cv2.IMREAD_GRAYSCALE) 


orb=cv2.ORB_create(nfeatures=100)
kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)

bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches=bf.match(des1,des2)
matches=sorted(matches, key= lambda x:x.distance)

# x,y co-ordinates of the matched points
list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
#print(list_kp1)
F_final,inlier_points1,inlier_points2=get_RANSAC(list_kp1,list_kp2,800)


# compute Fundamental matrix
x1 = [set1[i][0] for i in range(len(set1))]
y1 =[set1[i][1] for i in range(len(set1))]
x2 = [set2[i][0] for i in range(len(set2))]
y2 = [set2[i][1] for i in range(len(set2))]

A = []
for i in range(0,8):
    x = [x1[i]*x2[i],x1[i]*y2[i],x1[i],y1[i]*x2[i],y1[i]*y2[i],y1[i],x2[i],y2[i],1]
    A.append(x)

A = np.matrix(A)
u,d,v = np.linalg.svd(A)
F = v[-1,:]
F = F.reshape(3,3)
F_norm = F/(np.linalg.norm(F))
uf,sf,vf = np.linalg.svd(F_norm)
diag = np.matrix([[sf[0],0,0],[0,sf[2],0],[0,0,0]])
F_final = np.dot(np.dot(uf,diag),(vf.T))

#essential matrix

K = np.matrix([[964.828979,0, 643.788025],[0, 964.828979, 484.40799],[0,0,1]])
E = np.dot(np.dot(K.T,F),K)
ue,se,ve = np.linalg.svd(E)
w = np.matrix([[0,-1,0],[1,0,0],[0,0,1]])
E = np.dot(np.dot(ue,w),ve.T)
E = E/np.linalg.norm(E)

#get camera pose
R1 = np.dot(np.dot(ue,w),ve.T)
c = ue[:,2]
R2 = np.dot(np.dot(ue,w.T),ve.T)
R = []
C = []
if np.linalg.det(R1)<0:
    R.append(-R1)
    C.append(-c)
    R.append(-R1)
    C.append(c)
else:
    R.append(R1)
    C.append(c)
    R.append(R1)
    C.append(-c)

if np.linalg.det(R2)<0:
    R.append(-R2)
    C.append(-c)
    R.append(-R2)
    C.append(c)

else:
    R.append(R2)
    C.append(c)
    R.append(R2)
    C.append(-c)

