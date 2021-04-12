import cv2 
import numpy as np


def feature_match(img1,img2):


    orb = cv2.ORB_create(nfeatures=1000)
    kp1,des1=orb.detectAndCompute(img1,None)
    kp2,des2=orb.detectAndCompute(img2,None)
    
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches=bf.match(des1,des2)
    matches=sorted(matches, key= lambda x:x.distance)
    
    # x,y co-ordinates of the matched points
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
    
    return list_kp1,list_kp2

    
#essential matrix
def essential_matrix(F):
    K = np.matrix([[964.828979,0, 643.788025],[0, 964.828979, 484.40799],[0,0,1]])
    E = np.dot(np.dot(K.T,F),K)
    ue,se,ve = np.linalg.svd(E)
    w = np.matrix([[1,0,0],[0,1,0],[0,0,0]])
    E = ue*w*ve
    E = E/np.linalg.norm(E)
    return E


#get camera pose
def get_pose(E):
    w = np.matrix([[0,-1,0],[1,0,0],[0,0,1]])
    U,d,V = np.linalg.svd(E)
    R1 = np.dot(np.dot(U,w),V)
    c = U[:,2]
    R2 = np.dot(np.dot(U,w.T),V)
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
        
    return R,C
  

#def linear_triangulation(C1,R1,C2,R2,set1,set2):
#    x1 = [set1[i][0] for i in range(len(set1))]
#    y1 =[set1[i][1] for i in range(len(set1))]
#    x2 = [set2[i][0] for i in range(len(set2))]
#    y2 = [set2[i][1] for i in range(len(set2))]
#    X1 = np.column_stack((x1,y1))
#    X2 = np.column_stack((x2,y2))
#    R1 = np.matrix(R1)
#    C1 = np.matrix(C1)
#    X = np.zeros((8,3))
#    K = np.matrix([[964.828979,0, 643.788025],[0, 964.828979, 484.40799],[0,0,1]])
#    p1 = K*np.column_stack((R1,-R1*C1))
#    p2 = K*np.column_stack((R2,-R2*C2))
#    
#    for j in range(0,8):
#        A = np.array([[X1[j,0]*p1[2,:]-p1[0,:]],[X1[j,1]*p1[2,:]-p1[1,:]],[X2[j,0]*p2[2,:]-p2[0,:]],[X2[j,1]*p2[2,:]-p2[1,:]]])
#        A = np.matrix(A)
#        u,d,VT = np.linalg.svd(A)
#        m = VT.T
#        l = m[0:3,-1]
#        X[j] = l.T/(m[-1,-1])
#    return X

def linear_triangulation(C1,R1,set1,set2):
    x1 = [set1[i][0] for i in range(len(set1))]
    y1 =[set1[i][1] for i in range(len(set1))]
    x2 = [set2[i][0] for i in range(len(set2))]
    y2 = [set2[i][1] for i in range(len(set2))]
    Y1 = np.vstack((x1,y1))
    Y2 = np.vstack((x2,y2))
    n = len(Y1[0])
    x = np.vstack((Y1,np.ones(n)))
    y = np.vstack((Y2,np.ones(n)))
    K = np.matrix([[964.828979,0, 643.788025],[0, 964.828979, 484.40799],[0,0,1]])
    X1 = np.linalg.inv(K)*x
#    print(len(X1[1,:]))
    X2 = np.linalg.inv(K)*y
    _, m = X1.shape
    X = np.matrix(np.zeros((4,n)))
    X_new = np.matrix(np.zeros((4,n)))
    rot = R1
    t = C1
    #for i in range(0,4):
    #    rot = R[i]
    #    t = C[i]
    p1 = np.eye(3,4)
    p2 = np.array(np.hstack((rot,t)))
    H = np.vstack((p2,[0,0,0,1]))
    for j in range(0,m):
        A = np.array([[X1[0,j]*p1[2,:]-p1[0,:]],[X1[1,j]*p1[2,:]-p1[1,:]],[X2[0,j]*p2[2,:]-p2[0,:]],[X2[1,j]*p2[2,:]-p2[1,:]]])
        A = np.matrix(A)
        _,_,VT = np.linalg.svd(A)
        m = VT.T
        Y = m[:,-1]
        X[:,j]= Y/float(Y[3])
        X_new[:,j] = H*X[:,j]
        
    return X,X_new,X1
#get correct camera pose
#def correct_pose(Cset,Rset,Xset):
#    m =[]
#    for i in range(0,4):
#        C = Cset[i]
#        R = Rset[i]
#        X1 = np.matrix(Xset[i])
#        diff =  (X1-C.T)
#        ch = diff*(R[2,:].T)
#        chz = X1[:,2]
#        n = (np.sum(ch>0)) + np.sum((chz>0))
#        m.append(n)
#    
#    index = np.argmax(m)
#    C = Cset[index]
#    R = Rset[index]
#    X1 = Xset[index]
#    return C,R

def correct_pose(Cset,Rset,Xset,Xset_new):
    C = [0,0,0,0]
    C = np.reshape(C, (1,4))
   
    for i in range (0,4):
        Xset1 = Xset[i]
        Xset2 = Xset_new[i]
        C[0,i] = np.sum(Xset2[2,:] > 0) + np.sum(Xset1[2,:] > 0)
    

    if C[0,0] == 0 and C[0,1] == 0 and C[0,2] == 0 and C[0,3] == 0:
        R = np.eye(3)
        t = np.vstack((0,0,0))   
    else:
        index = np.argmax(C)
        R = Rset[index]
        t = Cset[index]
        if t[2] < 0:
            t = -t
#    Noise Reduction
    if np.abs(R[0,2]) < 0.001:
        R[0,2] = 0
    
    if np.abs(R[2,0]) < 0.001:
        R[2,0] = 0
      
    
    if np.abs(t[0])<0.01 or R[0,0] > 0.99:
        t = np.vstack((0,0,t[2]))
            
    return R, t
    