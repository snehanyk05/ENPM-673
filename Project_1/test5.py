import cv2
import numpy as np
from matplotlib import pyplot as plt


def findID(mat):
    while(mat[5,5]!=255):
        if(mat[5,5]!=255):
           X=np.flip(mat,axis=0)
           mat=X
        if(mat[5,5]!=255):
           X=np.flip(mat,axis=1)
           mat=X
        iD=0
        if(mat[3,3]==255):
            iD=iD+1;
        if(mat[3,4]==255):
            iD=iD+2;
        if(mat[4,3]==255):
            iD=iD+8;
        if(mat[4,4]==255):
            iD=iD+4;
    return iD;


def findCorners(contours):
    points = []
    for cnt in contours:
           approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
           points.append(approx)
           if len(approx)==4:
            cv2.drawContours(img,[cnt],0,(0,0,255))
    
    corner=[]
    for i in points[1]:
        x,y = i.ravel()
        corner.append((x,y))
    return corner
    
def findCoordinates(img,corner):
    min_x=max_x=corner[0][0]
    min_y=max_y=corner[0][1]
    
    for j in range(len(corner)):
        x = corner[j][0]
        y = corner[j][1]
        
        if(x<min_x):
            min_x=x
        if(x>max_x):
            max_x=x
        if(y<min_y):
            min_y=y
        if(y>max_y):
            max_y=y
        return min_x,min_y,max_x,max_y
        
def doHomography(img,corner):
    img1 = img
    img = cv2.imread('ref_marker.png')
    X_ref = np.array([[1,1],[200,1],[200,200],[1,200]])
    
    X_img = corner
    n = 4
    A = []
    for i in range(0,n):
        x = [X_img[i][0],X_img[i][1],1,0,0,0,-X_ref[i][0]*X_img[i][0],-X_ref[i][0]*X_img[i][1], -X_ref[i][0]]
        y = [0,0,0,X_img[i][0],X_img[i][1],1,-X_ref[i][1]*X_img[i][0],-X_ref[i][1]*X_img[i][1], -X_ref[i][1]]
        A.append(x)
        A.append(y)
    
    u,d,v = np.linalg.svd(A)
    h = v[-1,:]/(v[-1,-1])
    H = np.reshape(h,(3,3))
    im = cv2.warpPerspective(img1,H,(img.shape[1],img.shape[0]))
    
    return im,h

def getProjectionMat(h,K):
    
    K_inv = np.linalg.inv(K)
    lamBda = 1/((np.linalg.norm(K_inv * h[0])+np.linalg.norm(K_inv * h[1]))/2)
    H = np.reshape(h,(3,3))
    B=lamBda*(np.dot(K_inv,H))
    r1=lamBda*B[0]
    r2=lamBda*B[1]
    r3=np.cross(r1,r2)
    t=lamBda*B[2]
#    print(r1,r2,r3,t)
#    Rt
    Rt=np.vstack((r1,r2,r3,t))
    R=Rt.T
#    R=np.array([])
#    R.append(r1.T)
#    R.append(r2.T)
#    R.append(r3.T)
#    R.append(B.T)
    
    P=np.dot(K,R)
    print(P)
    return P,R,t

def mapLena(corner,img):
    im_gray = cv2.imread('map.jpg', cv2.IMREAD_GRAYSCALE)
    X_ref = [corner[1],corner[0],corner[3],corner[2]]
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)
    lp = np.array([[1,1],[512,1],[512,512],[1,512]])
    X = cv2.resize(im_bw,(8,8))
    if X[5,5] ==255:
        X_ref = [X_ref[0],X_ref[1],X_ref[2],X_ref[3]]
    if X[5,2]==255:
        X_ref = [X_ref[1],X_ref[2],X_ref[3],X_ref[0]]
    if X[2,5]==255:
        X_ref = [X_ref[3],X_ref[0],X_ref[1],X_ref[2]]
    if X[2,2] ==255:
        X_ref = [X_ref[2],X_ref[3],X_ref[0],X_ref[1]]
    
    
    lena = cv2.imread('Lena.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    #X_ref = [corner[1],corner[0],corner[3],corner[2]]
    X_img = lp
    n = 4
    A = []
    for i in range(0,n):
        x = [X_img[i][0],X_img[i][1],1,0,0,0,-X_ref[i][0]*X_img[i][0],-X_ref[i][0]*X_img[i][1], -X_ref[i][0]]
        y = [0,0,0,X_img[i][0],X_img[i][1],1,-X_ref[i][1]*X_img[i][0],-X_ref[i][1]*X_img[i][1], -X_ref[i][1]]
        A.append(x)
        A.append(y)
    
    u,d,v = np.linalg.svd(A)
    h = v[-1,:]/(v[-1,-1])
    H = np.reshape(h,(3,3))
    im = cv2.warpPerspective(lena,H,(img.shape[1],img.shape[0]),borderMode=cv2.BORDER_CONSTANT,borderValue=[0,0,0,0])
    return im


def draw_axis(img, R, t, K,corner):
    # unit is mm
    axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
    im=cv2.projectPoints(axis, R, t, K, corner)
    
    return im
    
img = cv2.imread('frame40.jpg')
gray = cv2.imread('frame40.jpg',0)
gray = cv2.medianBlur(gray,5)
Kt=np.array([[1406.08415449821,0,0],
   [2.20679787308599, 1417.99930662800,0],
    [1014.13643417416, 566.347754321696,1]])
K=Kt.T
ret,thresh = cv2.threshold(gray,240,255,1)

z,contours,h= cv2.findContours(thresh,1,2)

corner=findCorners(contours)
min_x,min_y,max_x,max_y=findCoordinates(img,corner)
roi = img[min_y:max_y,min_x:max_x]
cv2.imwrite("roi.png", roi)

im,h=doHomography(img,corner)
cv2.imwrite('map.jpg',im)
cv2.imshow('d',im)
cv2.waitKey(0)


    
lp = np.array([[1,1],[512,1],[512,512],[1,512]])
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.resize(imgray, (8, 8)) 
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

iD=findID(im_bw)

P,R,t=getProjectionMat(h,K)
img=draw_axis(img,R,t,K,corner)
#iml=mapLena(corner,img)
cv2.imshow('d',img)
cv2.waitKey(0)



iml=mapLena(corner,img)
cv2.imshow('d',iml)
cv2.waitKey(0)