
import cv2
import numpy as np
def findID(mat):
            iD =0
            if(X[5,5]==255):
                if X[3,3]==255:
                    iD = iD+1;
                if X[3,4]==255:
                    iD = iD+2;
                if([4,3]==255):
                    iD=iD+8;
                if(X[4,4]==255):
                    iD=iD+4;
            elif (X[5,2]==255):
                if X[3,4]==255:
                    iD = iD+1;
                if X[4,4]==255:
                    iD = iD+2;
                if X[4,3]==255:
                    iD = iD+4;
                if X[3,3]==255:
                    iD = iD+8;
            elif(X[2,5]==255):
                if X[4,4]==255:
                    iD = iD+1;
                if X[3,4]==255:
                    iD= iD+2;
                if X[3,3]==255:
                    iD = iD+4;
                if X[4,3]==255:
                    iD = iD+8;
            elif(X[2,2]==255):
                if X[4,3]==255:
                    iD = iD+1;
                if X[3,3]==255:
                    iD = iD+2;
                if X[3,4]==255:
                    iD = iD+4;
                if X[4,4]==255:
                    iD = iD+8;
            return iD;

'''Video for Id and Corners'''
vidcap = cv2.VideoCapture('Tag1.mp4')
success,image = vidcap.read()
count = 0
success = True
while success and count<200:
  cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  #print ('Read a new frame: '+ str(success))
  count += 1

img = cv2.imread('0.jpg')
height,width,layers=img.shape
video=cv2.VideoWriter('video.mp4',-1,20,(width,height)) 
for d in range(0,200):
    img = cv2.imread('%d.jpg'%d)
    gray = cv2.imread('%d.jpg'%d,0)
    gray = cv2.medianBlur(gray,5)
    ret,thresh = cv2.threshold(gray,240,255,1)
    
    z,contours,h= cv2.findContours(thresh,1,2)
    
    areas=[]
    for c in contours:
        areas.append(cv2.contourArea(c))
    points = []
    for cnt in contours:
           approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
           points.append(approx)
    #       if len(approx)==4:
    #        cv2.drawContours(img,[cnt],0,(0,0,255))
    
    corner=[]
    for i in points[2]:
        x,y = i.ravel()
        corner.append((x,y))
    
    for j in range(len(corner)):
        x = corner[j][0]
        y = corner[j][1]
        cv2.circle(img,(x,y),3,225,-1)
    cv2.imshow('img',img)
    cv2.waitKey(0)
#    cv2.imwrite('corner.jpg',img)
    
    img1 = cv2.imread('%d.jpg'%d)
    img = cv2.imread('ref_marker.png')
    X_ref = np.array([[1,1],[200,1],[200,200],[1,200]])
    
    X_img = [corner[0],corner[3],corner[2],corner[1]]
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
    cv2.imwrite('map.jpg'%d,im)
    cv2.imshow('d',im)
    cv2.waitKey(0)
    
    im_gray = cv2.imread('map.jpg', cv2.IMREAD_GRAYSCALE)
    X_ref = [corner[0],corner[3],corner[2],corner[1]]
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)
    lp = np.array([[1,1],[512,1],[512,512],[1,512]])
    X = cv2.resize(im_bw,(8,8))
    Y = (findID(X))
    iD = cv2.putText(img1,'ID:'+Y,(230,50),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,0))
    video.write(img)
    #cv2.destroyAllWindows()  
video.release()

'''Video for Lena'''
img = cv2.imread('0.jpg')
height,width,layers=img.shape
video1=cv2.VideoWriter('video.mp4',-1,20,(width,height)) 
for d in range(0,200):
    img1 = cv2.imread('%d.jpg'%d)
    img = cv2.imread('ref_marker.png')
    X_ref = np.array([[1,1],[200,1],[200,200],[1,200]])
    
    X_img = [corner[0],corner[3],corner[2],corner[1]]
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
    cv2.imwrite('map.jpg'%d,im)
    cv2.imshow('d',im)
    cv2.waitKey(0)
    
    im_gray = cv2.imread('map.jpg', cv2.IMREAD_GRAYSCALE)
    X_ref = [corner[0],corner[3],corner[2],corner[1]]
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
    
    img = cv2.imread('%d.jpg'%d)
    lena = cv2.imread('Lena.png')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
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
    im = cv2.warpPerspective(lena,H,(img.shape[1],img.shape[0]),img, borderMode=cv2.BORDER_TRANSPARENT)
    
    video1.write(im)
    ##cv2.destroyAllWindows()  
video.release()

''' Cube Projection'''
def getProjectionMat(h,K):
K_inv = np.linalg.inv(K)
lamBda = 1/((np.linalg.norm(K_inv * h[0])+np.linalg.norm(K_inv * h[1]))/2)
H = np.reshape(h,(3,3))
B=lamBda*(np.dot(K_inv,H))
r1=lamBda*B[0]
r2=lamBda*B[1]
r3=np.cross(r1,r2)
t=lamBda*B[2]
# print(r1,r2,r3,t)
# Rt
Rt=np.vstack((r1,r2,r3,t))
R=Rt.T
# R=np.array([])
# R.append(r1.T)
# R.append(r2.T)
# R.append(r3.T)
# R.append(B.T)
P=np.dot(K,R)
print(P)
return P,R,t
def draw_axis(img, R, t, K,corner):
# unit is mm
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
im=cv2.projectPoints(axis, R, t, K, corner)
return im




Kt=np.array([[1406.08415449821,0,0],
   [2.20679787308599, 1417.99930662800,0],
    [1014.13643417416, 566.347754321696,1]])
K=Kt.T
P,R,t=getProjectionMat(h,K)
img=draw_axis(img,R,t,K,corner)
