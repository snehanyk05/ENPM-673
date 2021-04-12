import cv2
import numpy as np

def hsv(img):
    img = cv2.medianBlur(img,5)
    im = np.zeros((600,600))
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img1 = cv2.normalize(imghsv, im , 0, 255, cv2.NORM_MINMAX).astype(float)
    img1 = img1/255
    
    s = img1[:,:,1]
    v  = img1[:,:,2]
    sthresh1 = np.zeros((600,600))
    for i in range(s.shape[0]):
       for j in range(s.shape[1]):
           if s[i,j]>=0.5 and s[i,j]<=0.9:
               sthresh1[i,j]= 1
           else:
                sthresh1[i,j]=0
    vthresh1 = np.zeros((600,600))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i,j]>=0.2 and v[i,j]<=0.75:
                vthresh1[i,j]=1
            else:
                vthresh1[i,j]=0
    red_mask = np.zeros((600,600))
    for i in range(600):
        for j in range(600):
            if sthresh1[i,j]==0 and vthresh1[i,j]==0:
                red_mask[i,j]=0
            if sthresh1[i,j]==0 and vthresh1[i,j]==1:
                red_mask[i,j]=0
            if sthresh1[i,j]==1 and vthresh1[i,j]==0:
                red_mask[i,j]=0
            if sthresh1[i,j]==1 and vthresh1[i,j]==1:
                red_mask[i,j]=1
    
    
    sthresh2 = np.zeros((600,600))
    for i in range(s.shape[0]):
       for j in range(s.shape[1]):
           if s[i,j]>=0.45 and s[i,j]<=0.8:
               sthresh2[i,j]= 1
           else:
                sthresh2[i,j]=0
    vthresh2 = np.zeros((600,600))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i,j]>=0.35 and v[i,j]<=1:
                vthresh2[i,j]=1
            else:
                vthresh2[i,j]=0
    blue_mask = np.zeros((600,600))
    for i in range(600):
        for j in range(600):
            if sthresh2[i,j]==0 and vthresh2[i,j]==0:
                blue_mask[i,j]=0
            if sthresh2[i,j]==0 and vthresh2[i,j]==1:
                blue_mask[i,j]=0
            if sthresh2[i,j]==1 and vthresh2[i,j]==0:
                blue_mask[i,j]=0
            if sthresh2[i,j]==1 and vthresh2[i,j]==1:
                blue_mask[i,j]=1
    
    return red_mask,blue_mask

            