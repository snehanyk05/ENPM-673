import numpy as np
import os
import cv2
from preprocessing import preprocess
from trainClassifier import trainClassifier
from MSER import mser
from hsv_threshold import hsv
from getsign import getsign

features, labels, classifier=trainClassifier()
frames = []
path = 'input\\'

for frame in os.listdir(path):
    frames.append(frame)
    
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('detect.mp4',fourcc,20,(600,600)) 
num = 0     
for m in range(1):
    print(m)
#    img = cv2.imread('input\\'+str(frames[m]))
    img = cv2.imread('input/image.033749.jpg')
    img = cv2.resize(img,(600,600))
    red_norm, blue_norm = preprocess(img)
    red_mask,blue_mask = hsv(img)
    cv2.imshow('red_mask',blue_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    region_red = mser(red_norm,'red')
    red_mser = np.zeros((600,600))
    for p in region_red:
        for i in range(len(p)):
            red_mser[p[i][1],p[i][0]]=1
#    cv2.imshow('red_mser',red_mser)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows
    
    region_blue = mser(blue_norm,'blue')
    blue_mser = np.zeros((600,600))
    for p in region_blue:
        for i in range(len(p)):
            blue_mser[p[i][1],p[i][0]]=1
    cv2.imshow('red_mser',blue_mser)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    red = np.zeros((600,600))
    for i in range(600):
        for j in range(600):
            if red_mser[i,j]==1 and red_mask[i,j]==1:
                red[i,j]=255
            if red_mser[i,j]==1 and red_mask[i,j] ==0:
                red[i,j]=0
            if red_mser[i,j] == 0 and red_mask[i,j]==1:
                red[i,j]==0
            if red_mser[i,j]==0 and red_mask[i,j]==0:
                red[i,j]==0
    red = red.astype(np.uint8)
#    cv2.imshow('red',red)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows
    x1,y1 = np.where(red==255)[0],np.where(red==255)[1]
    if len(x1)>0 and len(y1)>0:
        xmax1,xmin1 = np.max(y1),np.min(y1)
        ymax1,ymin1 = np.max(x1),np.min(x1)
        flag=getsign(img,'red',[xmax1,ymax1,xmin1,ymin1],classifier,features,labels)
        if flag==1:
            cv2.rectangle(img, (xmin1,ymax1), (xmax1,ymin1), (0, 255, 0), 1)
        video.write(img)
    else:
        video.write(img)
    blue = np.zeros((600,600))
    for i in range(600):
        for j in range(600):
            if blue_mser[i,j]==1 and blue_mask[i,j]==1:
                blue[i,j]=255
            if blue_mser[i,j]==1 and blue_mask[i,j] ==0:
                blue[i,j]=0
            if blue_mser[i,j] == 0 and blue_mask[i,j]==1:
                blue[i,j]==0
            if blue_mser[i,j]==0 and blue_mask[i,j]==0:
                blue[i,j]==0
    blue = blue.astype(np.uint8)
    cv2.imshow('red',blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    x2,y2 = np.where(blue==255)[0],np.where(blue==255)[1]
    if len(x2)>0 and len(y2)>0:
        xmax2,xmin2 = np.max(y2),np.min(y2)
        ymax2,ymin2 = np.max(x2),np.min(x2)
        flag=getsign(img,'blue',[xmax2,ymax2,xmin2,ymin2],classifier,features,labels)
        if flag==1:
            cv2.rectangle(img, (xmin2,ymax2), (xmax2,ymin2), (0, 255, 0), 1)
        video.write(img)
    else:
        video.write(img)
    num = num+1
    cv2.imshow('result',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
video.release()
