# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:27:32 2019

@author: Sneha
"""
import cv2
from skimage import exposure
from skimage import feature
from sklearn import svm
import numpy as np

def getsign(img,str,bbox,classifier,features,labels):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box_select= bbox;
    print("in red")
#    winSize = (64,64)
#    blockSize = (16,16)
#    blockStride = (4,4)
#    cellSize = (4,4)
#    nbins = 9
#    derivAperture = 1
#    winSigma = -1.
#    histogramNormType = 0
#    L2HysThreshold = 0.2
#    gammaCorrection = 1
#    nlevels = 64
#    signedGradient = True
#
#    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
#   

#for i in range (    =1: size(bbox,1)
#   box_select = bbox(i,:);
#    print(classifier)
#    cv2.imshow('result',gray)
#    cv2.waitKey(0)
    gray = cv2.medianBlur(gray,3)
    img_sign = gray[bbox[3]:bbox[1],bbox[2]:bbox[0]]
    if((bbox[3]-bbox[1])!=0 and (bbox[2]-bbox[0])!=0):
        img_sign=cv2.resize(img_sign,(64,64))
        img_sign = cv2.adaptiveThreshold(img_sign, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        (H, hogImage) = feature.hog(img_sign, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualize=True)
    #    cv2.imshow("HOG Image", hogImage)
    #    h=hog.compute(img_sign)
    #    h=h.ravel()
    #    print(h.shape)
        label=classifier.predict(H.reshape(1, -1))
        
    #    x=classifier.decision_function(H.reshape(1, -1))
        x=classifier.predict_proba(H.reshape(1, -1))
    #    print(features.shape, [H].shape)
    #    y=classifier.score([H],label)
    #    print(y)
        flag=0
        if(label is not None and label[0] != 'negatives'):
    ##        print(x)
    #        for i in x[0]:
            
                value=(max(x[0]))
                k=np.where(x[0] == value)
                i=k[0][0]
                if(str=='red'):
                    if(x[0][i]>0.5 and i<=4):
                        flag=1
                        print('Label',label[0],x)
                        im_detect=  cv2.imread(label[0]+'.png')
                        
                        img_shape=(img[bbox[3]:bbox[3]+64,bbox[2]+30:bbox[2]+30+64].shape)
                        if(img_shape[1]==0):
                            img_shape=(img[bbox[3]:bbox[3]+64,bbox[2]-30-64:bbox[2]-30].shape) 
                            im_detect=cv2.resize(im_detect,(img_shape[1],img_shape[0]))
                            
                            img[bbox[3]:bbox[3]+64,bbox[2]-30-64:bbox[2]-30]=im_detect
                        else:
                            im_detect=cv2.resize(im_detect,(img_shape[1],img_shape[0]))
                            
                            img[bbox[3]:bbox[3]+64,bbox[2]+30:bbox[2]+30+64]=im_detect
                elif(str=='blue'):
                    flag=1
                    if(x[0][i]>0.5 and i>4):
                        print('Label',label[0],x)
                        im_detect=  cv2.imread(label[0]+'.png')
                        
                        img_shape=(img[bbox[3]:bbox[3]+64,bbox[2]+30:bbox[2]+30+64].shape)
                        if(img_shape[1]==0):
                            img_shape=(img[bbox[3]:bbox[3]+64,bbox[2]-30-64:bbox[2]-30].shape) 
                            im_detect=cv2.resize(im_detect,(img_shape[1],img_shape[0]))
                            
                            img[bbox[3]:bbox[3]+64,bbox[2]-30-64:bbox[2]-30]=im_detect
                        else:
                            im_detect=cv2.resize(im_detect,(img_shape[1],img_shape[0]))
                            
                            img[bbox[3]:bbox[3]+64,bbox[2]+30:bbox[2]+30+64]=im_detect
        return flag
    else:
        return 0
#    cv2.imshow('result',img_sign)
#    cv2.waitKey(0)