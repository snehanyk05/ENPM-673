# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:42:36 2019

@author: Sneha
"""
from PIL import Image
import glob
import os
import cv2
from skimage import exposure
from skimage import feature
from sklearn import svm
import cv2



def trainClassifier():
    print("Here")
    for name in glob.glob('training_set/*.*'):
        print(name)
    winSize = (64,64)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (4,4)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    x=''
    trainingFeaturesSK,trainingFeaturesCV = dict(),dict()
    features,labels=[],[]
    for root, dirs, files in os.walk('training_selected/'):
#        print(root,dirs,files)
        for name in files:
#            print(name)
            if (name.endswith((".ppm")) or name.endswith((".jpg")) ):
#               print(root)
               if(x is not root):
                    
                    x=root
                    print(x)
                    arr=[]
                    arr1=[]
               img = cv2.imread(root+'/'+name)
               gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               median = cv2.medianBlur(gray,3)
               img=cv2.resize(median,(64,64))
#               print(img.shape)
#               cv2.imshow('result',img)
#               cv2.waitKey(0)
               img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

#               ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#               cv2.imshow('result',img)
#               cv2.waitKey(0)
#               print(root.split('/')[1])
#               trainingFeatures[root.split('/')[1]]=(hog.compute(img))
               #-------Through skimage------------
               (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualize=True)
#               hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
#               hogImage = hogImage.astype("uint8")
                
#               cv2.imshow("HOG Image", hogImage)
               features.append(H)
               labels.append(root.split('/')[1])
               arr1.append(H)
               trainingFeaturesSK[root.split('/')[1]]=arr1
              
               #-----------------------------------
              
               
               #----------Through opencv-------
               h=hog.compute(img)
#               h=h.ravel()
#               features.append(h)
#               labels.append(root.split('/')[1])
#               cv2.imshow('result',h)
#               cv2.waitKey(0)
               arr.append(h)
               trainingFeaturesCV[root.split('/')[1]]=arr
               # ------------------
     
    for root, dirs, files in os.walk('neg'):
#        print(root,dirs,files)
        print('neg')
        for name in files:
            
            if name.endswith((".jpg")):
#               print(root)
#               if(x is not root):
                    
#                    x=root
#               print(root+'/'+name)
               arr=[]
               arr1=[]
               img = cv2.imread(root+'/'+name)
               gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               median = cv2.medianBlur(gray,3)
               img=cv2.resize(median,(64,64))
#               print(img.shape)
#               cv2.imshow('result',img)
#               cv2.waitKey(0)
               img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

#               ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#               cv2.imshow('result',img)
#               cv2.waitKey(0)
#               print(root.split('/')[1])
#               trainingFeatures[root.split('/')[1]]=(hog.compute(img))
               #-------Through skimage------------
               (H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",visualize=True)
#               hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
#               hogImage = hogImage.astype("uint8")
                
#               cv2.imshow("HOG Image", hogImage)
               features.append(H)
               labels.append('negatives')
#    print(trainingFeaturesSK.keys())
    clf = svm.SVC(kernel='linear', probability=True, decision_function_shape='ovo')
#    clf.probability='True'
    clf.fit(features,labels)  
   
#    print(clf)
#cv.destroyAllWindows()
#    trainingSet = imageDatastore('../training_selected',   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
#    testSet     = imageDatastore('../testing_selected', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
#    
#    
#    numImages = numel(trainingSet.Files);
#    % trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
#    
#    for i = 1:numImages
#        img = readimage(trainingSet, i);
#        img = rgb2gray(img);
#        img = medfilt2(img, [3 3]);
#        img = imresize(img, [64 64]);    
#        % Apply pre-processing steps
#        img = imbinarize(img);
#        
#        trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', [4 4]);  
#    end
#    % Get labels for each image.
#    trainingLabels = trainingSet.Labels;
#    classifier = fitcecoc(trainingFeatures, trainingLabels);
    return features, labels,clf