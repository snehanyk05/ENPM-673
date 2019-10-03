import cv2
import numpy as np
import matplotlib.pyplot as plt

from part2 import *


fps = 5
video = cv2.VideoWriter('Em_buoy_detection.mp4',-1,fps,(10,10))
binary_data = '/TrainingFolder/CroppedBuoys/'

## Reading and processing all test frames

for n in range(0,50):
    
    I=cv2.imread('TestFolder/' +"%d.jpg" %n)
    I1=I

    red=I[:,:,0]
    green=I[:,:,1]
    blue=I[:,:,2]
    
    k=1
    L =np.size(I,0)
    W=np.size(I,1)
    R = np.reshape(red,L*W,1)
    G = np.reshape(green,L*W,1)
    B = np.reshape(blue,L*W,1)
    Intensities = [R, G, B]
    prob_map_R = np.zeros((L*W,1))
    prob_map_G = np.zeros((L*W,1))
    prob_map_Y = np.zeros((L*W,1))
    
    
    ## Multivariate Normal distribution function
    ## USE gaussian_nd HERE 

    for i in range(0,(mu_y.shape[0])):
        prob_map_Y = gaussian_ND(Intensities,mu_y[i,:],sigma_y[:,:,i]) + prob_map_Y
        
    for i in range(0,(mu_r.shape[0])):
        prob_map_R = gaussian_ND(Intensities,mu_r[i,:],sigma_r[:,:,i]) + prob_map_R
        
    for i in range (0,(mu_g.shape[0])):
        prob_map_G = gaussian_ND(Intensities,mu_g[i,:],sigma_g[:,:,i]) + prob_map_G
            
    prob_map_R = (prob_map_R/np.max(prob_map_R))/np.size(mu_r,1)
    prob_map_G = (prob_map_G/np.max(prob_map_G))/np.size(mu_g,1)
    prob_map_Y = (prob_map_Y/np.max(prob_map_Y))/np.size(mu_y,1)

    prob_map_R = np.reshape(prob_map_R,L,W)
    prob_map_G = np.reshape(prob_map_G,L,W)
    prob_map_Y = np.reshape(prob_map_Y,L,W)


    
    
    red_mask = prob_map_R > 40*np.mean(prob_map_R[:]) # prob_map_Y < np.mean(prob_map_Y[:])###########
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*5-1, 2*5-1))
    red_mask = cv2.dilate(red_mask,se)
    red_mask2=np.zeros(np.size(red_mask))

    red_comp = np.argwhere(red_mask)
    
    # calculate moments of binary image
    M = cv2.moments(red_comp)
    # calculate x,y coordinate of center
    cX = np.int(M["m10"] / M["m00"])
    cY = np.int(M["m01"] / M["m00"])
    for i in range(1,np.size(cX)):
        
        if ~(cX[i]>150 and cX[i]<400 and cY[i]>40 and cY[i]<600):
            red_comp.PixelIdxList[0,i] = []
            red_comp.NumObjects = red_comp.NumObjects -1
            
    if red_comp.NumObjects>0:
        pixels = red_comp.PixelIdxList
        [_,index] = np.max(pixels)
        
        M = cv2.moments(red_comp)
        # calculate x,y coordinate of center
        x = np.int(M["m10"] / M["m00"])
        y = np.int(M["m01"] / M["m00"])
        
        (xx,yy),(MA,ma),angle = cv2.fitEllipse(red_comp)

        r = np.max(MA,ma)/2
        cv2.circle([x,y],r, (0,0,255), -1)
        
        
        
        
    yellow_mask = prob_map_Y > 10*np.mean(prob_map_Y[:]) # prob_map_Y < 15*mean(prob_map_Y(:)); 
    yellow_mask = cv2.dilate(yellow_mask,se)
    yellow_mask2=np.zeros(np.size(yellow_mask))

    yellow_comp = np.argwhere(yellow_mask)
    
    # calculate moments of binary image
    M = cv2.moments(yellow_comp)
    # calculate x,y coordinate of center
    cX = np.int(M["m10"] / M["m00"])
    cY = np.int(M["m01"] / M["m00"])
    for i in range(1,np.size(cX)):
        
        if ~(cX[i]>150 and cX[i]<400 and cY[i]>40 and cY[i]<600):
            yellow_comp.PixelIdxList[0,i] = []
            yellow_comp.NumObjects = yellow_comp.NumObjects -1
            
    if yellow_comp.NumObjects>0:
        pixels = yellow_comp.PixelIdxList
        [_,index] = np.max(pixels)

        
        M = cv2.moments(yellow_comp)
        # calculate x,y coordinate of center
        x = np.int(M["m10"] / M["m00"])
        y = np.int(M["m01"] / M["m00"])
        
        (xx,yy),(MA,ma),angle = cv2.fitEllipse(yellow_comp)

        r = np.max(MA,ma)/2
        cv2.circle([x,y],r, (0,0,255), -1)
        
        

    if n<10:
        green_mask = prob_map_G > 100*np.mean(prob_map_G[:])
        green_mask = cv2.dilate(green_mask,se)
        green_mask2=np.zeros(np.size(green_mask))
    
        green_comp = np.argwhere(green_mask)
        
        # calculate moments of binary image
        M = cv2.moments(green_comp)
        # calculate x,y coordinate of center
        cX = np.int(M["m10"] / M["m00"])
        cY = np.int(M["m01"] / M["m00"])
        for i in range(1,np.size(cX)):
            
            if ~(cX[i]>150 and cX[i]<400 and cY[i]>40 and cY[i]<600):
                green_comp.PixelIdxList[0,i] = []
                green_comp.NumObjects = green_comp.NumObjects -1
                
        if green_comp.NumObjects>0:
            pixels = green_comp.PixelIdxList
            [_,index] = np.max(pixels)
            
            M = cv2.moments(green_comp)
            # calculate x,y coordinate of center
            x = np.int(M["m10"] / M["m00"])
            y = np.int(M["m01"] / M["m00"])
            
            (xx,yy),(MA,ma),angle = cv2.fitEllipse(green_comp)
    
            r = np.max(MA,ma)/2
            cv2.circle([x,y],r, (0,0,255), -1)
    if n<10:
        mask = yellow_mask2 or green_mask2 or red_mask2
    else:
        mask = yellow_mask2 or red_mask2
        
    
    # Write the video    
    result = cv2.imwrite(mask,binary_data(n))
    video.write(result);
    video.release


