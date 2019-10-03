# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 20:34:59 2019

@author: Shreya
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def final(img, left_lane, right_lane, H_inv,left_curve,right_curve,a):
    ''' Overlay the lanes on the undistorted image. Takes the undistorted image as one of the inputs.'''
    # get x and y values to plot the lanes
    y  = np.linspace(0, img.shape[0]-1, img.shape[0])
    # substituting y in the polynomial ay**2+by+c
    left_x = left_lane[0]*y**2 + left_lane[1]*y + left_lane[2]
    right_x = right_lane[0]*y**2 + right_lane[1]*y + right_lane[2]
    # create an image
    image = np.zeros((720, 1280, 3), dtype='uint8')
    # stack the x and y points in an array
    left_pts = np.array([np.transpose(np.vstack([left_x, y]))])
    right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x, y])))])
    # stack the left and right points
    pts = np.hstack((left_pts, right_pts))
    # draw the lanes on the image
    cv2.fillPoly(image, np.int_([pts]), (0,255, 0))
    # project back to original image space using the inverse homography
    newimage = cv2.warpPerspective(image, H_inv, (img.shape[1], img.shape[0]))
    # combine it with the undistorted image
    final_image = cv2.addWeighted(img, 1, newimage, 0.3, 0)
    # Put the radius of curvature on the image
    avg_curve = (left_curve + right_curve)/2
    label = 'Radius of curvature: %.1f m' % avg_curve
    final_image = cv2.putText(final_image, label, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    # put the turn prediction on the image
    a=a*1000 
    if a<(-0.09) and a>(-0.4):
        label = 'Left Curve'
        final_image= cv2.putText(final_image, label, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    if -0.04<a and a<0.07:
        label = 'Straight Curve'
        final_image= cv2.putText(final_image, label, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    else:
        label = 'Right Curve'
        final_image= cv2.putText(final_image, label, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    
    return final_image

def radius_of_curvature(left_lane_pts, right_lane_pts, nonzerox, nonzeroy):
    ''' Calculate the radius of curvature'''
    y = 719  # since height of the image is 720
    # Convert x and y from pixels space to meters
    ym = 30/700 # meters per pixel in y dimension
    xm = 3.7/700 # meters per pixel in x dimension
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_pts]
    lefty = nonzeroy[left_lane_pts]
    rightx = nonzerox[right_lane_pts]
    righty = nonzeroy[right_lane_pts]
    # Fit new polynomials to x,y in world space
    left_lane= np.polyfit(lefty*ym, leftx*xm, 2)
    right_lane = np.polyfit(righty*ym, rightx*xm, 2)
    
    # Calculate the new radii of curvature
    left_radius = ((1 + (2*left_lane[0]*y*ym + left_lane[1])**2)**1.5) / np.absolute(2*left_lane[0])
    right_radius = ((1 + (2*right_lane[0]*y*ym + right_lane[1])**2)**1.5) / np.absolute(2*right_lane[0])
    
    return left_radius, right_radius

'''Get fps'''
vidcap = cv2.VideoCapture('project_video.mp4') #challenge_video.mp4
num_frames = 100;     
print ("Capturing {0} frames".format(num_frames))
start = time.time()
for i in range(0, num_frames) :
     ret, frame = vidcap.read()
    # End time
end = time.time()
    # Time elapsed
seconds = end - start
print ("Time taken : {0} seconds".format(seconds))
    # Calculate frames per second
fps  = num_frames / seconds;
print ("Estimated frames per second : {0}".format(fps))
'''Get frames'''
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  #print ('Read a new frame: '+ str(success))
  count += 1


img=cv2.imread("0.jpg")
height , width , layers =  img.shape
video = cv2.VideoWriter('video2.mp4',-1,30,(width,height))
for d in range(count):
    print(d)
#    img=cv2.imread("0.jpg")
    img=cv2.imread("%d.jpg" % d) 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    K = np.matrix([[1.15422732e3,0,6.71627794e2],[0.,1.14818221e3,3.86046312e2],[0,0,1]])
    dist = np.array([[-2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02]])
    im = cv2.undistort(img,K,dist,None,K) #undistort the image
#    canny = cv2.Canny(im,75,255)
#    cv2.imshow('Seg',canny)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    h = img.shape[0] 
    w = img.shape[1]
    
    # manually select points on lanes
    #points for project_video
    src = np.float32([[200, 720],
    		[1100, 720],
    		[595, 450],
    		[685, 450]])
    dst = np.float32([[300, 720],
    		[980, 720],
    		[300, 0],
    		[980, 0]])

    # find homography to change perspective
    H,flag = cv2.findHomography(src,dst)
    H_inv = cv2.getPerspectiveTransform(dst,src)
    warp= cv2.warpPerspective(im,H,(w,h))
    
    blur = cv2.bilateralFilter(warp,9,75,75)
    median = cv2.medianBlur(blur,5)
    # get only pixels that are white and yellow corresponds to our lanes in the picture
    mask = cv2.inRange(median,np.array([0,0,170]),np.array([200,255,255]))
    
    # helps extract that part of the image
    seg = cv2.bitwise_and(im,im,mask=mask)
    # edge detection to get the lanes
    canny = cv2.Canny(seg,75,255)
#    cv2.imshow('Seg',canny)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#takes the bottom half of the image(roi) and adds the columns
    hist = np.sum(canny[canny.shape[0]//2:,:],axis=0)
    # create a black image to draw and visualize on
    out_img = (np.dstack((canny,canny,canny))*255) 
     #take the midpoint of the histogram
    midpoint = np.int(hist.shape[0] / 2)
    #gets the position of the left lane starting point
    left = np.argmax(hist[:midpoint]) 
    #gets the position of the right lane starting point
    right= np.argmax(hist[midpoint:]) + midpoint 
    
    # create windows to get the positions of the pixels corresponding to the lanes
    # Height of windows
    window_height = np.int(canny.shape[0]/9) #height can be changed by changing the denominator
    # Get x and y positions of all nonzero pixels in the image
    nonzero = canny.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Update the bottom points for each window
    leftx_current = left
    rightx_current = right
    # width of the window
    margin = 30
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Receive left and right lane pixel indices
    left_lane_pts = []
    right_lane_pts = []
    # Step through the windows one by one
    for window in range(9):
        # define the window boundary points
        win_y_low = canny.shape[0] - (window+1)*window_height
        win_y_high = canny.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
#        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
#        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        check=((nonzeroy >= win_y_low) & (nonzeroy < win_y_high))
        checkx=((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high))
        checkm=(check & checkx)
        checkmf=checkm.nonzero()
        good_left_pts = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_pts = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_pts.append(good_left_pts)
        right_lane_pts.append(good_right_pts)
        # > minpix pixels, recenter next window on their mean position
        if len(good_left_pts) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_pts]))
        if len(good_right_pts) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_pts]))
        # Concatenate the arrays of indices
    left_lane_pts = np.concatenate(left_lane_pts)
    right_lane_pts = np.concatenate(right_lane_pts)
    # Extract left and right line pixel positions which correspond to non-zero pixels
    leftx = nonzerox[left_lane_pts]
    lefty = nonzeroy[left_lane_pts]
    rightx = nonzerox[right_lane_pts]
    righty = nonzeroy[right_lane_pts]
    testlx=len(leftx) 
    testly=len(lefty)
    testry=len(righty)
    testrx=len(rightx)
    if(len(leftx) >0 and len(lefty) >0 and len(righty) >0 and len(rightx) >0):
            
            
            pright_lane_pts=right_lane_pts
            pleft_lane_pts=left_lane_pts
            pnonzeroy=nonzeroy
            pnonzerox=nonzerox
            pleftx = nonzerox[left_lane_pts]
            plefty = nonzeroy[left_lane_pts]
            prightx = nonzerox[right_lane_pts]
            prighty = nonzeroy[right_lane_pts]
            if(len(rightx)<250 or len(leftx)<250):
                if(len(rightx)>len(leftx)):
                    left_avg_x=np.average(leftx)
                    leftx=np.array([]);
                    for i in range(len(rightx)):
                        leftx=np.append(leftx,[left_avg_x])
                else:
                    right_avg_x=np.average(rightx)
                    rightx=np.array([]);
                    for i in range(len(leftx)):
                        rightx=np.append(rightx,[right_avg_x])
                if(len(righty)>len(lefty)):
                    left_avg_y=np.average(lefty)
                    for i in range(len(righty)-len(lefty)):
                        lefty=np.append(lefty,righty[i])
                else:
                    right_avg_y=np.average(righty)
                
                    for i in range(len(lefty)-len(righty)):
                        righty=np.append(righty,lefty[i])
            
            # get the centre line to get the slope to predict turns
            
    else:
            leftx = pleftx 
            lefty = plefty 
            rightx = prightx 
            righty = prighty
            right_lane_pts=pright_lane_pts
            left_lane_pts=pleft_lane_pts
            nonzeroy=pnonzeroy
            nonzerox=pnonzerox
    cen_avgx=[]
    cen_avgy=[]
    for i in range(len(righty)):
        for j in range(len(lefty)):
            if(righty[i]==lefty[j]):
                cen_avgx.append(((leftx[j]+rightx[i])/2))
                cen_avgy.append(((lefty[j])))

    center_fit =[0,0,0]
    if(len(cen_avgx) >0 and len(cen_avgy)>0):
        center_fit = np.polyfit(cen_avgy, cen_avgx, 2)
            
    left_lane = np.polyfit(lefty, leftx, 2)
    right_lane = np.polyfit(righty, rightx, 2)
        
    # call function to get the radius of curvature
    left_curve, right_curve = radius_of_curvature(left_lane_pts, right_lane_pts, nonzerox, nonzeroy)
        
            # call function to perform final visualization on top of original undistorted image
    result = final(im, left_lane, right_lane, H_inv,left_curve,right_curve,center_fit[0])        
#    cv2.imshow('Seg',result)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    plt.imshow(result)
    video.write(result)

video.release()