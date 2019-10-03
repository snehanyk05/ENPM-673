# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:32:49 2019

@author: Sneha
"""
import numpy as np
import cv2
import time
major_ver=(cv2.__version__)


vidcap = cv2.VideoCapture('Tag0.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS)
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
     
 
    # Number of frames to capture
params = cv2.SimpleBlobDetector_Params()
 
params.minThreshold = 5;
params.maxThreshold = 100;
params.filterByColor = True

 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
 
# Filter by Circularity
#params.filterByCircularity = True
#params.minCircularity = 0.1
# 
## Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.87
# 
## Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else :
    detector = cv2.SimpleBlobDetector_create(params)
    
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
success,image = vidcap.read()
count = 0
success = True
while success:
#  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

img=cv2.imread("frame0.jpg")
height , width , layers =  img.shape

video = cv2.VideoWriter('video1.mp4',-1,fps-90,(width,height))
for i in range(0,count):
  img=cv2.imread("frame%d.jpg" % i);
     
  blur = cv2.GaussianBlur(img,(5,5),0)
  imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


  edged = cv2.Canny(imgray,100,200)
  cnts, h,z = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  im = cv2.drawContours(edged, h, -1, (0,255,0), 3)

  imd = cv2.dilate(im,None)
  keypoints = detector.detect(imd)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
  im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  
  video.write(im_with_keypoints)
  

cv2.destroyAllWindows()
#cap = cv2.VideoCapture(1)


video.release()

#        imgplot = plt.imshow('dst',img)
#        plt.show()
            
