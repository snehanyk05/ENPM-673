# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:59:33 2019

@author: Sneha
"""

import cv2
import numpy as np

img = cv2.imread('frame500.jpg')
blur = cv2.GaussianBlur(img,(5,5),0)
imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


edged = cv2.Canny(imgray,100,200)
cv2.imshow('edges',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
( cnts, _) = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
#try: hierarchy = hierarchy[0]
#except: hierarchy = []
#
#height, width = edged.shape
#min_x, min_y = width, height
#max_x = max_y = 0
#
## computes the bounding box for the contour, and draws it on the frame,
#for contour, hier in zip(contours, hierarchy):
#    (x,y,w,h) = cv2.boundingRect(contour)
#    min_x, max_x = min(x, min_x), max(x+w, max_x)
#    min_y, max_y = min(y, min_y), max(y+h, max_y)
#    if w > 80 and h > 80:
#        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)
#
#if max_x - min_x > 0 and max_y - min_y > 0:
#    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
im = cv2.drawContours(edged, cnts, -1, (0,255,0), 3)
#cv2.imshow('contours',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imshow('contours',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
#im = cv2.drawContours(edged, cnts, -1, (0,255,0), 3)
#cv2.imshow('contours',im)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
imd = cv2.dilate(im,None)



#corners = cv2.goodFeaturesToTrack(imb,4,0.1,60)
#corners = np.int0(corners)
#for i in corners:
#    x,y = i.ravel()
#    cv2.circle(imb,(x,y),3,225,-1)
#cv2.imshow('corner',imb)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
dst = cv2.cornerHarris(imd,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
#ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)



# define the criteria to stop and refine the corners
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 4, 0.001)
#corners = cv2.cornerSubPix(imgray,np.float32(centroids),(5,5),(-1,-1),criteria)
#min_x = corners[0][1]
#min_y = corners[0][1]
#max_x = max_y = 0
#for c in corners:
#    print(c)
#    if(c[0]<min_x):
#     min_x=c[0]
#   
#    if(c[1]<min_y):
#     min_y=c[1]
#    
#    if(c[0]>max_x):
#     max_x=c[0]
#    
#    if(c[0]>max_y):
#     max_y=c[1]
    
    
#    (x,y,w,h) = cv2.boundingRect(c)
#    min_x, max_x = min(x, min_x), max(x+w, max_x)
#    min_y, max_y = min(y, min_y), max(y+h, max_y)
#print(int(min_x),int(min_y),int(max_x),int(max_y))
#cv2.circle(img,(int(min_y),int(min_x)), 50, (0,0,255), -1)
# Now draw them
#res = np.hstack((centroids,corners))
#res = np.int0(res)
#img[res[:,1],res[:,0]]=[0,0,255]
#img[res[:,3],res[:,2]] = [0,255,0]


#for i in corners:
#    x,y = i.ravel()
#    cv2.circle(imb,(x,y),3,225,-1)
#  blur = cv2.GaussianBlur(img,(9,9),0)     
        # Threshold for an optimal value, it may vary depending on the image.
#img = cv2.imread('frame500.jpg')
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(imgray,np.float32(centroids),(5,5),(-1,-1),criteria)
min_x1 = min_x2 = 1000
max_x1 = max_x2=0
min_y1 = min_y2 = 1000
max_y1 = max_y2= 0
min_tup_ur=min_tup_ul=min_tup_br=min_tup_bl=corners[3];

for i in range(3,len(corners)):
    print(corners[i])
#    if(corners[i][1]>=max_y1 or corners[i][0]<=min_x1):
##     print ('here')
#     min_x1=corners[i][0]
#     max_y1=corners[i][1]
#     min_tup_bl=corners[i]
#     
#    if(corners[i][1]<=min_y1 and corners[i][0]<=min_x2):
##     print ('here')
#     min_x2=corners[i][0]
#     min_y1=corners[i][1]
#     min_tup_ul=corners[i]
#     
#    
#    
#    if(corners[i][1]>=max_y2 or corners[i][0]>=max_x1):
##     print ('here')
#     max_x1=corners[i][0]
#     max_y2=corners[i][1]
#     min_tup_br=corners[i]
#    if(corners[i][1]<=min_y2 and corners[i][0]>=max_x2 ):
##     print ('here')
#     max_x2=corners[i][0]
#     min_y2=corners[i][1]
#     min_tup_ur=corners[i]
    if(corners[i][1]>max_y2 or corners[i][1]>max_y1):
        print('here')
        if(max_y2<=max_y1):
             max_y2=max_y1
             max_y1=corners[i][1]
             if(max_y2>max_y1):
                 temp=max_y2
                 max_y2=max_y1
                 max_y1=temp
#             print(max_y1,max_y2)
    if(corners[i][0]>max_x2 or corners[i][0]>max_x1):
       
        if(max_x2<=max_x1):
             max_x2=max_x1
             max_x1=corners[i][0] 
             if(max_x2>max_x1):
                 temp=max_x2
                 max_x2=max_x1
                 max_x1=temp
    if(corners[i][1]<min_y2 or corners[i][1]<min_y1):
       
        if(min_y2>=min_y1):
             min_y2=min_y1
             min_y1=corners[i][1]
             if(min_y2<min_y1):
                 temp=min_y2
                 min_y2=min_y1
                 min_y1=temp
    if(corners[i][0]<min_x2 or corners[i][0]<min_x1):
       
        if(min_x2>=min_x1):
             min_x2=min_x1
             min_x1=corners[i][0]
             if(min_x2<min_x1):
                 temp=min_x2
                 min_x2=min_x1
                 min_x1=temp
             
    
for i in range(1,len(corners)):
    if(corners[i][0]==min_x1):
     min_tup_1=corners[i]
    if(corners[i][0]==min_x2):
     min_tup_2=corners[i]
    if(corners[i][1]==min_y1):
     min_tup_3=corners[i]
    if(corners[i][1]==min_y2):
     min_tup_4=corners[i]
    if(corners[i][0]==max_x1):
     min_tup_5=corners[i]
    if(corners[i][0]==max_x2):
     min_tup_6=corners[i]
    if(corners[i][1]==max_y1):
     min_tup_7=corners[i]
    if(corners[i][1]==max_y2):
     min_tup_8=corners[i]  
     
print(min_x1,min_x2,max_x1,max_x2,min_y1,min_y2,max_y1,max_y2)
cv2.circle(img,(min_tup_1[0],min_tup_1[1]), 5, (0,255,0), -1)    
cv2.circle(img,(min_tup_2[0],min_tup_2[1]), 5, (0,255,0), -1)
cv2.circle(img,(min_tup_3[0],min_tup_3[1]), 5, (0,255,0), -1)
cv2.circle(img,(min_tup_4[0],min_tup_4[1]), 5, (0,255,0), -1)
cv2.circle(img,(min_tup_5[0],min_tup_5[1]), 5, (0,255,0), -1)    
cv2.circle(img,(min_tup_6[0],min_tup_6[1]), 5, (0,255,0), -1)
cv2.circle(img,(min_tup_7[0],min_tup_7[1]), 5, (0,255,0), -1)
cv2.circle(img,(min_tup_8[0],min_tup_8[1]), 5, (0,255,0), -1)
#cv2.circle(img,(max_x,min_y), 20, (0,255,0), -1)   
img[dst>0.01*dst.max()]=[0,0,255]


#cv2.circle(img,(points[0],points[1]), 50, (0,0,255), -1)
#cv2.imshow('dst',img)
cv2.imwrite('test200.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()