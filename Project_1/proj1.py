import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('frame1.jpg')
#img = cv2.medianBlur(img1,5)
blur = cv2.GaussianBlur(img1,(3,3),0)
imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(imgray,100,200)

( cnts,h,z) = cv2.findContours(edges, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im = cv2.drawContours(edges, h, -1, (0,255,0), 3)

cv2.imshow('edges',im)
cv2.waitKey(0)
cv2.destroyAllWindows()


areas=[]
min_area=10000;
min_h=0;
i=0;
for c in h:
    areas.append(cv2.contourArea(c))
    a=cv2.contourArea(c)
#    print(a,h)
    if(min_area>a and a>0):
#        print(c)
        min_area=a
        min_h=c
        
#    area = cv2.contourArea(c)
    
#    if(a):
#        cnt = h[int(a)]
#        area = cv2.drawContours(img1,cnt,-1, (0,255,0), 3)
#        cv2.imshow('edge',area)
#        cv2.waitKey(0)
print(min_h)
sorted_areas = sorted(areas)
print(h[int(sorted_areas[1])]);
cnt = min_h
M = cv2.moments(cnt)
print (M)
area = cv2.contourArea(cnt)
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
#rect = cv2.minAreaRect(cnt)
#box = cv2.boxPoints(rect)
#box = np.int0(box)
cv2.drawContours(img1,approx,-1,(0,0,255),3)
cv2.imshow('area',img1)
cv2.waitKey(0)
min_x1 = min_x2 = 1000
max_x1 = max_x2=0
min_y1 = min_y2 = 1000
max_y1 = max_y2= 0
min_tup_ur=min_tup_ul=min_tup_br=min_tup_bl=cnt[0];

#for i in range(0,len(cnt)):
    
#    print(cnt[i][0][0])
#    if(corners[i][1]>=max_y1 or corners[i][0]<=min_x1):
##     print ('here')
#     min_x1=corners[i][0]
#     max_y1=corners[i][1]
#     min_tup_bl=corners[i]
#     
#    if(corners[i][1]<=min_y1 and corners[i][0]<=min_x2):
##     print ('here')
#     min_x2=corners[i][0]
#     min_y1=cnt[i][1]
#     min_tup_ul=cnt[i]
#     
#    
#    
#    if(cnt[i][1]>=max_y2 or cnt[i][0]>=max_x1):
##     print ('here')
#     max_x1=cnt[i][0]
#     max_y2=cnt[i][1]
#     min_tup_br=cnt[i]
#    if(cnt[i][1]<=min_y2 and cnt[i][0]>=max_x2 ):
##     print ('here')
#     max_x2=cnt[i][0]
#     min_y2=cnt[i][1]
#     min_tup_ur=cnt[i]
#    if(cnt[i][1]>max_y2 or cnt[i][1]>max_y1):
#        print('here')
#        if(max_y2<=max_y1):
#             max_y2=max_y1
#             max_y1=cnt[i][1]
#             if(max_y2>max_y1):
#                 temp=max_y2
#                 max_y2=max_y1
#                 max_y1=temp
##             print(max_y1,max_y2)
#    if(cnt[i][0]>max_x2 or cnt[i][0]>max_x1):
#       
#        if(max_x2<=max_x1):
#             max_x2=max_x1
#             max_x1=cnt[i][0] 
#             if(max_x2>max_x1):
#                 temp=max_x2
#                 max_x2=max_x1
#                 max_x1=temp
#    if(cnt[i][1]<min_y2 or cnt[i][1]<min_y1):
#       
#        if(min_y2>=min_y1):
#             min_y2=min_y1
#             min_y1=cnt[i][1]
#             if(min_y2<min_y1):
#                 temp=min_y2
#                 min_y2=min_y1
#                 min_y1=temp
#    if(cnt[i][0]<min_x2 or cnt[i][0]<min_x1):
#       
#        if(min_x2>=min_x1):
#             min_x2=min_x1
#             min_x1=cnt[i][0]
#             if(min_x2<min_x1):
#                 temp=min_x2
#                 min_x2=min_x1
#                 min_x1=temp
#             
#    
#for i in range(1,len(cnt)):
#    if(cnt[i][0]==min_x1):
#     min_tup_1=cnt[i]
#    if(cnt[i][0]==min_x2):
#     min_tup_2=cnt[i]
#    if(cnt[i][1]==min_y1):
#     min_tup_3=cnt[i]
#    if(cnt[i][1]==min_y2):
#     min_tup_4=cnt[i]
#    if(cnt[i][0]==max_x1):
#     min_tup_5=cnt[i]
#    if(cnt[i][0]==max_x2):
#     min_tup_6=cnt[i]
#    if(cnt[i][1]==max_y1):
#     min_tup_7=cnt[i]
#    if(cnt[i][1]==max_y2):
#     min_tup_8=cnt[i]  
#     
#print(min_x1,min_x2,max_x1,max_x2,min_y1,min_y2,max_y1,max_y2)
#cv2.circle(img,(min_tup_1[0],min_tup_1[1]), 5, (0,255,0), -1)    
#cv2.circle(img,(min_tup_2[0],min_tup_2[1]), 5, (0,255,0), -1)
#cv2.circle(img,(min_tup_3[0],min_tup_3[1]), 5, (0,255,0), -1)
#cv2.circle(img,(min_tup_4[0],min_tup_4[1]), 5, (0,255,0), -1)
#cv2.circle(img,(min_tup_5[0],min_tup_5[1]), 5, (0,255,0), -1)    
#cv2.circle(img,(min_tup_6[0],min_tup_6[1]), 5, (0,255,0), -1)
#cv2.circle(img,(min_tup_7[0],min_tup_7[1]), 5, (0,255,0), -1)
#cv2.circle(img,(min_tup_8[0],min_tup_8[1]), 5, (0,255,0), -1)
#x,y,w,z = cv2.boundingRect(cnt)
#cv2.rectangle(img1,(x,y),(x+w,y+z),(0,255,0),2)
area = cv2.drawContours(img1,cnt,-1, (0,255,0), 3)

cv2.imshow('area',area)
cv2.waitKey(0)

gray = cv2.cvtColor(area,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,4,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img1,(x,y),3,225,-1)
cv2.imshow('im',img1)
cv2.waitKey(0)
