# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:29:21 2019

@author: Sneha
"""
import matplotlib.image as mpimg
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PIL.Image, PIL.ImageTk
refPt = []
color=''
roi_corners=[]
 
def click_and_crop(image,count,folder):
	# grab references to the global variables
    global refPt
    global color
    print(refPt)
    mask = np.zeros(image.shape, dtype=np.uint8)
            # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    roi_corners = np.array([refPt], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        # apply the mask
    masked_image = cv2.bitwise_and(image, mask)
        
        # save the result
#    print(color)
    
    cv2.imwrite(folder+color+"%d.jpg" % count,masked_image)
    refPt=[]
    
def cropRed():
    global color
    color='R_'
def cropGreen():
    global color
    color='G_'
def cropYellow():
    global color
    color='Y_' 
    
def onpick(event):
    refPt.append((event.x, event.y))
#    click_and_crop(event,event.x,event.y)

    return True

def cropImages(inputFolder,outputFolder,count):
        for i in range(count):
#            img=cv2.imread("%d.jpg" % i)
            print(i)
            root= tk.Tk() 
            cv_img = cv2.cvtColor(cv2.imread(inputFolder+"%d.jpg" % i), cv2.COLOR_BGR2RGB)
            
            
            image = cv2.imread(inputFolder+"%d.jpg" % i, -1)
            # mask defaulting to black for 3-channel and transparent for 4-channel
            # (of course replace corners with yours)
#            mask = np.zeros(image.shape, dtype=np.uint8)
#            # fill the ROI so it doesn't get wiped out when the mask is applied
#            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
#            ignore_mask_color = (255,)*channel_count
            
            # from Masterfool: use cv2.fillConvexPoly if you know it's convex
            
            
            
            
            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            height, width, no_channels = cv_img.shape
             
            # Create a canvas that can fit the above image
            canvas = tk.Canvas(root, width = width, height = height)
            canvas.pack()
             
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img),master=root)
             
             # Add a PhotoImage to the Canvas
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.bind("<Button 1>",onpick)
            btn_blur=tk.Button(root, text="Crop", width=50, command= lambda:click_and_crop(image,i,outputFolder))
            btn_blur.pack(anchor=tk.CENTER, expand=True)
            btn_blur=tk.Button(root, text="Red", width=50, command=cropRed)
            btn_blur.pack(anchor=tk.CENTER, expand=True)
            btn_blur=tk.Button(root, text="Yellow", width=50, command=cropYellow)
            btn_blur.pack(anchor=tk.CENTER, expand=True)
            btn_blur=tk.Button(root, text="Green", width=50, command=cropGreen)
            btn_blur.pack(anchor=tk.CENTER, expand=True)
    #        btn_blur=tk.Button(root, text="Next", width=50, command=NextImage)
    #        btn_blur.pack(anchor=tk.CENTER, expand=True)
            
            root.mainloop()
#        img=cv2.imread("%d.jpg" % i)
        
def computerHist(folder,count):
#    redHist=np.array([])
#    bb=np.array([])
#    br=np.array([])
#    bg=np.array([])
    red_histogram = np.zeros((255,3))
    green_histogram = np.zeros((255,3))
    yel_histogram = np.zeros((255,3))
    for l in range(51,100):
#            img=cv2.imread("%d.jpg" % i)
#            print(i)
#            root= tk.Tk() 
#            cv_img = cv2.cvtColor(cv2.imread(folder+"%d.jpg" % i), cv2.COLOR_BGR2RGB)
            
            
#            img = cv2.imread(folder+"G_%d.jpg" % i, -1)
            
            
            img = cv2.imread('TrainingFolder/Frames/%d.jpg' % l, -1)
            plane1 = img[:,:,0] #blue
            plane2 = img[:,:,1] #green
            plane3 = img[:,:,2] #red
#            RED
            im_r =cv2.imread(folder+"R_%d.jpg" % l, -1)
            

#            print(im)
            print(l)
            x_r = np.where(im_r>0)
            
            blue_r = []
            green_r = []
            red_r = []
            for i in range(len(x_r[0])):
                b_r= plane1[x_r[0][i]][x_r[1][i]]
                blue_r.append(b_r)
                g_r = plane2[x_r[0][i]][x_r[1][i]]
                green_r.append(g_r)
                r_r = plane3[x_r[0][i]][x_r[1][i]]
                red_r.append(r_r)
            
            blue_hist_r = np.histogram(blue_r,bins = range(0,256))
            green_hist_r = np.histogram(green_r,bins=range(0,256))
            red_hist_r = np.histogram(red_r,bins = range(0,256))
            red_histogram = red_histogram + np.column_stack((blue_hist_r[0],green_hist_r[0],red_hist_r[0]))
            
            
##            GREEN
##            print(i)
#            im_g =cv2.imread(folder+"G_%d.jpg" % l, -1)
#            
#            x_g = np.where(im_g>0)
#            
#            blue_g = []
#            green_g = []
#            red_g = []
#            for i in range(len(x_g[0])):
#                b_g= plane1[x_g[0][i]][x_g[1][i]]
#                blue_g.append(b_g)
#                g_g = plane2[x_g[0][i]][x_g[1][i]]
#                green_g.append(g_g)
#                r_g = plane3[x_g[0][i]][x_g[1][i]]
#                red_g.append(r_g)
#            
#            blue_hist_g = np.histogram(blue_g,bins = range(0,256))
#            green_hist_g = np.histogram(green_g,bins=range(0,256))
#            red_hist_g = np.histogram(red_g,bins = range(0,256))
#            green_histogram = green_histogram + np.column_stack((blue_hist_g[0],green_hist_g[0],red_hist_g[0]))
            
            
            
#            YELLOW
            im_y =cv2.imread(folder+"Y_%d.jpg" % l, -1)
            
            x_y = np.where(im_y>0)
            
            blue_y = []
            green_y = []
            red_y = []
            for i in range(len(x_y[0])):
                b_y= plane1[x_y[0][i]][x_y[1][i]]
                blue_y.append(b_y)
                g_y = plane2[x_y[0][i]][x_y[1][i]]
                green_y.append(g_y)
                r_y = plane3[x_y[0][i]][x_y[1][i]]
                red_y.append(r_y)
            
            blue_hist_y = np.histogram(blue_y,bins = range(0,256))
            green_hist_y = np.histogram(green_y,bins=range(0,256))
            red_hist_y = np.histogram(red_y,bins = range(0,256))
            yel_histogram = yel_histogram + np.column_stack((blue_hist_y[0],green_hist_y[0],red_hist_y[0]))
            

           
    red_histogram = red_histogram/50
    green_histogram = green_histogram/50
    yel_histogram = yel_histogram/50
    plt.bar(range(0,255),red_histogram[:,0],color ='blue')
    plt.bar(range(0,255),red_histogram[:,2],color ='red')
    plt.bar(range(0,255),red_histogram[:,1],color ='green')
    plt.show()
#    plt.bar(range(0,255),green_histogram[:,0],color ='blue')
#    plt.bar(range(0,255),green_histogram[:,2],color ='red')
#    plt.bar(range(0,255),green_histogram[:,1],color ='green')
#    plt.show()
    plt.bar(range(0,255),yel_histogram[:,0],color ='blue')
    plt.bar(range(0,255),yel_histogram[:,2],color ='red')
    plt.bar(range(0,255),yel_histogram[:,1],color ='green')
    plt.show()
    return (1)
    
vidcap = cv2.VideoCapture('detectbuoy.avi') #challenge_video.mp4

count = 0
countTest = 0
countTrain = 0
success = True
testFolder='TestFolder/'
trainFolder='TrainingFolder/Frames/'
cropFolder='TrainingFolder/CroppedFrames/'
while success:
  success,image = vidcap.read()
  if(count%4==1):
        cv2.imwrite(testFolder+"%d.jpg" % countTest, image)     # save frame as JPEG file
        countTest+=1
  else:
        cv2.imwrite(trainFolder+"%d.jpg" % countTrain, image) 
        countTrain+=1
  
  #print ('Read a new frame: '+ str(success))
  count += 1
#
cropImages(trainFolder,cropFolder,countTrain)
#
#redHist=computerHist(cropFolder,countTrain)
#img=cv2.imread("0.jpg")
#height , width , layers =  img.shape
#video = cv2.VideoWriter('video2.mp4',-1,30,(width,height))
#for d in range(count):
#    print(d)
##    img=cv2.imread("0.jpg")
#    img=cv2.imread("%d.jpg" % d)