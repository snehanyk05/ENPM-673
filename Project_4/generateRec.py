# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:52:30 2019

@author: Sneha
"""

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
 
def click_and_rec(image):
	# grab references to the global variables
    global refPt
    global color
    print(refPt)

    cv2.rectangle(image, refPt[0], refPt[1],(255,0,0),thickness=2)

    refPt=[]

    
def onpick(event):
    print(event.x, event.y)
    refPt.append((event.x, event.y))
#    click_and_crop(event,event.x,event.y)

    return True

def cropImages(inputFolder):
#        for i in range(count):
#            img=cv2.imread("%d.jpg" % i)
#            print(i)
            root= tk.Tk() 
            cv_img = cv2.cvtColor(cv2.imread(inputFolder+"frame0020.jpg"), cv2.COLOR_BGR2RGB)
            
            
            image = cv2.imread(inputFolder+"frame0020.jpg", -1)
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
            btn_blur=tk.Button(root, text="Rectangle", width=50, command= lambda:click_and_rec(image))
            btn_blur.pack(anchor=tk.CENTER, expand=True)

            
            root.mainloop()
            cv2.imshow('Seg',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

testFolder='data/car/'



cropImages(testFolder)
