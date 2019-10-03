import cv2
import numpy as np
import tkinter as tk
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
  
  print ('Read a new frame: '+ str(success))
  count += 1
cropImages(trainFolder,cropFolder,149)
