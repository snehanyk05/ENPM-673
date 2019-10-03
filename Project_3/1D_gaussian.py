import numpy as np
import cv2
from average_hist import average_hist

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)
    return res

r,g,y = average_hist()
'''mean and std for red buoy'''
mu_r = np.mean(r)
sigma_r = np.std(r)

'''mean and std for green buoy'''
mu_g = np.mean(g)
sigma_g = np.std(g)

'''mean and std for yellow buoy'''
mu_y = np.mean(y)
sigma_y = np.std(y)


def gaussian_1D(x,mu,sigma):
     return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-((x-mu)**2)/(2*sigma**2))
 
im=cv2.imread('TestFolder/'+"0.jpg")
height , width , layers =  im.shape
video = cv2.VideoWriter('1D_buoy_detection.mp4',-1,10,(width,height))
'''running on test data'''
for k in range(0,50):
    img = cv2.imread('TestFolder/'+'%d.jpg' %k)
    
    # adjust contrast of the image and change the range of pixels
    new_img = adjust_gamma(img,gamma=6.5)
    new_img = cv2.GaussianBlur(new_img,(5,5),7)      
    
    red = new_img[:,:,2]
    yellow = (new_img[:,:,2]+new_img[:,:,1])/2
    green = new_img[:,:,1]
    pixels = img.shape[0]*img.shape[1]
    img_red = (np.reshape(red,pixels,1))
    img_yellow = (np.reshape(yellow,pixels,1))
    img_green = (np.reshape(green,pixels,1))
    
    '''get probability for red buoy'''
    N1 = gaussian_1D(img_red,mu_r,sigma_r)
    redprob = np.reshape(N1,(N1.shape[0],1))
    redprob = redprob/max(redprob)
    redprob = np.reshape(redprob,(480,640))
    
    ''' get probability for yellow buoy'''
    N2 = gaussian_1D(img_yellow,mu_y,sigma_y)
    yellowprob = np.reshape(N2,(N2.shape[0],1))
    yellowprob = yellowprob/max(yellowprob[:])
    yellowprob = np.reshape(yellowprob,(480,640))
    
    ''' get probaility for green buoy'''
    N3 = gaussian_1D(img_green,mu_g,sigma_g)
    greenprob = np.reshape(N3,(N3.shape[0],1))
    greenprob = greenprob/max(greenprob[:])
    greenprob = np.reshape(greenprob,(480,640))
      
    ''' segment and get red buoy'''
    red_mask = np.zeros((480,640))
    for i in range(0,480):
        for j in range(0,640):
                if redprob[i][j]>0.97:
                    red_mask[i][j]=255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    red_mask = cv2.dilate(red_mask,kernel)
    red_mask = cv2.erode(red_mask,kernel)
    _,contours,_ = cv2.findContours(red_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    circle_red = cv2.drawContours(red_mask,contours,-1,(0,0,255),1)
    red_buoy = cv2.bitwise_and(img,img,mask=circle_red)
    
    ''' segment and get yellow buoy'''
    yellow_mask = np.zeros((480,640))
    for i in range(0,480):
        for j in range(0,640):
           if yellowprob[i][j]>0.97:
               yellow_mask[i][j]=255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    yellow_mask = cv2.dilate(yellow_mask,kernel)
    yellow_mask = cv2.erode(yellow_mask,kernel)
    _,contours,_ = cv2.findContours(yellow_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    circle_yellow = cv2.drawContours(yellow_mask,contours,-1,(0,255,255),1)
    yellow_buoy = cv2.bitwise_and(img,img,mask=circle_yellow)
    
    ''' segment and get green buoy'''
    green_mask = np.zeros((480,640))
    for i in range(0,480):
        for j in range(0,640):
           if greenprob[i][j]>0.97:
               green_mask[i][j]=255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    green_mask = cv2.dilate(green_mask,kernel)
    green_mask = cv2.erode(green_mask,kernel)
    _,contours,_ = cv2.findContours(yellow_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    circle_green = cv2.drawContours(yellow_mask,contours,-1,(0,255,0),1)
    green_buoy = cv2.bitwise_and(img,img,mask=circle_green)

    video.write(img)

video.release()