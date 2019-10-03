import cv2
import numpy as np
import matplotlib.pyplot as plt

'''Uncomment commented parts to see the histogram graphs'''
def average_hist(): 
    ''' For Red Buoy'''
    for k in range(51,101):
        img = cv2.imread('TrainingFolder/Frames/'+"%d.jpg" % k)
        im = cv2.imread('TrainingFolder/CroppedFrames/'+'R_%d.jpg' % k)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        
        red_histogram = np.zeros((255,3))
        red_count = 0
        
        plane1 = img[:,:,0] #blue
        plane2 = img[:,:,1] #green
        plane3 = img[:,:,2] #red
        
        x = np.where(im>0)
        blue = []
        green = []
        red = []
        red_dist = []
        green_dist = []
        blue_dist = []
        for i in range(len(x[0])):
            b= plane1[x[0][i]][x[1][i]]
            blue.append(b)
            g = plane2[x[0][i]][x[1][i]]
            green.append(g)
            r = plane3[x[0][i]][x[1][i]]
            red.append(r)
        red_dist.append(red)
        green_dist.append(green)
        blue_dist.append(blue)
        red_samples = red_dist + blue_dist +green_dist
        blue_hist = np.histogram(blue,bins = range(0,256))
        green_hist = np.histogram(green,bins=range(0,256))
        red_hist = np.histogram(red,bins = range(0,256))
        red_histogram = red_histogram + np.column_stack((blue_hist[0],green_hist[0],red_hist[0]))
        red_count = red_count+1
    red_histogram = red_histogram/red_count
#    plt.figure(1)
#    plt.bar(range(0,255),red_histogram[:,0],color ='blue')
#    plt.bar(range(0,255),red_histogram[:,2],color ='red')
#    plt.bar(range(0,255),red_histogram[:,1],color ='green')
        
    '''For Green Buoy'''
    for j in range(0,21):
            img = cv2.imread('TrainingFolder/Frames/'+"%d.jpg" % j)
            im = cv2.imread('TrainingFolder/CroppedFrames/'+'G_%d.jpg' % j)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            green_histogram = np.zeros((255,3))
            green_count = 0
            
            plane1 = img[:,:,0] #blue
            plane2 = img[:,:,1] #green
            plane3 = img[:,:,2] #red
            
            x = np.where(im>0)
            blue = []
            green = []
            red = []
            red_dist = []
            green_dist = []
            blue_dist = []
            for i in range(len(x[0])):
                b= plane1[x[0][i]][x[1][i]]
                blue.append(b)
                g = plane2[x[0][i]][x[1][i]]
                green.append(g)
                r = plane3[x[0][i]][x[1][i]]
                red.append(r)
            red_dist.append(red)
            green_dist.append(green)
            blue_dist.append(blue)
            green_samples = red_dist + blue_dist +green_dist
            blue_hist = np.histogram(blue,bins = range(0,256))
            green_hist = np.histogram(green,bins=range(0,256))
            red_hist = np.histogram(red,bins = range(0,256))
            green_histogram = green_histogram + np.column_stack((blue_hist[0],green_hist[0],red_hist[0]))
            green_count = green_count+1
    green_histogram = green_histogram/green_count
#    plt.figure(2)
#    plt.bar(range(0,255),green_histogram[:,0],color ='blue')
#    plt.bar(range(0,255),green_histogram[:,2],color ='red')
#    plt.bar(range(0,255),green_histogram[:,1],color ='green')
    '''For Yellow Buoy'''
    for l in range(51,101):
                img = cv2.imread('TrainingFolder/Frames/'+"%d.jpg" % l)
                im = cv2.imread('TrainingFolder/CroppedFrames/'+'Y_%d.jpg' % l)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
                yellow_histogram = np.zeros((255,3))
                yellow_count = 0
                
                plane1 = img[:,:,0] #blue
                plane2 = img[:,:,1] #green
                plane3 = img[:,:,2] #red
                
                x = np.where(im>0)
                blue = []
                green = []
                red = []
                red_dist = []
                green_dist = []
                blue_dist = []
                for i in range(len(x[0])):
                    b= plane1[x[0][i]][x[1][i]]
                    blue.append(b)
                    g = plane2[x[0][i]][x[1][i]]
                    green.append(g)
                    r = plane3[x[0][i]][x[1][i]]
                    red.append(r)
                red_dist.append(red)
                green_dist.append(green)
                blue_dist.append(blue)
                yellow_samples = red_dist + blue_dist +green_dist
                blue_hist = np.histogram(blue,bins = range(0,256))
                green_hist = np.histogram(green,bins=range(0,256))
                red_hist = np.histogram(red,bins = range(0,256))
                yellow_histogram = yellow_histogram + np.column_stack((blue_hist[0],green_hist[0],red_hist[0]))
                yellow_count = yellow_count+1
    yellow_histogram = yellow_histogram/yellow_count
#    plt.figure(3)
#    plt.bar(range(0,255),yellow_histogram[:,0],color ='blue')
#    plt.bar(range(0,255),yellow_histogram[:,2],color ='red')
#    plt.bar(range(0,255),yellow_histogram[:,1],color ='green')
    '''Returns the RGB distributions for the buoys'''
    return red_samples, green_samples, yellow_samples  
