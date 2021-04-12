import cv2
import numpy as np;
 
# Read image
#im = cv2.imread("frame200.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread('frame20.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(imgray,(5,5),0)
#lap=cv2.Laplacian(blur, cv2.CV_64F, 5)
#imgray = cv2.cvtColor(lap, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY);

#edged = cv2.Canny(binary,100,200)
#
#cnts, h,z = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#im = cv2.drawContours(edged, h, -1, (0,255,0), 3)
#
#imd = cv2.dilate(im,None)
#cv2.imshow('edges',imd)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.thresholdStep = 1;

params.filterByColor = True;
params.blobColor = 255;
params.filterByArea = True;
params.minArea = 3000;
params.maxArea = 150000;
params.filterByCircularity = False;
#    //params.minCircularity = "";

params.filterByConvexity = False;
#    //params.minConvexity = "";

params.filterByInertia = False;
#    //params.minInertiaRatio = "";
 
# Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else :
    detector = cv2.SimpleBlobDetector_create(params)
# Set up the detector with default parameters.

 
# Detect blobs.
keypoints = detector.detect(binary)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imwrite('test200.jpg', im_with_keypoints)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)