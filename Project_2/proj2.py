# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:27:06 2019

@author: Sneha
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:17:06 2019

@author: Sneha
"""

import cv2
import numpy as np
import matplotlib as plt
#vidcap = cv2.VideoCapture('project_video.mp4')
#success,image = vidcap.read()
#count = 0
#success = True
#while success:
#  cv2.imwrite("%d.jpg" % count, image)     # save frame as JPEG file
#  success,image = vidcap.read()
#  #print ('Read a new frame: '+ str(success))
#  count += 1
#count = 1257
#for d in range(count):
#    img = cv2.imread("%d.jpg"%d)
#    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#    K = np.matrix([[1.15422732e3,0,6.71627794e2],[0.,1.14818221e3,3.86046312e2],
# [0,0,1]])
#    dist = np.array([[-2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02]])
#    im = cv2.undistort(img,K,dist,None,K)
#    cv2.imwrite('frame%d.jpg'%d,im)
class Line():
	def __init__(self, n):
		"""
		n is the window size of the moving average
		"""
		self.n = n
		self.detected = False

		# Polynomial coefficients: x = A*y^2 + B*y + C
		# Each of A, B, C is a "list-queue" with max length n
		self.A = []
		self.B = []
		self.C = []
		# Average of above
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return (self.A_avg, self.B_avg, self.C_avg)

	def add_fit(self, fit_coeffs):
		"""
		Gets most recent line fit coefficients and updates internal smoothed coefficients
		fit_coeffs is a 3-element list of 2nd-order polynomial coefficients
		"""
		# Coefficient queue full?
		q_full = len(self.A) >= self.n
		print(q_full)

		# Append line fit coefficients
		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		# Pop from index 0 if full
		if q_full:
			_ = self.A.pop(0)
			_ = self.B.pop(0)
			_ = self.C.pop(0)
			print('here')

		# Simple average of line coefficients
		self.A_avg = np.mean(self.A)
		self.B_avg = np.mean(self.B)
		self.C_avg = np.mean(self.C)

		return (self.A_avg, self.B_avg, self.C_avg)
def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	cv2.imshow('Warp back',newwarp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# Combine the result with the original image
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	# Annotate lane curvature values and vehicle offset from center
#	avg_curve = (left_curve + right_curve)/2
#	label_str = 'Radius of curvature: %.1f m' % avg_curve
#	result = cv2.putText(result, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)
#
#	label_str = 'Vehicle offset from lane center: %.1f m' % vehicle_offset
#	result = cv2.putText(result, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

	return result
def calc_vehicle_offset(undist, left_fit, right_fit):
	"""
	Calculate vehicle offset from lane center, in meters
	"""
	# Calculate vehicle center offset in pixels
	bottom_y = undist.shape[0] - 1
	bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
	bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
	vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

	# Convert pixel offset to meters
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	vehicle_offset *= xm_per_pix

	return vehicle_offset
def calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
	"""
	Calculate radius of curvature in meters
	"""
	y_eval = 719  # 720p video/image, so last (lowest on screen) y index is 719

	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/700 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
#	print("Left X",leftx)
#	print("Left Y",lefty)
#	print("Right X",rightx)
#	print("RIght Y",righty)

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = 0
	right_curverad = 1
	# Now our radius of curvature is in meters

	return left_curverad, right_curverad

#‘’’ perspective transform and edge detection’’’
def warp_perspective(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    K = np.matrix([[1.15422732e3,0,6.71627794e2],[0.,1.14818221e3,3.86046312e2],[0,0,1]])
    dist = np.array([[-2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02]])
    im = cv2.undistort(img,K,dist,None,K) #undistort the image
    h = gray.shape[0] 
    w = gray.shape[1]
    
    # manually select points on lanes
    src = np.array([[200,720],[1100,720],[595,450],[685,450]])
    dst = np.array([[300,720],[980,720],[300,0],[980,0]])
    
    # find homography to change perspective
    H,flag = cv2.findHomography(src,dst)
    warp_image= cv2.warpPerspective(im,H,(w,h))
    return warp_image

def edge(warp):
    blur = cv2.bilateralFilter(warp,9,75,75)
    median = cv2.medianBlur(blur,5)
    # get only pixels that are white and yellow corresponds to our lanes in the picture
    mask = cv2.inRange(median,np.array([0,0,200]),np.array([200,255,255]))
    # helps extract that part of the image
    seg = cv2.bitwise_and(median,median,mask=mask)
#    blur = cv2.bilateralFilter(seg,9,75,75)
    # edge detection on that part of the image
    canny = cv2.Canny(seg,75,255)
    return canny

img = cv2.imread('126.jpg')

warp = warp_perspective(img)
cv2.imshow('warp',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
canny = edge(warp)
cv2.imshow('warp',canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


#cv2.imshow('Seg',canny)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#canny = cv2.Sobel(canny,cv2.CV_64F,1,0)

left_line = Line(n=5)
right_line = Line(n=5)


hist = np.sum(canny[canny.shape[0]//2:,:],axis=0) #takes the bottom half of the image, stated in pdf
out_img = (np.dstack((canny,canny,canny))*255) # creates a black-background?
midpoint = np.int(hist.shape[0] / 2) #gets the midpoint of the histogram
left = np.argmax(hist[:midpoint]) #gets the left lane starting point
rightx= np.argmax(hist[midpoint:]) + midpoint #gets the right lane starting point














nwindows = 9
	# Set height of windows
window_height = np.int(canny.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
nonzero = canny.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
leftx_current = left
rightx_current = rightx
	# Set the width of the windows +/- margin
margin = 20
	# Set minimum number of pixels found to recenter window
minpix = 50
	# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

	# Step through the windows one by one
for window in range(nwindows):
    
		# Identify window boundaries in x and y (and right and left)
        win_y_low = canny.shape[0] - (window+1)*window_height
        win_y_high = canny.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin 
        print(margin)
   
		# Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
#        cv2.imshow('Rec1',out_img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
#        cv2.imshow('Rec2',out_img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
		# Identify the nonzero pixels in x and y within the window
        check=((nonzeroy >= win_y_low) & (nonzeroy < win_y_high))
        checkx=((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high))
        checkm=(check & checkx)
        checkmf=checkm.nonzero()
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

	# Return a dict of relevant variables
#ret = {}
#ret['left_fit'] = left_fit
#ret['right_fit'] = right_fit
#ret['nonzerox'] = nonzerox
#ret['nonzeroy'] = nonzeroy
#ret['out_img'] = out_img
#ret['left_lane_inds'] = left_lane_inds
#ret['right_lane_inds'] = right_lane_inds
#
#left_fit = ret['left_fit']
#right_fit = ret['right_fit']
#nonzerox = ret['nonzerox']
#nonzeroy = ret['nonzeroy']
#left_lane_inds = ret['left_lane_inds']
#right_lane_inds = ret['right_lane_inds']

		# Get moving average of line fit coefficients
left_fit = left_line.add_fit(left_fit)
right_fit = right_line.add_fit(right_fit)
left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

vehicle_offset = 1

	# Perform final visualization on top of original undistorted image
result = final_viz(im, left_fit, right_fit, m_inv)
cv2.imshow('Result',result)
cv2.waitKey(0)
cv2.destroyAllWindows()