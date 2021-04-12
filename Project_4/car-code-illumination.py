import cv2
import numpy as np
import time
import os

# convert frames to video
image_folder = 'data/car'
video_name = 'car.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

img = []
parameters = [8, #maximum Iterations
				5, #initial Iterations
				7, #similarity Threshold Break
				19, #similarity Threshold
				100, #dissimilarity Threshold
				29, #featureX
				16, #featureY
				1535] #convergence Threshold

maxIter = parameters[0]
initIterations = parameters[1]
simThrb = parameters[2]
similarityThresh = parameters[3]
disThr = parameters[4]
featureX = parameters[5]
featureY = parameters[6]
convThr = parameters[7]

# Image gradient, Sobel filter
def imageGradient(img, kernelSize = 5, derivativeOrder = 1, filterScale = 1, outputType =  cv2.CV_64F):
	
	#sobel filter X
	derivativeOrderX = derivativeOrder
	derivativeOrderY = 0
	gradientImageX = cv2.Sobel(img, outputType, derivativeOrderX, derivativeOrderY, kernelSize, filterScale)

	#sobel filter Y
	derivativeOrderX, derivativeOrderY = derivativeOrderY, derivativeOrderX
	gradientImageY = cv2.Sobel(img, outputType, derivativeOrderX, derivativeOrderY, kernelSize, filterScale)
	
	return gradientImageX, gradientImageY

def increase_brightness(img, value=30):    ########### Part 3 Robustness to Illumination
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


# Positioned feature, described by a template, gradients, inverted Hessian
class Feature:
	x = None
	y = None
	x2 = None
	y2 = None
	width = None
	height = None

	hessianInv = [[0.0, 0.0],
				  [0.0, 0.0]]
	gradientX = None 
	gradientY = None 
	template = None

	dP = None
	e = None

	# Feature initialization
	def __init__(self, x, y, width, height, template, kernelSize = 5):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.halfHeight = height/2
		self.halfWidth = width/2
		self.area = self.width * self.height

		# crop a feature template
        
		self.template =  np.array(cv2.getRectSubPix(template, (width, height), self.center()), dtype=float)

		# padded template for kerneling
		templatePadded = cv2.getRectSubPix(template, (width + kernelSize, height + kernelSize), self.center())
		
		gradient = imageGradient(templatePadded, kernelSize)
		
		kernelOffset = int(np.floor(kernelSize/2))
		# Crop gradients by kernelOffset on each side
		self.gradientX = gradient[0][kernelOffset:kernelOffset+height, kernelOffset:kernelOffset+width]
		self.gradientY = gradient[1][kernelOffset:kernelOffset+height, kernelOffset:kernelOffset+width]
		
		# Calculate Hessian
		
		self.hessianInv = self.calculateInvertedHessian()

	# Center point of the feature
	def center(self, offset = [0.0, 0.0]):
		return (self.x + self.halfWidth + offset[0], self.y + self.halfHeight + offset[1])

	# Right-Bottom point of the feature
	def Rp(self):
		return (self.x + self.width, self.y + self.height)


	# Calculate Hessian matrix for gradient image window
	def calculateInvertedHessian(self):

		H = [[0.0,0.0],
			 [0.0,0.0]]

		# Sum Hessian
		for y in range(0, self.height):
			for x in range(0, self.width):
				H[0][0] += self.gradientX[y][x]*self.gradientX[y][x]
				H[0][1] += self.gradientX[y][x]*self.gradientY[y][x]
				H[1][0] += self.gradientX[y][x]*self.gradientY[y][x]
				H[1][1] += self.gradientY[y][x]*self.gradientY[y][x]

		return np.linalg.inv(H)


	# Calculate error image, #T(x) - I(x+p)
	def errorImage(self, image, offset = [0, 0]):
			
		E = self.template - cv2.getRectSubPix(image, (self.width, self.height), self.center(offset))
		e = np.sum(np.power(E,2))/(self.area)
		return E, e

	# Inverse translations tracker for the feature
	# Returns parameter chang
	def trackInverseTranslations(self, image, similarityThresh):
	
		dP = [0.0, 0.0]
		
		# Calculate error image
		E, e = self.errorImage(image)

		if e < similarityThresh:
			return dP, e

		# Calculate steepest descent parameter (offset) updates
		# S = sum matrix
		S = [[0.0], [0.0]]

		for y in range(0, self.height):
			for x in range(0, self.width):
				S[0][0] += self.gradientX[y][x] * E[y][x]
				S[1][0] += self.gradientY[y][x] * E[y][x]

		# Calculate new parameters (dP)
		# Multiply inverted hessian by steepest descent parameter updates (S)
		dP = np.squeeze(np.dot(self.hessianInv,S))
		self.e = e
		self.dP = dP
		self.x += dP[0]
		self.y += dP[1]

		return dP, e

class RegionOfInterest:
	
	features = []
	distance = {}
	longestDistance = 10

	# Add feature and record relative distance to other features in the ROI
	def addFeature(self, newFeature):
		
		self.features.append(newFeature)

		for f in self.features:
			if f == newFeature:
				continue
			
			c1 = f.center()
			c2 = newFeature.center()

			if (f in self.distance):
				self.distance[f][newFeature] = tuple(c1-c2)
			else:
				self.distance[f] = {newFeature : tuple(c1-c2)}


			if (newFeature in self.distance):
				self.distance[newFeature][f] = tuple(c2-c1)
			else:
				self.distance[newFeature] = {f : tuple(c2-c1)}


# Iterate tracking for features in ROI
def trackerIterator(roi, img, similarityThresh, maxIter, iterations, disThr, simThrb, convThr):

	for feature in roi.features:
		
		if feature is None:
			continue

		# error before tracking
		e = float("inf")
		# error after tracking and translating
		e2= float("inf")

		# Total max iterations
		mIter = maxIter

		# Max iterations with small movements
		i = iterations
		
		# tracking loop
		while(1):

			# Out of iterations - stagnation
			if mIter < 0 or i < 0:
				break

			mIter-=1

			# track and move a feature 
			dP, e = feature.trackInverseTranslations(img,simThrb)
			# new error after tracking and moving 
			E2, e2 = feature.errorImage(img, dP)
			# manually update the last error
			feature.e = e2

			# tracked feature movement distance 
			dPnorm = np.sqrt(dP[0]*dP[0]+dP[1]*dP[1])
			
			# stop tracking if too different
			if (e2 > disThr):
				break

			# stop tracking if feature run away
			if dPnorm > roi.longestDistance:
				break

			# stop tracking if already in a very similar place
			if (e2 < similarityThresh):
				break

			# Decrease stagnation iteration counter if feature has moved less than 0.1 pixels
			if dPnorm <= 0.1:
				i-=1
			else:
				i+=1

			# Stop tracking, if in relatively similar place and almost not moving anymore
			if e2 < convThr and dPnorm <= 10:#0.02:
				break

	return roi

# Add feature to the global ROI
def featureAdd(x,y):

	w = parameters[5]
	h = parameters[6]
	feature = Feature(x - int(w/2), y - int(h/2), w, h, img)

	if feature is not None:
		roi.addFeature(feature)

# Mouse click handler
# Add a new feature in place of the click
def click(event, x, y, flags, param):

	if event == cv2.EVENT_LBUTTONDOWN:
		featureAdd(x,y)

# Windows initialization
cv2.namedWindow('LKTracker', cv2.WINDOW_NORMAL)
cv2.setMouseCallback("LKTracker", click)
cap = cv2.VideoCapture('car.mp4')

# New and only region of interest
# All features would belong to this translational ROI
roi = RegionOfInterest()
# Get first frame, wait till Q is pressed.
ret, colorImg = cap.read()
img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
cv2.imshow('LKTracker',img)

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
height , width  =  img.shape
out = cv2.VideoWriter('./Car-Illumination' +'.avi',fourcc , 25, (width,height))

cv2.waitKey(0)

# Main loop
# Get a frame, track features
while(1):
	# Capture frame-by-frame
	ret, colorImg = cap.read()
    
	# Stop after the last frame
	if colorImg is None: 
		break
    
	img = increase_brightness(colorImg, value=30) ########### Part 3 Robustness to Illumination
    
	img = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)

    
	# Launch tracker iterator
	roi = trackerIterator(roi, img, parameters[3], parameters[0], parameters[1], parameters[4], parameters[2], parameters[7])

	# copy an image to draw features on it.
	showcaseImg = img.copy()

	# Draw feature rectangles
	for feature in roi.features:
		Rp = feature.Rp()
		rectangle = [feature.x, feature.y, Rp[0], Rp[1]]
		cv2.rectangle(showcaseImg,(int(rectangle[0]-80),int(rectangle[1]-20)),(int(rectangle[2]+80),int(rectangle[3])+90),(255,255,255))
	# Show new frame and output it into a video stream
	cv2.imshow('LKTracker',showcaseImg)
	out.write( cv2.cvtColor(showcaseImg,cv2.COLOR_GRAY2RGB))
	
	# Stop when Q is pressed
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
out.release()
cv2.destroyAllWindows()