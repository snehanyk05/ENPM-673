import cv2

method = cv2.TM_CCOEFF_NORMED
# Read the images from the file
small_image = cv2.imread('ref_marker.png')
large_image = cv2.imread('frame300.jpg')

result = cv2.matchTemplate(small_image, large_image, method)

# We want the minimum squared difference
mn,_,mnLoc,_ = cv2.minMaxLoc(result)

# Draw the rectangle:
# Extract the coordinates of our best match
MPx,MPy = mnLoc

# Step 2: Get the size of the template. This is the same size as the match.
trows,tcols = small_image.shape[:2]

# Step 3: Draw the rectangle on large_image
test = cv2.rectangle(large_image, (MPx,MPy),(MPx+tcols,MPy+trows),(0,0,255),2)

blur = cv2.GaussianBlur(test,(5,5),0)
imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
# Display the original image with the rectangle around the match.
#corners = cv2.goodFeaturesToTrack(imgray,4,0.1,60)
#corners = np.int0(corners)
#for i in corners:
#    x,y = i.ravel()
#    cv2.circle(img,(x,y),3,225,-1)
#cv2.imshow('corner',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imshow('output',large_image)

# The image is only displayed if we call this
cv2.waitKey(0)