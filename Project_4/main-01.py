# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:58:55 2019

@author: Sneha
"""

import cv2
import numpy as np


def jacobian(x_shape, y_shape):
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x, y = np.meshgrid(x, y)
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))

    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)
#     jacob = np.array[[x,0,y,0,1,0],[0,x,0,y,0,1]]
    return jacob


def affineLKtracker(img, tmp, rect, p):

    # Initialization
    rows, cols = tmp.shape
    warp_mat = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])

    # Calculate warp image
    warp_img = cv2.warpAffine(img, warp_mat, (img.shape[1],img.shape[0]), flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)
    diff = tmp.astype(int) - warp_img.astype(int)[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

    # Calculate warp gradient of image
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    #warp the gradient
    grad_x_warp = cv2.warpAffine(grad_x, warp_mat, (img.shape[1],img.shape[0]),
                                 flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    grad_y_warp = cv2.warpAffine(grad_y, warp_mat, (img.shape[1],img.shape[0]),
                                 flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

    
    # Calculate Jacobian for the 
    jacob = jacobian(cols, rows)
    
    grad = np.stack((grad_x_warp, grad_y_warp), axis=2)
    grad = np.expand_dims((grad), axis=2)
    #calculate steepest descent
    steepest_descents = np.matmul(grad, jacob)
    steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))

    # Compute Hessian matrix
    hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0,1))
 
    # Compute steepest-gradient-descent update
    diff = diff.reshape((rows, cols, 1, 1))
    update = (steepest_descents_trans * diff).sum((0,1))
    # calculate dp and update it
    d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape((-1))
    p += d_p

    return p

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [[x, y]]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append([x, y])
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, tuple(refPt[0]), tuple(refPt[1]), (0, 255, 0), 2)
        cv2.imshow("image", image)

frame_increase = 0
frame_num = str(20+frame_increase).zfill(4) + ".jpg"
cap = cv2.VideoCapture("data/car/frame" + frame_num)
im=cv2.imread('data/car/'+"frame0020.jpg")
height , width , layers =  im.shape
video = cv2.VideoWriter('Car1.mp4',-1,10,(width,height))
success, image = cap.read()

count = 0
p = np.zeros(6)
while success:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clone = image.copy()
    if count == 0:
        refPt = []
        cropping = False
        # load the image, clone it, and setup the mouse callback function
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        if len(refPt) == 2:
            crop = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            template = crop.copy()
            print("top_left(x,y) and bottom_right(x,y) is")
            print(refPt)
            cv2.waitKey(0)
        count += 1

    else:
        for k in range(50):
            p = affineLKtracker(clone, template, refPt, p)
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        newRefPt = []
        for i in range(2):
            new = refPt[i].copy()
            new.append(1)
            newRefPt.append(new)
        newRefPt = np.array(newRefPt)
        newRefPt = np.dot(warp_mat, newRefPt.T).astype(int).T
        img = cv2.rectangle(image, tuple(newRefPt[0]), tuple(newRefPt[1]), (0, 255, 0), 2)
#        cv2.imshow("", image)
#        cv2.waitKey(1)
        video.write(img)

    frame_increase += 1
    frame_num = str(20 + frame_increase).zfill(4) + ".jpg"
    print(frame_num)
    cap = cv2.VideoCapture("data/car/frame" + frame_num)
    success, image = cap.read()


video.release()
cv2.destroyAllWindows()


