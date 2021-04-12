import cv2
import numpy as np

def preprocess(img):
#    img = cv2.fastNlMeansDenoisingColored(img)
    R = img[:,:,2]
    R = cv2.medianBlur(R,5)
    G = img[:,:,1]
    G = cv2.medianBlur(G,5)
    B = img[:,:,0]
    B = cv2.medianBlur(B,5)
    im = np.zeros((600,600))
    R = cv2.normalize(R, im , 0, 255, cv2.NORM_MINMAX).astype(float)
    G = cv2.normalize(G,  im, 0, 255, cv2.NORM_MINMAX).astype(float)
    B = cv2.normalize(B,  im, 0, 255, cv2.NORM_MINMAX).astype(float)
    x1 = R-B
    y1 = R-G
    x2 = B-R
    y2 = B-G
    z1 = []
    z2 = []
    for j in range(x1.shape[0]):
        for k in range(x1.shape[1]):
            if x1[j,k] < y1[j,k]:
                z1.append(x1[j,k])
            else:
                z1.append(y1[j,k])
    z1 = np.reshape(z1,(600,600))
    red_norm = []
    for j in range(z1.shape[0]):
        for k in range(z1.shape[1]):
            if z1[j,k]>0:
                red_norm.append(z1[j,k])
            else:
                red_norm.append(0)
    red_norm = np.reshape(red_norm,(600,600)).astype(np.uint8)
    
    for j in range(x2.shape[0]):
        for k in range(x2.shape[1]):
            if x2[j,k] < y2[j,k]:
                z2.append(x2[j,k])
            else:
                z2.append(y2[j,k])
    z2 = np.reshape(z2,(600,600))
    blue_norm = []
    for j in range(z2.shape[0]):
        for k in range(z2.shape[1]):
            if z2[j,k]>0:
                blue_norm.append(z2[j,k])
            else:
                blue_norm.append(0)
    blue_norm = np.reshape(blue_norm,(600,600)).astype(np.uint8)
    
    return red_norm,blue_norm