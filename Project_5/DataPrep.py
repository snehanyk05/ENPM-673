import cv2
import glob
import numpy as np
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

model_dir = 'Oxford_dataset/model/'
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(model_dir)

count = 0
for img in glob.glob('Oxford_dataset/stereo/centre/*.png'):
    image = cv2.imread(img,cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    color_image = cv2.cvtColor(image,cv2.COLOR_BayerGR2BGR)
    dist = UndistortImage(color_image,LUT)
    cv2.imwrite('data/'+'%d.png'%count,dist)
    count = count+1

K = np.matrix([[fx,0,cx],[0,fy,cy],[0,0,1]])
