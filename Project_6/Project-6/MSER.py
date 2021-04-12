import cv2

def mser(img,str):
    if str == 'blue':
        mser = cv2.MSER_create(_min_area=100,_max_area=1000)
    if str=='red':
        mser = cv2.MSER_create(_min_area=400,_max_area=800)
    regions, _ = mser.detectRegions(img)

    return regions