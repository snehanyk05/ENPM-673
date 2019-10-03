import cv2
import tkinter as tk
import numpy as np
import PIL.Image, PIL.ImageTk
refPt = []
color=''
roi_corners=[]
 
def click_and_rec(image):
	# grab references to the global variables
    global refPt
    global color
    print(refPt)
    cv2.rectangle(image, refPt[0], refPt[1],(255,0,0),thickness=2)
    refPt=[]

    
def onpick(event):
    print(event.x, event.y)
    refPt.append((event.x, event.y))
    return True

def getRect(inputFolder):
            root= tk.Tk() 
            cv_img = cv2.cvtColor(cv2.imread(inputFolder+"0019.jpg"), cv2.COLOR_BGR2RGB)
            
            
            image = cv2.imread(inputFolder+"0019.jpg", -1)

            height, width, no_channels = cv_img.shape
             
            # Create a canvas that can fit the above image
            canvas = tk.Canvas(root, width = width, height = height)
            canvas.pack()
             
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img),master=root)
             
             # Add a PhotoImage to the Canvas
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.bind("<Button 1>",onpick)
            btn_blur=tk.Button(root, text="Rectangle", width=50, command= lambda:click_and_rec(image))
            btn_blur.pack(anchor=tk.CENTER, expand=True)

            
            root.mainloop()
            cv2.imshow('Seg',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
testFolder='data/vase/'
#getRect(testFolder) # click on top-left and bottom-right points. To get the rectangle.


def jacobian(x_shape, y_shape):
    # get jacobian of the template size.
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x, y = np.meshgrid(x, y) 
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))

    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)

    return jacob

def affineLKtracker(img, tmp, rect, p_prev):
        thresh = 0.5
        d_p_norm = np.inf
        tmp = (tmp-np.mean(tmp))/np.std(tmp)
        tmp = tmp[rect[0][1]:rect[3][1], rect[0][0]:rect[1][0]]
        rows, cols = tmp.shape
        img = (img-np.mean(img))/np.std(img)
        while(d_p_norm>=thresh):
            warp_mat = np.array([[1+p_prev[0], p_prev[2], p_prev[4]], [p_prev[1], 1+p_prev[3], p_prev[5]]])
            warp_img = cv2.warpAffine(img, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC)[rect[0][1]:rect[3][1], rect[0][0]:rect[1][0]]
            diff = tmp.astype(int) - warp_img.astype(int)
        
            # Calculate warp gradient of image
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=7)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=7)
            #warp the gradient
            grad_x_warp = cv2.warpAffine(grad_x, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC)[rect[0][1]:rect[3][1], rect[0][0]:rect[1][0]]
            grad_y_warp = cv2.warpAffine(grad_y, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC)[rect[0][1]:rect[3][1], rect[0][0]:rect[1][0]]
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
            p_prev += d_p
            d_p_norm = np.linalg.norm(d_p)
            print(d_p_norm)

        return p_prev

frame_increase = 0
frame_num = str(19+frame_increase).zfill(4) + ".jpg"
cap = cv2.VideoCapture("data/vase/" + frame_num)
im=cv2.imread('data/vase/'+"0019.jpg")
height , width , layers =  im.shape
video = cv2.VideoWriter('Vase.mp4',-10,10,(width,height))
success, image = cap.read()

count = 0
while success:
    template = cv2.imread('data/vase/'+'0019.jpg')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img = cv2.imread('data/vase/'+frame_num)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # rectangle co-ordinates for vase, calculated from the event handling part.
    rect = [[126,97],[169,96],[171,146],[128,148]]
    p_prev = np.zeros(6)
    p = affineLKtracker(img, template,rect,p_prev)
    warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
    newrect = []
    for i in range(4):
        new = rect[i].copy()
        new.append(1)
        newrect.append(new)
    newrect = np.array(newrect)
    newrect = np.dot(np.linalg.pinv(warp_mat).T, (newrect.T)).astype(int).T
    newrect = np.array((newrect))
    newrect = newrect.reshape((-1,1,2))
    # using the new warped rectangle co-ordinated to get the bounding box.
    final = cv2.polylines(image,[newrect] , True,(0, 255, 0))
    video.write(final)  
    frame_increase += 1
    frame_num = str(19 + frame_increase).zfill(4) + ".jpg"
    print(frame_num)
    p_prev = p
    cap = cv2.VideoCapture("data/vase/" + frame_num)
    success, image = cap.read()


video.release()