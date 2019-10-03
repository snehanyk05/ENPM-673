import cv2
import numpy as np
import matplotlib.pyplot as plt

def em(X,N,d,r):
    
    m =  X.shape[0]
    values = np.random.permutation(m);
    mu = X[values[0:N]]+r;
    Mu=mu
#    print('Mu',Mu.shape)
    sigma = np.zeros((3,3,N));

    for j in range(0,N):
        w=np.cov(X).shape
       
        sigma[:,:,j]=(np.cov(np.transpose(X))+np.eye(d,d)*r)
#        print('In loop',(np.cov(X)+np.eye(d,d)*r).shape)
#    print(sigma.shape)
    phi = np.ones(N)
  
    pdf = np.zeros((X.shape[0],N));
    
    for k in range(0,1000):
    
        for j in range(0,N):
#           print('Mu',Mu[j])
           pdf[:,j] = gaussian_ND(X, Mu[j,:], sigma[:,:,j]);
         
        temp_pdf = pdf * phi;
        
        sum1=np.sum(temp_pdf, 1)
        sum1=sum1.reshape(-1,1)
#        print(sum1.shape) 
        weights = np.divide(temp_pdf,sum1)
#        print(weights.shape)
        mu_previous = Mu; 
#        X=X.reshape(-1,1)
        for j in range(0,N):
           
            phi[j] = np.mean(weights[:, j], 0);
            y=np.transpose(weights[:, j,np.newaxis])
           
            Mu[j,:] = np.dot(y,X)
            Mu[j,:] = Mu[j,:] / np.sum(weights[:, j], 0)
            
            
            temp = np.zeros((d, d))+np.eye(3,3)*r;
            Xm = X - Mu[j, :]
            for i in range (0,m):
                temp = temp + (weights[i, j] * (np.transpose(Xm[i, :]) * Xm[i, :]))
            
            sigma[:,:,j] = temp / np.sum(weights[:, j])
            
        if (np.array_equal(Mu,mu_previous)):
            break;
    
#    temp_pdf = pdf .* phi;
#    weights = temp_pdf./sum(temp_pdf, 2);
#    mu_previous = mu;  
    return (Mu,sigma)
def gaussian_ND(X, mu, Sigma):

    n = X.shape[1]
 
    
    diff = X - mu
   
#
#    print('Fir',(np.transpose(diff)).shape)
#    print('SEc',(np.linalg.inv(Sigma).shape))
    t=np.multiply(np.dot((diff),np.linalg.inv(Sigma)),diff)
    u=(-1/2 * np.sum((t),axis=1))
    s=np.linalg.det(Sigma)
    x=(np.sqrt(((2 * np.pi)**n)*np.linalg.det(Sigma)))
   
    N1 = (np.sqrt(((2 * np.pi)**n)*np.linalg.det(Sigma))) * np.exp( -1/2 * np.sum((t),axis=1))
    
#    N1=N1.reshape(-1,1)
   
    return u
def gaussian_1D(x,mu,Sigma):
    N1 = 1/(Sigma*np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * Sigma**2))
    
    return N1
Training_data  = 'TrainingFolder/Frames/'
Cropped_buoys = 'TrainingFolder/CroppedFrames/'
outputfolder = 'Output/Part2/'
###folder = @(i) fullfile(sprintf('../../Output/Part2/%s_hist.jpg',i)); #######

#######################################


red_buoy_rc = np.zeros((256,1))
red_buoy_rg = np.zeros((256,1))
red_buoy_rb = np.zeros((256,1))
yellow_buoy_rc = np.zeros((256,1))
yellow_buoy_rg = np.zeros((256,1))
yellow_buoy_rb = np.zeros((256,1))
green_buoy_rc = np.zeros((256,1))
green_buoy_rg = np.zeros((256,1))
green_buoy_rb = np.zeros((256,1))
red_samples = np.empty(0)
green_samples = np.empty(0)
yellow_samples = np.empty(0)
thresh = 30
numofsamples = 30


########## Training Data Buoys
for k in range(1,12):
    
    I = cv2.imread(Training_data +"%d.jpg" %k) ############
    
    red = I[:,:,0]
    green = I[:,:,1]
    blue = I[:,:,2]
    
    ## Red buoy
    maskR = cv2.imread(Cropped_buoys+'R_%d.jpg' %k)
    maskR = maskR[:, :, 0]
    #imshow(maskR);
    
    imR_R = red[maskR>thresh]
    imR_G = green[maskR>thresh]
    imR_B = blue[maskR>thresh]

    foreground_mask = (maskR>thresh)
    seg = I* np.reshape(np.tile(foreground_mask, [3,1,1]),[480,640,3])
    
    #figure(2);imshow(seg);
    R = seg[:,:,0]
    G = seg[:,:,1]
    B = seg[:,:,2]
   
    [red_count, _] = np.histogram(R[R > 0],254)
    [green_count, _] = np.histogram(G[G > 0],254)
    [blue_count, _] = np.histogram(B[B > 0],254)
    for j in range(0, 254):
        red_buoy_rc[j] = red_buoy_rc[j] + red_count[j]
        red_buoy_rg[j] = red_buoy_rg[j] + green_count[j]
        red_buoy_rb[j] = red_buoy_rb[j] + blue_count[j]
   
#    r=(imR_R)
#    r=np.append(np.append(r,imR_B),imR_G)
#    red_samples = np.append(red_samples,r)
    red_samples=np.column_stack((imR_R,imR_G,imR_B))
    
    
    ##################################################
    
    ## Yellow buoy

    maskY = cv2.imread(Cropped_buoys+'Y_%d.jpg' %k)
    maskY = maskY[:, :, 0]
    
    sample_ind_Y = np.where(maskY > thresh)############
    #imshow(maskR);
    
    RY = red[maskY>thresh]
    GY = green[maskY>thresh]
    BY = blue[maskY>thresh]
    
    foreground_mask = (maskY>thresh)
    seg = I* np.reshape(np.tile(foreground_mask, [3,1,1]),[480,640,3])
    
    #figure(2);imshow(seg);
    R = seg[:,:,0]
    G = seg[:,:,1]
    B = seg[:,:,2]
   
    [red_count, _] = np.histogram(R[R > 0],254)
    [green_count, _] = np.histogram(G[G > 0],254)
    [blue_count, _] = np.histogram(B[B > 0],254)
    for j in range(0, 254):
        yellow_buoy_rc[j] = yellow_buoy_rc[j] + red_count[j]
        yellow_buoy_rg[j] = yellow_buoy_rg[j] + green_count[j]
        yellow_buoy_rb[j] = yellow_buoy_rb[j] + blue_count[j]
        
    yellow_samples=np.column_stack((RY,GY,BY))
    
     ## Green buoy
     
    maskG = cv2.imread(Cropped_buoys+'G_%d.jpg' %k)
    maskG = maskG[:, :, 0]
    
    sample_ind_G = np.where(maskG > thresh)############
    #imshow(maskR);
    
    RG = red[maskG>thresh]
    GG = green[maskG>thresh]
    BG = blue[maskG>thresh]
    
    foreground_mask = (maskG>thresh)
    seg = I* np.reshape(np.tile(foreground_mask, [3,1,1]),[480,640,3])
    
    #figure(2);imshow(seg);
    R = seg[:,:,0]
    G = seg[:,:,1]
    B = seg[:,:,2]
   
    [red_count, _] = np.histogram(R[R > 0],254)
    [green_count, _] = np.histogram(G[G > 0],254)
    [blue_count, _] = np.histogram(B[B > 0],254)
    for j in range(0, 254):
        green_buoy_rc[j] = green_buoy_rc[j] + red_count[j]
        green_buoy_rg[j] = green_buoy_rg[j] + green_count[j]
        green_buoy_rb[j] = green_buoy_rb[j] + blue_count[j]
#    g=(RG)
#    g=np.append(np.append(g,BG),GG)
#    green_samples = np.append(green_samples,g)
    green_samples=np.column_stack((RG,BG,GG))    
#    green_samples = [green_samples, [RG, GG, BG]]
    
    ## Histogram Visualization
red_buoy_rc = red_buoy_rc / thresh
red_buoy_rg = red_buoy_rg / thresh
red_buoy_rb = red_buoy_rb / thresh

yellow_buoy_rc = yellow_buoy_rc / thresh
yellow_buoy_rg = yellow_buoy_rg / thresh
yellow_buoy_rb = yellow_buoy_rb / thresh

green_buoy_rc = green_buoy_rc / thresh
green_buoy_rg = green_buoy_rg / thresh
green_buoy_rb = green_buoy_rb / thresh



#figure(1);
#x = (0:1:255)';
#title('Histogram for Red colured buoy')
#area(x, red_buoy_rc, 'FaceColor', 'r')
#xlim([0 255])
#hold on
#area(x, red_buoy_rg, 'FaceColor', 'g')
#area(x, red_buoy_rb, 'FaceColor', 'b')
#hgexport(gcf, fullfile(outputfolder, 'R_hist.jpg'), hgexport('factorystyle'), 'Format', 'jpeg');
#hold off
#pause(0.1)
#
#figure(2);
#title('Histogram for Yellow colured buoy')
#area(x, yellow_buoy_rc, 'FaceColor', 'r')
#xlim([0 255])
#hold on
#area(x, yellow_buoy_rg, 'FaceColor', 'g')
#area(x, yellow_buoy_rb, 'FaceColor', 'b')
#hold off
#pause(0.1)
#hgexport(gcf, fullfile(outputfolder, 'Y_hist.jpg'), hgexport('factorystyle'), 'Format', 'jpeg');
#
#
#figure(3);
#title('Histogram for Green colured buoy')
#area(x, green_buoy_rc, 'FaceColor', 'r')
#xlim([0 255])
#hold on
#area(x, green_buoy_rg, 'FaceColor', 'g')
#area(x, green_buoy_rb, 'FaceColor', 'b')
#hold off
#pause(0.1)
#hgexport(gcf, fullfile(outputfolder, 'G_hist.jpg'), hgexport('factorystyle'), 'Format', 'jpeg');


N=4;
data = red_samples;
[mu_r,sigma_r] = em(data,N,3,10**-7);
data = green_samples;
[mu_g,sigma_g] = em(data,N,3,10**-7);
data = yellow_samples;
[mu_y,sigma_y] = em(data,N,3,10**-7);


#save('Parameters_5N_3d.mat','mean_red','sigma_red','mean_yellow','sigma_yellow','mean_green','sigma_green');
    