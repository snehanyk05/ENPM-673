# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:40:40 2019

@author: Sneha
"""
import numpy as np
import random
def get_fundamental_matrix(set1,set2):
#    set1 = (random.choices(set1,k=8))
#    set2 = (random.choices(set2,k=8))
    x1 = [set1[i][0] for i in range(len(set1))]
    y1 =[set1[i][1] for i in range(len(set1))]
    x2 = [set2[i][0] for i in range(len(set2))]
    y2 = [set2[i][1] for i in range(len(set2))]
    
    centre_x1 = np.mean(x1)
    centre_y1 = np.mean(y1)
    x1 = x1-centre_x1*np.ones((len(x1)))
    y1 = y1-centre_y1*np.ones((len(y1)))
    
    avg_dist1 = np.sqrt(np.sum(x1**2+y1**2))/len(x1)
    scaling_1 = np.sqrt(2)/avg_dist1
    x1 = scaling_1*x1
    y1 = scaling_1*y1
    
    scaled1 = np.matrix([[scaling_1,0,(-scaling_1*centre_x1)],[0,scaling_1,(-scaling_1*centre_y1)],[0,0,1]])
    
    centre_x2 = np.mean(x2)
    centre_y2 = np.mean(y2)
    x2 = x2-centre_x2*np.ones((len(x2)))
    y2 = y2-centre_y2*np.ones((len(y2)))
    avg_dist2 = np.sqrt(np.sum(x2**2+y2**2))/len(x2)
    scaling_2 = np.sqrt(2)/avg_dist2
    x2 = scaling_2*x2
    y2 = scaling_2*y2
    scaled2 = np.matrix([[scaling_2,0,(-scaling_2*centre_x2)],[0,scaling_2,(-scaling_2*centre_y2)],[0,0,1]])
    A = []
    for i in range(0,8):
        x = [x1[i]*x2[i],x1[i]*y2[i],x1[i],y1[i]*x2[i],y1[i]*y2[i],y1[i],x2[i],y2[i],1]
        A.append(x)
    
    A = np.matrix(A)
    u,d,vt = np.linalg.svd(A)
    v = vt.T
    F = v[:,-1]
    F = F.reshape(3,3).T
    F_norm = F/(np.linalg.norm(F))
    uf,sf,vft = np.linalg.svd(F_norm)
    vf = vft
    diag = np.matrix([[sf[0],0,0],[0,sf[1],0],[0,0,0]])
    F_final = np.dot(np.dot(uf,diag),(vf))
    F = np.dot(np.dot(scaled2.T,F_final),scaled1)
    
#    return F,set1,set2
    
    return F
    
def get_RANSAC(ss,rr,matchedpoints1, matchedpoints2,N):
    F_final = np.zeros((3,3));
    sz = (len(matchedpoints1))
#    print(sz)
    index = 0;
    set1,set2,F,err,inlier_index=[],[],[],[],[]
    for n in range (N):
#        %Select 8 Points at random
        # select 8 random points
        s=ss[n]
        r=rr[n]
        
        set1.append(s)
        set2.append(r)
#        print(type( (random.choices(matchedpoints2,k=8))))
#        set1.append([(726.0, 446.4000244140625), (859.963623046875, 459.8416748046875), (387.0, 449.0), (725.7600708007812, 442.3680419921875), (865.2000122070312, 459.6000061035156), (201.60000610351562, 222.00001525878906), (591.0, 452.0), (591.0, 452.0)])
#        set2.append([(725.7600708007812, 449.280029296875), (874.8935546875, 471.78558349609375), (559.8720703125, 330.04803466796875), (701.7064208984375, 444.9117431640625), (660.0, 343.0), (1000.3048706054688, 271.2269592285156), (672.1920776367188, 451.008056640625), (878.377197265625, 467.80426025390625)])
#        ind = randi(sz,1,8);
#        %ind = randperm(sz,8);
#        x{n} = matchedpoints1(ind);
#        y{n} = matchedpoints2(ind);
        F.append(get_fundamental_matrix(set1[n],set2[n]))
#        print('F',F[n])
        temp_err=[]
        for i in range(8):
           x2 = np.array([set2[n][i][0],set2[n][i][1],1])
#           x2=x2.T
           x1 = np.array([set1[n][i][0],set1[n][i][1],1])
           x1=x1[:,np.newaxis]
           x2=x2[:,np.newaxis]
#           print('x1',x1)
#           epip_line = F[n] * x1
           
           epip_line = np.dot(F[n] , x1)
           
           if(np.sqrt(epip_line[0]**2 + epip_line[1]**2)!=0):
               temp_err.append(np.abs((x2.T) * (F[n] * x1)) / (np.sqrt(epip_line[0]**2 + epip_line[1]**2)))
           else:
               temp_err.append(np.inf)
               print('epip_line',epip_line)
#           print(temp_err)
#        end
#        print("temp Err",temp_err)
        
        err.append(sum(temp_err)/8)
#        min_err=err[n]
#        print('Err',(min(err)))
        
        if err[n]<0.7:
#           print(err[n])
           inlier_index.append(n)
#           index = index+1
#       end
#    end
#    
#    err[n]={0:1,1:2}
#    print("err",min(err))

    min_index = err.index(min(err))
#    print('Min',min_index)
#    print('Final',F[min_index])
    F_final = F[min_index]
#    x_r = set1[min_index]
#    x_l = set2[min_index]
#    inlier_index = np.unique(inlier_index)
    print('Index',(inlier_index))
    inlier_points1 = np.array(set1[inlier_index[0]])
    inlier_points2 = np.array(set2[inlier_index[0]])
#    print((inlier_points1))
    for index in range(1,len(inlier_index)-1):
        
        inlier_points1 = np.concatenate((inlier_points1,set1[inlier_index[index]]),axis=0)
        inlier_points2 = np.concatenate((inlier_points2,set2[inlier_index[index]]),axis=0)
#        inlier_points2 = vertcat(inlier_points2,y{inlier_index(index)});
#    end
   
    return F_final,inlier_points1,inlier_points2