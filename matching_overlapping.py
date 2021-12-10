"""
Importing the libraries to be used
"""
import cv2
import numpy as np
import os


"""
A list has been created to automatically use the images to be used in the program process.
"""
# !!!!!!!!!********** PLEASE CHANGE THE PATH OF IMAGES *********!!!!!!!!!!!!!
path = "C://Users//MONSTER//Desktop//Matching_Overlapping_Kadir_DURGUN//mf_images"
dir_list = os.listdir(path)

"""
A wide field image was defined and a font was determined to be used in the overlapping process.
"""
wide = cv2.imread('wf.JPG')
wide_gray = cv2.cvtColor(wide,cv2.COLOR_BGR2GRAY)
wide_cropped=wide_gray[1500:3000,0:6000]
wide_final=wide_gray

font = cv2.FONT_HERSHEY_SIMPLEX
org = (800, 600)
fontScale = 3
color = (255, 0, 0)
thickness = 2


# A loop has been created to operate in the mf_images folder.
for file_name in dir_list:

    # The reason I use try except is that it couldn't complete the process
    #  because it gives errors in several of the matching operations.
    try:
        # !!!!!!!!!************ PLEASE CHANGE THE PATH OF IMAGES **************!!!!!!!!!!
        med = cv2.imread("C://Users//MONSTER//Desktop//Matching_Overlapping_Kadir_DURGUN//mf_images//"+ file_name)
        med_gray = cv2.cvtColor(med,cv2.COLOR_BGR2GRAY)
        resize_med = cv2.resize(med_gray,None,fx = 1/12,fy = 1/12,interpolation = cv2.INTER_AREA)
        med_final = resize_med


        # find the keypoints and descriptors with SIFT  
        # Here images has been used as gray scale for better detection
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(med_final,None)
        kp2, des2 = sift.detectAndCompute(wide_final,None)

        # FLANN parameters
        # I did optimization for parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 11)
        search_params = dict(checks=250)   
        # Matching
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        ## Matches has been showned ##

        # Creating a mask for showing better matches
        matchesMask = [[0,0] for i in range(len(matches))]
        
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.75*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)

        img3 = cv2.drawMatchesKnn(med_final,kp1,wide_final,kp2,matches,None, **draw_params)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        # I extracted the coordinates for overlapping and training the image points.
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        # I used homography to get the M transformation matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = med_final.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Here I did transform to shift train image to query image
        dst = cv2.perspectiveTransform(pts,M)
        # for now I only displayed it using polylines, depending on what you need you can use these points to do something else
        im2 = cv2.polylines(wide_final,[np.int32(dst)],True,(0,0,255),10, cv2.LINE_AA)

        
        #Here, I have printed which medium field image is on the places displayed by the polyline.
        imageT = cv2.putText(im2,file_name,(int(dst[0][0][0]),int(dst[0][0][1])),font,fontScale,color,thickness,cv2.LINE_AA)
        
        #Finally, I completed the process by saving the overlapped image.
        cv2.imwrite("Overlap.JPG", imageT)
        
        

    except:
        pass