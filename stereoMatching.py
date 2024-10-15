import cv2
import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
import sys

def stereoMatching(img1,img2,h,w):
    DMap = np.zeros(img1.shape[:2])
    imgH,imgW = img1.shape[:2]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    for scanH in range(0,imgH,h):
        for scanW in range(0,imgW,w):
            template = img1[scanH:scanH+h,scanW:scanW+w]
            croppedImg2 = img2[scanH:scanH+h,0:imgW]

            # match
            result = cv2.matchTemplate(croppedImg2, template, cv2.TM_CCOEFF_NORMED)

            # find the best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # print(max_loc[0])
            for i in range(scanH,min(scanH+h, imgH),1):
                for j in range(scanW,min(scanW+w, imgW),1):
                    DMap[i,j] = max_loc[0]-scanW
    return DMap

