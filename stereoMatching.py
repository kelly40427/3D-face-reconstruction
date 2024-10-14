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

            print(max_loc[0])
            for i in range(scanH,min(scanH+h, imgH),1):
                for j in range(scanW,min(scanW+w, imgW),1):
                    DMap[i,j] = max_loc[0]-scanW
    return DMap

img1 = cv.imread('rectified1.jpg')
img2 = cv.imread('rectified2.jpg')
DMap = stereoMatching(img1,img2,16,16)

DMap_clipped = np.clip(DMap, 0, 255)
# 使用 imshow 显示矩阵
plt.imshow(DMap, cmap='viridis', interpolation='nearest')  # 选择 'viridis' 颜色映射
plt.colorbar()  # 添加颜色条

# 保存图片到文件
plt.savefig('DMap.png')

# 显示图片
plt.show()