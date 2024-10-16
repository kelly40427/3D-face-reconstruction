import cv2
import numpy as np

class stereoMatching:
    def stereoMatching(self, img1, img2, h, w):
        DMap = np.zeros(img1.shape[:2])
        imgH, imgW = img1.shape[:2]
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        for scanH in range(0, imgH, h):
            for scanW in range(0, imgW, w):
                template = img1[scanH:scanH+h, scanW:scanW+w]
                croppedImg2 = img2[scanH:scanH+h, 0:imgW]

                # match
                result = cv2.matchTemplate(croppedImg2, template, cv2.TM_CCOEFF_NORMED)

                # find the best match
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                for i in range(scanH, min(scanH+h, imgH)):
                    for j in range(scanW, min(scanW+w, imgW)):
                        DMap[i, j] = max_loc[0] - scanW
        return DMap
    
    
    def stereoMatchingBM(self, img1, img2):
        # Convert images to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Parameters for StereoSGBM
        numDisparities = 16*3  # Must be a multiple of 16
        blockSize = 15  # Block size for matching

        # Initialize StereoSGBM object with corrected P1 and P2
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=numDisparities,
            blockSize=blockSize,
            P1=8 * 3 * 9 ** 2,  # Smaller penalty on disparity changes
            P2=32 * 3 * 9 ** 2,  # Larger penalty on disparity changes
            disp12MaxDiff=1,
            uniquenessRatio=1,
            speckleWindowSize=50,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        disparity = stereo.compute(img1_gray, img2_gray).astype(np.float32) / 16.0

        # Apply bilateral filter to smooth disparity map
        # disparity = cv2.bilateralFilter(disparity, 9, 75, 75)
        
        # Normalize for better visualization
        # disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # disparity = np.uint8(disparity)
        
        return disparity

    
