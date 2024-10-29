import cv2
import numpy as np
from cv2 import ximgproc

class stereoMatching:

    def __init__(self, disparity_threshold=1.0):
        self.disparity_threshold = disparity_threshold

    def stereoMatching(self, img1, img2, h, w):
        DMap = np.zeros(img1.shape[:2])
        imgH, imgW = img1.shape[:2]
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        for scanH in range(0, imgH, h):
            for scanW in range(0, imgW, w):
                template = img2[scanH:scanH+h, scanW:scanW+w]
                croppedImg2 = img1[scanH:scanH+h, 0:imgW]

                # match
                result = cv2.matchTemplate(croppedImg2, template, cv2.TM_CCOEFF_NORMED)

                # find the best match
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                for i in range(scanH, min(scanH+h, imgH)):
                    for j in range(scanW, min(scanW+w, imgW)):
                        DMap[i, j] = max_loc[0] - scanW

        # Define a kernel for morphological operations
        kernel = np.ones((20, 20), np.uint8)

        # Apply morphological closing
        DMap = cv2.morphologyEx(DMap, cv2.MORPH_CLOSE, kernel)

        DMap[DMap>200] = 200
        return DMap
    
    
    def stereoMatchingBM(self, img1, img2, minDisparity, numDisparities, blockSize, uniquenessRatio):
        # Convert images to grayscale
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Parameters for StereoSGBM
        #minDisparity = 276
        #numDisparities = 16*4  # Must be a multiple of 16
        blockSize = 5  # Block size for matching

        # Initialize StereoSGBM object with corrected P1 and P2
        stereo = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=blockSize,
            P1=8 * 3 * blockSize ** 2,  # Smaller penalty on disparity changes
            P2=32 * 3 * blockSize ** 2,  # Larger penalty on disparity changes
            disp12MaxDiff=1,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=50,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        disparity = stereo.compute(img1_gray, img2_gray).astype(np.float32) / 16.0
        disparity[disparity < 0] = 0

        # Define a kernel for morphological operations
        kernel = np.ones((25, 25), np.uint8)

        # Apply morphological closing
        disparity = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)
        
        # Apply bilateral filter to smooth disparity map
        # disparity = cv2.bilateralFilter(disparity, 9, 5, 5)

        # Adaptive Weighting with Guided Filtering
        disparity = self.guided_filtering(disparity, img1_gray)
        
        # Normalize for better visualization
        # disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # disparity = np.uint8(disparity)
        print(f"Disparity Map Min: {np.min(disparity)}, Max: {np.max(disparity)}")
        
        return minDisparity+numDisparities, disparity
    
    def unreliable_disparity_mask(self, disparity_map):

        unreliable_mask = np.zeros_like(disparity_map, dtype=np.uint8)
        unreliable_mask[disparity_map < self.disparity_threshold] = 255  


        return unreliable_mask


    def filter_disparity(self, disparity_map, unreliable_mask):

        filtered_disparity = np.copy(disparity_map)
        filtered_disparity[unreliable_mask == 255] = 0  

        return filtered_disparity
        
    def guided_filtering(self, disparity, guide_image, radius=15, eps=1e-6):
        """Applies guided filtering to smooth the disparity map based on the guide image."""
        # Use OpenCV's guided filter if available
        if hasattr(ximgproc, 'guidedFilter'):
            guided_disparity = ximgproc.guidedFilter(guide_image, disparity, radius, eps)
        else:
            # If ximgproc is unavailable, fall back to bilateral filter as a rough approximation
            guided_disparity = cv2.bilateralFilter(disparity, d=9, sigmaColor=5, sigmaSpace=5)

        return guided_disparity
    
    
