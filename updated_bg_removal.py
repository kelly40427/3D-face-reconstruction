import cv2
import numpy as np
import os
import glob
from tqdm import tqdm


class ImprovedBackground:
    def __init__(self):
        # Initialize the face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("WARNING: Failed to load face detector")

    def batch_remove_background(self, subject_path, output_folder):
        # make sure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # acquire all image files
        file_list = []
        for sub_folder in ['subject*Left', 'subject*Middle', 'subject*Right']:
            file_list.extend(glob.glob(os.path.join(subject_path, sub_folder, '*.jpg')))

        # create log file
        log_file = os.path.join(output_folder, 'processing_log.txt')

        # show progress using tqdm
        with tqdm(total=len(file_list), desc="处理图像") as pbar:
            for file_path_in in file_list:
                try:
                    img = cv2.imread(file_path_in)
                    if img is None:
                        self._log_error(log_file, f"Unable to read the image: {file_path_in}")
                        pbar.update(1)
                        continue

                    filename = os.path.basename(file_path_in)
                    # PNG
                    output_filename = os.path.splitext(filename)[0] + '.png'

                    # process the image
                    # bald people and left images need special parameters, for better bg removal result
                    is_bald = False
                    is_left = False

                    if 'Left' in filename:
                        is_left=True
                    if 'subject4' in filename:
                        is_bald=True

                    result_image = self.remove_background(img, is_bald=is_bald, is_left=is_left)


                    # output_path = os.path.join(output_folder, filename)
                    # cv2.imwrite(output_path, result_image)

                    # PNG
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, result_image)

                    self._log_success(log_file, f"Success to process: {filename}")

                except Exception as e:
                    self._log_error(log_file, f"Failed to process {filename}: {str(e)}")

                pbar.update(1)

    def remove_background(self, image, min_contour_area=2000, is_left=False, is_bald=False):

        original = image.copy()

        # 1. preprocessing - denoise
        denoised = cv2.bilateralFilter(image, 9, 75, 75)

        # 2. Face detection
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return image

        # 3. Create a mask for GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        mask.fill(cv2.GC_BGD)

        for (x, y, w, h) in faces:

            # expand the face region
            if is_left:
                x = max(0, x - int(w * 0.05))
                if not is_bald:
                    w = int(w * 1.2)
                y = max(0, y - int(h * 0.25))
                h = int(h * 1.5)

                # Center region as sure foreground
                center_x = x + w // 6

            else:
                y = max(0, y - int(h * 0.3))
                h = int(h * 1.6)
                if is_bald:
                    w = int(w)
                    x = max(0, x - int(w * 0.05))
                else:
                    w = int(w * 1.4)
                    x = max(0, x - int(w * 0.2))

                center_x = x + w // 4
            # Center region as sure foreground
            center_y = y + h // 4
            center_h = h // 2
            center_w = w // 2

            # make the entire face region as possible foreground
            mask[y:y + h, x:x + w] = cv2.GC_PR_FGD
            mask[center_y:center_y + center_h, center_x:center_x + center_w] = cv2.GC_FGD

        # 4. GrabCut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(denoised, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        # 5. acquire foreground mask
        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

        # 6. morphology - close then open to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

        # 7. keep the largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask2, connectivity=8)
        max_area = 0
        max_label = 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[i, cv2.CC_STAT_AREA]
                max_label = i

        mask2[labels != max_label] = 0

        # 9. apply the mask
        # mask_3ch = cv2.merge([mask2, mask2, mask2])
        # result = cv2.bitwise_and(original, mask_3ch)

        # V2 png
        # 8. smooth the mask - using bigger kernel
        mask2 = cv2.GaussianBlur(mask2, (11, 11), 0)

        # 9. export the image with alpha channel
        # converting to BGRA
        result = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
        # using mask2 as alpha channel
        result[:, :, 3] = mask2

        return result

    def _log_success(self, log_file, message):
        with open(log_file, 'a') as f:
            f.write(f"[SUCCESS] {message}\n")

    def _log_error(self, log_file, message):
        with open(log_file, 'a') as f:
            f.write(f"[ERROR] {message}\n")

if __name__ == '__main__':
    # batch process all images
    bg_remover = ImprovedBackground()
    subject_path = "ipcv_project3/subject4"
    output_folder = "subject1output9"
    bg_remover.batch_remove_background(subject_path, output_folder)
