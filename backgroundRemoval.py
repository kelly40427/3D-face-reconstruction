import cv2
import numpy as np
import os
import glob

class background:
    def get_coordinates(self, image):
        global rect, leftButtonDown, leftButtonUp

        rect = [0, 0, 0, 0]
        leftButtonDown = False
        leftButtonUp = True

        # Mouse event callback function
        def on_mouse(event, x, y, flags, param):
            global rect, leftButtonDown, leftButtonUp
            if event == cv2.EVENT_LBUTTONDOWN:
                rect[0] = x
                rect[1] = y
                rect[2] = x
                rect[3] = y
                leftButtonDown = True
                leftButtonUp = False

            if event == cv2.EVENT_MOUSEMOVE:
                if leftButtonDown and not leftButtonUp:
                    rect[2] = x
                    rect[3] = y

            if event == cv2.EVENT_LBUTTONUP:
                if leftButtonDown and not leftButtonUp:
                    rect[0], rect[2] = min(rect[0], rect[2]), max(rect[0], rect[2])
                    rect[1], rect[3] = min(rect[1], rect[3]), max(rect[1], rect[3])
                    leftButtonDown = False
                    leftButtonUp = True
                    cv2.destroyWindow('Select ROI')

        # Create a window and set mouse callback
        cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select ROI', on_mouse)

        while True:
            img_copy = image.copy()
            if leftButtonDown and not leftButtonUp:
                cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow('Select ROI', img_copy)

            # Exit when left button is released
            if leftButtonUp and not leftButtonDown and (rect[2] - rect[0] > 0 and rect[3] - rect[1] > 0):
                break

            if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key press
                rect = [0, 0, 0, 0]  # Reset rect if the user cancels
                break

        cv2.destroyWindow('Select ROI')

        return (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])  # Returns (x, y, width, height)

    def batch_remove_background(self, subject_path, rect, output_folder):
        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Get all image files from 'Left', 'Middle', and 'Right' folders
        file_list = []
        for sub_folder in ['subject*Left', 'subject*Middle', 'subject*Right']:
            file_list.extend(glob.glob(os.path.join(subject_path, sub_folder, '*.jpg')))

        # Loop through all images in the folder
        for file_path_in in file_list:
            img_rb = cv2.imread(file_path_in)  # Read the image
            if img_rb is None:
                continue  # Skip files that can't be read as images

            filename = os.path.basename(file_path_in)  # Extract the filename

            # Apply GrabCut background removal using the provided rect
            try:
                if rect[2] > 0 and rect[3] > 0:  # Ensure valid rect dimensions
                    result_image = self.remove_background(img_rb, rect)
                else:
                    print(f"Skipping {filename} due to invalid ROI selection.")
                    continue
            except cv2.error as e:
                print(f"Error processing {filename}: {e}")
                continue

            # Save the resulting image to the output folder
            file_path_out = os.path.join(output_folder, filename)
            cv2.imwrite(file_path_out, result_image)

            print(f"Processed {filename} and saved to {file_path_out}")

    def remove_background(self, image, rect):
        if rect[2] <= 0 or rect[3] <= 0:
            raise ValueError("Invalid rectangle dimensions for GrabCut.")

        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Apply GrabCut algorithm
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Create mask where the background is set to 0, and the foreground to 1
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply mask to the image
        result_image = image * mask2[:, :, np.newaxis]

        return result_image
