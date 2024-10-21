import cv2
import numpy as np
import os
import glob

class background:
    def batch_remove_background(self, subject_path, output_folder):
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

            # Apply background removal
            try:
                result_image = self.remove_background(img_rb)
            except cv2.error as e:
                print(f"Error processing {filename}: {e}")
                continue

            # Save the resulting image to the output folder
            file_path_out = os.path.join(output_folder, filename)
            cv2.imwrite(file_path_out, result_image)

            print(f"Processed {filename} and saved to {file_path_out}")

    # def remove_background(self, image, min_contour_area=7000):
    #     # Step 1: Convert the image to grayscale
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # Step 2: Use a combination of Canny edge detection and thresholding to better isolate the subject
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
    #     # Apply Canny edge detection to detect edges
    #     edges = cv2.Canny(blurred, 400, 700)

    #     # Apply thresholding to further separate foreground and background
    #     _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    #     # Combine edges and threshold to create a mask
    #     combined_mask = cv2.bitwise_or(edges, thresh)

    #     # Step 3: Clean up the mask using morphological operations
    #     kernel = np.ones((5, 5), np.uint8)
    #     mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    #     # Step 4: Invert the mask (foreground becomes white, background becomes black)
    #     mask_inv = cv2.bitwise_not(mask_cleaned)

    #     # Step 5: Contour detection to find objects in the mask
    #     contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # Step 6: Filter out small objects by contour area
    #     filtered_mask = np.zeros_like(mask_inv)  # Empty mask to store the filtered objects
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area > min_contour_area:  # Only keep contours larger than the min_contour_area
    #             cv2.drawContours(filtered_mask, [contour], -1, (255), thickness=cv2.FILLED)

    #     # Step 7: Use the filtered mask to isolate the foreground
    #     mask_inv_3ch = cv2.merge([filtered_mask, filtered_mask, filtered_mask])
    #     result_image = cv2.bitwise_and(image, mask_inv_3ch)

    #     return result_image

    def remove_background(self, image, min_contour_area=2000):
        # Step 1: Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Use a combination of Canny edge detection and thresholding to better isolate the subject
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adjust edge detection to be even less aggressive
        edges = cv2.Canny(blurred, 70, 200)

        # Apply binary thresholding with a slightly higher threshold
        _, thresh = cv2.threshold(blurred, 113, 255, cv2.THRESH_BINARY)

        # Combine edges and threshold to create a mask
        combined_mask = cv2.bitwise_or(edges, thresh)

        # Step 3: Clean up the mask using less aggressive or no morphological operations
        kernel = np.ones((2, 2), np.uint8)  # Optional: Can skip morphology or make the kernel smaller
        mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Step 4: Invert the mask (foreground becomes white, background becomes black)
        mask_inv = cv2.bitwise_not(mask_cleaned)

        # Step 5: Contour detection to find objects in the mask
        contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 6: Filter out small objects by contour area
        filtered_mask = np.zeros_like(mask_inv)  # Empty mask to store the filtered objects
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:  # Adjust the area threshold to keep more details
                cv2.drawContours(filtered_mask, [contour], -1, (255), thickness=cv2.FILLED)

        # Step 7: Optionally blur the mask for smoother edges
        filtered_mask = cv2.GaussianBlur(filtered_mask, (3, 3), 0)

        # Use the filtered mask to isolate the foreground
        mask_inv_3ch = cv2.merge([filtered_mask, filtered_mask, filtered_mask])
        result_image = cv2.bitwise_and(image, mask_inv_3ch)

        return result_image


