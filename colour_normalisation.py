import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ColourNorm:
    def normalize_images(self, images):
        
        # Convert images to float32 NumPy arrays for computation
        images = [np.array(image).astype(np.float32) for image in images]

        # Stack images to compute global mean and standard deviation across all images
        stacked_images = np.stack(images, axis=0)
        
        # Calculate the global mean and standard deviation for each color channel
        global_mean = np.mean(stacked_images, axis=(0, 1, 2))
        global_std = np.std(stacked_images, axis=(0, 1, 2))
        
        # Normalize each image based on the global mean and standard deviation
        normalized_images = [self.normalize_image(image, global_mean, global_std) for image in images]
        
        # Clip values to stay within valid image range [0, 255] and convert back to uint8
        normalized_images = [np.clip(img, 0, 255).astype(np.uint8) for img in normalized_images]

        return normalized_images

    def normalize_image(self, image, global_mean, global_std):
            # Normalize the image using global mean and standard deviation
            normalized_image = (image - global_mean) / (global_std + 1e-8)
            # Rescale to the range 0-255 for display
            normalized_image = ((normalized_image - normalized_image.min()) * (255 / (normalized_image.max() - normalized_image.min())))
            return normalized_image

