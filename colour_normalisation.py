import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def normalize_images(images):
    
    def normalize_image(image, global_mean, global_std):
        # Normalize the image using global mean and standard deviation
        normalized_image = (image - global_mean) / (global_std + 1e-8)
        # Rescale to the range 0-255 for display
        normalized_image = ((normalized_image - normalized_image.min()) * (255 / (normalized_image.max() - normalized_image.min())))
        return normalized_image
    
    # Convert images to float32 NumPy arrays for computation
    images = [np.array(image).astype(np.float32) for image in images]

    # Stack images to compute global mean and standard deviation across all images
    stacked_images = np.stack(images, axis=0)
    
    # Calculate the global mean and standard deviation for each color channel
    global_mean = np.mean(stacked_images, axis=(0, 1, 2))
    global_std = np.std(stacked_images, axis=(0, 1, 2))
    
    # Normalize each image based on the global mean and standard deviation
    normalized_images = [normalize_image(image, global_mean, global_std) for image in images]
    
    # Clip values to stay within valid image range [0, 255] and convert back to uint8
    normalized_images = [np.clip(img, 0, 255).astype(np.uint8) for img in normalized_images]

    return normalized_images

# Load the images
image_path1 = r"C:\Users\lobke\OneDrive - University of Twente\Documents\BME5\image processing and computer vision\project\ipcv_project3\subject1\subject1Left\subject1_Left_1.jpg"
image1 = Image.open(image_path1)
image_path2 = r"C:\Users\lobke\OneDrive - University of Twente\Documents\BME5\image processing and computer vision\project\ipcv_project3\subject1\subject1Middle\subject1_Middle_1.jpg"
image2 = Image.open(image_path2)
image_path3 = r"C:\Users\lobke\OneDrive - University of Twente\Documents\BME5\image processing and computer vision\project\ipcv_project3\subject1\subject1Right\subject1_Right_1.jpg"
image3 = Image.open(image_path3)

images = [image1, image2, image3]
normalized_images = normalize_images(images)

# Function to plot color histograms for all three images after normalization
def plot_normalized_histograms(normalized_images):
    colors = ('r', 'g', 'b')  # Define color channels
    channel_ids = (0, 1, 2)   # Channels for RGB
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # One row, three columns for each image's histograms
    
    # Iterate over each image
    for idx, (image, ax) in enumerate(zip(normalized_images, axes)):
        for channel_id, color in zip(channel_ids, colors):
            ax.hist(image[:, :, channel_id].ravel(), bins=256, color=color, alpha=0.5, label=f'{color.upper()} channel')
        ax.set_title(f'Normalized Image {idx + 1} - Color Distribution')
        ax.set_xlim([0, 256])
        ax.legend()

    plt.tight_layout()
    plt.show()

# Plot histograms for all three normalized images
plot_normalized_histograms(normalized_images)

# Plot original and normalized images side by side
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot original images
for i, image in enumerate(images):
    axes[0, i].imshow(image)
    axes[0, i].set_title(f"Original Image {i+1}")
    axes[0, i].axis('off')

# Plot normalized images
for i, image in enumerate(normalized_images):
    axes[1, i].imshow(image)
    axes[1, i].set_title(f"Normalized Image {i+1}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()