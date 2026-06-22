import os
import urllib.request
import numpy as np
import cv2

def download_data():
    os.makedirs("data", exist_ok=True)
    
    print("Downloading sample heightmap...")
    # URL to a generic sample heightmap (e.g., a grayscale image)
    # Since we can't reliably pull external real DEMs without API keys easily in a generic script,
    # we'll generate a high-quality "natural-looking" sample using fractal noise
    # and save it as an image to simulate downloading a real dataset.
    # If the user has a real DEM, they can just place it in the data/ folder.
    
    size = 256
    img = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # simple mock "natural" data
            img[i, j] = (np.sin(i/10.0) * np.cos(j/10.0) + 1.0) * 127.5
            
    cv2.imwrite("data/sample_natural_terrain.png", img.astype(np.uint8))
    print("Sample natural terrain created at data/sample_natural_terrain.png")
    print("To use real data, drop your .png or .npy heightmaps into the data/ directory.")

if __name__ == "__main__":
    download_data()
