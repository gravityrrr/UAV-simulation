import cv2
import numpy as np
import os

def make_white_transparent(image_path):
    if not os.path.exists(image_path):
        return
        
    # Read image with alpha channel if exists, or just color
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert to BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
    # Find white pixels (where R, G, B are all > 240)
    white_mask = (img[:, :, 0] > 240) & (img[:, :, 1] > 240) & (img[:, :, 2] > 240)
    
    # Set alpha channel to 0 for white pixels
    img[white_mask, 3] = 0
    
    cv2.imwrite(image_path, img)
    print(f"Made white transparent for {image_path}")

make_white_transparent("c:/Users/ravir/drdo/uav_rl_sim/assets/drone.png")
make_white_transparent("c:/Users/ravir/drdo/uav_rl_sim/assets/tree.png")
