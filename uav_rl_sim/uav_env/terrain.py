import numpy as np
import noise
import cv2
import os

class TerrainGenerator:
    def __init__(self, config):
        self.grid_size = config.get("grid_size", 100)
        self.roughness = config.get("terrain_roughness", 0.5)
        self.num_obstacles = config.get("num_obstacles", 30)
        
    def generate_synthetic(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # 1. Generate Base Heightmap using Perlin Noise
        heightmap = np.zeros((self.grid_size, self.grid_size))
        scale = 100.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        
        base_z = np.random.randint(0, 100)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # mapped value from -1 to 1 mostly
                val = noise.pnoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=base_z)
                heightmap[i][j] = val
                
        # Normalize to 0-20 meters roughly
        heightmap = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap) + 1e-6)
        heightmap *= (20.0 * self.roughness)
        
        # 2. Place obstacles (cylinders representing trees/buildings)
        # 0 means no obstacle, >0 means obstacle height
        obstacles = np.zeros((self.grid_size, self.grid_size))
        
        for _ in range(self.num_obstacles):
            x = np.random.randint(5, self.grid_size - 5)
            y = np.random.randint(5, self.grid_size - 5)
            radius = np.random.randint(1, 4)
            height = np.random.uniform(5.0, 30.0) # obstacle height above terrain
            
            # draw circle of obstacle
            y_indices, x_indices = np.ogrid[-x:self.grid_size-x, -y:self.grid_size-y]
            mask = x_indices**2 + y_indices**2 <= radius**2
            obstacles[mask] = np.maximum(obstacles[mask], heightmap[mask] + height)
            
        return heightmap, obstacles

    def load_natural(self, filepath):
        """
        Loads natural data (e.g. grayscale image where intensity = height)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Terrain file {filepath} not found.")
            
        if filepath.endswith('.npy'):
            data = np.load(filepath)
            # Assuming channel 0 is height, channel 1 is obstacles if shape is (2, H, W)
            if len(data.shape) == 3 and data.shape[0] == 2:
                heightmap = cv2.resize(data[0], (self.grid_size, self.grid_size))
                obstacles = cv2.resize(data[1], (self.grid_size, self.grid_size))
            else:
                heightmap = cv2.resize(data, (self.grid_size, self.grid_size))
                obstacles = np.zeros_like(heightmap)
            return heightmap, obstacles
        elif filepath.endswith('.png') or filepath.endswith('.jpg'):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.grid_size, self.grid_size))
            heightmap = img.astype(np.float32) / 255.0 * 50.0 # arbitrary max height 50m
            obstacles = np.zeros_like(heightmap)
            return heightmap, obstacles
        else:
            raise ValueError("Unsupported format. Use .npy or .png/.jpg")
