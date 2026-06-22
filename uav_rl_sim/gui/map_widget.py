import sys
# pyrefly: ignore [missing-import]
import numpy as np
import cv2
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem
from PyQt6.QtGui import QPixmap, QImage, QColor, QPen, QBrush, QTransform, QPainter
from PyQt6.QtCore import Qt, QRectF
import os

class MapWidget(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self.grid_size = 100
        self.cell_size = 8 # pixels per grid cell
        
        # Items
        self.terrain_item = None
        self.obstacle_items = []
        self.uav_item = None
        self.goal_item = None
        self.trajectory_lines = []
        
        # Load Assets
        self.drone_pixmap = self._load_asset("assets/jet.png", size=32)
        self.tree_pixmap = self._load_asset("assets/tree.png", size=16)
        self.custom_colormap = self._create_terrain_colormap()
        
    def _create_terrain_colormap(self):
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        # Keypoints: B, G, R
        colors = [
            (0.0, (128, 0, 0)),       # Deep Water
            (0.1, (255, 128, 0)),     # Shallow Water
            (0.15, (128, 178, 194)),  # Sand
            (0.3, (34, 139, 34)),     # Grass
            (0.7, (19, 69, 139)),     # Rock/Mountain
            (1.0, (255, 255, 255))    # Snow
        ]
        
        for i in range(256):
            val = i / 255.0
            for j in range(len(colors)-1):
                if colors[j][0] <= val <= colors[j+1][0]:
                    t = (val - colors[j][0]) / (colors[j+1][0] - colors[j][0] + 1e-6)
                    c1 = np.array(colors[j][1])
                    c2 = np.array(colors[j+1][1])
                    c = c1 * (1 - t) + c2 * t
                    colormap[i, 0, :] = c.astype(np.uint8)
                    break
        return colormap
        
    def _load_asset(self, path, size=32):
        if os.path.exists(path):
            pix = QPixmap(path)
            return pix.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        return None
        
    def set_environment(self, heightmap, obstacles):
        self.scene.clear()
        self.obstacle_items.clear()
        self.trajectory_lines.clear()
        
        self.grid_size = heightmap.shape[0]
        
        # 1. Render Terrain Base
        # Convert heightmap to colormap
        norm_h = np.clip(heightmap / 30.0, 0, 1) # Normalize roughly
        colormap_img = cv2.applyColorMap((norm_h * 255).astype(np.uint8), self.custom_colormap)
        colormap_img = cv2.cvtColor(colormap_img, cv2.COLOR_BGR2RGB)
        
        # Scale up
        colormap_img = cv2.resize(colormap_img, (self.grid_size * self.cell_size, self.grid_size * self.cell_size), interpolation=cv2.INTER_LINEAR)
        
        h, w, ch = colormap_img.shape
        bytes_per_line = ch * w
        qimg = QImage(colormap_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.terrain_item = self.scene.addPixmap(QPixmap.fromImage(qimg))
        
        # 2. Render Obstacles
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if obstacles[i, j] > 0:
                    x = i * self.cell_size
                    y = j * self.cell_size
                    if self.tree_pixmap:
                        item = self.scene.addPixmap(self.tree_pixmap)
                        # Center the pixmap
                        item.setPos(x - self.tree_pixmap.width()/2, y - self.tree_pixmap.height()/2)
                    else:
                        item = self.scene.addEllipse(x-4, y-4, 8, 8, QPen(Qt.GlobalColor.darkGreen), QBrush(Qt.GlobalColor.green))
                    self.obstacle_items.append(item)
                    
        # 3. Add Goal
        self.goal_item = self.scene.addEllipse(0, 0, 16, 16, QPen(Qt.GlobalColor.red, 2), QBrush(Qt.GlobalColor.yellow))
        self.goal_item.setZValue(10)
        
        # 4. Add UAV
        if self.drone_pixmap:
            self.uav_item = self.scene.addPixmap(self.drone_pixmap)
            self.uav_item.setTransformOriginPoint(self.drone_pixmap.width()/2, self.drone_pixmap.height()/2)
        else:
            self.uav_item = self.scene.addEllipse(0, 0, 16, 16, QPen(Qt.GlobalColor.cyan, 2), QBrush(Qt.GlobalColor.blue))
            
        self.uav_item.setZValue(20)
        
    def update_state(self, uav_pos, goal_pos, uav_vel, trajectory):
        # Update Goal
        gx = goal_pos[0] * self.cell_size
        gy = goal_pos[1] * self.cell_size
        if isinstance(self.goal_item, QGraphicsEllipseItem):
            self.goal_item.setPos(gx - 8, gy - 8)
            
        # Update UAV
        ux = uav_pos[0] * self.cell_size
        uy = uav_pos[1] * self.cell_size
        
        if isinstance(self.uav_item, QGraphicsPixmapItem):
            self.uav_item.setPos(ux - self.drone_pixmap.width()/2, uy - self.drone_pixmap.height()/2)
            # Rotate based on velocity
            if np.linalg.norm(uav_vel[:2]) > 0.1:
                angle = np.degrees(np.arctan2(uav_vel[1], uav_vel[0]))
                self.uav_item.setRotation(angle + 90) # Adjust based on drone image orientation
        else:
            self.uav_item.setPos(ux - 8, uy - 8)
            
        # Draw Trajectory
        for line in self.trajectory_lines:
            self.scene.removeItem(line)
        self.trajectory_lines.clear()
        
        if len(trajectory) > 1:
            pen = QPen(QColor(0, 255, 255, 150), 2)
            for i in range(max(0, len(trajectory)-50), len(trajectory)-1):
                p1 = trajectory[i]
                p2 = trajectory[i+1]
                line = self.scene.addLine(
                    p1[0] * self.cell_size, p1[1] * self.cell_size,
                    p2[0] * self.cell_size, p2[1] * self.cell_size,
                    pen
                )
                self.trajectory_lines.append(line)
                
        # Center view
        self.centerOn(ux, uy)
