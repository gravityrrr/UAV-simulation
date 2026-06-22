import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem
from PyQt6.QtGui import QPixmap, QImage, QColor, QPen, QBrush, QTransform, QPainter
from PyQt6.QtCore import Qt, QRectF
import os

class MapWidget2D(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self.grid_size = 100
        self.cell_size = 8
        
        self.terrain_item = None
        self.obstacle_items = []
        self.uav_item = None
        self.goal_item = None
        self.trajectory_lines = []
        
        self.drone_pixmap = self._load_asset("assets/jet.png", size=32)
        self.tree_pixmap = self._load_asset("assets/tree.png", size=16)
        self.custom_colormap = self._create_terrain_colormap()
        
    def _create_terrain_colormap(self):
        colors = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            if i < 50:
                colors[i] = [200, 100, 20] # Deep Water (BGR)
            elif i < 76:
                colors[i] = [150, 200, 50] # Sand
            elif i < 153:
                colors[i] = [50, 150, 50] # Grass
            elif i < 204:
                colors[i] = [100, 100, 100] # Rock
            else:
                colors[i] = [250, 250, 250] # Snow
        return colors
        
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
        
        norm_h = np.clip((heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap) + 1e-5) * 255, 0, 255).astype(np.uint8)
        colormap_img = cv2.applyColorMap(norm_h, self.custom_colormap)
        colormap_img = cv2.cvtColor(colormap_img, cv2.COLOR_BGR2RGB)
        
        colormap_img = cv2.resize(colormap_img, (self.grid_size * self.cell_size, self.grid_size * self.cell_size), interpolation=cv2.INTER_LINEAR)
        
        h, w, ch = colormap_img.shape
        bytes_per_line = ch * w
        q_img = QImage(colormap_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        self.terrain_item = self.scene.addPixmap(QPixmap.fromImage(q_img))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if obstacles[i, j] > 0:
                    px = i * self.cell_size
                    py = j * self.cell_size
                    if self.tree_pixmap:
                        item = self.scene.addPixmap(self.tree_pixmap)
                        item.setPos(px - self.tree_pixmap.width()/2, py - self.tree_pixmap.height()/2)
                    else:
                        item = self.scene.addRect(px, py, self.cell_size, self.cell_size, QPen(Qt.GlobalColor.darkGreen), QBrush(Qt.GlobalColor.darkGreen))
                    self.obstacle_items.append(item)
                    item.setZValue(10)
                    
        self.goal_item = self.scene.addEllipse(0, 0, 16, 16, QPen(Qt.GlobalColor.cyan, 2), QBrush(Qt.GlobalColor.red))
        self.goal_item.setZValue(15)
        
        if self.drone_pixmap:
            self.uav_item = self.scene.addPixmap(self.drone_pixmap)
        else:
            self.uav_item = self.scene.addEllipse(0, 0, 16, 16, QPen(Qt.GlobalColor.cyan, 2), QBrush(Qt.GlobalColor.blue))
            
        self.uav_item.setZValue(20)
        
    def update_state(self, uav_pos, goal_pos, uav_vel, trajectory):
        gx = goal_pos[0] * self.cell_size
        gy = goal_pos[1] * self.cell_size
        if isinstance(self.goal_item, QGraphicsEllipseItem):
            self.goal_item.setPos(gx - 8, gy - 8)
            
        ux = uav_pos[0] * self.cell_size
        uy = uav_pos[1] * self.cell_size
        
        if isinstance(self.uav_item, QGraphicsPixmapItem):
            self.uav_item.setPos(ux - self.drone_pixmap.width()/2, uy - self.drone_pixmap.height()/2)
            if np.linalg.norm(uav_vel[:2]) > 0.1:
                angle = np.degrees(np.arctan2(uav_vel[1], uav_vel[0]))
                self.uav_item.setRotation(angle + 90)
        else:
            self.uav_item.setPos(ux - 8, uy - 8)
            
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
                
        self.centerOn(ux, uy)
