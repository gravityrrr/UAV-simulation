import sys
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import pyqtgraph.opengl as gl

class MapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D View Widget
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 120
        self.view.opts['elevation'] = 45
        self.view.opts['azimuth'] = 45
        self.view.opts['center'] = pg.Vector(50, 50, 0)
        self.layout.addWidget(self.view)
        
        self.view.setBackgroundColor((20, 20, 30)) # Dark sleek background
        
        # Add 3D Grid for depth perception
        self.grid = gl.GLGridItem()
        self.grid.setSize(150, 150)
        self.grid.setSpacing(10, 10)
        self.grid.translate(50, 50, -5)
        self.view.addItem(self.grid)
        
        self.grid_size = 100
        
        # 3D Items
        self.surface = None
        self.obstacle_meshes = []
        self.uav_mesh = None
        self.goal_mesh = None
        self.trajectory_line = None
        
        self.trajectory_points = []
        
    def set_environment(self, heightmap, obstacles):
        self.view.clear()
        
        # 1. Add Terrain Surface
        z = heightmap
        
        # Calculate colors based on height
        min_z, max_z = np.min(z), np.max(z)
        if max_z == min_z:
            max_z = min_z + 1.0
            
        colors = np.zeros((self.grid_size, self.grid_size, 4))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                norm_z = (z[i, j] - min_z) / (max_z - min_z)
                # Simple blue -> sand -> green -> rock -> snow colormap
                if norm_z < 0.2:
                    colors[i, j] = [0.1, 0.3, 0.8, 1.0] # Water
                elif norm_z < 0.3:
                    colors[i, j] = [0.8, 0.8, 0.4, 1.0] # Sand
                elif norm_z < 0.6:
                    colors[i, j] = [0.2, 0.6, 0.2, 1.0] # Grass
                elif norm_z < 0.8:
                    colors[i, j] = [0.5, 0.5, 0.5, 1.0] # Rock
                else:
                    colors[i, j] = [0.9, 0.9, 0.9, 1.0] # Snow
                    
        self.surface = gl.GLSurfacePlotItem(z=z, colors=colors, computeNormals=True, smooth=True, shader='shaded')
        self.surface.scale(1, 1, 1.5)
        self.surface.translate(-0.5, -0.5, 0)
        self.view.addItem(self.surface)
        
        # 2. Add Obstacles (Sleek sci-fi building blocks)
        for mesh in self.obstacle_meshes:
            self.view.removeItem(mesh)
        self.obstacle_meshes.clear()
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if obstacles[i, j] > 0:
                    th = z[i, j]
                    oh = obstacles[i, j]
                    height = max(1.0, oh - th)
                    
                    # Create a solid building block
                    box = gl.GLBoxItem(size=pg.Vector(1.0, 1.0, height * 1.5), color=(80, 100, 120, 255))
                    box.translate(i - 0.5, j - 0.5, th * 1.5)
                    self.view.addItem(box)
                    self.obstacle_meshes.append(box)
            
        # 3. Custom Low-Poly Jet Mesh for UAV
        verts = np.array([
            [ 0,  2,  0], # 0 Nose
            [-1.5, -1,  0], # 1 Left Wing
            [ 1.5, -1,  0], # 2 Right Wing
            [ 0, -1,  1], # 3 Tail fin
            [ 0, -0.5, -0.5]  # 4 Bottom
        ])
        faces = np.array([
            [0, 1, 3], [0, 3, 2], [0, 2, 4], [0, 4, 1],
            [1, 2, 3], [1, 4, 2]
        ])
        jet_md = gl.MeshData(vertexes=verts, faces=faces)
        self.uav_mesh = gl.GLMeshItem(meshdata=jet_md, smooth=False, color=(0.2, 0.8, 1.0, 1.0), drawEdges=True, edgeColor=(1, 1, 1, 1))
        self.view.addItem(self.uav_mesh)
        
        # Goal as a bright Diamond
        diamond_md = gl.MeshData.sphere(rows=4, cols=4) # low poly sphere looks like diamond
        self.goal_mesh = gl.GLMeshItem(meshdata=diamond_md, smooth=False, color=(1.0, 0.2, 0.2, 0.8), drawEdges=True, edgeColor=(1, 0, 0, 1))
        self.view.addItem(self.goal_mesh)
        
        # 4. Trajectory
        self.trajectory_points = []
        self.trajectory_line = gl.GLLinePlotItem(color=(0, 1, 1, 0.8), width=2, antialias=True)
        self.view.addItem(self.trajectory_line)

    def update_state(self, uav_pos, goal_pos, uav_vel, trajectory):
        # Update UAV position and orientation
        self.uav_mesh.resetTransform()
        
        # Calculate Heading from velocity (fallback to 0 if stationary)
        angle = 0
        if np.linalg.norm(uav_vel[:2]) > 0.1:
            angle = np.degrees(np.arctan2(uav_vel[1], uav_vel[0]))
            
        self.uav_mesh.scale(1.5, 1.5, 1.5)
        self.uav_mesh.rotate(angle - 90, 0, 0, 1) # point jet in direction of travel
        self.uav_mesh.translate(uav_pos[0], uav_pos[1], uav_pos[2] * 1.5)
        
        # Update Goal position
        self.goal_mesh.resetTransform()
        self.goal_mesh.scale(2.5, 2.5, 3.0)
        self.goal_mesh.translate(goal_pos[0], goal_pos[1], goal_pos[2] * 1.5)
        
        # Update Trajectory
        if len(trajectory) > 1:
            pts = np.array(trajectory)
            pts[:, 2] *= 1.5 # scale Z
            self.trajectory_line.setData(pos=pts)
