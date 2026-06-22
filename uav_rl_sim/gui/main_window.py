import sys
import yaml
import os
import time
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QFormLayout, 
                             QGroupBox, QTabWidget, QSpinBox, QDoubleSpinBox, 
                             QFileDialog, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from .map_widget import MapWidget
from .plot_widget import TrainingPlotWidget
from uav_env.env import UAVEnv
from agents.ppo import PPOAgent
from agents.sac import SACAgent

class TrainingWorker(QThread):
    update_plot_signal = pyqtSignal(float, float, float) # reward, loss, success
    update_map_signal = pyqtSignal(object, object, object, object) # pos, goal, vel, trajectory
    setup_env_signal = pyqtSignal(object, object) # heightmap, obstacles
    finished_signal = pyqtSignal()
    
    def __init__(self, config, agent_type="PPO"):
        super().__init__()
        self.config = config
        self.agent_type = agent_type
        self.running = False
        
        self.env = UAVEnv(self.config['env'])
        self.env.goal_reward = self.config['rewards']['goal_reward']
        self.env.collision_penalty = self.config['rewards']['collision_penalty']
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        if self.agent_type == "PPO":
            self.agent = PPOAgent(state_dim, action_dim, self.config['training'])
        else:
            self.agent = SACAgent(state_dim, action_dim, self.config['training'])
            
        self.max_episodes = self.config['training']['max_episodes']
        
    def run(self):
        self.running = True
        success_count = 0
        
        for episode in range(1, self.max_episodes + 1):
            if not self.running:
                break
                
            state, _ = self.env.reset()
            self.setup_env_signal.emit(self.env.heightmap, self.env.obstacles)
            
            episode_reward = 0
            loss = 0
            steps = 0
            
            while True:
                if not self.running:
                    break
                    
                # Action Selection
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Store / Update
                if self.agent_type == "PPO":
                    self.agent.store_reward(reward, terminated)
                else:
                    self.agent.store_transition(state, action, reward, next_state, terminated)
                    if steps % 10 == 0:
                        a_loss, c_loss = self.agent.update()
                        loss += c_loss
                        
                state = next_state
                episode_reward += reward
                steps += 1
                
                # UI Update (throttle map updates to 10 Hz)
                if steps % 3 == 0:
                    self.update_map_signal.emit(self.env.uav_pos, self.env.goal_pos, self.env.uav_vel, self.env.trajectory)
                    time.sleep(0.02) # Give UI time to process
                    
                if terminated or truncated:
                    if info.get('reason') == 'reached_goal':
                        success_count += 1
                    break
                    
            if self.agent_type == "PPO":
                loss = self.agent.update()
            else:
                loss = loss / max(1, (steps // 10))
                
            success_rate = (success_count / episode) * 100
            self.update_plot_signal.emit(episode_reward, loss, success_rate)
            
            # Save periodic
            if episode % self.config['training'].get('save_freq', 100) == 0:
                os.makedirs("checkpoints", exist_ok=True)
                self.agent.save(f"checkpoints/{self.agent_type}_ep{episode}.pt")
                
        self.finished_signal.emit()

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Autonomous Navigation RL Simulation")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #121212; color: white;")
        
        # Load Config
        with open("configs/default_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
            
        self.worker = None
        
        self._init_ui()
        
    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # --- LEFT PANEL: CONTROLS ---
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        
        # Algorithm Group
        algo_group = QGroupBox("Algorithm")
        algo_group.setStyleSheet("color: white;")
        algo_layout = QFormLayout()
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["PPO", "SAC"])
        self.algo_combo.setStyleSheet("background-color: #333; color: white;")
        algo_layout.addRow("Agent:", self.algo_combo)
        algo_group.setLayout(algo_layout)
        control_layout.addWidget(algo_group)
        
        # Env Params
        env_group = QGroupBox("Environment")
        env_group.setStyleSheet("color: white;")
        env_layout = QFormLayout()
        
        self.rough_spin = QDoubleSpinBox()
        self.rough_spin.setRange(0.1, 2.0)
        self.rough_spin.setSingleStep(0.1)
        self.rough_spin.setValue(self.config['env']['terrain_roughness'])
        self.rough_spin.setStyleSheet("background-color: #333; color: white;")
        
        self.obs_spin = QSpinBox()
        self.obs_spin.setRange(0, 100)
        self.obs_spin.setValue(self.config['env']['num_obstacles'])
        self.obs_spin.setStyleSheet("background-color: #333; color: white;")
        
        env_layout.addRow("Roughness:", self.rough_spin)
        env_layout.addRow("Obstacles:", self.obs_spin)
        env_group.setLayout(env_layout)
        control_layout.addWidget(env_group)
        
        # Actions
        btn_style = "QPushButton { background-color: #0078D7; color: white; padding: 8px; border-radius: 4px; font-weight: bold; } QPushButton:hover { background-color: #005A9E; }"
        stop_style = "QPushButton { background-color: #D13438; color: white; padding: 8px; border-radius: 4px; font-weight: bold; } QPushButton:hover { background-color: #A80000; }"
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setStyleSheet(btn_style)
        self.start_btn.clicked.connect(self.start_training)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(stop_style)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        
        # --- RIGHT PANEL: VISUALIZATION ---
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #333; } QTabBar::tab { background: #222; color: white; padding: 8px; } QTabBar::tab:selected { background: #0078D7; }")
        
        self.map_widget = MapWidget()
        self.plot_widget = TrainingPlotWidget()
        
        self.tabs.addTab(self.map_widget, "Environment View")
        self.tabs.addTab(self.plot_widget, "Training Metrics")
        
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.tabs, stretch=1)
        
    def start_training(self):
        # Update config
        self.config['env']['terrain_roughness'] = self.rough_spin.value()
        self.config['env']['num_obstacles'] = self.obs_spin.value()
        
        agent_type = self.algo_combo.currentText()
        
        self.worker = TrainingWorker(self.config, agent_type)
        self.worker.update_plot_signal.connect(self.plot_widget.update_plots)
        self.worker.setup_env_signal.connect(self.map_widget.set_environment)
        self.worker.update_map_signal.connect(self.map_widget.update_state)
        self.worker.finished_signal.connect(self.training_finished)
        
        self.worker.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.algo_combo.setEnabled(False)
        self.tabs.setCurrentIndex(0) # Switch to map
        
    def stop_training(self):
        if self.worker:
            self.worker.stop()
            self.stop_btn.setText("Stopping...")
            self.stop_btn.setEnabled(False)
            
    def training_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.algo_combo.setEnabled(True)
        QMessageBox.information(self, "Finished", "Training stopped/finished.")
