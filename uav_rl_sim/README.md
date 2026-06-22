# UAV Autonomous Navigation RL Simulation

This is a complete, offline-capable Python research project for simulation-based UAV autonomous navigation in complex terrain using reinforcement learning.

It features a custom Gymnasium-compatible environment simulating 2.5D terrain heightmaps, obstacles, and wind, alongside custom PyTorch implementations of Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC). A high-performance PyQt6 GUI provides real-time visualization and control.

## Project Structure
- `main.py`: Entry point for the GUI application.
- `uav_env/`: Custom Gymnasium environment and terrain generation (Perlin noise and obstacle placement).
- `agents/`: PyTorch implementations of PPO, SAC, Actor-Critic networks, and a Replay Buffer.
- `gui/`: PyQt6 user interface with Matplotlib plots and QGraphicsScene 2D map views.
- `configs/`: YAML configuration files for hyperparameters.
- `assets/`: Image textures for rendering the environment beautifully.

## Setup Instructions (Offline Capable)

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Create Virtual Environment
Open your terminal and run:
```bash
python -m venv venv
```

### 3. Activate Virtual Environment
- **Windows:**
```bash
.\venv\Scripts\activate
```
- **Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
Since this project is offline-capable, once these are installed, you will not need internet access to run the training or simulation.
```bash
pip install -r requirements.txt
```

*(Note: If you are on a fully offline machine, you can download these packages as wheels on another computer using `pip download -r requirements.txt`, transfer the folder via USB, and install them using `pip install --no-index --find-links /path/to/folder -r requirements.txt`)*

### 5. Run the Application
```bash
python main.py
```

## Using the Application
1. **Configure Environment:** In the left panel, adjust the terrain roughness and obstacle density.
2. **Select Algorithm:** Choose between PPO (on-policy) and SAC (off-policy).
3. **Start Training:** Click "Start Training". The map view on the right will show a highly detailed 2.5D overhead view of the UAV navigating the generated terrain.
4. **Monitor Metrics:** Switch to the "Training Metrics" tab to view real-time plots of Episode Reward, Training Loss, and Success Rate.
5. **Real Data:** To use natural terrain data instead of synthetic, drop `.png` heightmap images or `.npy` arrays into the `data/` folder and modify `data.mode` in `configs/default_config.yaml`. (You can run `python download_sample_data.py` to create a mock natural dataset).

## Safety Disclaimer
This project is for simulation and research purposes only. It does not contain code for real-world drone flight controllers, arming, or deployment.
