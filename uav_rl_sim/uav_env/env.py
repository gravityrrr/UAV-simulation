import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .terrain import TerrainGenerator

class UAVEnv(gym.Env):
    """
    Custom 2.5D UAV Environment compatible with Gymnasium.
    State space: 
        UAV Pos (x, y, z), Vel (vx, vy, vz)
        Relative Goal (dx, dy, dz)
        Local sensor array (flattened 5x5 grid of terrain height)
        Wind (wx, wy)
    Action space:
        Accels (ax, ay, az) continuous [-1, 1]
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
            
        self.grid_size = config.get("grid_size", 100)
        self.max_steps = config.get("max_steps", 1000)
        self.dt = config.get("dt", 0.1)
        self.uav_max_speed = config.get("uav_max_speed", 5.0)
        self.uav_max_accel = config.get("uav_max_accel", 2.0)
        self.sensor_range = config.get("sensor_range", 5) # 5x5 grid
        self.action_scale = config.get("action_scale", 1.0)
        self.wind_max = config.get("wind_max_speed", 2.0)
        
        # Reward configs
        self.goal_reward = config.get("goal_reward", 100.0)
        self.collision_penalty = config.get("collision_penalty", -100.0)
        self.oob_penalty = config.get("out_of_bounds_penalty", -50.0)
        self.step_penalty = config.get("step_penalty", -0.1)
        self.dist_reward_scale = config.get("distance_reward_scale", 1.0)
        
        self.terrain_gen = TerrainGenerator(config)
        
        # State: 3 (pos) + 3 (vel) + 3 (rel goal) + 2 (wind) + sensor_range^2 (sensor array)
        obs_dim = 11 + self.sensor_range**2
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Actions: ax, ay, az
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.heightmap = None
        self.obstacles = None
        self.wind = np.zeros(2)
        
        self.uav_pos = np.zeros(3)
        self.uav_vel = np.zeros(3)
        self.goal_pos = np.zeros(3)
        
        self.current_step = 0
        self.trajectory = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate new terrain for domain randomization
        self.heightmap, self.obstacles = self.terrain_gen.generate_synthetic(seed=seed)
        
        # Random wind
        self.wind = np.random.uniform(-self.wind_max, self.wind_max, size=2)
        
        # Spawn UAV at safe random location
        while True:
            sx = np.random.randint(10, self.grid_size - 10)
            sy = np.random.randint(10, self.grid_size - 10)
            if self.obstacles[sx, sy] == 0: # Not on an obstacle
                th = self.heightmap[sx, sy]
                self.uav_pos = np.array([float(sx), float(sy), th + 10.0]) # 10m above ground
                break
                
        # Spawn Goal at safe random location far enough
        while True:
            gx = np.random.randint(10, self.grid_size - 10)
            gy = np.random.randint(10, self.grid_size - 10)
            dist = np.linalg.norm(np.array([gx, gy]) - self.uav_pos[:2])
            if self.obstacles[gx, gy] == 0 and dist > self.grid_size * 0.4:
                th = self.heightmap[gx, gy]
                self.goal_pos = np.array([float(gx), float(gy), th + 10.0])
                break
                
        self.uav_vel = np.zeros(3)
        self.current_step = 0
        self.trajectory = [self.uav_pos.copy()]
        
        self.last_dist = np.linalg.norm(self.goal_pos - self.uav_pos)
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Rel goal
        rel_goal = self.goal_pos - self.uav_pos
        
        # Local terrain sensor
        sensor_half = self.sensor_range // 2
        sensor_data = np.zeros((self.sensor_range, self.sensor_range))
        
        px, py = int(np.round(self.uav_pos[0])), int(np.round(self.uav_pos[1]))
        
        for i in range(self.sensor_range):
            for j in range(self.sensor_range):
                gx = px + i - sensor_half
                gy = py + j - sensor_half
                
                # Check bounds
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    # Sensor reading is the max of terrain or obstacle height relative to UAV
                    t_height = max(self.heightmap[gx, gy], self.obstacles[gx, gy])
                    sensor_data[i, j] = t_height - self.uav_pos[2]
                else:
                    sensor_data[i, j] = 50.0 # Virtual walls
                    
        # Normalize observations to roughly [-1, 1] range to stabilize neural network
        flat_sensor = sensor_data.flatten() / 50.0
        norm_pos = self.uav_pos / float(self.grid_size)
        norm_vel = self.uav_vel / self.uav_max_speed
        norm_rel_goal = rel_goal / float(self.grid_size)
        norm_wind = self.wind / max(1.0, self.wind_max)
        
        obs = np.concatenate([
            norm_pos,
            norm_vel,
            norm_rel_goal,
            norm_wind,
            flat_sensor
        ]).astype(np.float32)
        
        return obs
        
    def step(self, action):
        self.current_step += 1
        
        # Apply actions
        accel = np.clip(action, -1.0, 1.0) * self.uav_max_accel
        
        # Wind perturbation on XY
        effective_accel = np.copy(accel)
        effective_accel[0] += self.wind[0] * 0.1
        effective_accel[1] += self.wind[1] * 0.1
        
        # Update physics
        self.uav_vel += effective_accel * self.dt
        
        # Apply air resistance / drag
        drag_coefficient = 0.5
        self.uav_vel -= self.uav_vel * drag_coefficient * self.dt
        
        # Speed limit
        speed = np.linalg.norm(self.uav_vel)
        if speed > self.uav_max_speed:
            self.uav_vel = (self.uav_vel / speed) * self.uav_max_speed
            
        self.uav_pos += self.uav_vel * self.dt
        self.trajectory.append(self.uav_pos.copy())
        
        # Determine reward and done flags
        reward = self.step_penalty
        terminated = False
        truncated = False
        info = {'reason': 'timeout'}
        
        # 1. Check Out of Bounds
        if not (0 <= self.uav_pos[0] < self.grid_size - 1 and 0 <= self.uav_pos[1] < self.grid_size - 1):
            reward += self.oob_penalty
            terminated = True
            info['reason'] = 'out_of_bounds'
            return self._get_obs(), reward, terminated, truncated, info
            
        # 2. Check Collision
        px, py = int(np.round(self.uav_pos[0])), int(np.round(self.uav_pos[1]))
        ground_height = max(self.heightmap[px, py], self.obstacles[px, py])
        if self.uav_pos[2] <= ground_height:
            reward += self.collision_penalty
            terminated = True
            info['reason'] = 'collision'
            return self._get_obs(), reward, terminated, truncated, info
            
        # 3. Check Ceiling
        if self.uav_pos[2] > 100.0:
            reward += self.oob_penalty
            terminated = True
            info['reason'] = 'ceiling_hit'
            return self._get_obs(), reward, terminated, truncated, info
            
        # 4. Check Goal
        dist_to_goal_2d = np.linalg.norm(self.goal_pos[:2] - self.uav_pos[:2])
        if dist_to_goal_2d < 3.0: # Reached goal radius
            reward += self.goal_reward
            terminated = True
            info['reason'] = 'reached_goal'
            return self._get_obs(), float(reward), terminated, truncated, info
            
        # Shaping: reward heavily for moving closer to goal, penalize for moving away
        dist_to_goal = np.linalg.norm(self.goal_pos - self.uav_pos)
        dist_diff = self.last_dist - dist_to_goal
        reward += dist_diff * 5.0  # Increased scaling so it prioritizes reaching target
        self.last_dist = dist_to_goal
        
        # Penalty for flying too close to ground
        clearance = self.uav_pos[2] - ground_height
        if clearance < 5.0:
            reward -= 0.5 * (5.0 - clearance) # Penalize proportional to closeness
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        info['reason'] = 'flying'
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        # We handle rendering in the PyQt6 GUI separately.
        pass
