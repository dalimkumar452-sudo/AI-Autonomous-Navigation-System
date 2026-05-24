import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class ProNavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    
    def __init__(self):
        super().__init__()
        self.width, self.height = 800, 600
        
        # Action Space (3 discrete moves): 0=Forward, 1=Turn Left, 2=Turn Right
        self.action_space = spaces.Discrete(3)
        
        # Obs Space: [x, y, angle, goal_x, goal_y, dist_to_goal, 5 LiDAR sensor readings]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        
        self.robot_pos = [50.0, 50.0]
        self.robot_angle = 0.0
        self.goal_pos = [700.0, 500.0]
        self.path_history = []
        
        # Static Obstacles (Pro Level Feature: Added Wall blocks like screenshot)
        self.obstacles = [
            pygame.Rect(150, 150, 40, 300), # Left Wall
            pygame.Rect(500, 150, 40, 300), # Right Wall
            pygame.Rect(300, 450, 200, 40), # Bottom Barrier
        ]
        
        # Pygame setup
        self.window = None
        self.clock = None
        self.font = None
        self.current_reward = 0
        self.lidar_distances = [200] * 5 # Max sensor range is 200
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = [50.0, 50.0]
        self.robot_angle = 0.0
        # Reset to a random goal dynamically to ensure AI learns navigation, not just a path
        self.goal_pos = [np.random.randint(650, 750), np.random.randint(100, 550)]
        self.path_history = []
        self.current_reward = 0
        return self._get_obs(), {}
        
    def _cast_ray(self, angle):
        max_dist = 200
        x, y = self.robot_pos
        for d in range(1, max_dist, 4):
            rx = x + d * math.cos(angle)
            ry = y + d * math.sin(angle)
            # Wall bounds check
            if rx < 0 or rx > self.width or ry < 0 or ry > self.height:
                return d
            # Obstacle check
            for obs in self.obstacles:
                if obs.collidepoint(rx, ry):
                    return d
        return max_dist

    def _get_lidar(self):
        angles = [-np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4] # 5 Sensor Beams
        distances = []
        for a in angles:
            ray_angle = self.robot_angle + a
            dist = self._cast_ray(ray_angle)
            distances.append(dist)
        self.lidar_distances = distances
        return distances

    def _get_obs(self):
        dist_goal = math.hypot(self.goal_pos[0] - self.robot_pos[0], self.goal_pos[1] - self.robot_pos[1])
        lidars = self._get_lidar()
        obs = [
            self.robot_pos[0], self.robot_pos[1], self.robot_angle,
            self.goal_pos[0], self.goal_pos[1], dist_goal
        ] + lidars
        return np.array(obs, dtype=np.float32)
        
    def step(self, action):
        speed = 6.0
        
        if action == 1:
            self.robot_angle -= 0.2
        elif action == 2:
            self.robot_angle += 0.2
            
        self.robot_pos[0] += speed * math.cos(self.robot_angle)
        self.robot_pos[1] += speed * math.sin(self.robot_angle)
        
        # Save path history for pro visualization
        self.path_history.append((self.robot_pos[0], self.robot_pos[1]))
        if len(self.path_history) > 100:
            self.path_history.pop(0)
            
        dist_goal = math.hypot(self.goal_pos[0] - self.robot_pos[0], self.goal_pos[1] - self.robot_pos[1])
        lidars = self._get_lidar()
        
        reward = -0.1 # Time penalty
        terminated = False
        
        # Win Condition: Reached the goal
        if dist_goal < 25:
            reward = 100.0
            terminated = True
            
        # Lose Condition 1: Screen Boundary Collision
        if (self.robot_pos[0] < 0 or self.robot_pos[0] > self.width or 
            self.robot_pos[1] < 0 or self.robot_pos[1] > self.height):
            reward = -50.0
            terminated = True
            
        # Lose Condition 2: Obstacle Collision (Crashed into Wall)
        for obs in self.obstacles:
            if obs.collidepoint(self.robot_pos[0], self.robot_pos[1]):
                reward = -50.0
                terminated = True
        
        # Reward Shaper: Penalize getting too close to walls dynamically
        if min(lidars) < 30:
            reward -= 1.0
            
        self.current_reward = reward
        return self._get_obs(), reward, terminated, False, {}
        
    def render(self):
        if self.window is None:
            pygame.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AI Autonomous Navigation System (Pro Version - LiDAR Active)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Consolas", 18)
            
        pygame.event.pump()
        # Dark Theme Background (Indie look)
        self.window.fill((22, 26, 38)) 
        
        # Draw Obstacles (Walls in Red, like screenshot blocks)
        for obs in self.obstacles:
            pygame.draw.rect(self.window, (231, 76, 60), obs)
            
        # Draw Goal with Ring visual
        pygame.draw.circle(self.window, (46, 204, 113), (int(self.goal_pos[0]), int(self.goal_pos[1])), 15)
        pygame.draw.circle(self.window, (46, 204, 113), (int(self.goal_pos[0]), int(self.goal_pos[1])), 25, 2)
        
        # Draw LiDAR Sensor Beams (Yellow Laser Beams - like pro-simulation)
        angles = [-np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4]
        for i, a in enumerate(angles):
            ray_angle = self.robot_angle + a
            dist = self.lidar_distances[i]
            end_x = self.robot_pos[0] + dist * math.cos(ray_angle)
            end_y = self.robot_pos[1] + dist * math.sin(ray_angle)
            pygame.draw.line(self.window, (241, 196, 15), self.robot_pos, (end_x, end_y), 1)
        
        # Draw Robot with Trailing Line (Path History)
        if len(self.path_history) > 1:
            pygame.draw.lines(self.window, (41, 128, 185), False, self.path_history, 2)
        pygame.draw.circle(self.window, (52, 152, 219), (int(self.robot_pos[0]), int(self.robot_pos[1])), 15)
        
        # TELEMETRY DASHBOARD OVERLAY (Industry Requirement)
        dist_val = math.hypot(self.goal_pos[0] - self.robot_pos[0], self.goal_pos[1] - self.robot_pos[1])
        
        text1 = self.font.render("TELEMETRY DASHBOARD", True, (0, 255, 255))
        text2 = self.font.render(f"Goal Dist : {dist_val:.1f}m", True, (255, 255, 255))
        text3 = self.font.render(f"Reward    : {self.current_reward:.1f}", True, (255, 255, 255))
        text4 = self.font.render(f"Sensors   : ONLINE", True, (46, 204, 113))
        
        self.window.blit(text1, (10, 10))
        self.window.blit(text2, (10, 30))
        self.window.blit(text3, (10, 50))
        self.window.blit(text4, (10, 70))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        if self.window is not None:
            pygame.quit()