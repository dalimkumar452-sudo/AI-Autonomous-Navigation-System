import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class GridNavEnv(gym.Env):
    # Set FPS to 10 so we can clearly see it moving block by block
    metadata = {"render_modes": ["human"], "render_fps": 10} 
    
    def __init__(self):
        super().__init__()
        self.grid_size = 15   # 15x15 Grid
        self.cell_size = 40   # Each block is 40 pixels
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size
        
        # Action Space: 4 Discrete moves -> 0:Up, 1:Down, 2:Left, 3:Right
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: [agent_x, agent_y, goal_x, goal_y]
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1, shape=(4,), dtype=np.float32
        )
        
        # Static Obstacles (Black Blocks from your screenshot)
        # Format: (x_column, y_row)
        self.obstacles = [
            (3, 4), (3, 5), (3, 6),  # Vertical block left
            (10, 8), (11, 8),        # Horizontal block middle
            (7, 12), (8, 12)         # Horizontal block bottom
        ]
        
        self.window = None
        self.clock = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [1, 1]   # Starting position (Top Left)
        self.goal_pos = [13, 13]  # Goal position (Bottom Right)
        return self._get_obs(), {}
        
    def _get_obs(self):
        return np.array([
            self.agent_pos[0], self.agent_pos[1], 
            self.goal_pos[0], self.goal_pos[1]
        ], dtype=np.float32)
        
    def step(self, action):
        # Calculate intended new position
        new_pos = list(self.agent_pos)
        if action == 0: new_pos[1] -= 1   # Up
        elif action == 1: new_pos[1] += 1 # Down
        elif action == 2: new_pos[0] -= 1 # Left
        elif action == 3: new_pos[0] += 1 # Right
        
        reward = -1.0 # Standard step penalty to encourage fastest route
        terminated = False
        
        # Check if hitting the outer boundary
        if (new_pos[0] < 0 or new_pos[0] >= self.grid_size or 
            new_pos[1] < 0 or new_pos[1] >= self.grid_size):
            reward = -50.0
            terminated = True
            
        # Check if hitting a black obstacle
        elif tuple(new_pos) in self.obstacles:
            reward = -50.0
            terminated = True
            
        else:
            # Move is valid
            self.agent_pos = new_pos
            
        # Check if reached the Red Goal
        if self.agent_pos == self.goal_pos:
            reward = 100.0
            terminated = True
            
        return self._get_obs(), reward, terminated, False, {}
        
    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AI Grid Navigation System")
            self.clock = pygame.time.Clock()
            
        pygame.event.pump()
        # White Background
        self.window.fill((255, 255, 255)) 
        
        # Draw Light Gray Grid Lines
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.window, (220, 220, 220), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.window, (220, 220, 220), (0, y), (self.width, y))
            
        # Draw Black Obstacles
        for obs in self.obstacles:
            rect = pygame.Rect(obs[0] * self.cell_size, obs[1] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.window, (0, 0, 0), rect)
            
        # Draw Red Goal
        goal_rect = pygame.Rect(self.goal_pos[0] * self.cell_size, self.goal_pos[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.window, (231, 76, 60), goal_rect)
        
        # Draw Green Agent
        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.window, (46, 204, 113), agent_rect)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        if self.window is not None:
            pygame.quit()