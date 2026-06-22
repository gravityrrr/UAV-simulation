import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from .networks import ActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.get("learning_rate", 3e-4))
        
        self.gamma = config.get("gamma", 0.99)
        self.clip_param = config.get("ppo_clip_param", 0.2)
        self.ppo_epochs = config.get("ppo_epochs", 10)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        
        # Buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.policy.act(state)
            
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        
        return action.cpu().numpy().flatten()
        
    def store_reward(self, reward, is_terminal):
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        
    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = rewards.unsqueeze(1)
        
        old_states = torch.cat(self.states).detach()
        old_actions = torch.cat(self.actions).detach()
        old_log_probs = torch.cat(self.log_probs).detach()
        
        for _ in range(self.ppo_epochs):
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            advantages = rewards - state_values.detach()
            
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_param, 1+self.clip_param) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values, rewards)
            
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

        return loss.item()
        
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
        
    def load(self, filepath):
        if os.path.exists(filepath):
            self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
