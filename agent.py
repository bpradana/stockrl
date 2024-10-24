import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim


class DQNAgent:
    def __init__(
        self,
        q_network,
        target_network,
        replay_buffer,
        epsilon_strategy,
        gamma=0.99,
        lr=1e-3,
        device="auto",
    ):
        self.device = self._get_device(device)
        self.q_network = q_network.to(self.device)
        self.target_network = target_network.to(self.device)
        self.replay_buffer = replay_buffer
        self.epsilon_strategy = epsilon_strategy
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor for future rewards
        self.current_step = 0

    def select_action(self, state, epsilon):
        # Convert state to tensor and pass through the Q-network
        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        # Select action based on epsilon-greedy strategy
        if random.random() < epsilon:
            return random.choice([0, 1, 2])  # Random action
        else:
            # Predict Q-values from the Q-network
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()  # Choose action with the highest Q-value

    def update(self, batch_size):
        # Sample a batch of experiences from the replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.stack(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32).to(
            self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q-values from the Q-network
        q_values = self.q_network(states).gather(1, actions)

        # Target Q-values from the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss (Mean Squared Error)
        loss = F.mse_loss(q_values, target_q_values)

        # Backpropagation to update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()

        return loss.item()

    def _get_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
