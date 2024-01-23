import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Brain:
    def __init__(self,
                 model,
                 action_size=3,
                 memory_size=2000,
                 gamma=0.9,
                 learning_rate=0.001,
                 epsilon=1,
                 epsilon_decay=0.99,
                 epsilon_min=0.01):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.memory = []
        self.memory_size = memory_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = action_size

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[1:]

    def train(self, batch_size=1, update_epsilon=False):
        if batch_size == 1:
            batch = [self.memory[-1]]
        else:
            batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        self.__train(batch, update_epsilon)

    def __train(self, batch, update_epsilon=True):
        states = torch.tensor(np.array([tup[0] for tup in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([tup[1] for tup in batch]), dtype=torch.long)
        rewards = torch.tensor(np.array([tup[2] for tup in batch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([tup[3] for tup in batch]), dtype=torch.float32)
        done = torch.tensor(np.array([tup[4] for tup in batch]), dtype=torch.bool)

        target = rewards + self.gamma * torch.max(self.model(next_states), dim=1).values
        target[done] = rewards[done]

        prediction = self.model(states)
        target_f = prediction.clone()
        target_f[range(len(batch)), actions] = target

        loss = self.loss(prediction, target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if len(batch) != 1 and self.epsilon > self.epsilon_min and update_epsilon:
            self.epsilon = max(self.epsilon_decay * self.epsilon, self.epsilon_min)

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return torch.argmax(self.model(state)).item()

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
