import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Qネットワークの定義
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# エージェントの定義
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # 割引率
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).detach().clone()
            
            # 修正部分: target_fの形状を[action_size]に合わせて明示的に設定
            target_f = target_f.view(1, -1)  # (1, action_size) の形状にする
            target_f[0, action] = target  # アクションに対応したターゲットを代入
            
            print(f"state: {state}, action: {action}, reward: {reward}, next_state: {next_state}, done: {done}")
            print(f"target: {target}, target_f: {target_f}")
            
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            print(f"loss: {loss.item()}")
            loss.backward()
            self.optimizer.step()
        
        # 探索率を減少
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# 環境の設定
env = gym.make('MountainCar-v0', render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32
EPISODES = 1000

# メインループ
for e in range(EPISODES):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # 状態のみを取得
    state = np.reshape(state, [1, state_size])
    for time in range(5000):
        env.render()  # 環境を表示
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        reward = reward+abs(next_state[0])+next_state[0] if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done or truncated:
            print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
env.close()