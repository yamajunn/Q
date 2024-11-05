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
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQNエージェントの定義
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 割引率: 将来の報酬の現在価値を計算するためのパラメータ
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 探索率の最小値
        self.epsilon_decay = 0.9999  # 探索率の減衰率
        self.learning_rate = 0.001  # 学習率
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
        return torch.argmax(act_values, dim=1).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 環境を作成
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32
EPISODES = 1000

for e in range(EPISODES):
    state = env.reset()[0]  # stateをタプルから取得
    total_reward = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        reward = reward - abs(next_state[0]) - abs(next_state[2]) if (not done) and (not truncated) else -30
        agent.remember(state, action, reward, next_state, done or truncated)
        state = next_state
        total_reward += reward
        if len(agent.memory) > batch_size and time % 10 == 0:  # 10ステップごとにバッチ学習を行う
            agent.replay(batch_size)
            agent.replay(batch_size)
        if done or truncated:
            print(f"episode: {e}/{EPISODES}, score: {total_reward:.2f}, e: {agent.epsilon:.2}")
            break
env.close()

# 学習が完了したら一つの環境でテスト
test_env = gym.make('CartPole-v1', render_mode='human')
state = test_env.reset()[0]
done = False

for _ in range(1000):
    test_env.render()  # テスト中にレンダリング
    action = agent.act(state)  # actionを整数に変換
    next_state, reward, done, truncated, _ = test_env.step(action)
    state = next_state
    if done or truncated:
        test_env.reset()  # エピソードが終了したらリセット

test_env.close()
