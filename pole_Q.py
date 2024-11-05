import gym
import numpy as np

# 環境を作成
env = gym.make('CartPole-v1', render_mode='human')

# Qテーブルの初期化
state_bins = [10, 10, 10, 10]  # 各状態変数のビンの数
state_space_size = tuple(state_bins)
action_space_size = env.action_space.n
q_table = np.zeros(state_space_size + (action_space_size,))
print(q_table.shape)

# ハイパーパラメータ
alpha = 0.1  # 学習率
gamma = 0.99  # 割引率
epsilon = 1.0  # 探索率
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 1000

# 状態を離散化する関数
def discretize_state(state):
    bins = np.array(state_bins)
    state_adj = (state - env.observation_space.low) * bins / (env.observation_space.high - env.observation_space.low)
    return tuple(state_adj.astype(int))

# Q学習のメインループ
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    truncated = False
    total_reward = 0

    env.render()  # エピソードごとに1回だけレンダリング

    while not done and not truncated:
        # ε-グリーディ法でアクションを選択
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # ランダムなアクションを選択
        else:
            action = np.argmax(q_table[state])  # Q値が最大のアクションを選択

        next_state, reward, done, truncated, info = env.step(action)  # 環境にアクションを適用
        next_state = discretize_state(next_state)
        total_reward += reward

        # Q値の更新
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error

        state = next_state

    # εを減衰
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()  # 環境を閉じる
