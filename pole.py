import gym

# 環境を作成
env = gym.make('CartPole-v1', render_mode='human')

# 環境をリセット
state = env.reset()

for _ in range(1000):
    env.render()  # 環境を描画
    action = env.action_space.sample()  # ランダムなアクションを選択
    state, reward, done, truncated, info = env.step(action)  # 環境にアクションを適用
    print(state, reward, done, truncated, info)

    if done or truncated:
        state = env.reset()  # エピソードが終了したらリセット

env.close()  # 環境を閉じる
