import gym

# MountainCar-v0 環境を作成
env = gym.make('MountainCar-v0', render_mode='human')

# 環境をリセット
state = env.reset()[0]  # stateをタプルから取得

done = False
while not done:
    env.render()  # 環境をレンダリング
    action = env.action_space.sample()  # ランダムに行動を選択
    next_state, reward, done, truncated, info = env.step(action)  # 行動を実行して次の状態を取得
    state = next_state  # 状態を更新
    done = done or truncated  # どちらかがTrueならエピソード終了

env.close()  # 環境を閉じる