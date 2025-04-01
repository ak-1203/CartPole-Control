import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import pickle


from cartpole_q_train import state_tiler, default_q_value 

env = gym.make("CartPole-v1")

Q = defaultdict(default_q_value)
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)


# testing
test_episodes=500
test_rew=[]
test_steps=[]
for epiIndex in range(test_episodes):
    state,_=env.reset()
    state=state_tiler(state)
    total_rew=0
    total_steps=0
    done = False
    while(not done):
        action=np.argmax(Q[state])

        next_state,reward,done,truncated,_=env.step(action)
        total_rew+=1
        total_steps+=1
        next_state=state_tiler(next_state)
        state=next_state
        if done or truncated:
            break
    test_rew.append(total_rew)
    test_steps.append(total_steps)

plt.plot(test_rew)
plt.plot(pd.Series(test_rew).rolling(50).mean())
plt.title("Testing Progress")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

plt.plot(test_steps)
plt.plot(pd.Series(test_steps).rolling(50).mean())
plt.title("Testing Progress")
plt.xlabel("Episode")
plt.ylabel("Total Steps")
plt.show()
