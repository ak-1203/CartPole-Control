import gymnasium as gym
from a2c_model import ActorCriticAgent
import matplotlib.pyplot as plt  

env = gym.make("CartPole-v1")
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

agent=ActorCriticAgent(state_dim,action_dim)
num_episodes=500
max_steps=200
episode_rewards=[]
for episode in range(num_episodes):
    state=env.reset()[0]
    done=False
    step_count = 0
    total_reward=0
    while(not done and step_count < max_steps):
        action,action_prob=agent.select_actions(state)
        next_state,reward,done,_,_ = env.step(action)
        agent.update(state,action_prob,reward,next_state,done)
        state=next_state
        total_reward+=reward
        step_count += 1
    print(f"Episode {episode + 1}: Total Reward: {total_reward}")    
    episode_rewards.append(total_reward)

plt.title("A2C (cartpole)")
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.plot(episode_rewards)
plt.show()

#Testing

import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
return_history = [] 
epi_num=250
for episode in range(epi_num):
    state, _ = env.reset()  
    done = False
    total_reward = 0

    while not done:  
        action,_ = agent.select_actions(state) 
        state, reward, done, _, _ = env.step(action)  
        total_reward += reward

    return_history.append(total_reward)  

env.close() 


plt.figure(figsize=(12,6))
plt.title(f"Testing for {epi_num} Episodes")
plt.xlabel("Episode")
plt.ylabel("rewards")
plt.plot(return_history)
plt.show()

print("Final Score:", total_reward)  # Print the final reward of the last episode

plt.show()


##Final render

env = gym.make("CartPole-v1",render_mode="human")
total_reward=0
done=False
env.reset()
while not done:
        env.render()  
        action,_ = agent.select_actions(state)  
        state, reward, done, _, _ = env.step(action)  
        total_reward += reward

env.close()