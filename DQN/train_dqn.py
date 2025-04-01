import gymnasium as gym

from dqn_model import DQN_agent

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQN_agent(state_dim, action_dim)

episodes = 500
max_steps = 200
target_update_freq = 10
reward_history = []

for episode in range(episodes):
    state, _ = env.reset() 
    total_reward = 0

    for t in range(max_steps):
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  

        agent.memory.store(state, action, reward, next_state, done)

        agent.update()

        state = next_state
        total_reward += reward

        if done:
            break

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

    if episode % target_update_freq == 0:
        agent.update_target_network()

    reward_history.append(total_reward)
    print(f"Episode {episode}: Reward = {total_reward}")

env.close()


##Testing 

import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
return_history = []
epi_num=500
for episode in range(epi_num):
    state, _ = env.reset()  
    done = False
    total_reward = 0

    while not done:
        env.render()  
        action = agent.select_action(state)  
        state, reward, done, _, _ = env.step(action)  
        total_reward += reward

    return_history.append(total_reward)  

env.close() 

plt.title(f"Testing for {epi_num} Episodes")
plt.xlabel("Episode")
plt.ylabel("rewards")
plt.plot(return_history)
plt.show()

print("Final Score:", total_reward)  # Print the final reward of the last episode

plt.show()


##render after trained  

env = gym.make("CartPole-v1",render_mode='human')
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    env.render()
    action = agent.select_action(state)  # Use trained policy
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

env.close()
print("Final Score:", total_reward)