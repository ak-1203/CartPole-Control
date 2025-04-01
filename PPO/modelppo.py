import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import os

log_dir = "./ppo_cartpole_logs/"
os.makedirs(log_dir, exist_ok=True)
model_path = os.path.join(log_dir, "best_model")

env = make_vec_env("CartPole-v1", n_envs=4, monitor_dir=log_dir)

eval_env = make_vec_env("CartPole-v1", n_envs=1, monitor_dir=os.path.join(log_dir, "eval"))

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    tensorboard_log="./ppo_cartpole_tensorboard/",
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=495, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=stop_callback,
    eval_freq=2000,
    n_eval_episodes=10,
    best_model_save_path=model_path,
    log_path=log_dir,
    verbose=1,
)

model.learn(
    total_timesteps=100_000,
    callback=eval_callback,
    tb_log_name="ppo_cartpole_run"
)

final_model_path = os.path.join(log_dir, "final_model")
model.save(final_model_path)
print(f"Model saved to {final_model_path}")

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_training_results(log_folder, title='Learning Curve'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    
    x = x[len(x) - len(y):]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Rewards')
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    return fig, ax

plot_training_results(log_dir, title='PPO CartPole Training Rewards')
plt.show()

# ===== Testing Phase =====
# Load the best model for testing
print("Loading best model for testing...")
if os.path.exists(os.path.join(model_path, "best_model.zip")):
    best_model = PPO.load(os.path.join(model_path, "best_model"))
else:
    print("Best model not found, using final model")
    best_model = PPO.load(final_model_path)

# Test the model
test_env = gym.make("CartPole-v1")
episode_rewards = []
num_test_episodes = 50

for episode in range(num_test_episodes):
    obs = test_env.reset()
    done = False
    total_reward = 0
    
    while not done:

        if isinstance(obs, tuple):
            obs = obs[0]
    
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
            
        action, _ = best_model.predict(obs, deterministic=True)

        if hasattr(action, "__len__"):
            action = action[0]
            
        obs, reward, done, _ ,_ = test_env.step(action)

        if isinstance(done, tuple):
            done = done[0]
        
        total_reward += reward

    
    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

test_env.close()