import gymnasium as gym
from stable_baselines3 import PPO

# Load and run saved model for screen recording

env = gym.make("CartPole-v1", render_mode="human")

loaded_model = PPO.load("ppo_cartpole_model_0.003")

obs, info = env.reset()

for _ in range(1000):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        obs, info = env.reset()

env.close()