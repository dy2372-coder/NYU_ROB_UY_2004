import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback


# ============================================================
# Assignment 4 - Question 2
# PPO Reinforcement Learning for CartPole-v1
# ============================================================

# Change this variable depending on what you want to run:
# "train" = train one model
# "run_best" = load and visualize the best trained model
MODE = "run_best"


# ============================================================
# Training setup
# ============================================================

learning_rate_value = 0.003   # Best learning rate based on my results

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 20000,
    "env_name": "CartPole-v1",
    "learning_rate": learning_rate_value,
}


if MODE == "train":
    env = gym.make("CartPole-v1", render_mode="human")

    run = wandb.init(
        project="assignment_04a",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate_value,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    model.save("ppo_cartpole_model_0.003")

    run.finish()
    env.close()


# ============================================================
# Load and run the best saved model
# This section corresponds to parts i-o of the assignment.
# ============================================================

if MODE == "run_best":
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


# ============================================================
# Question 2p Answer
# ============================================================

# I tested five learning rates: 0.00003, 0.0003, 0.003, 0.03, and 0.3.
# I evaluated the results by comparing the W&B train/loss plots, the final
# rollout/ep_rew_mean values, the final rollout/ep_len_mean values, and the
# visual stability of the CartPole model during deployment.
#
# Results:
# 0.00003: final reward = 53.55, final episode length = 53.55, train/loss ≈ 56.1
# 0.0003: final reward = 145, final episode length = 145, train/loss = 1.04
# 0.003: final reward = 166, final episode length = 166, train/loss = 1.7
# 0.03: final reward = 162, final episode length = 162, train/loss = 1.85
# 0.3: final reward = 9.22, final episode length = 9.22, train/loss = 7.34
#
# The best learning rate was 0.003 because it produced the highest final
# mean reward and highest final mean episode length. Since the CartPole
# environment gives +1 reward for every timestep the pole remains balanced,
# the higher reward means that this model kept the pole balanced for longer.
# The 0.03 learning rate was close, but 0.003 performed slightly better.
# The 0.3 learning rate performed poorly, which suggests that the learning
# rate was too large for stable learning.