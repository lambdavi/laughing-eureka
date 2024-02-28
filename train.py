import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env

env = gym.make("LocoMujoco", env_name="UnitreeA1.simple")

model = PPO(policy="MlpPolicy", batch_size=45, n_epochs=5,clip_range=0.2, ent_coef=0.0035,target_kl=0.1, env=env)
model.learn(1_000_000, log_interval=1, progress_bar=True)
model.save("final_model")
