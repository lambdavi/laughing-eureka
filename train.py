
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym
import torch

# define what ever reward function you want
def my_reward_function(state, action, next_state):
    return -np.mean(action)     # here we just return the negative mean of the action

def make_env():
    return gym.make("LocoMujoco", env_name="UnitreeA1.simple")

env = make_vec_env(make_env, n_envs=1)


"""model = PPO(
    policy="MlpPolicy", 
    batch_size=128, 
    n_epochs=5, 
    clip_range=0.2, 
    ent_coef=0.0035,
    target_kl=0.1, 
    env=env, 
    verbose=1,

)"""
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1,
    policy_kwargs=dict(
        net_arch=[512, 512, 512, 512], 
        activation_fn=torch.nn.ELU,
        n_critics=2,
    ),
    learning_rate=1e-4,
    learning_starts=1000,
    tensorboard_log="./logs/sac/"
)

model.learn(250_000, log_interval=100, progress_bar=True)
model.save("final_model")
