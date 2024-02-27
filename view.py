import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

#env = make_vec_env((lambda: gym.make("LocoMujoco", env_name="UnitreeA1.simple")), n_envs=2)
env = gym.make("LocoMujoco", env_name="UnitreeA1.simple")

model = PPO(policy="MlpPolicy", env=env, verbose=0)
model.learn(10_000_000, log_interval=1, progress_bar=True)

action_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

print(action_dim)
print(obs_dim)

nstate, _ = env.reset()
env.render()
terminated = False
i = 0

while True:
    if i == 1000 or terminated:
        nstate, _ = env.reset()
        i = 0
    action = model.predict(observation=nstate)[0]
    nstate, reward, terminated, truncated, info = env.step(action)

    env.render()
    i += 1