import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym

# create the environment and task
def my_reward_function(state, action, next_state):
    return -np.mean(action)


env = gym.make("LocoMujoco", env_name="UnitreeA1.simple")
# get the dataset for the chosen environment and task

action_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

print(action_dim)
print(obs_dim)

env.reset()
env.render()
terminated = False
i = 0

while True:
    if i == 1000 or terminated:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, terminated, truncated, info = env.step(action)

    env.render()
    i += 1