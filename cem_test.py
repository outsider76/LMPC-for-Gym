import gymnasium as gym
import numpy as np
import torch
import logging
import math
from cem import cem
from gymnasium import wrappers, logger as gym_log

from common import utils



gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 15  # T
    N_ELITES = 20
    N_SAMPLES = 500  # K
    SAMPLE_ITER = 3  # M
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double


    dynamics, running_cost = utils.search4dynamics(ENV_NAME)





    def train(new_data):
        pass


    downward_start = True
    env = gym.make(ENV_NAME, render_mode="rgb_array")  # bypass the default TimeLimit wrapper
    env.reset()


    env.reset()

    nx = 3
    nu = 1
    cem_gym = cem.CEM(dynamics, running_cost, nx, nu, num_samples=N_SAMPLES, num_iterations=SAMPLE_ITER,
                      horizon=TIMESTEPS, device=d, num_elite=N_ELITES,
                      u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=d))
    total_reward, data = cem.run_cem(cem_gym, env, train)
    logger.info("Total reward %f", total_reward)