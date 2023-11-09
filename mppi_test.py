import time
import gymnasium as gym
import imageio
import numpy as np
import torch
import logging
import math
from mppi.mppi import MPPI
from mppi import mppi
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
    N_SAMPLES = 1000  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    d = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.double

    noise_sigma = torch.tensor(10, device=d, dtype=dtype)
    # noise_sigma = torch.tensor([[10, 0], [0, 10]], device=d, dtype=dtype)
    lambda_ = 1.



    def train(new_data):
        pass


    env = gym.make(ENV_NAME, render_mode="rgb_array")  # bypass the default TimeLimit wrapper
    state, info = env.reset()


    # Hopefully, the controller could be produced through
    # mppi_pendulum = MPPI(env_name,
                            # noise_sigma, 
                            # num_samples=N_SAMPLES, 
                            # horizon=TIMESTEPS,
                            # lambda_=lambda_)

    mppi_gym = MPPI(ENV_NAME, 
                        noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                        lambda_=lambda_)













    def run_mppi(mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True):

        images = []
        img = env.render()
        images.append(img)

        state, info = env.reset()

        dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d)
        total_reward = 0
        for i in range(iter):
            command_start = time.perf_counter()
            action = mppi.command(state)
            elapsed = time.perf_counter() - command_start
            state, r, _, _, _ = env.step(action.cpu().numpy())
            total_reward += r
            logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
            
            img = env.render()
            images.append(img)

            di = i % retrain_after_iter
            if di == 0 and i > 0:
                retrain_dynamics(dataset)
                # don't have to clear dataset since it'll be overridden, but useful for debugging
                dataset.zero_()
            dataset[di, :mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
            dataset[di, mppi.nx:] = action
        
        imageio.mimsave('SAC.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
        return total_reward, dataset
    
    total_reward = run_mppi(mppi_gym, env, train)

    logger.info("Total reward %f", total_reward)