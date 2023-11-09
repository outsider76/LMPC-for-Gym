import os
import sys

import gymnasium as gym

# "Pendulum-v1"

def search4dynamics(env_name):
    dynamics_path = "./dynamics/analytical/"
    model_path = dynamics_path + env_name
    # Analytical dynamics for this env does not exist
    if not os.path.isdir(model_path):
        print("No Analytical Model!")
    else:
        sys.path.append(model_path)
        import model
        print("Analytical Model Loaded!")
        return model.dynamics, model.cost

# Return the corresponding dimension of state space and action space
def env_info(env_name):
    test_gym = gym.make(env_name)
    info = {}
    # Identify the exact class
    observation_space = test_gym.observation_space
    action_space = test_gym.action_space

    #TODO: Only for box now
    info["observation_dim"] = observation_space.shape[0]
    info["action_dim"] = action_space.shape[0]

    return info

info = env_info("Pendulum-v1")

print(info["observation_dim"])