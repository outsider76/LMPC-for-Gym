import math
import torch
import numpy as np



# Basic parameters
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5  # actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0
tau = 0.02  # seconds between state updates
kinematics_integrator = "euler"

theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4

def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def get_th(x, y):
    sign = np.sign(y)
    th = np.multiply(np.arccos(x), sign)
    return th

def dynamics(state, action):
    
    # Get states and actions
    x = state[:, 0].view(-1, 1)
    x_dot = state[:, 1].view(-1, 1)
    theta = state[:, 2].view(-1, 1)
    theta_dot = state[:, 3].view(-1, 1)

    force = action * force_mag
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (
        force + polemass_length * theta_dot**2 * sintheta
    ) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if kinematics_integrator == "euler":
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + tau * xacc
        x = x + tau * x_dot
        theta_dot = theta_dot + tau * thetaacc
        theta = theta + tau * theta_dot

    # Arrange for a new state
    state = torch.cat((x, x_dot, theta, theta_dot), dim=1)

    return state



def cost(state, action):
    theta = state[:, 2].view(-1, 1)

    cost = theta ** 2
    return cost