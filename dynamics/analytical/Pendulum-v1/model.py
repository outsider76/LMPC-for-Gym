import math
import torch
import numpy as np

#TODO: Make it compatiable using cuda and cou devices


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def get_th(x, y):
    sign = torch.sign(y)
    th = torch.multiply(torch.arccos(x), sign)
    return th

def dynamics(state, perturbed_action):
    
    # Basic parameters
    g = 10
    m = 1
    l = 1
    dt = 0.05

    # Get states and actions
    x = state[:, 0].view(-1, 1)
    y = state[:, 1].view(-1, 1)
    th = get_th(x, y)
    thdot = state[:, 2].view(-1, 1)

    u = perturbed_action
    u = torch.clamp(u, -2, 2)

    # Do one step
    newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3. / (m * l ** 2) * u) * dt
    newth = th + newthdot * dt
    newx = torch.cos(newth)
    newy = torch.sin(newth)
    newthdot = torch.clamp(newthdot, -8, 8)

    # Arrange for a new state
    state = torch.cat((newx, newy, newthdot), dim=1)

    return state



def cost(state, action):
    x = state[:, 0]
    y = state[:, 1]
    th = get_th(x, y)
    thdot = state[:, 2]

    action = action[:, 0]
    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (action**2)
    return cost