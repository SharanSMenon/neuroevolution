"""
An implementation of OpenAI's Evolition Strategies algorithm in PyTorch. 
The algorithm is designed to train a neural network to solve BipedalWalker-v3. 
Run this on a CPU cluster
"""

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.multiprocessing import Pool
from tqdm import tqdm

SIGMA = 0.01
LR = 0.001
POPULATION_SIZE=50
ITERATIONS = 1000


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


def fitness_func(solution, model, env):
    with torch.no_grad():
        nn.utils.vector_to_parameters(solution.float(), model.parameters())
        episode_reward = 0
        state = env.reset()
        for i in range(10000):
            action = model(torch.FloatTensor(state)).detach().numpy()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break

        return episode_reward


def jitter(mother_params, state_dict):
    params_try = mother_params + SIGMA*state_dict
    return params_try


def calc_population_fitness(pop, mother_params, model, env):
    torch.multiprocessing.freeze_support()
    fitness = POOL.starmap(fitness_func, [(jitter(
        mother_params, param), model, env) for param in pop])
    return torch.tensor(fitness)


if __name__ == '__main__':
    POOL = Pool(8)
    env = gym.make("BipedalWalker-v3")
    with torch.no_grad():
        model = nn.Sequential(*[
            nn.Linear(24, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 4),
            nn.Tanh()
        ])
        model.apply(weights_init)
        mother_params = model.parameters()
        mother_vector = nn.utils.parameters_to_vector(mother_params)
        n_params = nn.utils.parameters_to_vector(model.parameters()).shape[0]
        print(f"Number of parameters: {n_params}")  # Number of parameters: 6020
        for iteration in tqdm(range(ITERATIONS)):
            pop = torch.from_numpy(np.random.randn(POPULATION_SIZE, n_params))
            fitness = calc_population_fitness(pop, mother_vector, model, env)
            normalized_fitness = (fitness - torch.mean(fitness)) / torch.std(fitness)
            mother_vector = mother_vector + (LR / (POPULATION_SIZE * SIGMA)) * torch.from_numpy(np.dot(pop.t().numpy(), normalized_fitness.numpy()))
            if iteration % 50 == 0:
                reward = fitness_func(mother_vector, model, env)
                print(f"Iteration: {iteration}, Reward: {reward}")
        torch.save(model.state_dict(), "bipedal_walker.pt")


