import os
import neat
import gym
import numpy as np
import visualize

env = gym.make('BipedalWalker-v3')

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        observation = env.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0
        done = False
        while not done:
            # Ensure observation[0] is an ndarray and flatten it
            if isinstance(observation[0], np.ndarray):
                observation_data = observation[0].flatten()
                if len(observation_data) != 24:
                    raise ValueError(f"Expected observation of length 24, got {len(observation_data)}")
            else:
                raise TypeError(f"Expected ndarray for observation_data, got {type(observation[0])}")

            action = net.activate(observation_data)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            fitness += reward
        genome.fitness = fitness

def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)

    print('\nBest genome:\n{!s}'.format(winner))

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
